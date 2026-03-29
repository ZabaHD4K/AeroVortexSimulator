#include "lbm2d.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA] %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    } \
} while(0)

// ── D2Q9 constants ──────────────────────────────────
// Directions: 0=center, 1=E, 2=N, 3=W, 4=S, 5=NE, 6=NW, 7=SW, 8=SE
__constant__ int cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
__constant__ int cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };
__constant__ float w[9] = {
    4.0f/9.0f,
    1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};
// Opposite direction index for bounce-back
__constant__ int opp[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };

// ── Equilibrium distribution ────────────────────────
__device__ float feq(int i, float rho, float ux, float uy) {
    float cu = (float)cx[i] * ux + (float)cy[i] * uy;
    float usq = ux * ux + uy * uy;
    return w[i] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * usq);
}

// ── Collision + Streaming kernel (fused) ────────────
__global__ void lbmCollideStream(
    const float* __restrict__ f,
    float* __restrict__ fNew,
    float* __restrict__ ux_out,
    float* __restrict__ uy_out,
    float* __restrict__ rho_out,
    int nx, int ny, float omega, float lidU)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    int idx = y * nx + x;

    // ── Compute macroscopic quantities ──────────
    float rho = 0.0f, uxx = 0.0f, uyy = 0.0f;
    for (int i = 0; i < 9; i++) {
        float fi = f[i * nx * ny + idx];
        rho += fi;
        uxx += (float)cx[i] * fi;
        uyy += (float)cy[i] * fi;
    }
    if (rho > 1e-10f) {
        uxx /= rho;
        uyy /= rho;
    }

    // ── Boundary conditions ─────────────────────
    bool isWall = (x == 0 || x == nx - 1 || y == 0);
    bool isLid  = (y == ny - 1);

    if (isLid) {
        // Top lid: moving wall with prescribed velocity
        uxx = lidU;
        uyy = 0.0f;
        // Zou-He boundary: recompute rho from known distributions
        rho = 1.0f; // approximately
    }

    if (isWall) {
        uxx = 0.0f;
        uyy = 0.0f;
    }

    // Store macroscopic fields
    ux_out[idx] = uxx;
    uy_out[idx] = uyy;
    rho_out[idx] = rho;

    // ── Collide (BGK) ───────────────────────────
    float fPost[9];
    for (int i = 0; i < 9; i++) {
        float fi = f[i * nx * ny + idx];
        float fEq = feq(i, rho, uxx, uyy);
        fPost[i] = fi - omega * (fi - fEq);
    }

    // ── Stream ──────────────────────────────────
    if (isWall || isLid) {
        // Bounce-back for walls: reverse directions
        for (int i = 0; i < 9; i++) {
            int oppI = opp[i];
            fNew[oppI * nx * ny + idx] = fPost[i];
        }
        // For lid, apply Zou-He velocity correction
        if (isLid) {
            for (int i = 0; i < 9; i++) {
                fNew[i * nx * ny + idx] = feq(i, rho, lidU, 0.0f);
            }
        }
    } else {
        // Interior: stream to neighbors
        for (int i = 0; i < 9; i++) {
            int xn = x + cx[i];
            int yn = y + cy[i];
            if (xn >= 0 && xn < nx && yn >= 0 && yn < ny) {
                int nidx = yn * nx + xn;
                fNew[i * nx * ny + nidx] = fPost[i];
            }
        }
    }
}

// ── LBM2D implementation ────────────────────────────

bool LBM2D::init(int width, int height) {
    nx = width;
    ny = height;
    currentStep = 0;

    size_t fieldSize = (size_t)nx * ny * sizeof(float);
    size_t fSize = 9 * fieldSize;

    CUDA_CHECK(cudaMalloc(&d_f, fSize));
    CUDA_CHECK(cudaMalloc(&d_fTemp, fSize));
    CUDA_CHECK(cudaMalloc(&d_ux, fieldSize));
    CUDA_CHECK(cudaMalloc(&d_uy, fieldSize));
    CUDA_CHECK(cudaMalloc(&d_rho, fieldSize));

    h_velMag   = new float[nx * ny];
    h_vorticity = new float[nx * ny];
    h_density  = new float[nx * ny];

    reset();

    printf("[LBM2D] Initialized %dx%d grid (%zu KB GPU memory)\n",
           nx, ny, (2 * fSize + 3 * fieldSize) / 1024);
    return true;
}

void LBM2D::shutdown() {
    if (d_f)     { cudaFree(d_f);     d_f = nullptr; }
    if (d_fTemp) { cudaFree(d_fTemp); d_fTemp = nullptr; }
    if (d_ux)    { cudaFree(d_ux);    d_ux = nullptr; }
    if (d_uy)    { cudaFree(d_uy);    d_uy = nullptr; }
    if (d_rho)   { cudaFree(d_rho);   d_rho = nullptr; }

    delete[] h_velMag;   h_velMag = nullptr;
    delete[] h_vorticity; h_vorticity = nullptr;
    delete[] h_density;  h_density = nullptr;
}

void LBM2D::reset() {
    currentStep = 0;
    // Initialize all distributions to equilibrium at rest (rho=1, u=0)
    size_t N = (size_t)nx * ny;
    float* h_f = new float[9 * N];

    // Host-side D2Q9 weights
    float weights[9] = {
        4.0f/9.0f,
        1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
        1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
    };

    for (int i = 0; i < 9; i++) {
        for (size_t j = 0; j < N; j++) {
            h_f[i * N + j] = weights[i]; // feq(rho=1, u=0) = w_i
        }
    }

    CUDA_CHECK(cudaMemcpy(d_f, h_f, 9 * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fTemp, h_f, 9 * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_ux, 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_uy, 0, N * sizeof(float)));

    // Init rho to 1.0
    for (size_t j = 0; j < N; j++) h_f[j] = 1.0f; // reuse buffer
    CUDA_CHECK(cudaMemcpy(d_rho, h_f, N * sizeof(float), cudaMemcpyHostToDevice));

    delete[] h_f;
}

void LBM2D::setTau(float t) { tau = t; }
void LBM2D::setLidVelocity(float u) { lidVelocity = u; }

void LBM2D::step() {
    float omega = 1.0f / tau;
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    lbmCollideStream<<<grid, block>>>(d_f, d_fTemp, d_ux, d_uy, d_rho,
                                       nx, ny, omega, lidVelocity);

    // Swap buffers
    float* tmp = d_f;
    d_f = d_fTemp;
    d_fTemp = tmp;

    currentStep++;
}

void LBM2D::stepMultiple(int n) {
    for (int i = 0; i < n; i++)
        step();
    CUDA_CHECK(cudaDeviceSynchronize());
}

const float* LBM2D::getVelocityMagnitude() {
    size_t N = (size_t)nx * ny;
    float* h_ux = new float[N];
    float* h_uy = new float[N];

    CUDA_CHECK(cudaMemcpy(h_ux, d_ux, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_uy, d_uy, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; i++)
        h_velMag[i] = sqrtf(h_ux[i] * h_ux[i] + h_uy[i] * h_uy[i]);

    delete[] h_ux;
    delete[] h_uy;
    return h_velMag;
}

const float* LBM2D::getVorticityField() {
    size_t N = (size_t)nx * ny;
    float* h_ux = new float[N];
    float* h_uy = new float[N];

    CUDA_CHECK(cudaMemcpy(h_ux, d_ux, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_uy, d_uy, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Vorticity = duy/dx - dux/dy (central differences)
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            int idx = y * nx + x;
            float duy_dx = 0, dux_dy = 0;

            if (x > 0 && x < nx - 1)
                duy_dx = (h_uy[y * nx + (x + 1)] - h_uy[y * nx + (x - 1)]) * 0.5f;
            if (y > 0 && y < ny - 1)
                dux_dy = (h_ux[(y + 1) * nx + x] - h_ux[(y - 1) * nx + x]) * 0.5f;

            h_vorticity[idx] = duy_dx - dux_dy;
        }
    }

    delete[] h_ux;
    delete[] h_uy;
    return h_vorticity;
}

const float* LBM2D::getDensityField() {
    size_t N = (size_t)nx * ny;
    CUDA_CHECK(cudaMemcpy(h_density, d_rho, N * sizeof(float), cudaMemcpyDeviceToHost));
    return h_density;
}
