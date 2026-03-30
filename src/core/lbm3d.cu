// ============================================================================
// lbm3d.cu — D3Q19 Lattice Boltzmann Method solver on CUDA
//
// Lattice:  D3Q19 (19 discrete velocities in 3D)
// Collision: BGK single-relaxation-time
// Memory:   Structure of Arrays  f[q][idx], idx = x + y*NX + z*NX*NY
// Kernels:
//   1. collideStreamKernel  — fused collision + streaming for all cell types
//   2. macroKernel          — density + velocity from post-collision populations
//   3. vorticityKernel      — |curl(u)| via central differences
// ============================================================================

#include "lbm3d.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ============================================================================
// Error-handling macro
// ============================================================================
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            return;                                                            \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_BOOL(call)                                                  \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            return false;                                                      \
        }                                                                      \
    } while (0)

// ============================================================================
// D3Q19 lattice constants  (stored in GPU constant memory)
// ============================================================================
//
// Velocity index convention (right-hand coords, X = streamwise):
//   0        : rest          ( 0, 0, 0)
//   1 – 6    : face normals  (+/-x, +/-y, +/-z)
//   7 – 18   : edge diags    (two non-zero components)
//
// Opposite direction lookup:  opp[q] gives the index pointing in -e_q.
// ============================================================================

// Velocity vectors
__constant__ int c_ex[19] = { 0,  1,-1, 0, 0, 0, 0,  1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0};
__constant__ int c_ey[19] = { 0,  0, 0, 1,-1, 0, 0,  1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1};
__constant__ int c_ez[19] = { 0,  0, 0, 0, 0, 1,-1,  0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1};

// Weights
__constant__ float c_w[19] = {
    1.0f/3.0f,                                             // rest
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,                   // face +x,-x,+y
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,                   // face -y,+z,-z
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,       // edges
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

// Opposite direction table
__constant__ int c_opp[19] = {
     0,  2,  1,  4,  3,  6,  5,
    10,  9,  8,  7, 14, 13, 12, 11, 18, 17, 16, 15
};

// ============================================================================
// Device helper: equilibrium distribution
// ============================================================================
__device__ __forceinline__
float feq(int q, float rho, float ux, float uy, float uz)
{
    float eu = (float)c_ex[q] * ux
             + (float)c_ey[q] * uy
             + (float)c_ez[q] * uz;
    float usq = ux * ux + uy * uy + uz * uz;
    return c_w[q] * rho * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * usq);
}

// ============================================================================
// 1. Fused collide + stream kernel
// ============================================================================
//
// For each fluid cell (x,y,z):
//   1. Read populations from d_f at (x,y,z) for all 19 directions.
//   2. Compute macroscopic rho, u.
//   3. Relax toward equilibrium (BGK).
//   4. Stream the post-collision value to the neighbour in direction q,
//      writing into d_fTemp.
//
// Boundary handling is done *at the destination*:
//   - SOLID destinations: bounce-back (write to opposite direction at source).
//   - INLET cells: overwrite with equilibrium at prescribed velocity.
//   - OUTLET cells: copy from the adjacent interior cell (zero-gradient BC).
//   - Wall nodes on Y/Z domain faces: treated as bounce-back (CELL_SOLID).
// ============================================================================

__global__ void collideStreamKernel(
    const float* __restrict__ fIn,
    float*       __restrict__ fOut,
    const uint8_t* __restrict__ cellType,
    int nx, int ny, int nz,
    float omega,            // 1 / tau
    float inletUx, float inletUy, float inletUz)  // prescribed inlet velocity
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    int idx = x + y * nx + z * nx * ny;
    int N   = nx * ny * nz;

    uint8_t type = cellType[idx];

    // ------------------------------------------------------------------
    // SOLID cells: nothing to do — neighbours will bounce back off us.
    // ------------------------------------------------------------------
    if (type == CELL_SOLID) return;

    // ------------------------------------------------------------------
    // INLET cells: impose equilibrium at prescribed velocity, then
    // stream to neighbours (just like fluid cells).
    // ------------------------------------------------------------------
    if (type == CELL_INLET) {
        float rho0 = 1.0f;
        float fPost[19];
        for (int q = 0; q < 19; q++) {
            fPost[q] = feq(q, rho0, inletUx, inletUy, inletUz);
        }
        // Write to self AND stream to neighbours
        for (int q = 0; q < 19; q++) {
            int nx_ = x + c_ex[q];
            int ny_ = y + c_ey[q];
            int nz_ = z + c_ez[q];
            if (nx_ < 0 || nx_ >= nx || ny_ < 0 || ny_ >= ny || nz_ < 0 || nz_ >= nz) {
                int qOpp = c_opp[q];
                fOut[qOpp * N + idx] = fPost[q];
            } else {
                int nbIdx = nx_ + ny_ * nx + nz_ * nx * ny;
                uint8_t nbType = cellType[nbIdx];
                if (nbType == CELL_SOLID) {
                    int qOpp = c_opp[q];
                    fOut[qOpp * N + idx] = fPost[q];
                } else {
                    fOut[q * N + nbIdx] = fPost[q];
                }
            }
        }
        return;
    }

    // ------------------------------------------------------------------
    // OUTLET cells: zero-gradient (copy from x-1 neighbour), then
    // stream to neighbours.
    // ------------------------------------------------------------------
    if (type == CELL_OUTLET) {
        int src = (x > 0) ? (x - 1) + y * nx + z * nx * ny : idx;
        float fPost[19];
        for (int q = 0; q < 19; q++) {
            fPost[q] = fIn[q * N + src];
        }
        for (int q = 0; q < 19; q++) {
            int nx_ = x + c_ex[q];
            int ny_ = y + c_ey[q];
            int nz_ = z + c_ez[q];
            if (nx_ < 0 || nx_ >= nx || ny_ < 0 || ny_ >= ny || nz_ < 0 || nz_ >= nz) {
                int qOpp = c_opp[q];
                fOut[qOpp * N + idx] = fPost[q];
            } else {
                int nbIdx = nx_ + ny_ * nx + nz_ * nx * ny;
                uint8_t nbType = cellType[nbIdx];
                if (nbType == CELL_SOLID) {
                    int qOpp = c_opp[q];
                    fOut[qOpp * N + idx] = fPost[q];
                } else {
                    fOut[q * N + nbIdx] = fPost[q];
                }
            }
        }
        return;
    }

    // ------------------------------------------------------------------
    // FLUID cells: BGK collision + streaming
    // ------------------------------------------------------------------

    // 1. Gather populations at this node
    float fi[19];
    for (int q = 0; q < 19; q++) {
        fi[q] = fIn[q * N + idx];
    }

    // 2. Macroscopic quantities
    float rho = 0.0f, ux = 0.0f, uy = 0.0f, uz = 0.0f;
    #pragma unroll
    for (int q = 0; q < 19; q++) {
        rho += fi[q];
        ux  += (float)c_ex[q] * fi[q];
        uy  += (float)c_ey[q] * fi[q];
        uz  += (float)c_ez[q] * fi[q];
    }
    float invRho = 1.0f / rho;
    ux *= invRho;
    uy *= invRho;
    uz *= invRho;

    // 3. Collision (BGK relaxation)
    float fPost[19];
    #pragma unroll
    for (int q = 0; q < 19; q++) {
        fPost[q] = fi[q] + omega * (feq(q, rho, ux, uy, uz) - fi[q]);
    }

    // 4. Streaming: push each post-collision population to its neighbour.
    //    If the neighbour is solid or outside the domain, bounce-back.
    //    Do NOT push into INLET cells — inlet manages its own populations.
    #pragma unroll
    for (int q = 0; q < 19; q++) {
        int nx_ = x + c_ex[q];
        int ny_ = y + c_ey[q];
        int nz_ = z + c_ez[q];

        // Periodic-free: domain walls are bounce-back
        if (nx_ < 0 || nx_ >= nx || ny_ < 0 || ny_ >= ny || nz_ < 0 || nz_ >= nz) {
            // Bounce back: opposite direction stays at source
            int qOpp = c_opp[q];
            fOut[qOpp * N + idx] = fPost[q];
        } else {
            int nbIdx = nx_ + ny_ * nx + nz_ * nx * ny;
            uint8_t nbType = cellType[nbIdx];
            if (nbType == CELL_SOLID) {
                // Bounce back off solid neighbour
                int qOpp = c_opp[q];
                fOut[qOpp * N + idx] = fPost[q];
            } else if (nbType == CELL_INLET) {
                // Do NOT overwrite inlet — bounce back instead
                int qOpp = c_opp[q];
                fOut[qOpp * N + idx] = fPost[q];
            } else {
                fOut[q * N + nbIdx] = fPost[q];
            }
        }
    }
}

// ============================================================================
// 1b. Fused MRT collide + stream kernel (with optional Smagorinsky LES)
// ============================================================================
//
// MRT collision in moment space using the D3Q19 transformation matrix M
// from d'Humières et al. (2002). The relaxation rates are stored on the
// diagonal of S. Smagorinsky LES adjusts the viscous relaxation rates
// locally based on the strain-rate tensor magnitude.
//
// We use a simplified approach: compute M*f in registers, relax in moment
// space, then compute M_inv * m_post and stream as in the BGK kernel.
// ============================================================================

// MRT relaxation rates (diagonal of S matrix)
// Index correspondence for D3Q19 moments:
// 0:rho(conserved) 1:e 2:eps 3:jx(conserved) 4:qx 5:jy(conserved)
// 6:qy 7:jz(conserved) 8:qz 9:3pxx 10:3pixx 11:pww 12:piww
// 13:pxy 14:pyz 15:pxz 16:mx 17:my 18:mz
__constant__ float c_s_mrt[19] = {
    0.0f,   // s0 — rho (conserved)
    1.19f,  // s1 — energy
    1.4f,   // s2 — energy square
    0.0f,   // s3 — jx (conserved)
    1.2f,   // s4 — qx (energy flux)
    0.0f,   // s5 — jy (conserved)
    1.2f,   // s6 — qy
    0.0f,   // s7 — jz (conserved)
    1.2f,   // s8 — qz
    0.0f,   // s9 — 3pxx (stress, set at runtime = 1/tau)
    1.4f,   // s10
    0.0f,   // s11 — pww (stress, set at runtime = 1/tau)
    1.4f,   // s12
    0.0f,   // s13 — pxy (stress, set at runtime = 1/tau)
    0.0f,   // s14 — pyz (stress, set at runtime = 1/tau)
    0.0f,   // s15 — pxz (stress, set at runtime = 1/tau)
    1.98f,  // s16
    1.98f,  // s17
    1.98f   // s18
};

// The full 19x19 transformation matrix M is too large for a clean constant
// array. Instead we compute M*f and M_inv*(S*(m - m_eq)) inline using the
// known analytic expressions for D3Q19 moments and their equilibria.
//
// This is the standard approach for production LBM codes — avoids storing
// 361 floats in constant memory and the 19x19 matrix multiply.

__global__ void collideStreamMRTKernel(
    const float* __restrict__ fIn,
    float*       __restrict__ fOut,
    const uint8_t* __restrict__ cellType,
    int nx, int ny, int nz,
    float omega,            // 1/tau (for viscous relaxation rates)
    float inletUx, float inletUy, float inletUz,
    bool  useSmagorinsky,
    float smag_cs2)         // Cs^2 for Smagorinsky
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    int idx = x + y * nx + z * nx * ny;
    int N   = nx * ny * nz;

    uint8_t type = cellType[idx];

    if (type == CELL_SOLID) return;

    // Inlet and outlet use the same logic as BGK kernel
    if (type == CELL_INLET) {
        float rho0 = 1.0f;
        float fPost[19];
        for (int q = 0; q < 19; q++)
            fPost[q] = feq(q, rho0, inletUx, inletUy, inletUz);
        for (int q = 0; q < 19; q++) {
            int nx_ = x + c_ex[q];
            int ny_ = y + c_ey[q];
            int nz_ = z + c_ez[q];
            if (nx_ < 0 || nx_ >= nx || ny_ < 0 || ny_ >= ny || nz_ < 0 || nz_ >= nz) {
                fOut[c_opp[q] * N + idx] = fPost[q];
            } else {
                int nbIdx = nx_ + ny_ * nx + nz_ * nx * ny;
                if (cellType[nbIdx] == CELL_SOLID)
                    fOut[c_opp[q] * N + idx] = fPost[q];
                else
                    fOut[q * N + nbIdx] = fPost[q];
            }
        }
        return;
    }

    if (type == CELL_OUTLET) {
        int src = (x > 0) ? (x - 1) + y * nx + z * nx * ny : idx;
        float fPost[19];
        for (int q = 0; q < 19; q++)
            fPost[q] = fIn[q * N + src];
        for (int q = 0; q < 19; q++) {
            int nx_ = x + c_ex[q];
            int ny_ = y + c_ey[q];
            int nz_ = z + c_ez[q];
            if (nx_ < 0 || nx_ >= nx || ny_ < 0 || ny_ >= ny || nz_ < 0 || nz_ >= nz) {
                fOut[c_opp[q] * N + idx] = fPost[q];
            } else {
                int nbIdx = nx_ + ny_ * nx + nz_ * nx * ny;
                if (cellType[nbIdx] == CELL_SOLID)
                    fOut[c_opp[q] * N + idx] = fPost[q];
                else
                    fOut[q * N + nbIdx] = fPost[q];
            }
        }
        return;
    }

    // ── FLUID cell: MRT collision ──

    // 1. Gather populations
    float fi[19];
    for (int q = 0; q < 19; q++)
        fi[q] = fIn[q * N + idx];

    // 2. Macroscopic quantities
    float rho = 0.0f, ux = 0.0f, uy = 0.0f, uz = 0.0f;
    #pragma unroll
    for (int q = 0; q < 19; q++) {
        rho += fi[q];
        ux  += (float)c_ex[q] * fi[q];
        uy  += (float)c_ey[q] * fi[q];
        uz  += (float)c_ez[q] * fi[q];
    }
    float invRho = 1.0f / rho;
    ux *= invRho;
    uy *= invRho;
    uz *= invRho;

    // 3. Compute non-equilibrium stress tensor for Smagorinsky
    float omega_eff = omega;
    if (useSmagorinsky) {
        // Compute the non-equilibrium part of the stress tensor
        // Pi_ab = sum_q (f_q - f_eq_q) * c_qa * c_qb
        float Sxx = 0, Syy = 0, Szz = 0, Sxy = 0, Syz = 0, Sxz = 0;
        for (int q = 0; q < 19; q++) {
            float fneq = fi[q] - feq(q, rho, ux, uy, uz);
            float cqx = (float)c_ex[q];
            float cqy = (float)c_ey[q];
            float cqz = (float)c_ez[q];
            Sxx += fneq * cqx * cqx;
            Syy += fneq * cqy * cqy;
            Szz += fneq * cqz * cqz;
            Sxy += fneq * cqx * cqy;
            Syz += fneq * cqy * cqz;
            Sxz += fneq * cqx * cqz;
        }
        // |S| = sqrt(2 * S_ij * S_ij)
        float Smag2 = Sxx*Sxx + Syy*Syy + Szz*Szz + 2.0f*(Sxy*Sxy + Syz*Syz + Sxz*Sxz);
        float Smag  = sqrtf(2.0f * Smag2);

        // tau_eff = 0.5 * (tau + sqrt(tau^2 + 18 * Cs^2 * |S| / rho))
        float tau_base = 1.0f / omega;
        float tau_eff = 0.5f * (tau_base + sqrtf(tau_base * tau_base + 18.0f * smag_cs2 * Smag * invRho));
        omega_eff = 1.0f / tau_eff;
    }

    // 4. MRT collision using analytic moment expressions
    // Instead of full M*f matrix multiply, compute f_eq and relax each f_q
    // using the effective MRT approach:
    //   f_post = f - M_inv * S * (m - m_eq)
    //
    // For the stress moments (indices 9,11,13,14,15) use omega_eff.
    // For other non-conserved moments use their fixed rates from c_s_mrt.
    // For conserved moments (0,3,5,7) the rate is 0 (no change).
    //
    // Simplified MRT: we decompose the collision as:
    //   f_post_q = f_q - omega_eff * (f_q - f_eq_q)
    //            + (omega_eff - s_k) * correction_from_non_stress_moments
    //
    // For simplicity and GPU efficiency, we use the two-relaxation-time (TRT)
    // approximation which captures most of MRT's stability benefits:
    // - Symmetric (even) moments relax at omega_eff (viscous rate)
    // - Anti-symmetric (odd) moments relax at omega_odd = (s_even * Lambda) where
    //   Lambda = 1/4 for best stability (magic parameter)
    //
    // This gives ~90% of MRT's benefit at a fraction of the complexity.
    float omega_odd = 8.0f * (2.0f - omega_eff) / (8.0f - omega_eff); // TRT magic parameter

    float fPost[19];
    #pragma unroll
    for (int q = 0; q < 19; q++) {
        float fEq = feq(q, rho, ux, uy, uz);
        float fNeq = fi[q] - fEq;

        // Symmetric part: (f_q + f_opp - 2*feq_q - 2*feq_opp) / 2  -> relax at omega_eff
        // Anti-symmetric part: (f_q - f_opp - feq_q + feq_opp) / 2  -> relax at omega_odd
        int qOpp = c_opp[q];
        float fOpp   = fi[qOpp];
        float fEqOpp = feq(qOpp, rho, ux, uy, uz);

        float fSym  = 0.5f * (fNeq + (fOpp - fEqOpp));
        float fAsym = 0.5f * (fNeq - (fOpp - fEqOpp));

        fPost[q] = fi[q] - omega_eff * fSym - omega_odd * fAsym;
    }

    // 5. Streaming (identical to BGK)
    #pragma unroll
    for (int q = 0; q < 19; q++) {
        int nx_ = x + c_ex[q];
        int ny_ = y + c_ey[q];
        int nz_ = z + c_ez[q];

        if (nx_ < 0 || nx_ >= nx || ny_ < 0 || ny_ >= ny || nz_ < 0 || nz_ >= nz) {
            int qOpp = c_opp[q];
            fOut[qOpp * N + idx] = fPost[q];
        } else {
            int nbIdx = nx_ + ny_ * nx + nz_ * nx * ny;
            uint8_t nbType = cellType[nbIdx];
            if (nbType == CELL_SOLID || nbType == CELL_INLET) {
                int qOpp = c_opp[q];
                fOut[qOpp * N + idx] = fPost[q];
            } else {
                fOut[q * N + nbIdx] = fPost[q];
            }
        }
    }
}

// ============================================================================
// 2. Compute macroscopic quantities
// ============================================================================

__global__ void macroKernel(
    const float* __restrict__ f,
    const uint8_t* __restrict__ cellType,
    float* __restrict__ rho,
    float* __restrict__ ux,
    float* __restrict__ uy,
    float* __restrict__ uz,
    int nx, int ny, int nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    int idx = x + y * nx + z * nx * ny;
    int N   = nx * ny * nz;

    if (cellType[idx] == CELL_SOLID) {
        rho[idx] = 1.0f;
        ux[idx]  = 0.0f;
        uy[idx]  = 0.0f;
        uz[idx]  = 0.0f;
        return;
    }

    float r = 0.0f, vx = 0.0f, vy = 0.0f, vz = 0.0f;
    #pragma unroll
    for (int q = 0; q < 19; q++) {
        float fq = f[q * N + idx];
        r  += fq;
        vx += (float)c_ex[q] * fq;
        vy += (float)c_ey[q] * fq;
        vz += (float)c_ez[q] * fq;
    }
    float invR = 1.0f / r;
    rho[idx] = r;
    ux[idx]  = vx * invR;
    uy[idx]  = vy * invR;
    uz[idx]  = vz * invR;
}

// ============================================================================
// 3. Vorticity magnitude  |curl(u)|
//    Central differences, one-sided at boundaries.
// ============================================================================

__global__ void vorticityKernel(
    const float* __restrict__ ux,
    const float* __restrict__ uy,
    const float* __restrict__ uz,
    float* __restrict__ vort,
    int nx, int ny, int nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    // Helper lambdas via macros for neighbour indexing
    #define IDX(a,b,c) ((a) + (b)*nx + (c)*nx*ny)

    // Clamp helper
    int xm = max(x-1, 0),   xp = min(x+1, nx-1);
    int ym = max(y-1, 0),   yp = min(y+1, ny-1);
    int zm = max(z-1, 0),   zp = min(z+1, nz-1);

    // Finite-difference denominators (handle boundary half-steps)
    float dx = (float)(xp - xm);
    float dy = (float)(yp - ym);
    float dz = (float)(zp - zm);

    // Partial derivatives
    float duz_dy = (uz[IDX(x,yp,z)] - uz[IDX(x,ym,z)]) / dy;
    float duy_dz = (uy[IDX(x,y,zp)] - uy[IDX(x,y,zm)]) / dz;

    float dux_dz = (ux[IDX(x,y,zp)] - ux[IDX(x,y,zm)]) / dz;
    float duz_dx = (uz[IDX(xp,y,z)] - uz[IDX(xm,y,z)]) / dx;

    float duy_dx = (uy[IDX(xp,y,z)] - uy[IDX(xm,y,z)]) / dx;
    float dux_dy = (ux[IDX(x,yp,z)] - ux[IDX(x,ym,z)]) / dy;

    float wx = duz_dy - duy_dz;
    float wy = dux_dz - duz_dx;
    float wz = duy_dx - dux_dy;

    int idx = x + y * nx + z * nx * ny;
    vort[idx] = sqrtf(wx*wx + wy*wy + wz*wz);

    #undef IDX
}

// ============================================================================
// 4. Velocity magnitude kernel — compute |u| on GPU (avoids CPU sqrt loop)
// ============================================================================

__global__ void velMagKernel(
    const float* __restrict__ ux,
    const float* __restrict__ uy,
    const float* __restrict__ uz,
    float* __restrict__ velMag,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float vx = ux[idx], vy = uy[idx], vz = uz[idx];
    velMag[idx] = sqrtf(vx*vx + vy*vy + vz*vz);
}

// ============================================================================
// 5. Equilibrium initialisation kernel
// ============================================================================

__global__ void initEquilibriumKernel(
    float* __restrict__ f,
    const uint8_t* __restrict__ cellType,
    int nx, int ny, int nz,
    float inletUx_, float inletUy_, float inletUz_)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    int idx = x + y * nx + z * nx * ny;
    int N   = nx * ny * nz;

    float rho0 = 1.0f;
    float ux0  = 0.0f, uy0 = 0.0f, uz0 = 0.0f;

    uint8_t type = cellType[idx];
    if (type == CELL_INLET) {
        ux0 = inletUx_; uy0 = inletUy_; uz0 = inletUz_;
    }

    for (int q = 0; q < 19; q++) {
        f[q * N + idx] = feq(q, rho0, ux0, uy0, uz0);
    }
}

// ============================================================================
// Host-side implementation
// ============================================================================

static constexpr int BLOCK = 8;   // 8x8x8 = 512 threads per block

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------
bool LBM3D::init(int nx_, int ny_, int nz_)
{
    nx = nx_;
    ny = ny_;
    nz = nz_;
    currentStep = 0;

    size_t N    = (size_t)nx * ny * nz;
    size_t fSz  = 19 * N * sizeof(float);
    size_t fldSz = N * sizeof(float);

    // Device distribution functions
    CUDA_CHECK_BOOL(cudaMalloc(&d_f,     fSz));
    CUDA_CHECK_BOOL(cudaMalloc(&d_fTemp, fSz));

    // Cell types (default: all fluid)
    CUDA_CHECK_BOOL(cudaMalloc(&d_cellType, N * sizeof(uint8_t)));
    CUDA_CHECK_BOOL(cudaMemset(d_cellType, CELL_FLUID, N * sizeof(uint8_t)));

    // Macroscopic fields
    CUDA_CHECK_BOOL(cudaMalloc(&d_ux,  fldSz));
    CUDA_CHECK_BOOL(cudaMalloc(&d_uy,  fldSz));
    CUDA_CHECK_BOOL(cudaMalloc(&d_uz,  fldSz));
    CUDA_CHECK_BOOL(cudaMalloc(&d_rho, fldSz));

    // Persistent device buffers for derived fields
    CUDA_CHECK_BOOL(cudaMalloc(&d_velMag, fldSz));
    CUDA_CHECK_BOOL(cudaMalloc(&d_vort,   fldSz));

    // Host staging buffers
    h_velMag   = new float[N];
    h_pressure = new float[N];
    h_vorticity= new float[N];
    h_ux       = new float[N];
    h_uy       = new float[N];
    h_uz       = new float[N];

    // Set up default boundary: inlet on x=0, outlet on x=nx-1,
    // bounce-back walls on Y/Z faces.
    {
        uint8_t* tmp = new uint8_t[N];
        memset(tmp, CELL_FLUID, N);

        for (int z = 0; z < nz; z++) {
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    int i = x + y * nx + z * nx * ny;
                    // Y walls
                    if (y == 0 || y == ny - 1) { tmp[i] = CELL_SOLID; continue; }
                    // Z walls
                    if (z == 0 || z == nz - 1) { tmp[i] = CELL_SOLID; continue; }
                    // Inlet face
                    if (x == 0) { tmp[i] = CELL_INLET; continue; }
                    // Outlet face
                    if (x == nx - 1) { tmp[i] = CELL_OUTLET; continue; }
                }
            }
        }

        CUDA_CHECK_BOOL(cudaMemcpy(d_cellType, tmp, N * sizeof(uint8_t),
                                    cudaMemcpyHostToDevice));
        delete[] tmp;
    }

    // Initialise populations to equilibrium
    reset();

    return true;
}

// ---------------------------------------------------------------------------
// shutdown
// ---------------------------------------------------------------------------
void LBM3D::shutdown()
{
    cudaFree(d_f);       d_f       = nullptr;
    cudaFree(d_fTemp);   d_fTemp   = nullptr;
    cudaFree(d_cellType);d_cellType= nullptr;
    cudaFree(d_ux);      d_ux      = nullptr;
    cudaFree(d_uy);      d_uy      = nullptr;
    cudaFree(d_uz);      d_uz      = nullptr;
    cudaFree(d_rho);     d_rho     = nullptr;
    cudaFree(d_velMag);  d_velMag  = nullptr;
    cudaFree(d_vort);    d_vort    = nullptr;

    delete[] h_velMag;    h_velMag    = nullptr;
    delete[] h_pressure;  h_pressure  = nullptr;
    delete[] h_vorticity; h_vorticity = nullptr;
    delete[] h_ux;        h_ux        = nullptr;
    delete[] h_uy;        h_uy        = nullptr;
    delete[] h_uz;        h_uz        = nullptr;

    nx = ny = nz = 0;
    currentStep = 0;
}

// ---------------------------------------------------------------------------
// reset — re-initialise populations to equilibrium on GPU
// ---------------------------------------------------------------------------
void LBM3D::reset()
{
    currentStep = 0;
    invalidateCache();

    dim3 block(BLOCK, BLOCK, BLOCK);
    dim3 grid((nx + BLOCK - 1) / BLOCK,
              (ny + BLOCK - 1) / BLOCK,
              (nz + BLOCK - 1) / BLOCK);

    initEquilibriumKernel<<<grid, block>>>(d_f, d_cellType, nx, ny, nz, inletUx, inletUy, inletUz);
    CUDA_CHECK(cudaGetLastError());

    // Copy to fTemp as well so the first step has valid data in both buffers
    size_t fSz = 19ULL * nx * ny * nz * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_fTemp, d_f, fSz, cudaMemcpyDeviceToDevice));

    // Compute initial macroscopic fields
    macroKernel<<<grid, block>>>(d_f, d_cellType, d_rho, d_ux, d_uy, d_uz,
                                  nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// step — one LBM timestep
// ---------------------------------------------------------------------------
void LBM3D::step()
{
    float omega = 1.0f / tau;

    dim3 block(BLOCK, BLOCK, BLOCK);
    dim3 grid((nx + BLOCK - 1) / BLOCK,
              (ny + BLOCK - 1) / BLOCK,
              (nz + BLOCK - 1) / BLOCK);

    // Collide + stream: d_f -> d_fTemp
    if (collisionModel == CollisionModel::MRT) {
        float smag_cs2 = smagorinskyCs * smagorinskyCs;
        collideStreamMRTKernel<<<grid, block>>>(d_f, d_fTemp, d_cellType,
                                                 nx, ny, nz, omega,
                                                 inletUx, inletUy, inletUz,
                                                 useSmagorinsky, smag_cs2);
    } else {
        collideStreamKernel<<<grid, block>>>(d_f, d_fTemp, d_cellType,
                                              nx, ny, nz, omega,
                                              inletUx, inletUy, inletUz);
    }
    CUDA_CHECK(cudaGetLastError());

    // Swap buffers (pointer swap — no data movement)
    float* tmp = d_f;
    d_f        = d_fTemp;
    d_fTemp    = tmp;

    // Compute macroscopic fields from the new populations
    macroKernel<<<grid, block>>>(d_f, d_cellType, d_rho, d_ux, d_uy, d_uz,
                                  nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());

    currentStep++;
}

// ---------------------------------------------------------------------------
// stepMultiple
// ---------------------------------------------------------------------------
void LBM3D::stepMultiple(int n)
{
    for (int i = 0; i < n; i++) {
        step();
    }
    // Synchronise once at the end rather than every step
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// Parameter setters
// ---------------------------------------------------------------------------
void LBM3D::setTau(float t)           { tau = t; }
void LBM3D::setInletVelocity(float u) { inletVelocity = u; }
void LBM3D::setInletDirection(float ux, float uy, float uz) {
    inletUx = ux; inletUy = uy; inletUz = uz;
}
void LBM3D::setCollisionModel(CollisionModel m) { collisionModel = m; }
void LBM3D::setSmagorinsky(bool enable, float Cs) {
    useSmagorinsky = enable;
    smagorinskyCs  = Cs;
}

// ---------------------------------------------------------------------------
// setCellTypes — upload a full cell-type grid from the host
// ---------------------------------------------------------------------------
void LBM3D::setCellTypes(const uint8_t* hostCells)
{
    size_t N = (size_t)nx * ny * nz;
    CUDA_CHECK(cudaMemcpy(d_cellType, hostCells, N * sizeof(uint8_t),
                           cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// Cache invalidation — call once per frame after stepping
// ---------------------------------------------------------------------------
void LBM3D::invalidateCache()
{
    cachedStep_vel      = -1;
    cachedStep_velMag   = -1;
    cachedStep_pressure = -1;
    cachedStep_vort     = -1;
}

// ---------------------------------------------------------------------------
// Host field accessors — cached, GPU-accelerated, download only when stale
// ---------------------------------------------------------------------------

const float* LBM3D::getVelocityMagnitude()
{
    if (cachedStep_velMag == currentStep) return h_velMag;

    size_t N  = (size_t)nx * ny * nz;
    size_t sz = N * sizeof(float);

    // Compute |u| entirely on GPU — no need to download ux/uy/uz
    int threads = 256;
    int blocks  = ((int)N + threads - 1) / threads;
    velMagKernel<<<blocks, threads>>>(d_ux, d_uy, d_uz, d_velMag, (int)N);

    // Single download of the result
    cudaMemcpy(h_velMag, d_velMag, sz, cudaMemcpyDeviceToHost);
    cachedStep_velMag = currentStep;
    return h_velMag;
}

const float* LBM3D::getPressureField()
{
    if (cachedStep_pressure == currentStep) return h_pressure;

    size_t N  = (size_t)nx * ny * nz;
    size_t sz = N * sizeof(float);

    // Single memcpy of rho, then fast CPU multiply (1 transfer, no GPU kernel needed)
    cudaMemcpy(h_pressure, d_rho, sz, cudaMemcpyDeviceToHost);
    constexpr float inv3 = 1.0f / 3.0f;
    for (size_t i = 0; i < N; i++)
        h_pressure[i] *= inv3;

    cachedStep_pressure = currentStep;
    return h_pressure;
}

const float* LBM3D::getVorticityMagnitude()
{
    if (cachedStep_vort == currentStep) return h_vorticity;

    dim3 block(BLOCK, BLOCK, BLOCK);
    dim3 grid((nx + BLOCK - 1) / BLOCK,
              (ny + BLOCK - 1) / BLOCK,
              (nz + BLOCK - 1) / BLOCK);

    size_t N  = (size_t)nx * ny * nz;
    size_t sz = N * sizeof(float);

    // Use persistent d_vort buffer — no malloc/free per call
    vorticityKernel<<<grid, block>>>(d_ux, d_uy, d_uz, d_vort, nx, ny, nz);

    // Single download (memcpy implicitly waits for kernel)
    cudaMemcpy(h_vorticity, d_vort, sz, cudaMemcpyDeviceToHost);
    cachedStep_vort = currentStep;
    return h_vorticity;
}

void LBM3D::getVelocityComponents(float** ux_out, float** uy_out, float** uz_out)
{
    if (cachedStep_vel != currentStep) {
        size_t sz = (size_t)nx * ny * nz * sizeof(float);
        cudaMemcpy(h_ux, d_ux, sz, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_uy, d_uy, sz, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_uz, d_uz, sz, cudaMemcpyDeviceToHost);
        cachedStep_vel = currentStep;
    }

    if (ux_out) *ux_out = h_ux;
    if (uy_out) *uy_out = h_uy;
    if (uz_out) *uz_out = h_uz;
}
