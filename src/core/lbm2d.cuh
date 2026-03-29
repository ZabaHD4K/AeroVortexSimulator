#pragma once

// LBM 2D solver using D2Q9 lattice with BGK collision operator
// Runs entirely on GPU via CUDA

struct LBM2DParams {
    int nx = 256;           // grid width
    int ny = 256;           // grid height
    float tau = 0.6f;       // relaxation time (viscosity = (tau - 0.5) / 3)
    float lidVelocity = 0.1f; // top lid velocity (lattice units)
    bool running = false;
    int stepsPerFrame = 10;
};

class LBM2D {
public:
    bool init(int nx, int ny);
    void shutdown();
    void reset();
    void step();               // one LBM timestep
    void stepMultiple(int n);  // multiple steps

    void setTau(float tau);
    void setLidVelocity(float u);

    // Copy field data to host for visualization
    // Returns pointer to host array of size nx*ny (float)
    const float* getVelocityMagnitude();
    const float* getVorticityField();
    const float* getDensityField();

    int getNx() const { return nx; }
    int getNy() const { return ny; }
    int getStep() const { return currentStep; }

private:
    int nx = 0, ny = 0;
    float tau = 0.6f;
    float lidVelocity = 0.1f;
    int currentStep = 0;

    // GPU arrays: f[9 * nx * ny], fTemp[9 * nx * ny]
    float* d_f = nullptr;
    float* d_fTemp = nullptr;

    // Host output buffers
    float* h_velMag = nullptr;
    float* h_vorticity = nullptr;
    float* h_density = nullptr;

    // GPU output buffers
    float* d_ux = nullptr;
    float* d_uy = nullptr;
    float* d_rho = nullptr;
};
