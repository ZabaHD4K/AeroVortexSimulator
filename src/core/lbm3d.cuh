#pragma once

// ============================================================================
// LBM 3D solver — D3Q19 lattice with BGK / MRT collision + Smagorinsky LES
// Runs entirely on GPU via CUDA
//
// Memory layout: Structure of Arrays (SoA)
//   f[q][idx]  where q in [0,19), idx = x + y*nx + z*nx*ny
// This gives coalesced reads when threads walk along the X axis.
//
// Boundary conditions:
//   - Bounce-back on solid cells and Y/Z walls
//   - Prescribed-velocity inlet
//   - Zero-gradient (Neumann) outlet
//
// Collision models:
//   - BGK: single relaxation time (original)
//   - MRT: multiple relaxation time (d'Humières 2002)
//   - Smagorinsky LES: subgrid-scale turbulence model (optional, works with both)
// ============================================================================

#include <cstdint>

// ---------------------------------------------------------------------------
// Cell classification
// ---------------------------------------------------------------------------
enum CellType : uint8_t {
    CELL_FLUID  = 0,
    CELL_SOLID  = 1,
    CELL_INLET  = 2,
    CELL_OUTLET = 3
};

// ---------------------------------------------------------------------------
// Wind direction — which axis/face the wind comes from
// ---------------------------------------------------------------------------
enum WindDirection : int {
    WIND_POS_X = 0,   // Front  (+X)
    WIND_NEG_X = 1,   // Back   (-X)
    WIND_POS_Z = 2,   // Right  (+Z)
    WIND_NEG_Z = 3,   // Left   (-Z)
    WIND_POS_Y = 4,   // Above  (+Y → down)
    WIND_NEG_Y = 5    // Below  (-Y → up)
};

// ---------------------------------------------------------------------------
// Collision model
// ---------------------------------------------------------------------------
enum class CollisionModel : int {
    BGK = 0,
    MRT = 1
};

// ---------------------------------------------------------------------------
// Simulation parameters (handy POD for the UI layer)
// ---------------------------------------------------------------------------
struct LBM3DParams {
    int   nx = 200, ny = 100, nz = 100;
    float tau           = 0.8f;
    float inletVelocity = 0.05f;     // lattice units, magnitude
    bool  running       = false;
    int   stepsPerFrame = 5;
    WindDirection windDir = WIND_POS_X;

    // Collision model
    CollisionModel collisionModel = CollisionModel::MRT;

    // Smagorinsky LES
    bool  useSmagorinsky = true;
    float smagorinskyCs  = 0.1f;     // Smagorinsky constant
};

// ---------------------------------------------------------------------------
// Reynolds-number helper (lattice units)
//   Re = u * L / nu,  nu = (tau - 0.5) / 3
// ---------------------------------------------------------------------------
inline float reynoldsNumber(float u, float L, float tau)
{
    float nu = (tau - 0.5f) / 3.0f;
    return (nu > 0.0f) ? (u * L / nu) : 0.0f;
}

// ---------------------------------------------------------------------------
// Solver class
// ---------------------------------------------------------------------------
class LBM3D {
public:
    // Lifecycle ---------------------------------------------------------------
    bool init(int nx, int ny, int nz);
    void shutdown();
    void reset();

    // Stepping ----------------------------------------------------------------
    void step();
    void stepMultiple(int n);

    // Parameters --------------------------------------------------------------
    void setTau(float tau);
    void setInletVelocity(float u);
    void setInletDirection(float ux, float uy, float uz);
    void setCellTypes(const uint8_t* hostCells);   // upload cell-type grid
    void setCollisionModel(CollisionModel m);
    void setSmagorinsky(bool enable, float Cs = 0.1f);

    // Host-side field access (cached — only downloads when data is stale) -----
    const float* getVelocityMagnitude();           // |u|, nx*ny*nz
    const float* getPressureField();               // rho / 3, nx*ny*nz
    const float* getVorticityMagnitude();           // |curl u|, nx*ny*nz
    void getVelocityComponents(float** ux, float** uy, float** uz);

    // Call once per frame before any field access to mark caches stale
    void invalidateCache();

    // Device pointers for CUDA interop (zero-copy rendering, etc.) -----------
    float*       getDeviceUx()        { return d_ux; }
    float*       getDeviceUy()        { return d_uy; }
    float*       getDeviceUz()        { return d_uz; }
    float*       getDeviceRho()       { return d_rho; }
    const uint8_t* getDeviceCellTypes() { return d_cellType; }

    // Meta --------------------------------------------------------------------
    int getNx()   const { return nx; }
    int getNy()   const { return ny; }
    int getNz()   const { return nz; }
    int getStep() const { return currentStep; }

private:
    // Grid dimensions
    int nx = 0, ny = 0, nz = 0;

    // Physical parameters
    float tau           = 0.8f;
    float inletVelocity = 0.05f;
    float inletUx = 0.05f, inletUy = 0.0f, inletUz = 0.0f;
    int   currentStep   = 0;

    // Collision model
    CollisionModel collisionModel = CollisionModel::MRT;
    bool  useSmagorinsky = true;
    float smagorinskyCs  = 0.1f;

    // --- Device arrays -------------------------------------------------------

    // Distribution functions: 19 * N floats each, SoA layout
    float* d_f     = nullptr;       // current populations
    float* d_fTemp = nullptr;       // post-stream populations (ping-pong)

    // Per-cell type flag
    uint8_t* d_cellType = nullptr;

    // Macroscopic fields (N floats each)
    float* d_ux  = nullptr;
    float* d_uy  = nullptr;
    float* d_uz  = nullptr;
    float* d_rho = nullptr;

    // Persistent device buffers for derived fields (no malloc/free per call)
    float* d_velMag = nullptr;
    float* d_vort   = nullptr;

    // --- Host staging buffers ------------------------------------------------
    float* h_velMag   = nullptr;
    float* h_pressure = nullptr;
    float* h_vorticity= nullptr;
    float* h_ux       = nullptr;
    float* h_uy       = nullptr;
    float* h_uz       = nullptr;

    // --- Cache tracking (avoid redundant GPU→CPU transfers) ------------------
    int cachedStep_vel     = -1;  // step when velocity components were last downloaded
    int cachedStep_velMag  = -1;  // step when velocity magnitude was last computed
    int cachedStep_pressure= -1;  // step when pressure was last downloaded
    int cachedStep_vort    = -1;  // step when vorticity was last computed
};
