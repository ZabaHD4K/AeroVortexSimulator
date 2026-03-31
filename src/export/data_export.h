#pragma once
#include <string>
#include <cstdint>
#include <vector>

// Forward declarations
struct AeroCoefficients;

// ── Simulation report data ─────────────────────────────────
struct SimReportData {
    // Model info
    std::string modelName;
    int totalVertices = 0;
    int totalTriangles = 0;
    int numMeshes = 0;
    float boundingRadius = 0.0f;

    // Grid / domain
    int nx = 0, ny = 0, nz = 0;
    float domainScale = 0.0f;
    float voxelSize = 0.0f;
    int solidCells = 0, fluidCells = 0, inletCells = 0, outletCells = 0;

    // Physics parameters
    float tau = 0.0f;
    float inletVelocity = 0.0f;
    int windDir = 0;        // WindDirection enum value
    int collisionModel = 0; // 0=BGK, 1=MRT
    bool smagorinsky = false;
    float smagorinskyCs = 0.0f;
    float reynoldsNumber = 0.0f;
    float viscosity = 0.0f;

    // Simulation state
    int currentStep = 0;
    int stepsPerFrame = 0;

    // Aerodynamic coefficients
    float Cd = 0, Cl = 0, Cs = 0;
    float Fx = 0, Fy = 0, Fz = 0;

    // Cd/Cl history
    std::vector<float> cdHistory;
    std::vector<float> clHistory;

    // Validation
    bool validationActive = false;
    float expectedCd = 0, expectedCl = 0;
    const char* validationSource = "";

    // Flow field statistics (computed from fields)
    float velMin = 0, velMax = 0, velMean = 0;
    float presMin = 0, presMax = 0, presMean = 0;
    float vortMin = 0, vortMax = 0, vortMean = 0;
};

class DataExporter {
public:
    // Export 3D fields to VTK structured grid (.vts) for ParaView
    static bool exportVTK(const std::string& filename,
                          const float* velocityMag, const float* pressure,
                          const float* vorticity,
                          const float* ux, const float* uy, const float* uz,
                          const uint8_t* cellTypes,
                          int nx, int ny, int nz);

    // Export aero coefficient history to CSV
    static bool exportCSV(const std::string& filename,
                          const std::vector<float>& cdHistory,
                          const std::vector<float>& clHistory,
                          float tau, float inletVelocity);

    // Append one row to a running aero CSV (creates file + header if absent)
    static bool appendAeroCSV(const std::string& filename,
                              int step, float Cd, float Cl, float Cs,
                              float Fx, float Fy, float Fz);

    // ── Full simulation report (HTML) ──────────────────────
    static bool exportReport(const std::string& filename,
                             const SimReportData& data);

    // ── Detailed flow field statistics CSV ──────────────────
    static bool exportFlowFieldCSV(const std::string& filename,
                                   const float* velocityMag, const float* pressure,
                                   const float* vorticity,
                                   const float* ux, const float* uy, const float* uz,
                                   const uint8_t* cellTypes,
                                   int nx, int ny, int nz,
                                   float voxelSize);

    // Save current GL framebuffer as BMP
    static bool saveScreenshot(const std::string& filename, int width, int height);

    // "prefix_YYYYMMDD_HHMMSS.ext"
    static std::string generateFilename(const std::string& prefix,
                                        const std::string& extension);

    // Compute flow field statistics for report
    static void computeFieldStats(const float* field, const uint8_t* cellTypes,
                                  int N, float& outMin, float& outMax, float& outMean);
};
