#pragma once
#include <string>
#include <cstdint>
#include <vector>

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

    // Save current GL framebuffer as BMP
    static bool saveScreenshot(const std::string& filename, int width, int height);

    // "prefix_YYYYMMDD_HHMMSS.ext"
    static std::string generateFilename(const std::string& prefix,
                                        const std::string& extension);
};
