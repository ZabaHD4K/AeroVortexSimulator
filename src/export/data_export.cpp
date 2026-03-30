#include "data_export.h"
#include <glad/gl.h>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <filesystem>

// ============================================================================
// VTK Export — VTK XML StructuredGrid (.vts)
// ============================================================================

bool DataExporter::exportVTK(
    const std::string& filename,
    const float* velocityMag, const float* pressure, const float* vorticity,
    const float* ux, const float* uy, const float* uz,
    const uint8_t* cellTypes,
    int nx, int ny, int nz)
{
    std::ofstream f(filename);
    if (!f.is_open()) return false;

    int N = nx * ny * nz;

    f << "<?xml version=\"1.0\"?>\n";
    f << "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    f << "  <StructuredGrid WholeExtent=\"0 " << nx-1 << " 0 " << ny-1 << " 0 " << nz-1 << "\">\n";
    f << "    <Piece Extent=\"0 " << nx-1 << " 0 " << ny-1 << " 0 " << nz-1 << "\">\n";

    // ── PointData ──
    f << "      <PointData>\n";

    if (velocityMag) {
        f << "        <DataArray type=\"Float32\" Name=\"velocity_magnitude\" format=\"ascii\">\n";
        for (int i = 0; i < N; i++) f << velocityMag[i] << " ";
        f << "\n        </DataArray>\n";
    }

    if (pressure) {
        f << "        <DataArray type=\"Float32\" Name=\"pressure\" format=\"ascii\">\n";
        for (int i = 0; i < N; i++) f << pressure[i] << " ";
        f << "\n        </DataArray>\n";
    }

    if (vorticity) {
        f << "        <DataArray type=\"Float32\" Name=\"vorticity\" format=\"ascii\">\n";
        for (int i = 0; i < N; i++) f << vorticity[i] << " ";
        f << "\n        </DataArray>\n";
    }

    if (ux && uy && uz) {
        f << "        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (int i = 0; i < N; i++)
            f << ux[i] << " " << uy[i] << " " << uz[i] << " ";
        f << "\n        </DataArray>\n";
    }

    if (cellTypes) {
        f << "        <DataArray type=\"UInt8\" Name=\"cell_type\" format=\"ascii\">\n";
        for (int i = 0; i < N; i++) f << (int)cellTypes[i] << " ";
        f << "\n        </DataArray>\n";
    }

    f << "      </PointData>\n";

    // ── Points ──
    f << "      <Points>\n";
    f << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++)
                f << x << " " << y << " " << z << " ";
    f << "\n        </DataArray>\n";
    f << "      </Points>\n";

    f << "    </Piece>\n";
    f << "  </StructuredGrid>\n";
    f << "</VTKFile>\n";

    return f.good();
}

// ============================================================================
// CSV Export — Aero coefficient history
// ============================================================================

bool DataExporter::exportCSV(
    const std::string& filename,
    const std::vector<float>& cdHistory,
    const std::vector<float>& clHistory,
    float tau, float inletVelocity)
{
    std::ofstream f(filename);
    if (!f.is_open()) return false;

    f << "# AeroVortex Simulator — Coefficient History\n";
    f << "# tau=" << tau << ", inlet_velocity=" << inletVelocity << "\n";
    f << "sample,Cd,Cl\n";

    int count = (int)std::min(cdHistory.size(), clHistory.size());
    for (int i = 0; i < count; i++) {
        f << i << "," << cdHistory[i] << "," << clHistory[i] << "\n";
    }

    return f.good();
}

bool DataExporter::appendAeroCSV(
    const std::string& filename,
    int step, float Cd, float Cl, float Cs,
    float Fx, float Fy, float Fz)
{
    bool exists = std::filesystem::exists(filename);
    std::ofstream f(filename, std::ios::app);
    if (!f.is_open()) return false;

    if (!exists) {
        f << "step,Cd,Cl,Cs,Fx,Fy,Fz\n";
    }
    f << step << "," << Cd << "," << Cl << "," << Cs
      << "," << Fx << "," << Fy << "," << Fz << "\n";

    return f.good();
}

// ============================================================================
// Screenshot — BMP format (no external dependencies)
// ============================================================================

bool DataExporter::saveScreenshot(const std::string& filename, int width, int height) {
    int rowBytes = width * 3;
    int rowPad   = (4 - (rowBytes % 4)) % 4;
    int dataSize = (rowBytes + rowPad) * height;
    int fileSize = 54 + dataSize;

    std::vector<uint8_t> pixels(width * height * 3);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) return false;

    // ── BMP File Header (14 bytes) ──
    uint8_t fileHeader[14] = {};
    fileHeader[0] = 'B'; fileHeader[1] = 'M';
    memcpy(fileHeader + 2, &fileSize, 4);
    int offset = 54;
    memcpy(fileHeader + 10, &offset, 4);
    f.write((char*)fileHeader, 14);

    // ── BMP Info Header (40 bytes) ──
    uint8_t infoHeader[40] = {};
    int headerSize = 40;
    short planes = 1, bpp = 24;
    memcpy(infoHeader + 0, &headerSize, 4);
    memcpy(infoHeader + 4, &width, 4);
    memcpy(infoHeader + 8, &height, 4);
    memcpy(infoHeader + 12, &planes, 2);
    memcpy(infoHeader + 14, &bpp, 2);
    memcpy(infoHeader + 20, &dataSize, 4);
    f.write((char*)infoHeader, 40);

    // ── Pixel data (BGR, bottom-up = matches GL default) ──
    uint8_t pad[3] = {0, 0, 0};
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            uint8_t bgr[3] = { pixels[idx + 2], pixels[idx + 1], pixels[idx] };
            f.write((char*)bgr, 3);
        }
        if (rowPad > 0) f.write((char*)pad, rowPad);
    }

    return f.good();
}

// ============================================================================
// Filename generation with timestamp
// ============================================================================

std::string DataExporter::generateFilename(const std::string& prefix,
                                            const std::string& extension) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = {};
#ifdef _WIN32
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    char buf[64];
    snprintf(buf, sizeof(buf), "%s_%04d%02d%02d_%02d%02d%02d.%s",
             prefix.c_str(),
             tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec,
             extension.c_str());
    return std::string(buf);
}
