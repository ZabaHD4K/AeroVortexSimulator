#include "data_export.h"
#include "core/log.h"
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
    // At least one data field and cellTypes must be provided
    if (!velocityMag && !pressure && !vorticity && !ux) {
        LOG_ERROR("exportVTK: no data fields provided");
        return false;
    }
    if (!cellTypes) {
        LOG_ERROR("exportVTK: cellTypes is null");
        return false;
    }
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        LOG_ERROR("exportVTK: invalid grid dimensions");
        return false;
    }

    std::ofstream f(filename);
    if (!f.is_open()) {
        LOG_ERROR("exportVTK: failed to open " << filename);
        return false;
    }

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

    bool ok = f.good();
    if (ok) LOG_INFO("exportVTK: saved " << filename);
    else    LOG_ERROR("exportVTK: write error for " << filename);
    return ok;
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
    if (!f.is_open()) {
        LOG_ERROR("exportCSV: failed to open " << filename);
        return false;
    }

    f << "# AeroVortex Simulator — Coefficient History\n";
    f << "# tau=" << tau << ", inlet_velocity=" << inletVelocity << "\n";
    f << "sample,Cd,Cl\n";

    int count = (int)std::min(cdHistory.size(), clHistory.size());
    for (int i = 0; i < count; i++) {
        f << i << "," << cdHistory[i] << "," << clHistory[i] << "\n";
    }

    bool ok = f.good();
    if (ok) LOG_INFO("exportCSV: saved " << filename << " (" << count << " samples)");
    else    LOG_ERROR("exportCSV: write error for " << filename);
    return ok;
}

bool DataExporter::appendAeroCSV(
    const std::string& filename,
    int step, float Cd, float Cl, float Cs,
    float Fx, float Fy, float Fz)
{
    bool exists = std::filesystem::exists(filename);
    std::ofstream f(filename, std::ios::app);
    if (!f.is_open()) {
        LOG_ERROR("appendAeroCSV: failed to open " << filename);
        return false;
    }

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
    if (width <= 0 || height <= 0) {
        LOG_ERROR("saveScreenshot: invalid dimensions " << width << "x" << height);
        return false;
    }

    int rowBytes = width * 3;
    int rowPad   = (4 - (rowBytes % 4)) % 4;
    int dataSize = (rowBytes + rowPad) * height;
    int fileSize = 54 + dataSize;

    std::vector<uint8_t> pixels(width * height * 3);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        LOG_ERROR("saveScreenshot: failed to open " << filename);
        return false;
    }

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

    bool ok = f.good();
    if (ok) LOG_INFO("saveScreenshot: saved " << filename << " (" << width << "x" << height << ")");
    else    LOG_ERROR("saveScreenshot: write error for " << filename);
    return ok;
}

// ============================================================================
// Field statistics helper
// ============================================================================

void DataExporter::computeFieldStats(const float* field, const uint8_t* cellTypes,
                                      int N, float& outMin, float& outMax, float& outMean)
{
    outMin = 0.0f;
    outMax = 0.0f;
    outMean = 0.0f;
    if (!field || N <= 0) return;

    outMin = 1e30f;
    outMax = -1e30f;
    int count = 0;

    for (int i = 0; i < N; i++) {
        if (cellTypes && cellTypes[i] == 1) continue; // skip solid
        float v = field[i];
        if (v < outMin) outMin = v;
        if (v > outMax) outMax = v;
        outMean += v;
        count++;
    }
    if (count > 0) outMean /= (float)count;
    else { outMin = 0; outMax = 0; }
}

// ============================================================================
// Full simulation report — HTML
// ============================================================================

static std::string windDirName(int dir) {
    switch (dir) {
        case 0: return "Front (+X)";
        case 1: return "Back (-X)";
        case 2: return "Right (+Z)";
        case 3: return "Left (-Z)";
        case 4: return "Top (+Y)";
        case 5: return "Bottom (-Y)";
        default: return "Unknown";
    }
}

static std::string collisionModelName(int cm) {
    return cm == 0 ? "BGK" : "MRT (TRT)";
}

static std::string currentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = {};
#ifdef _WIN32
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    char buf[64];
    snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d",
             tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec);
    return std::string(buf);
}

bool DataExporter::exportReport(const std::string& filename, const SimReportData& d)
{
    // Validate report data has reasonable values
    if (d.nx <= 0 || d.ny <= 0 || d.nz <= 0) {
        LOG_ERROR("exportReport: invalid grid dimensions");
        return false;
    }
    if (d.tau <= 0.5f) {
        LOG_ERROR("exportReport: invalid tau value " << d.tau);
        return false;
    }
    if (d.inletVelocity < 0.0f) {
        LOG_ERROR("exportReport: invalid inlet velocity");
        return false;
    }

    std::ofstream f(filename);
    if (!f.is_open()) {
        LOG_ERROR("exportReport: failed to open " << filename);
        return false;
    }

    f << std::fixed;

    int totalCells = d.nx * d.ny * d.nz;
    float memMB = totalCells * (19.0f * 2 + 5) * 4.0f / (1024.0f * 1024.0f);

    // Convergence analysis
    float cdLast = 0, cdAvg10 = 0, cdStd10 = 0;
    float clLast = 0, clAvg10 = 0;
    bool converged = false;
    if (!d.cdHistory.empty()) {
        cdLast = d.cdHistory.back();
        clLast = d.clHistory.empty() ? 0 : d.clHistory.back();

        int n10 = std::min(10, (int)d.cdHistory.size());
        for (int i = (int)d.cdHistory.size() - n10; i < (int)d.cdHistory.size(); i++) {
            cdAvg10 += d.cdHistory[i];
            if (i < (int)d.clHistory.size()) clAvg10 += d.clHistory[i];
        }
        cdAvg10 /= n10;
        clAvg10 /= n10;

        for (int i = (int)d.cdHistory.size() - n10; i < (int)d.cdHistory.size(); i++) {
            float diff = d.cdHistory[i] - cdAvg10;
            cdStd10 += diff * diff;
        }
        cdStd10 = std::sqrt(cdStd10 / n10);
        converged = (n10 >= 5 && cdStd10 < 0.01f * std::abs(cdAvg10 + 1e-9f));
    }

    // ── SVG sparkline for Cd/Cl history ──
    auto makeSvgSparkline = [](const std::vector<float>& data, const std::string& color, int w, int h) -> std::string {
        if (data.size() < 2) return "";
        float mn = *std::min_element(data.begin(), data.end());
        float mx = *std::max_element(data.begin(), data.end());
        float range = mx - mn;
        if (range < 1e-9f) range = 1.0f;

        std::ostringstream s;
        s << "<svg width=\"" << w << "\" height=\"" << h << "\" style=\"background:#1a1a2e;border-radius:4px;\">";
        s << "<polyline fill=\"none\" stroke=\"" << color << "\" stroke-width=\"1.5\" points=\"";
        for (int i = 0; i < (int)data.size(); i++) {
            float x = (float)i / (float)(data.size() - 1) * (w - 4) + 2;
            float y = (h - 2) - (data[i] - mn) / range * (h - 4);
            s << std::fixed << std::setprecision(1) << x << "," << y << " ";
        }
        s << "\"/></svg>";
        return s.str();
    };

    // ── HTML output ──
    f << R"(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AeroVortex Simulation Report</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:'Segoe UI',system-ui,sans-serif; background:#0f0f1a; color:#e0e0e0; padding:24px; line-height:1.6; }
  .container { max-width:960px; margin:0 auto; }
  h1 { color:#4fc3f7; font-size:28px; margin-bottom:4px; }
  h2 { color:#81d4fa; font-size:18px; margin:28px 0 12px; border-bottom:1px solid #333; padding-bottom:6px; }
  .subtitle { color:#888; font-size:13px; margin-bottom:20px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px; }
  .card { background:#1a1a2e; border:1px solid #2a2a44; border-radius:8px; padding:16px; }
  .card-full { grid-column:1/-1; }
  .label { color:#888; font-size:12px; text-transform:uppercase; letter-spacing:0.5px; }
  .value { font-size:20px; font-weight:600; color:#fff; }
  .value-sm { font-size:15px; font-weight:500; color:#ddd; }
  table { width:100%; border-collapse:collapse; margin-top:8px; }
  th { text-align:left; color:#888; font-size:12px; text-transform:uppercase; padding:6px 12px; border-bottom:1px solid #333; }
  td { padding:6px 12px; border-bottom:1px solid #222; font-size:14px; }
  .good { color:#66bb6a; }
  .warn { color:#ffa726; }
  .bad  { color:#ef5350; }
  .tag  { display:inline-block; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:600; }
  .tag-blue { background:#1565c0; color:#bbdefb; }
  .tag-green { background:#2e7d32; color:#c8e6c9; }
  .tag-orange { background:#e65100; color:#ffe0b2; }
  .tag-red { background:#b71c1c; color:#ffcdd2; }
  .stats-row { display:flex; gap:24px; margin:8px 0; }
  .stats-row .stat { flex:1; }
  .mono { font-family:'Cascadia Code','Fira Code',monospace; }
  .footer { margin-top:32px; color:#555; font-size:12px; text-align:center; border-top:1px solid #222; padding-top:16px; }
  svg { display:block; margin:8px 0; }
</style>
</head>
<body>
<div class="container">
)";

    f << "<h1>AeroVortex Simulation Report</h1>\n";
    f << "<div class=\"subtitle\">Generated: " << currentTimestamp() << "</div>\n";

    // ── Model Info ──
    f << "<h2>Model</h2>\n";
    f << "<div class=\"grid\">\n";
    f << "  <div class=\"card\">\n";
    f << "    <div class=\"label\">Model Name</div>\n";
    f << "    <div class=\"value\">" << d.modelName << "</div>\n";
    f << "  </div>\n";
    f << "  <div class=\"card\">\n";
    f << "    <div class=\"label\">Geometry</div>\n";
    f << "    <div class=\"value-sm\">" << d.numMeshes << " meshes &middot; "
      << d.totalVertices << " vertices &middot; " << d.totalTriangles << " triangles</div>\n";
    f << "    <div class=\"value-sm\">Bounding radius: " << std::setprecision(4) << d.boundingRadius << "</div>\n";
    f << "  </div>\n";
    f << "</div>\n";

    // ── Domain & Grid ──
    f << "<h2>Domain &amp; Grid</h2>\n";
    f << "<div class=\"grid\">\n";

    f << "  <div class=\"card\">\n";
    f << "    <div class=\"label\">Grid Resolution</div>\n";
    f << "    <div class=\"value\">" << d.nx << " x " << d.ny << " x " << d.nz << "</div>\n";
    f << "    <div class=\"value-sm\">" << totalCells << " total cells &middot; ~"
      << std::setprecision(0) << memMB << " MB VRAM</div>\n";
    f << "  </div>\n";

    f << "  <div class=\"card\">\n";
    f << "    <div class=\"label\">Voxel Size / Domain Scale</div>\n";
    f << "    <div class=\"value-sm\">Voxel: " << std::setprecision(6) << d.voxelSize << " units</div>\n";
    f << "    <div class=\"value-sm\">Scale: " << std::setprecision(2) << d.domainScale << "</div>\n";
    f << "  </div>\n";

    f << "  <div class=\"card card-full\">\n";
    f << "    <div class=\"label\">Cell Classification</div>\n";
    f << "    <table><tr><th>Type</th><th>Count</th><th>%</th></tr>\n";
    auto cellPct = [&](int count) { return totalCells > 0 ? 100.0f * count / totalCells : 0.0f; };
    f << "    <tr><td>Solid</td><td class=\"mono\">" << d.solidCells << "</td><td>" << std::setprecision(1) << cellPct(d.solidCells) << "%</td></tr>\n";
    f << "    <tr><td>Fluid</td><td class=\"mono\">" << d.fluidCells << "</td><td>" << std::setprecision(1) << cellPct(d.fluidCells) << "%</td></tr>\n";
    f << "    <tr><td>Inlet</td><td class=\"mono\">" << d.inletCells << "</td><td>" << std::setprecision(1) << cellPct(d.inletCells) << "%</td></tr>\n";
    f << "    <tr><td>Outlet</td><td class=\"mono\">" << d.outletCells << "</td><td>" << std::setprecision(1) << cellPct(d.outletCells) << "%</td></tr>\n";
    f << "    </table>\n";
    f << "  </div>\n";
    f << "</div>\n";

    // ── Physics Parameters ──
    f << "<h2>Physics Parameters</h2>\n";
    f << "<div class=\"grid\">\n";

    f << "  <div class=\"card\">\n";
    f << "    <div class=\"label\">Wind</div>\n";
    f << "    <div class=\"value-sm\">Direction: " << windDirName(d.windDir) << "</div>\n";
    f << "    <div class=\"value-sm\">Inlet velocity: <span class=\"mono\">" << std::setprecision(4) << d.inletVelocity << "</span> (lattice units)</div>\n";
    f << "  </div>\n";

    f << "  <div class=\"card\">\n";
    f << "    <div class=\"label\">Relaxation</div>\n";
    f << "    <div class=\"value-sm\">Tau: <span class=\"mono\">" << std::setprecision(4) << d.tau << "</span></div>\n";
    f << "    <div class=\"value-sm\">Kinematic viscosity: <span class=\"mono\">" << std::setprecision(6) << d.viscosity << "</span></div>\n";
    f << "  </div>\n";

    f << "  <div class=\"card\">\n";
    f << "    <div class=\"label\">Collision Model</div>\n";
    f << "    <div class=\"value-sm\">" << collisionModelName(d.collisionModel) << "</div>\n";
    if (d.smagorinsky)
        f << "    <div class=\"value-sm\">Smagorinsky LES: <span class=\"tag tag-blue\">ON</span> Cs = " << std::setprecision(3) << d.smagorinskyCs << "</div>\n";
    else
        f << "    <div class=\"value-sm\">Smagorinsky LES: OFF</div>\n";
    f << "  </div>\n";

    f << "  <div class=\"card\">\n";
    f << "    <div class=\"label\">Reynolds Number</div>\n";
    f << "    <div class=\"value\">" << std::setprecision(0) << d.reynoldsNumber << "</div>\n";
    if (d.reynoldsNumber < 1)
        f << "    <div class=\"value-sm bad\">Too low for simulation</div>\n";
    else if (d.reynoldsNumber < 50)
        f << "    <div class=\"value-sm warn\">Laminar (no vortex shedding)</div>\n";
    else if (d.reynoldsNumber < 300)
        f << "    <div class=\"value-sm good\">Vortex shedding regime</div>\n";
    else
        f << "    <div class=\"value-sm good\">Turbulent transition</div>\n";
    f << "  </div>\n";

    f << "</div>\n";

    // ── Simulation State ──
    f << "<h2>Simulation State</h2>\n";
    f << "<div class=\"grid\">\n";

    f << "  <div class=\"card\">\n";
    f << "    <div class=\"label\">Progress</div>\n";
    f << "    <div class=\"value\">" << d.currentStep << " <span style=\"font-size:14px;color:#888;\">steps</span></div>\n";
    f << "    <div class=\"value-sm\">" << d.stepsPerFrame << " steps/frame</div>\n";
    f << "  </div>\n";

    f << "  <div class=\"card\">\n";
    f << "    <div class=\"label\">Convergence</div>\n";
    if (converged)
        f << "    <div class=\"value-sm\"><span class=\"tag tag-green\">CONVERGED</span></div>\n";
    else if (d.currentStep > 500)
        f << "    <div class=\"value-sm\"><span class=\"tag tag-orange\">CONVERGING</span></div>\n";
    else
        f << "    <div class=\"value-sm\"><span class=\"tag tag-blue\">DEVELOPING</span></div>\n";
    f << "    <div class=\"value-sm\">Cd std (last 10): " << std::setprecision(6) << cdStd10 << "</div>\n";
    f << "  </div>\n";

    f << "</div>\n";

    // ── Aerodynamic Results ──
    f << "<h2>Aerodynamic Coefficients</h2>\n";
    f << "<div class=\"grid\">\n";

    f << "  <div class=\"card\">\n";
    f << "    <div class=\"label\">Drag Coefficient (Cd)</div>\n";
    f << "    <div class=\"value\">" << std::setprecision(4) << d.Cd << "</div>\n";
    if (!d.cdHistory.empty())
        f << "    <div class=\"value-sm\">Avg (last 10): " << std::setprecision(4) << cdAvg10 << "</div>\n";
    f << "  </div>\n";

    f << "  <div class=\"card\">\n";
    f << "    <div class=\"label\">Lift Coefficient (Cl)</div>\n";
    f << "    <div class=\"value\">" << std::setprecision(4) << d.Cl << "</div>\n";
    if (!d.clHistory.empty())
        f << "    <div class=\"value-sm\">Avg (last 10): " << std::setprecision(4) << clAvg10 << "</div>\n";
    f << "  </div>\n";

    f << "  <div class=\"card\">\n";
    f << "    <div class=\"label\">Side Force Coefficient (Cs)</div>\n";
    f << "    <div class=\"value\">" << std::setprecision(4) << d.Cs << "</div>\n";
    f << "  </div>\n";

    f << "  <div class=\"card\">\n";
    f << "    <div class=\"label\">Forces (lattice units)</div>\n";
    f << "    <div class=\"value-sm\">Fx = " << std::setprecision(6) << d.Fx << "</div>\n";
    f << "    <div class=\"value-sm\">Fy = " << std::setprecision(6) << d.Fy << "</div>\n";
    f << "    <div class=\"value-sm\">Fz = " << std::setprecision(6) << d.Fz << "</div>\n";
    f << "  </div>\n";

    // Cd/Cl history charts
    if (d.cdHistory.size() >= 2) {
        f << "  <div class=\"card card-full\">\n";
        f << "    <div class=\"label\">Cd History (" << d.cdHistory.size() << " samples)</div>\n";
        f << "    " << makeSvgSparkline(d.cdHistory, "#4fc3f7", 900, 80) << "\n";
        f << "    <div class=\"label\" style=\"margin-top:12px;\">Cl History</div>\n";
        f << "    " << makeSvgSparkline(d.clHistory, "#66bb6a", 900, 80) << "\n";
        f << "  </div>\n";
    }

    f << "</div>\n";

    // ── Validation ──
    if (d.validationActive) {
        f << "<h2>Validation</h2>\n";
        f << "<div class=\"grid\">\n";

        float cdErr = (std::abs(d.expectedCd) > 1e-6f) ?
            std::abs(d.Cd - d.expectedCd) / d.expectedCd * 100.0f : 0.0f;
        float clErr = std::abs(d.Cl - d.expectedCl);

        f << "  <div class=\"card\">\n";
        f << "    <div class=\"label\">Reference</div>\n";
        f << "    <div class=\"value-sm\">" << d.validationSource << "</div>\n";
        f << "    <div class=\"value-sm\">Expected Cd: " << std::setprecision(4) << d.expectedCd << "</div>\n";
        f << "    <div class=\"value-sm\">Expected Cl: " << std::setprecision(4) << d.expectedCl << "</div>\n";
        f << "  </div>\n";

        f << "  <div class=\"card\">\n";
        f << "    <div class=\"label\">Error</div>\n";
        std::string errClass = (cdErr < 20.0f) ? "good" : (cdErr < 50.0f) ? "warn" : "bad";
        f << "    <div class=\"value-sm " << errClass << "\">Cd error: " << std::setprecision(1) << cdErr << "%</div>\n";
        f << "    <div class=\"value-sm\">Cl deviation: " << std::setprecision(4) << clErr << "</div>\n";
        f << "  </div>\n";

        f << "</div>\n";
    }

    // ── Flow Field Statistics ──
    f << "<h2>Flow Field Statistics</h2>\n";
    f << "<div class=\"card\">\n";
    f << "<table>\n";
    f << "<tr><th>Field</th><th>Min</th><th>Max</th><th>Mean</th></tr>\n";
    f << "<tr><td>Velocity magnitude</td><td class=\"mono\">" << std::setprecision(6) << d.velMin
      << "</td><td class=\"mono\">" << d.velMax << "</td><td class=\"mono\">" << d.velMean << "</td></tr>\n";
    f << "<tr><td>Pressure (rho/3)</td><td class=\"mono\">" << std::setprecision(6) << d.presMin
      << "</td><td class=\"mono\">" << d.presMax << "</td><td class=\"mono\">" << d.presMean << "</td></tr>\n";
    f << "<tr><td>Vorticity |curl u|</td><td class=\"mono\">" << std::setprecision(6) << d.vortMin
      << "</td><td class=\"mono\">" << d.vortMax << "</td><td class=\"mono\">" << d.vortMean << "</td></tr>\n";
    f << "</table>\n";
    f << "</div>\n";

    // ── Footer ──
    f << R"(
<div class="footer">
  AeroVortex Simulator &mdash; D3Q19 Lattice Boltzmann Method on CUDA GPU<br>
  Report generated automatically
</div>
</div>
</body>
</html>
)";

    bool ok = f.good();
    if (ok) LOG_INFO("exportReport: saved " << filename);
    else    LOG_ERROR("exportReport: write error for " << filename);
    return ok;
}

// ============================================================================
// Detailed flow field statistics CSV (per-slice data)
// ============================================================================

bool DataExporter::exportFlowFieldCSV(
    const std::string& filename,
    const float* velocityMag, const float* pressure, const float* vorticity,
    const float* ux, const float* uy, const float* uz,
    const uint8_t* cellTypes,
    int nx, int ny, int nz, float voxelSize)
{
    if (!cellTypes) {
        LOG_ERROR("exportFlowFieldCSV: cellTypes is null");
        return false;
    }
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        LOG_ERROR("exportFlowFieldCSV: invalid grid dimensions");
        return false;
    }

    std::ofstream f(filename);
    if (!f.is_open()) {
        LOG_ERROR("exportFlowFieldCSV: failed to open " << filename);
        return false;
    }

    f << std::fixed << std::setprecision(6);
    f << "# AeroVortex Simulator — Flow Field Statistics per X-slice\n";
    f << "# voxelSize=" << voxelSize << ", grid=" << nx << "x" << ny << "x" << nz << "\n";
    f << "x_slice,x_world,fluid_cells,solid_cells,vel_min,vel_max,vel_mean,pres_min,pres_max,pres_mean,vort_min,vort_max,vort_mean,ux_mean,uy_mean,uz_mean\n";

    for (int x = 0; x < nx; x++) {
        float vMin = 1e30f, vMax = -1e30f, vSum = 0;
        float pMin = 1e30f, pMax = -1e30f, pSum = 0;
        float wMin = 1e30f, wMax = -1e30f, wSum = 0;
        float uxSum = 0, uySum = 0, uzSum = 0;
        int nFluid = 0, nSolid = 0;

        for (int z = 0; z < nz; z++) {
            for (int y = 0; y < ny; y++) {
                int idx = x + y * nx + z * nx * ny;
                if (cellTypes[idx] == 1) { nSolid++; continue; }
                nFluid++;

                if (velocityMag) {
                    float v = velocityMag[idx];
                    if (v < vMin) vMin = v;
                    if (v > vMax) vMax = v;
                    vSum += v;
                }
                if (pressure) {
                    float p = pressure[idx];
                    if (p < pMin) pMin = p;
                    if (p > pMax) pMax = p;
                    pSum += p;
                }
                if (vorticity) {
                    float w = vorticity[idx];
                    if (w < wMin) wMin = w;
                    if (w > wMax) wMax = w;
                    wSum += w;
                }
                if (ux) uxSum += ux[idx];
                if (uy) uySum += uy[idx];
                if (uz) uzSum += uz[idx];
            }
        }

        float inv = nFluid > 0 ? 1.0f / nFluid : 0.0f;
        if (nFluid == 0) { vMin = vMax = pMin = pMax = wMin = wMax = 0; }
        float xWorld = (x - nx * 0.5f) * voxelSize;

        f << x << "," << xWorld << "," << nFluid << "," << nSolid << ","
          << vMin << "," << vMax << "," << vSum * inv << ","
          << pMin << "," << pMax << "," << pSum * inv << ","
          << wMin << "," << wMax << "," << wSum * inv << ","
          << uxSum * inv << "," << uySum * inv << "," << uzSum * inv << "\n";
    }

    bool ok = f.good();
    if (ok) LOG_INFO("exportFlowFieldCSV: saved " << filename);
    else    LOG_ERROR("exportFlowFieldCSV: write error for " << filename);
    return ok;
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
