#include "gui.h"
#include "app.h"
#include "geometry/primitives.h"
#include <imgui.h>
#include <filesystem>
#include <algorithm>
#include <cstdio>
#include <cfloat>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>

namespace fs = std::filesystem;

static const char* supportedExts[] = {
    ".stl", ".obj", ".fbx", ".ply", ".dae", ".3ds", ".gltf", ".glb"
};

static bool isSupportedModel(const std::string& ext) {
    std::string lower = ext;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    for (auto& e : supportedExts)
        if (lower == e) return true;
    return false;
}

std::string openFileDialog() {
    char filename[MAX_PATH] = "";
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = "3D Models\0*.stl;*.obj;*.fbx;*.ply;*.dae;*.3ds;*.gltf;*.glb\0All Files\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = "Load 3D Model";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameA(&ofn)) return std::string(filename);
    return "";
}

void Gui::init(const std::string& modelsPath) {
    modelsDir = modelsPath;
    refreshModelLibrary();
}

void Gui::refreshModelLibrary() {
    libraryModels.clear();
    if (modelsDir.empty() || !fs::exists(modelsDir)) return;
    for (auto& entry : fs::directory_iterator(modelsDir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        if (!isSupportedModel(ext)) continue;
        ModelEntry me;
        me.name = entry.path().stem().string();
        me.path = entry.path().string();
        me.extension = ext;
        me.sizeMB = (float)entry.file_size() / (1024.0f * 1024.0f);
        libraryModels.push_back(me);
    }
    std::sort(libraryModels.begin(), libraryModels.end(),
        [](const ModelEntry& a, const ModelEntry& b) { return a.name < b.name; });
}

void Gui::render(App& app) {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(340, 0), ImGuiCond_FirstUseEver);

    ImGui::Begin("AeroVortex Simulator");

    // ── MODEL ────────────────────────────────────
    if (app.model.has_value()) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "%s", app.model->name.c_str());
    }

    for (auto& entry : libraryModels) {
        char label[256];
        snprintf(label, sizeof(label), "%s", entry.name.c_str());
        if (ImGui::Button(label, ImVec2(-1, 0))) {
            selectedLibraryPath = entry.path;
            libraryLoadRequested = true;
        }
    }

    if (ImGui::Button("Browse...", ImVec2(-1, 0))) loadRequested = true;

    // ── TEST MODELS (Validation) ─────────────────
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Validation Models")) {
        ImGui::TextWrapped("Procedural geometries with known Cd/Cl for solver validation.");
        if (ImGui::Button("Sphere", ImVec2(-1, 0))) testSphereRequested = true;
        ImGui::SameLine();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Cd ~ 0.47 (Re 100-10000)");
        if (ImGui::Button("Cylinder", ImVec2(-1, 0))) testCylinderRequested = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Cd ~ 1.17 (Re 100-10000)");
        if (ImGui::Button("NACA 0012", ImVec2(-1, 0))) testNACARequested = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Cd ~ 0.012, Cl ~ 0 at 0 AoA");
    }

    ImGui::Spacing();
    ImGui::Separator();

    // ── DOMAIN SETTINGS ──────────────────────────
    if (!app.simInitialized) {
        ImGui::Spacing();
        if (ImGui::CollapsingHeader("Domain Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
            auto& p = app.lbm3dParams;

            // Resolution presets
            ImGui::Text("Resolution Preset:");
            if (ImGui::Button("Fast (96)", ImVec2(80, 0))) { p.nx = 128; p.ny = 64; p.nz = 64; }
            ImGui::SameLine();
            if (ImGui::Button("Medium (150)", ImVec2(95, 0))) { p.nx = 200; p.ny = 100; p.nz = 100; }
            ImGui::SameLine();
            if (ImGui::Button("High (256)", ImVec2(80, 0))) { p.nx = 300; p.ny = 150; p.nz = 150; }

            ImGui::SliderInt("NX", &p.nx, 32, 400);
            ImGui::SliderInt("NY", &p.ny, 32, 300);
            ImGui::SliderInt("NZ", &p.nz, 32, 300);
            ImGui::SliderFloat("Domain Scale", &app.domainScale, 0.2f, 0.6f, "%.2f");
            ImGui::TextDisabled("Scale: how much of the domain the model fills");

            int cells = p.nx * p.ny * p.nz;
            float memMB = cells * (19.0f * 2 + 5) * 4.0f / (1024.0f * 1024.0f);
            ImGui::Text("Cells: %d  (~%.0f MB VRAM)", cells, memMB);

            size_t estMB = ((size_t)app.lbm3dParams.nx * app.lbm3dParams.ny * app.lbm3dParams.nz * 83) / (1024*1024);
            ImGui::TextColored(estMB > 2000 ? ImVec4(1,0.3f,0.3f,1) : ImVec4(0.5f,0.5f,0.5f,1),
                               "Est. VRAM: %zu MB", estMB);
        }
    }

    // ── WIND TUNNEL ──────────────────────────────
    if (!app.model.has_value()) {
        ImGui::TextDisabled("Load a model first");
    } else if (!app.simInitialized) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.7f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.85f, 0.3f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.05f, 0.5f, 0.15f, 1.0f));
        if (ImGui::Button("START WIND TUNNEL", ImVec2(-1, 45)))
            windStartRequested = true;
        ImGui::PopStyleColor(3);
    } else {
        auto& p = app.lbm3dParams;

        // ── Simulation controls ──
        if (p.running) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.1f, 1.0f));
            if (ImGui::Button("STOP", ImVec2(-1, 35))) p.running = false;
            ImGui::PopStyleColor();
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.7f, 0.2f, 1.0f));
            if (ImGui::Button("RESUME", ImVec2(-1, 35))) p.running = true;
            ImGui::PopStyleColor();
        }

        ImGui::SliderFloat("Wind Speed", &p.inletVelocity, 0.01f, 0.1f, "%.3f");
        ImGui::SliderFloat("Tau", &p.tau, 0.51f, 2.0f, "%.3f");
        ImGui::SliderInt("Steps/Frame", &p.stepsPerFrame, 1, 20);

        // Wind direction
        ImGui::Spacing();
        ImGui::Text("Wind Direction:");
        static const char* windLabels[] = {
            "Front (+X)", "Back (-X)", "Right (+Z)", "Left (-Z)", "Top (+Y)", "Bottom (-Y)"
        };
        int currentDir = (int)p.windDir;
        if (ImGui::Combo("##winddir", &currentDir, windLabels, 6)) {
            p.windDir = (WindDirection)currentDir;
            windDirChangeRequested = true;
        }

        // ── Physics ──
        ImGui::Spacing();
        if (ImGui::CollapsingHeader("Physics")) {
            static const char* collisionLabels[] = { "BGK", "MRT (TRT)" };
            int cm = (int)p.collisionModel;
            if (ImGui::Combo("Collision", &cm, collisionLabels, 2))
                p.collisionModel = (CollisionModel)cm;

            ImGui::Checkbox("Smagorinsky LES", &p.useSmagorinsky);
            if (p.useSmagorinsky)
                ImGui::SliderFloat("Cs", &p.smagorinskyCs, 0.05f, 0.3f, "%.2f");

            float Re = reynoldsNumber(p.inletVelocity, (float)p.ny * 0.4f, p.tau);
            ImGui::Text("Re ~ %.0f", Re);
        }

        // ── Visualization toggles ──
        ImGui::Spacing();
        if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Model", &app.showModel);
            ImGui::Checkbox("Streamlines", &app.showStreamlines);
            if (app.showStreamlines)
                ImGui::SliderInt("Num Lines", &app.numStreamlines, 25, 1000);

            ImGui::Checkbox("Slice Plane", &app.showSlicePlane);
            if (app.showSlicePlane) {
                static const char* axisLabels[] = { "X", "Y", "Z" };
                ImGui::Combo("Axis", &app.sliceAxis, axisLabels, 3);
                int maxIdx = (app.sliceAxis == 0) ? p.nx-1 : (app.sliceAxis == 1) ? p.ny-1 : p.nz-1;
                ImGui::SliderInt("Slice", &app.sliceIndex, 0, maxIdx);
                static const char* fieldLabels[] = { "Velocity", "Pressure", "Vorticity" };
                ImGui::Combo("Field##slice", &app.sliceField, fieldLabels, 3);
            }

            ImGui::Checkbox("Surface Pressure", &app.showSurfacePressure);
            ImGui::Checkbox("Particles", &app.showParticles);

            ImGui::Checkbox("Volume Render", &app.showVolume);
            if (app.showVolume) {
                static const char* vfLabels[] = { "Velocity", "Pressure", "Vorticity" };
                ImGui::Combo("Field##vol", &app.volumeField, vfLabels, 3);
                ImGui::SliderFloat("Density", &app.volumeRenderer.densityScale, 1.0f, 50.0f);
                ImGui::SliderFloat("Opacity##vol", &app.volumeRenderer.opacity, 0.1f, 1.0f);
            }
        }

        // ── Aero coefficients + Validation ──
        ImGui::Spacing();
        if (ImGui::CollapsingHeader("Aerodynamic Coefficients", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Step: %d", app.lbm3d.getStep());
            ImGui::Text("RMS Change: %.2e", app.convergenceRMS);
            if (app.converged) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "[Converged]");
            }
            ImGui::Text("Cd: %.4f  Cl: %.4f  Cs: %.4f",
                         app.aeroCoeffs.Cd, app.aeroCoeffs.Cl, app.aeroCoeffs.Cs);
            ImGui::Text("Fx: %.4f  Fy: %.4f  Fz: %.4f",
                         app.aeroCoeffs.Fx, app.aeroCoeffs.Fy, app.aeroCoeffs.Fz);

            // Show validation comparison if using a test model
            if (app.validationActive) {
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Validation:");
                ImGui::Text("Expected Cd: %.4f", app.validationRef.expectedCd);
                ImGui::Text("Expected Cl: %.4f", app.validationRef.expectedCl);
                float cdErr = (app.validationRef.expectedCd > 1e-6f) ?
                    std::abs(app.aeroCoeffs.Cd - app.validationRef.expectedCd) / app.validationRef.expectedCd * 100.0f : 0.0f;
                float clErr = std::abs(app.aeroCoeffs.Cl - app.validationRef.expectedCl);
                if (cdErr < 20.0f)
                    ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Cd error: %.1f%%", cdErr);
                else if (cdErr < 50.0f)
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Cd error: %.1f%%", cdErr);
                else
                    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.2f, 1.0f), "Cd error: %.1f%% (converging...)", cdErr);
                ImGui::Text("Cl deviation: %.4f", clErr);
                ImGui::TextDisabled("Ref: %s", app.validationRef.source);
                ImGui::TextDisabled("Valid Re: %.0f - %.0f",
                    app.validationRef.reRange[0], app.validationRef.reRange[1]);
            }

            // Cd/Cl history plot
            if (!app.cdHistory.empty()) {
                ImGui::PlotLines("Cd", app.cdHistory.data(), (int)app.cdHistory.size(),
                                 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(0, 40));
                ImGui::PlotLines("Cl", app.clHistory.data(), (int)app.clHistory.size(),
                                 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(0, 40));
            }
        }

        // ── Export ──
        ImGui::Spacing();
        if (ImGui::CollapsingHeader("Export")) {
            if (ImGui::Button("Screenshot (BMP)", ImVec2(-1, 0)))
                screenshotRequested = true;

            ImGui::Spacing();
            ImGui::TextDisabled("Data");
            if (ImGui::Button("Export VTK (ParaView)", ImVec2(-1, 0)))
                exportVTKRequested = true;
            if (ImGui::Button("Export CSV (Cd/Cl)", ImVec2(-1, 0)))
                exportCSVRequested = true;
            if (ImGui::Button("Export Flow Field CSV", ImVec2(-1, 0)))
                exportFlowCSVRequested = true;

            ImGui::Spacing();
            ImGui::TextDisabled("Reports");
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.4f, 0.7f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.5f, 0.85f, 1.0f));
            if (ImGui::Button("Full Report (HTML)", ImVec2(-1, 0)))
                exportReportRequested = true;
            ImGui::PopStyleColor(2);
        }

        if (ImGui::Button("Reset Simulation", ImVec2(-1, 0)))
            simResetRequested = true;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.4f, 1.0f), "FPS: %.0f", ImGui::GetIO().Framerate);

    ImGui::End();
}
