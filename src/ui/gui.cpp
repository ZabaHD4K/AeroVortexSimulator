#include "gui.h"
#include "app.h"
#include <imgui.h>
#include <filesystem>
#include <algorithm>
#include <cstdio>

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
    ImGui::SetNextWindowSize(ImVec2(320, 0), ImGuiCond_FirstUseEver);

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

    ImGui::Spacing();

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

        // Wind direction selector
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

        ImGui::Spacing();
        ImGui::Text("Step: %d  |  Cd: %.3f  Cl: %.3f",
                     app.lbm3d.getStep(), app.aeroCoeffs.Cd, app.aeroCoeffs.Cl);
    }

    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.4f, 1.0f), "FPS: %.0f", ImGui::GetIO().Framerate);

    ImGui::End();
}
