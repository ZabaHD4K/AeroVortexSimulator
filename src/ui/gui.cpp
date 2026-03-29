#include "gui.h"
#include <imgui.h>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>

std::string openFileDialog() {
    char filename[MAX_PATH] = "";
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = "3D Models\0*.stl;*.obj;*.fbx;*.ply;*.dae;*.3ds;*.gltf;*.glb\0All Files\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = "Load 3D Model";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileNameA(&ofn))
        return std::string(filename);
    return "";
}

void Gui::render(Renderer& renderer, Camera& cam, const Model* model) {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320, 0), ImGuiCond_FirstUseEver);

    ImGui::Begin("Aero3D Simulator");

    // ── Model ────────────────────────────────────
    ImGui::SeparatorText("Model");

    if (ImGui::Button("Load Model (.stl/.obj/...)", ImVec2(-1, 30)))
        loadRequested = true;

    if (model) {
        ImGui::Text("Name: %s", model->name.c_str());
        int totalVerts = 0, totalTris = 0;
        for (auto& m : model->meshes) {
            totalVerts += (int)m.vertices.size();
            totalTris += (int)m.indices.size() / 3;
        }
        ImGui::Text("Meshes: %d", (int)model->meshes.size());
        ImGui::Text("Vertices: %s", std::to_string(totalVerts).c_str());
        ImGui::Text("Triangles: %s", std::to_string(totalTris).c_str());
    } else {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No model loaded");
    }

    // ── Render settings ──────────────────────────
    ImGui::SeparatorText("Render");

    ImGui::ColorEdit3("Model Color", &renderer.modelColor.x);
    ImGui::Checkbox("Wireframe", &renderer.wireframe);
    ImGui::Checkbox("Show Grid", &renderer.showGrid);
    ImGui::DragFloat3("Light Dir", &renderer.lightDir.x, 0.01f, -1.0f, 1.0f);

    // ── Camera ───────────────────────────────────
    ImGui::SeparatorText("Camera");

    ImGui::Text("Distance: %.2f", cam.distance);
    ImGui::Text("Yaw: %.1f  Pitch: %.1f", cam.yaw, cam.pitch);
    if (ImGui::Button("Reset Camera"))
        cam.reset();

    ImGui::SameLine();
    if (ImGui::Button("Front")) { cam.yaw = 0; cam.pitch = 0; }
    ImGui::SameLine();
    if (ImGui::Button("Side"))  { cam.yaw = 90; cam.pitch = 0; }
    ImGui::SameLine();
    if (ImGui::Button("Top"))   { cam.yaw = 0; cam.pitch = 89; }

    // ── Info ─────────────────────────────────────
    ImGui::SeparatorText("Info");
    ImGui::Text("FPS: %.0f", ImGui::GetIO().Framerate);

    ImGui::End();
}
