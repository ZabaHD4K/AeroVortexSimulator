#pragma once
#include "visualization/renderer.h"
#include "visualization/camera.h"
#include "ui/gui.h"
#include "geometry/mesh.h"
#include <optional>

struct GLFWwindow;

class App {
public:
    bool init(GLFWwindow* window);
    void shutdown();
    void update(float dt);
    void render();

    // Drag & drop (called from main callback)
    void onDrop(int count, const char** paths);

private:
    GLFWwindow* window = nullptr;
    Renderer renderer;
    Camera camera;
    Gui gui;
    std::optional<Model> model;

    // Mouse state for polling
    double lastMouseX = 0, lastMouseY = 0;
    bool firstMouse = true;

    void handleInput();
    void loadModelFromPath(const std::string& path);
};
