#include "app.h"
#include "geometry/model_loader.h"
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>

bool App::init(GLFWwindow* win) {
    window = win;
    if (!renderer.init()) {
        std::cerr << "[App] Failed to init renderer" << std::endl;
        return false;
    }
    return true;
}

void App::shutdown() {
    if (model)
        freeModelGPU(*model);
    renderer.shutdown();
}

void App::handleInput() {
    // Skip if ImGui wants the mouse
    if (ImGui::GetIO().WantCaptureMouse) {
        firstMouse = true;
        return;
    }

    double mx, my;
    glfwGetCursorPos(window, &mx, &my);

    if (firstMouse) {
        lastMouseX = mx;
        lastMouseY = my;
        firstMouse = false;
    }

    float dx = (float)(mx - lastMouseX);
    float dy = (float)(my - lastMouseY);
    lastMouseX = mx;
    lastMouseY = my;

    // Orbit with left mouse
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
        camera.orbit(dx, -dy);

    // Pan with right mouse or middle mouse
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS ||
        glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS)
        camera.pan(dx, dy);

    // Scroll zoom
    float scroll = ImGui::GetIO().MouseWheel;
    if (scroll != 0.0f)
        camera.zoom(scroll);

    // Keyboard shortcuts (only when ImGui doesn't want keyboard)
    if (!ImGui::GetIO().WantCaptureKeyboard) {
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
            camera.reset();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

void App::update(float dt) {
    (void)dt;
    handleInput();

    if (gui.wantsLoadModel()) {
        gui.clearLoadRequest();
        std::string path = openFileDialog();
        if (!path.empty())
            loadModelFromPath(path);
    }
}

void App::render() {
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    if (w == 0 || h == 0) return;

    glViewport(0, 0, w, h);
    glClearColor(0.18f, 0.20f, 0.24f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    float aspect = (float)w / (float)h;

    renderer.renderGrid(camera, aspect);

    if (model)
        renderer.renderModel(*model, camera, aspect);

    // ImGui
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    gui.render(renderer, camera, model ? &*model : nullptr);

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void App::onDrop(int count, const char** paths) {
    if (count > 0)
        loadModelFromPath(paths[0]);
}

void App::loadModelFromPath(const std::string& path) {
    auto newModel = loadModel(path);
    if (newModel) {
        if (model) freeModelGPU(*model);
        model = std::move(*newModel);
        camera.reset();
        std::cout << "[App] Model loaded: " << model->name << std::endl;
    } else {
        std::cerr << "[App] Failed to load: " << path << std::endl;
    }
}
