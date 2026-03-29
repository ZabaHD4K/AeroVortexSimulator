#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>
#include "app.h"

// Global app pointer for callbacks
static App* g_app = nullptr;

static void dropCallback(GLFWwindow*, int count, const char** paths) {
    g_app->onDrop(count, paths);
}

int main() {
    // ── GLFW init ────────────────────────────────
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW" << std::endl;
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(1600, 900, "Aero3D Simulator", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // ── GLAD ─────────────────────────────────────
    if (!gladLoadGL(glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD" << std::endl;
        return 1;
    }
    std::cout << "[OpenGL] " << glGetString(GL_VERSION) << std::endl;
    std::cout << "[GPU]    " << glGetString(GL_RENDERER) << std::endl;

    // ── ImGui ────────────────────────────────────
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\segoeui.ttf", 18.0f);

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;

    // install_callbacks=true: ImGui handles all GLFW input
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 460");

    // Drop callback (ImGui doesn't handle this)
    glfwSetDropCallback(window, dropCallback);

    // ── App ──────────────────────────────────────
    App app;
    g_app = &app;

    if (!app.init(window)) {
        std::cerr << "Failed to init app" << std::endl;
        return 1;
    }

    // ── Main loop ────────────────────────────────
    double lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        double now = glfwGetTime();
        float dt = (float)(now - lastTime);
        lastTime = now;
        if (dt > 0.1f) dt = 0.1f;

        app.update(dt);
        app.render();

        glfwSwapBuffers(window);
    }

    // ── Cleanup ──────────────────────────────────
    app.shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
