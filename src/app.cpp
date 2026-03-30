#include "app.h"
#include "geometry/model_loader.h"
#include "core/voxelizer.h"
#include "core/aero_forces.h"
#include "export/data_export.h"
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

static std::string findModelsDir() {
    if (fs::exists("models")) return fs::absolute("models").string();
    if (fs::exists("../models")) return fs::absolute("../models").string();
    return "";
}

bool App::init(GLFWwindow* win) {
    window = win;

    if (!renderer.init()) { std::cerr << "[App] Failed to init renderer\n"; return false; }
    if (!field2d.init())  { std::cerr << "[App] Failed to init field2d\n"; return false; }
    if (!streamlines.init()) { std::cerr << "[App] Failed to init streamlines\n"; return false; }
    if (!slicePlane.init())  { std::cerr << "[App] Failed to init slice plane\n"; return false; }
    if (!surfacePressure.init()) { std::cerr << "[App] Failed to init surface pressure\n"; return false; }
    if (!particles.init()) { std::cerr << "[App] Failed to init particles\n"; return false; }
    if (!volumeRenderer.init()) { std::cerr << "[App] Failed to init volume renderer\n"; return false; }

    if (!lbm2d.init(lbm2dParams.nx, lbm2dParams.ny)) {
        std::cerr << "[App] Failed to init LBM2D\n"; return false;
    }

    std::string modelsPath = findModelsDir();
    if (!modelsPath.empty())
        std::cout << "[App] Models directory: " << modelsPath << std::endl;
    gui.init(modelsPath);

    return true;
}

void App::shutdown() {
    if (model) freeModelGPU(*model);
    if (simInitialized) lbm3d.shutdown();
    lbm2d.shutdown();
    volumeRenderer.shutdown();
    particles.shutdown();
    surfacePressure.shutdown();
    slicePlane.shutdown();
    streamlines.shutdown();
    field2d.shutdown();
    renderer.shutdown();
}

void App::handleInput() {
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

    if (mode != AppMode::LBM_2D) {
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
            camera.orbit(dx, -dy);
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS ||
            glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS)
            camera.pan(dx, dy);
        float scroll = ImGui::GetIO().MouseWheel;
        if (scroll != 0.0f) camera.zoom(scroll);
    }

    if (!ImGui::GetIO().WantCaptureKeyboard) {
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) camera.reset();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

void App::initSimulation() {
    if (!model) return;

    std::cout << "[App] Voxelizing model for simulation..." << std::endl;
    VoxelGrid grid = voxelizeModel(*model, lbm3dParams.nx, lbm3dParams.ny, lbm3dParams.nz,
                                   domainScale, lbm3dParams.windDir);
    voxelGrid = grid.cells;

    // Init LBM3D
    if (simInitialized) lbm3d.shutdown();
    if (!lbm3d.init(lbm3dParams.nx, lbm3dParams.ny, lbm3dParams.nz)) {
        std::cerr << "[App] Failed to init LBM3D\n";
        return;
    }
    lbm3d.setCellTypes(voxelGrid.data());
    lbm3d.setTau(lbm3dParams.tau);
    lbm3d.setInletVelocity(lbm3dParams.inletVelocity);
    lbm3d.setCollisionModel(lbm3dParams.collisionModel);
    lbm3d.setSmagorinsky(lbm3dParams.useSmagorinsky, lbm3dParams.smagorinskyCs);

    // Set 3-component inlet velocity based on wind direction
    float u = lbm3dParams.inletVelocity;
    float iux = 0, iuy = 0, iuz = 0;
    switch (lbm3dParams.windDir) {
        case WIND_POS_X: iux =  u; break;
        case WIND_NEG_X: iux = -u; break;
        case WIND_POS_Z: iuz =  u; break;
        case WIND_NEG_Z: iuz = -u; break;
        case WIND_POS_Y: iuy =  u; break;
        case WIND_NEG_Y: iuy = -u; break;
    }
    lbm3d.setInletDirection(iux, iuy, iuz);
    lbm3d.reset();  // re-init populations with correct cell types

    simInitialized = true;
    sliceIndex = lbm3dParams.nx / 2;
    cdHistory.clear();
    clHistory.clear();

    std::cout << "[App] Simulation initialized: "
              << lbm3dParams.nx << "x" << lbm3dParams.ny << "x" << lbm3dParams.nz
              << " grid" << std::endl;
}

void App::updateSimulation(float dt) {
    if (!simInitialized || !lbm3dParams.running) return;

    lbm3d.setTau(lbm3dParams.tau);
    lbm3d.setInletVelocity(lbm3dParams.inletVelocity);
    lbm3d.setCollisionModel(lbm3dParams.collisionModel);
    lbm3d.setSmagorinsky(lbm3dParams.useSmagorinsky, lbm3dParams.smagorinskyCs);
    {
        float u = lbm3dParams.inletVelocity;
        float iux = 0, iuy = 0, iuz = 0;
        switch (lbm3dParams.windDir) {
            case WIND_POS_X: iux =  u; break;
            case WIND_NEG_X: iux = -u; break;
            case WIND_POS_Z: iuz =  u; break;
            case WIND_NEG_Z: iuz = -u; break;
            case WIND_POS_Y: iuy =  u; break;
            case WIND_NEG_Y: iuy = -u; break;
        }
        lbm3d.setInletDirection(iux, iuy, iuz);
    }
    lbm3d.stepMultiple(lbm3dParams.stepsPerFrame);

    // Invalidate field caches after stepping — forces fresh download on next access
    lbm3d.invalidateCache();

    // Update particles (uses cached velocity components)
    if (showParticles) {
        float *ux, *uy, *uz;
        lbm3d.getVelocityComponents(&ux, &uy, &uz);
        particles.update(dt, ux, uy, uz,
                         lbm3dParams.nx, lbm3dParams.ny, lbm3dParams.nz,
                         voxelGrid.data());
    }

    // Calculate aero coefficients every 50 steps (uses cached pressure)
    if (lbm3d.getStep() % 50 == 0) {
        const float* pressure = lbm3d.getPressureField();
        aeroCoeffs = calculateAeroCoefficients(
            pressure, voxelGrid.data(),
            lbm3dParams.nx, lbm3dParams.ny, lbm3dParams.nz,
            lbm3dParams.inletVelocity, domainScale);

        cdHistory.push_back(aeroCoeffs.Cd);
        clHistory.push_back(aeroCoeffs.Cl);
        if (cdHistory.size() > 500) {
            cdHistory.erase(cdHistory.begin());
            clHistory.erase(clHistory.begin());
        }
    }
}

void App::renderSimulation3D(float aspect) {
    glm::mat4 view = camera.getViewMatrix();
    glm::mat4 proj = camera.getProjectionMatrix(aspect);
    glm::mat4 mvp = proj * view;

    int nx = lbm3dParams.nx, ny = lbm3dParams.ny, nz = lbm3dParams.nz;

    // Render model
    if (showModel && model) {
        renderer.renderModel(*model, camera, aspect);
    }

    if (simInitialized) {
        float *ux, *uy, *uz;
        lbm3d.getVelocityComponents(&ux, &uy, &uz);

        // Streamlines
        if (showStreamlines) {
            if (lbm3d.getStep() % 50 == 0 || streamlines.getLineCount() == 0) {
                streamlines.generate(ux, uy, uz, nx, ny, nz,
                                     voxelGrid.data(), numStreamlines, 4000,
                                     lbm3dParams.windDir);
            }
            streamlines.render(mvp, lbm3dParams.inletVelocity);
        }

        // Slice plane
        if (showSlicePlane) {
            const float* sliceField3D = nullptr;
            float sMin = 0, sMax = 0.1f;
            if (sliceField == 0) {
                sliceField3D = lbm3d.getVelocityMagnitude();
                sMax = lbm3dParams.inletVelocity * 1.5f;
            } else if (sliceField == 1) {
                sliceField3D = lbm3d.getPressureField();
                sMin = 0.33f - 0.01f; sMax = 0.33f + 0.01f;
            } else {
                sliceField3D = lbm3d.getVorticityMagnitude();
                sMax = 0.05f;
            }
            slicePlane.render(sliceField3D, nx, ny, nz,
                              (SliceAxis)sliceAxis, sliceIndex,
                              mvp, sMin, sMax, voxelGrid.data());
        }

        // Surface pressure
        if (showSurfacePressure && model) {
            const float* pField = lbm3d.getPressureField();
            surfacePressure.render(*model, pField, nx, ny, nz, domainScale,
                                   mvp, camera.getPosition(),
                                   0.33f - 0.01f, 0.33f + 0.01f);
        }

        // Volume rendering
        if (showVolume) {
            const float* volField = nullptr;
            float vMin = 0, vMax = 0.1f;
            if (volumeField == 0) {
                volField = lbm3d.getVelocityMagnitude();
                vMax = lbm3dParams.inletVelocity * 1.5f;
            } else if (volumeField == 1) {
                volField = lbm3d.getPressureField();
                vMin = 0.33f - 0.01f; vMax = 0.33f + 0.01f;
            } else {
                volField = lbm3d.getVorticityMagnitude();
                vMax = 0.05f;
            }
            volumeRenderer.uploadField(volField, nx, ny, nz);
            volumeRenderer.render(view, proj, camera.getPosition(), vMin, vMax);
        }

        // Particles
        if (showParticles) {
            particles.render(mvp, lbm3dParams.inletVelocity * 1.5f);
        }
    }

    // Grid floor
    renderer.renderGrid(camera, aspect);
}

void App::update(float dt) {
    handleInput();

    // GUI actions
    if (gui.wantsLoadModel()) {
        gui.clearLoadRequest();
        std::string path = openFileDialog();
        if (!path.empty()) loadModelFromPath(path);
    }
    if (gui.wantsLoadFromLibrary()) {
        std::string path = gui.getLibrarySelection();
        gui.clearLibraryRequest();
        loadModelFromPath(path);
    }
    if (gui.wantsLBMReset()) {
        gui.clearLBMReset();
        if (mode == AppMode::LBM_2D) lbm2d.reset();
    }
    if (gui.wantsSimInit()) {
        gui.clearSimInit();
        initSimulation();
    }
    if (gui.wantsWindStart()) {
        gui.clearWindStart();
        initSimulation();
        if (simInitialized) {
            mode = AppMode::SIMULATION_3D;
            lbm3dParams.running = true;
            showStreamlines = true;
            showParticles = false;
            showModel = true;
            showSlicePlane = false;
            showSurfacePressure = false;
            // Camera: behind-left, looking along the wind direction (+X)
            camera.yaw = 155.0f;
            camera.pitch = 18.0f;
            camera.distance = 2.8f;
            camera.target = glm::vec3(0.0f);
        }
    }
    if (gui.wantsWindDirChange()) {
        gui.clearWindDirChange();
        if (simInitialized && model) {
            bool wasRunning = lbm3dParams.running;
            initSimulation();
            if (simInitialized) {
                lbm3dParams.running = wasRunning;
            }
        }
    }
    if (gui.wantsSimReset()) {
        gui.clearSimReset();
        if (simInitialized) {
            lbm3d.reset();
            lbm3d.setCellTypes(voxelGrid.data());
            cdHistory.clear();
            clHistory.clear();
        }
    }
    if (gui.wantsScreenshot()) {
        gui.clearScreenshot();
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        std::string fname = DataExporter::generateFilename("screenshot", "bmp");
        if (DataExporter::saveScreenshot(fname, w, h))
            std::cout << "[Export] Screenshot saved: " << fname << std::endl;
    }
    if (gui.wantsExportVTK()) {
        gui.clearExportVTK();
        if (simInitialized) {
            std::string fname = DataExporter::generateFilename("field", "vts");
            const float* velMag = lbm3d.getVelocityMagnitude();
            const float* pres   = lbm3d.getPressureField();
            const float* vort   = lbm3d.getVorticityMagnitude();
            float *ux, *uy, *uz;
            lbm3d.getVelocityComponents(&ux, &uy, &uz);
            if (DataExporter::exportVTK(fname, velMag, pres, vort, ux, uy, uz,
                                         voxelGrid.data(),
                                         lbm3dParams.nx, lbm3dParams.ny, lbm3dParams.nz))
                std::cout << "[Export] VTK saved: " << fname << std::endl;
        }
    }
    if (gui.wantsExportCSV()) {
        gui.clearExportCSV();
        if (!cdHistory.empty()) {
            std::string fname = DataExporter::generateFilename("coefficients", "csv");
            if (DataExporter::exportCSV(fname, cdHistory, clHistory,
                                         lbm3dParams.tau, lbm3dParams.inletVelocity))
                std::cout << "[Export] CSV saved: " << fname << std::endl;
        }
    }

    // LBM 2D
    if (mode == AppMode::LBM_2D && lbm2dParams.running) {
        lbm2d.setTau(lbm2dParams.tau);
        lbm2d.setLidVelocity(lbm2dParams.lidVelocity);
        lbm2d.stepMultiple(lbm2dParams.stepsPerFrame);
    }

    // LBM 3D
    if (mode == AppMode::SIMULATION_3D) {
        updateSimulation(dt);
    }
}

void App::render() {
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    if (w == 0 || h == 0) return;

    glViewport(0, 0, w, h);
    glClearColor(0.14f, 0.16f, 0.20f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    float aspect = (float)w / (float)h;

    switch (mode) {
    case AppMode::MODEL_VIEWER:
        renderer.renderGrid(camera, aspect);
        if (model) renderer.renderModel(*model, camera, aspect);
        break;

    case AppMode::LBM_2D: {
        const float* field = lbm2d.getVelocityMagnitude();
        field2d.render(field, lbm2d.getNx(), lbm2d.getNy(), 0.0f,
                       lbm2dParams.lidVelocity * 1.2f);
        break;
    }

    case AppMode::SIMULATION_3D:
        renderSimulation3D(aspect);
        break;
    }

    // ImGui
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    gui.render(*this);

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void App::onDrop(int count, const char** paths) {
    if (count > 0) loadModelFromPath(paths[0]);
}

void App::loadModelFromPath(const std::string& path) {
    auto newModel = loadModel(path);
    if (newModel) {
        if (model) freeModelGPU(*model);
        if (simInitialized) { lbm3d.shutdown(); simInitialized = false; }
        model = std::move(*newModel);
        camera.reset();
        lbm3dParams.running = false;
        cdHistory.clear();
        clHistory.clear();
        if (mode == AppMode::SIMULATION_3D) mode = AppMode::MODEL_VIEWER;
        std::cout << "[App] Model loaded: " << model->name << std::endl;
    } else {
        std::cerr << "[App] Failed to load: " << path << std::endl;
    }
}
