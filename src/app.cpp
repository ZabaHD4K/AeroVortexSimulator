#include "app.h"
#include "geometry/model_loader.h"
#include "geometry/primitives.h"
#include "core/voxelizer.h"
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
#include <set>

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
    voxelSize = grid.voxelSize;

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
    lbm3d.setWindDirection(lbm3dParams.windDir);

    // Initialize at ramp-start velocity (60%) to avoid impulse shock
    // The ramp in updateSimulation() will bring it to 100% over 150 steps
    float u = lbm3dParams.inletVelocity * 0.6f;
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
    lbm3d.reset();  // re-init populations at low velocity (matching ramp start)

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
    lbm3d.setWindDirection(lbm3dParams.windDir);
    {
        // Gradual ramp-up: over the first 300 steps, linearly increase velocity
        // from 20% to 100%. Prevents the shock wave that destabilizes the solver.
        int step = lbm3d.getStep();
        float ramp = 1.0f;
        if (step < 150) {
            ramp = 0.6f + 0.4f * (float)step / 150.0f;
        }

        float u = lbm3dParams.inletVelocity * ramp;
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

    // Read convergence metric (already computed at end of stepMultiple)
    convergenceRMS = lbm3d.computeRMSChange();
    converged = lbm3d.isConverged(1e-5f);

    // Invalidate field caches after stepping — forces fresh download on next access
    lbm3d.invalidateCache();

    // Compute normalized inlet direction for coloring
    glm::vec3 inletDir(0.0f);
    switch (lbm3dParams.windDir) {
        case WIND_POS_X: inletDir = glm::vec3( 1, 0, 0); break;
        case WIND_NEG_X: inletDir = glm::vec3(-1, 0, 0); break;
        case WIND_POS_Z: inletDir = glm::vec3( 0, 0, 1); break;
        case WIND_NEG_Z: inletDir = glm::vec3( 0, 0,-1); break;
        case WIND_POS_Y: inletDir = glm::vec3( 0, 1, 0); break;
        case WIND_NEG_Y: inletDir = glm::vec3( 0,-1, 0); break;
    }

    // Update particles (uses cached velocity components)
    if (showParticles) {
        float *ux, *uy, *uz;
        lbm3d.getVelocityComponents(&ux, &uy, &uz);
        particles.update(dt, ux, uy, uz,
                         lbm3dParams.nx, lbm3dParams.ny, lbm3dParams.nz,
                         voxelGrid.data(), lbm3dParams.windDir, inletDir);
    }

    // Calculate aero coefficients every 50 steps via momentum exchange (GPU).
    // Skip first 500 steps — flow needs partial development.
    if (lbm3d.getStep() > 500 && lbm3d.getStep() % 50 == 0) {
        // Momentum exchange gives accurate forces directly in lattice units
        float rawFx = 0, rawFy = 0, rawFz = 0;
        lbm3d.computeForces(rawFx, rawFy, rawFz);

        // Frontal area from voxel grid (project solid onto YZ plane)
        std::set<int> projected;
        int nx_ = lbm3dParams.nx, ny_ = lbm3dParams.ny, nz_ = lbm3dParams.nz;
        for (int z = 0; z < nz_; z++)
            for (int y = 0; y < ny_; y++)
                for (int x = 0; x < nx_; x++)
                    if (voxelGrid[z * nx_ * ny_ + y * nx_ + x] == 1)
                        projected.insert(y * nz_ + z);
        float frontalArea = std::max((float)projected.size(), 1.0f);
        float q = 0.5f * 1.0f * lbm3dParams.inletVelocity * lbm3dParams.inletVelocity;

        AeroCoefficients raw;
        raw.Fx = rawFx;
        raw.Fy = rawFy;
        raw.Fz = rawFz;
        if (q > 1e-10f) {
            raw.Cd = rawFx / (q * frontalArea);
            raw.Cl = rawFy / (q * frontalArea);
            raw.Cs = rawFz / (q * frontalArea);
        }

        // Exponential moving average (alpha=0.08) for stable display
        const float alpha = 0.08f;
        if (cdHistory.empty()) {
            aeroCoeffs = raw;
        } else {
            aeroCoeffs.Cd = aeroCoeffs.Cd + alpha * (raw.Cd - aeroCoeffs.Cd);
            aeroCoeffs.Cl = aeroCoeffs.Cl + alpha * (raw.Cl - aeroCoeffs.Cl);
            aeroCoeffs.Cs = aeroCoeffs.Cs + alpha * (raw.Cs - aeroCoeffs.Cs);
            aeroCoeffs.Fx = aeroCoeffs.Fx + alpha * (raw.Fx - aeroCoeffs.Fx);
            aeroCoeffs.Fy = aeroCoeffs.Fy + alpha * (raw.Fy - aeroCoeffs.Fy);
            aeroCoeffs.Fz = aeroCoeffs.Fz + alpha * (raw.Fz - aeroCoeffs.Fz);
        }

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

        // Compute inlet direction for coloring
        glm::vec3 inletDir(0.0f);
        switch (lbm3dParams.windDir) {
            case WIND_POS_X: inletDir = glm::vec3( 1, 0, 0); break;
            case WIND_NEG_X: inletDir = glm::vec3(-1, 0, 0); break;
            case WIND_POS_Z: inletDir = glm::vec3( 0, 0, 1); break;
            case WIND_NEG_Z: inletDir = glm::vec3( 0, 0,-1); break;
            case WIND_POS_Y: inletDir = glm::vec3( 0, 1, 0); break;
            case WIND_NEG_Y: inletDir = glm::vec3( 0,-1, 0); break;
        }

        // Streamlines
        if (showStreamlines) {
            if (lbm3d.getStep() % 50 == 0 || streamlines.getLineCount() == 0) {
                streamlines.generate(ux, uy, uz, nx, ny, nz,
                                     voxelGrid.data(), numStreamlines, 4000,
                                     lbm3dParams.windDir, voxelSize, inletDir);
            }
            streamlines.render(mvp);
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
                              mvp, sMin, sMax, voxelGrid.data(), voxelSize);
        }

        // Surface pressure
        if (showSurfacePressure && model) {
            const float* pField = lbm3d.getPressureField();
            surfacePressure.render(*model, pField, nx, ny, nz, voxelSize,
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
            volumeRenderer.uploadField(volField, nx, ny, nz, voxelSize);
            volumeRenderer.render(view, proj, camera.getPosition(), vMin, vMax);
        }

        // Particles
        if (showParticles) {
            particles.render(mvp, lbm3dParams.inletVelocity * 1.5f, voxelSize);
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

    // Test model generation
    auto loadTestModel = [&](Model&& newModel, ValidationRef ref) {
        if (model) freeModelGPU(*model);
        if (simInitialized) { lbm3d.shutdown(); simInitialized = false; }
        model = std::move(newModel);
        validationActive = true;
        validationRef = ref;
        camera.reset();
        lbm3dParams.running = false;
        cdHistory.clear();
        clHistory.clear();
        if (mode == AppMode::SIMULATION_3D) mode = AppMode::MODEL_VIEWER;
        std::cout << "[App] Test model loaded: " << model->name << std::endl;
    };
    if (gui.wantsTestSphere()) {
        gui.clearTestSphere();
        loadTestModel(generateSphere(), getSphereRef());
    }
    if (gui.wantsTestCylinder()) {
        gui.clearTestCylinder();
        loadTestModel(generateCylinder(), getCylinderRef());
    }
    if (gui.wantsTestNACA()) {
        gui.clearTestNACA();
        loadTestModel(generateNACA0012(), getNACA0012Ref());
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
    if (gui.wantsExportFlowCSV()) {
        gui.clearExportFlowCSV();
        if (simInitialized) {
            std::string fname = DataExporter::generateFilename("flowfield", "csv");
            const float* velMag = lbm3d.getVelocityMagnitude();
            const float* pres   = lbm3d.getPressureField();
            const float* vort   = lbm3d.getVorticityMagnitude();
            float *ux, *uy, *uz;
            lbm3d.getVelocityComponents(&ux, &uy, &uz);
            if (DataExporter::exportFlowFieldCSV(fname, velMag, pres, vort, ux, uy, uz,
                                                  voxelGrid.data(),
                                                  lbm3dParams.nx, lbm3dParams.ny, lbm3dParams.nz,
                                                  voxelSize))
                std::cout << "[Export] Flow field CSV saved: " << fname << std::endl;
        }
    }
    if (gui.wantsExportReport()) {
        gui.clearExportReport();
        if (simInitialized) {
            std::string fname = DataExporter::generateFilename("report", "html");

            // Build report data
            SimReportData rd;
            if (model) {
                rd.modelName = model->name;
                rd.boundingRadius = model->radius;
                rd.numMeshes = (int)model->meshes.size();
                for (auto& m : model->meshes) {
                    rd.totalVertices += (int)m.vertices.size();
                    rd.totalTriangles += (int)m.indices.size() / 3;
                }
            }

            rd.nx = lbm3dParams.nx;
            rd.ny = lbm3dParams.ny;
            rd.nz = lbm3dParams.nz;
            rd.domainScale = domainScale;
            rd.voxelSize = voxelSize;

            // Cell counts
            for (auto c : voxelGrid) {
                switch (c) {
                    case 0: rd.fluidCells++;  break;
                    case 1: rd.solidCells++;  break;
                    case 2: rd.inletCells++;  break;
                    case 3: rd.outletCells++; break;
                }
            }

            rd.tau = lbm3dParams.tau;
            rd.inletVelocity = lbm3dParams.inletVelocity;
            rd.windDir = (int)lbm3dParams.windDir;
            rd.collisionModel = (int)lbm3dParams.collisionModel;
            rd.smagorinsky = lbm3dParams.useSmagorinsky;
            rd.smagorinskyCs = lbm3dParams.smagorinskyCs;
            rd.viscosity = (lbm3dParams.tau - 0.5f) / 3.0f;
            rd.reynoldsNumber = reynoldsNumber(lbm3dParams.inletVelocity,
                                                (float)lbm3dParams.ny * 0.4f,
                                                lbm3dParams.tau);

            rd.currentStep = lbm3d.getStep();
            rd.stepsPerFrame = lbm3dParams.stepsPerFrame;

            rd.Cd = aeroCoeffs.Cd;
            rd.Cl = aeroCoeffs.Cl;
            rd.Cs = aeroCoeffs.Cs;
            rd.Fx = aeroCoeffs.Fx;
            rd.Fy = aeroCoeffs.Fy;
            rd.Fz = aeroCoeffs.Fz;

            rd.cdHistory = cdHistory;
            rd.clHistory = clHistory;

            rd.validationActive = validationActive;
            if (validationActive) {
                rd.expectedCd = validationRef.expectedCd;
                rd.expectedCl = validationRef.expectedCl;
                rd.validationSource = validationRef.source;
            }

            // Compute flow field statistics
            const float* velMag = lbm3d.getVelocityMagnitude();
            const float* pres   = lbm3d.getPressureField();
            const float* vort   = lbm3d.getVorticityMagnitude();
            int N = lbm3dParams.nx * lbm3dParams.ny * lbm3dParams.nz;
            DataExporter::computeFieldStats(velMag, voxelGrid.data(), N,
                                            rd.velMin, rd.velMax, rd.velMean);
            DataExporter::computeFieldStats(pres, voxelGrid.data(), N,
                                            rd.presMin, rd.presMax, rd.presMean);
            DataExporter::computeFieldStats(vort, voxelGrid.data(), N,
                                            rd.vortMin, rd.vortMax, rd.vortMean);

            if (DataExporter::exportReport(fname, rd))
                std::cout << "[Export] Report saved: " << fname << std::endl;
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
        validationActive = false;
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
