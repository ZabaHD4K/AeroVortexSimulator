#pragma once
#include "visualization/renderer.h"
#include "visualization/camera.h"
#include "visualization/field2d.h"
#include "visualization/streamlines.h"
#include "visualization/slice_plane.h"
#include "visualization/surface_pressure.h"
#include "visualization/particles.h"
#include "visualization/volume_renderer.h"
#include "ui/gui.h"
#include "geometry/mesh.h"
#include "core/lbm3d.cuh"
#include "core/lbm2d.cuh"
#include "core/aero_forces.h"
#include <optional>
#include <vector>

struct GLFWwindow;

enum class AppMode {
    MODEL_VIEWER,
    LBM_2D,
    SIMULATION_3D
};

class App {
    friend class Gui;
public:
    bool init(GLFWwindow* window);
    void shutdown();
    void update(float dt);
    void render();

    void onDrop(int count, const char** paths);

    // Exposed for GUI
    AppMode mode = AppMode::MODEL_VIEWER;
    LBM2DParams lbm2dParams;
    LBM3DParams lbm3dParams;
    AeroCoefficients aeroCoeffs;

    // Visualization toggles
    bool showStreamlines = true;
    bool showSlicePlane = true;
    bool showSurfacePressure = true;
    bool showParticles = false;
    bool showModel = true;
    bool showVolume = false;

    // Slice plane config
    int sliceAxis = 0;   // 0=X, 1=Y, 2=Z
    int sliceIndex = 50;
    int sliceField = 0;  // 0=velocity, 1=pressure, 2=vorticity

    // Streamline config
    int numStreamlines = 300;

    // Volume rendering config
    int volumeField = 0; // 0=velocity, 1=pressure, 2=vorticity

    // Simulation state
    bool simInitialized = false;
    float domainScale = 0.4f;

    // Aero coefficient history for plotting
    std::vector<float> cdHistory;
    std::vector<float> clHistory;

private:
    GLFWwindow* window = nullptr;
    Renderer renderer;
    Camera camera;
    Gui gui;
    Field2DRenderer field2d;
    StreamlineRenderer streamlines;
    SlicePlaneRenderer slicePlane;
    SurfacePressureRenderer surfacePressure;
    ParticleRenderer particles;
    VolumeRenderer volumeRenderer;
    LBM2D lbm2d;
    LBM3D lbm3d;
    std::optional<Model> model;

    // Voxel grid (host copy)
    std::vector<uint8_t> voxelGrid;

    // Mouse state
    double lastMouseX = 0, lastMouseY = 0;
    bool firstMouse = true;

    void handleInput();
    void loadModelFromPath(const std::string& path);
    void initSimulation();  // voxelize model + setup LBM3D
    void updateSimulation(float dt);
    void renderSimulation3D(float aspect);
};
