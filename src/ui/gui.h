#pragma once
#include "visualization/renderer.h"
#include "visualization/camera.h"
#include "geometry/mesh.h"
#include <string>

class Gui {
public:
    void render(Renderer& renderer, Camera& cam, const Model* model);
    bool wantsLoadModel() const { return loadRequested; }
    void clearLoadRequest() { loadRequested = false; }

private:
    bool loadRequested = false;
};

// Opens native Windows file dialog, returns path or empty string
std::string openFileDialog();
