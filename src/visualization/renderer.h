#pragma once
#include "camera.h"
#include "geometry/mesh.h"
#include <glm/glm.hpp>

class Renderer {
public:
    bool init();
    void shutdown();
    void renderModel(const Model& model, const Camera& cam, float aspect);
    void renderGrid(const Camera& cam, float aspect);

    // Visual settings
    glm::vec3 modelColor{0.6f, 0.65f, 0.7f};
    glm::vec3 lightDir{0.4f, 0.8f, 0.4f};
    bool wireframe = false;
    bool showGrid = true;

private:
    unsigned int meshShader = 0;
    unsigned int gridShader = 0;
    unsigned int gridVAO = 0, gridVBO = 0;
    int gridVertCount = 0;

    unsigned int compileShader(const char* vertSrc, const char* fragSrc);
    void buildGrid();
};
