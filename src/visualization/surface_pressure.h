#pragma once
#include "geometry/mesh.h"
#include <glm/glm.hpp>
#include <vector>

class SurfacePressureRenderer {
public:
    bool init();
    void shutdown();

    // Sample pressure field at model vertices and render
    // model: the loaded 3D model
    // pressure: 3D pressure field (nx*ny*nz)
    // nx,ny,nz: grid dimensions
    // domainScale: same as used in voxelization
    void render(const Model& model, const float* pressure,
                int nx, int ny, int nz, float voxelSize,
                const glm::mat4& mvp, const glm::vec3& camPos,
                float minP, float maxP);

private:
    unsigned int shader = 0;
    // Uses model's existing VAO but with a custom shader that colors by pressure
    // We need a separate VBO for pressure values per vertex
    unsigned int pressureVBO = 0;
    std::vector<float> vertexPressures;
    unsigned int customVAO = 0;
    unsigned int posVBO = 0;
    unsigned int normalVBO = 0;
    unsigned int pressureAttribVBO = 0;
    unsigned int EBO = 0;
    int indexCount = 0;

    // Track which model we built geometry for (to avoid rebuilding pos/normal each frame)
    const Model* cachedModel = nullptr;
};
