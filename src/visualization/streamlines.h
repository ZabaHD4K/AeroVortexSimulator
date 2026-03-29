#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include "core/lbm3d.cuh"

class StreamlineRenderer {
public:
    bool init();
    void shutdown();

    // Generate streamlines from velocity field
    // ux,uy,uz: host arrays of size nx*ny*nz
    // nx,ny,nz: grid dimensions
    // numLines: how many streamlines to seed
    // maxSteps: max integration steps per line
    void generate(const float* ux, const float* uy, const float* uz,
                  int nx, int ny, int nz, const uint8_t* cellTypes,
                  int numLines = 600, int maxSteps = 2000,
                  WindDirection windDir = WIND_POS_X);

    // Render all streamlines
    // mvp: model-view-projection matrix
    // Maps grid coordinates to [-1,1] normalized space matching the model viewer
    void render(const glm::mat4& mvp, float maxVel);

    int getLineCount() const { return lineCount; }

private:
    unsigned int shader = 0;
    unsigned int VAO = 0, VBO = 0;
    int vertexCount = 0;
    int lineCount = 0;

    struct StreamVertex {
        glm::vec3 pos;
        float velocity; // magnitude, for coloring
        float alpha;    // opacity (fades along the line)
    };

    // Segment offsets for GL_LINE_STRIP rendering (one per streamline)
    std::vector<int> lineOffsets; // start index
    std::vector<int> lineLengths; // vertex count

    // RK4 integration of velocity field
    glm::vec3 interpolateVelocity(const float* ux, const float* uy, const float* uz,
                                   int nx, int ny, int nz, glm::vec3 pos);
};
