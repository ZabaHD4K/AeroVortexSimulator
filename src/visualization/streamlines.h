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
    // inletDir: normalized freestream direction (for deviation coloring)
    void generate(const float* ux, const float* uy, const float* uz,
                  int nx, int ny, int nz, const uint8_t* cellTypes,
                  int numLines = 300, int maxSteps = 4000,
                  WindDirection windDir = WIND_POS_X,
                  float voxelSize = 0.01f,
                  glm::vec3 inletDir = glm::vec3(1,0,0));

    void render(const glm::mat4& mvp);

    int getLineCount() const { return lineCount; }

private:
    unsigned int shader = 0;
    unsigned int VAO = 0, VBO = 0;
    int vertexCount = 0;
    int lineCount = 0;

    struct StreamVertex {
        glm::vec3 pos;
        float deviation; // angle deviation from freestream [0,1]
        float alpha;     // opacity (fades along the line)
    };

    // Segment offsets for GL_LINE_STRIP rendering (one per streamline)
    std::vector<int> lineOffsets;
    std::vector<int> lineLengths;

    // RK4 integration of velocity field
    glm::vec3 interpolateVelocity(const float* ux, const float* uy, const float* uz,
                                   int nx, int ny, int nz, glm::vec3 pos);
};
