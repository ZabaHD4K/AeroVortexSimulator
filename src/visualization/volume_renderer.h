#pragma once
#include <glm/glm.hpp>

enum class VolumeColormap { VIRIDIS, COOLWARM, INFERNO };

class VolumeRenderer {
public:
    bool init();
    void shutdown();

    // Upload a 3D scalar field as a GL_TEXTURE_3D
    void uploadField(const float* data, int nx, int ny, int nz);

    // Render the volume via front-face ray marching of a proxy cube
    void render(const glm::mat4& view, const glm::mat4& proj,
                const glm::vec3& camPos,
                float minVal, float maxVal);

    // Tunables
    VolumeColormap colormap = VolumeColormap::VIRIDIS;
    float opacity     = 0.5f;
    float densityScale = 15.0f;
    float stepSize     = 0.004f;

private:
    unsigned int shader  = 0;
    unsigned int VAO     = 0;
    unsigned int VBO     = 0;
    unsigned int EBO     = 0;
    unsigned int tex3D   = 0;
    int          texNx = 0, texNy = 0, texNz = 0;

    // Volume AABB in world space (set by uploadField)
    glm::vec3 boxMin{-1}, boxMax{1};
};
