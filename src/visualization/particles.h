#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include "core/lbm3d.cuh"

class ParticleRenderer {
public:
    bool init();
    void shutdown();

    // Advect existing particles and inject new ones along wind jets
    // inletDir: normalized freestream direction (for deviation coloring)
    void update(float dt, const float* ux, const float* uy, const float* uz,
                int nx, int ny, int nz, const uint8_t* cellTypes,
                WindDirection windDir = WIND_POS_X,
                glm::vec3 inletDir = glm::vec3(1,0,0));

    // Render wind jets as connected line strips
    void render(const glm::mat4& mvp, float maxVel, float voxelSize = 0.01f);

    int numJets = 200;         // max wind jet streams (auto-sized to model)

private:
    unsigned int shader = 0;
    unsigned int VAO = 0, VBO = 0;

    struct Particle {
        glm::vec3 pos;
        float velocity;   // magnitude for legacy
        float deviation;  // angle deviation from freestream [0,1]
        int jetId;
        int bounces = 0;
        float age = 0.0f; // frames alive
    };
    std::vector<Particle> particles;

    // Fixed seed points for jets (transverse-axis positions at inlet)
    std::vector<glm::vec2> jetSeeds;
    bool seedsGenerated = false;
    int lastNy = 0, lastNz = 0;

    // Cached grid dimensions
    int cachedNx = 0, cachedNy = 0, cachedNz = 0;

    // Cached wind direction for injection
    WindDirection cachedWindDir = WIND_POS_X;

    void generateSeeds(int nx, int ny, int nz, const uint8_t* cellTypes,
                       WindDirection windDir);

    glm::vec3 interpolateVelocity(const float* ux, const float* uy, const float* uz,
                                   int nx, int ny, int nz, glm::vec3 pos);
};
