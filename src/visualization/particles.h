#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

class ParticleRenderer {
public:
    bool init();
    void shutdown();

    // Advect existing particles and inject new ones along wind jets
    void update(float dt, const float* ux, const float* uy, const float* uz,
                int nx, int ny, int nz, const uint8_t* cellTypes);

    // Render wind jets as connected line strips
    void render(const glm::mat4& mvp, float maxVel);

    int numJets = 120;         // number of wind jet streams

private:
    unsigned int shader = 0;
    unsigned int VAO = 0, VBO = 0;

    struct Particle {
        glm::vec3 pos;
        float velocity;  // magnitude for coloring
        int jetId;       // which jet this particle belongs to
    };
    std::vector<Particle> particles;

    // Fixed seed points for jets (Y-Z positions at inlet)
    std::vector<glm::vec2> jetSeeds;
    bool seedsGenerated = false;
    int lastNy = 0, lastNz = 0;

    // Cached grid dimensions
    int cachedNx = 0, cachedNy = 0, cachedNz = 0;

    void generateSeeds(int ny, int nz);

    glm::vec3 interpolateVelocity(const float* ux, const float* uy, const float* uz,
                                   int nx, int ny, int nz, glm::vec3 pos);
};
