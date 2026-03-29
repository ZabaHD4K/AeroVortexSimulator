#include "particles.h"
#include <glad/gl.h>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>

// ── Shaders ─────────────────────────────────────────

static const char* jetVertSrc = R"(
#version 460 core
layout(location=0) in vec3 aPos;
layout(location=1) in float aVelocity;

uniform mat4 uMVP;
uniform float uMaxVel;

out float vT;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vT = clamp(aVelocity / max(uMaxVel, 1e-6), 0.0, 1.0);
}
)";

static const char* jetFragSrc = R"(
#version 460 core
in float vT;
out vec4 FragColor;

vec3 viridis(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.267, 0.004, 0.329);
    vec3 c1 = vec3(0.282, 0.140, 0.458);
    vec3 c2 = vec3(0.212, 0.359, 0.551);
    vec3 c3 = vec3(0.127, 0.566, 0.551);
    vec3 c4 = vec3(0.267, 0.749, 0.440);
    vec3 c5 = vec3(0.741, 0.873, 0.150);
    vec3 c6 = vec3(0.993, 0.906, 0.144);
    if (t < 0.167) return mix(c0, c1, t / 0.167);
    if (t < 0.333) return mix(c1, c2, (t - 0.167) / 0.167);
    if (t < 0.500) return mix(c2, c3, (t - 0.333) / 0.167);
    if (t < 0.667) return mix(c3, c4, (t - 0.500) / 0.167);
    if (t < 0.833) return mix(c4, c5, (t - 0.667) / 0.167);
    return mix(c5, c6, (t - 0.833) / 0.167);
}

void main() {
    FragColor = vec4(viridis(vT), 0.85);
}
)";

// ── Shader compilation ──────────────────────────────

static unsigned int compileShaderProgram(const char* vSrc, const char* fSrc) {
    auto compile = [](GLenum type, const char* src) -> unsigned int {
        unsigned int s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        int ok;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[512];
            glGetShaderInfoLog(s, 512, nullptr, log);
            std::cerr << "[JetShader] Compile error:\n" << log << std::endl;
        }
        return s;
    };

    unsigned int vs = compile(GL_VERTEX_SHADER, vSrc);
    unsigned int fs = compile(GL_FRAGMENT_SHADER, fSrc);
    unsigned int prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    int ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(prog, 512, nullptr, log);
        std::cerr << "[JetShader] Link error:\n" << log << std::endl;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

// ── Grid-to-world mapping ───────────────────────────

static glm::vec3 gridToWorld(float gx, float gy, float gz, int nx, int ny, int nz) {
    float maxDim = (float)std::max({nx, ny, nz});
    return glm::vec3(
        (gx - nx * 0.5f) / (maxDim * 0.5f),
        (gy - ny * 0.5f) / (maxDim * 0.5f),
        (gz - nz * 0.5f) / (maxDim * 0.5f)
    );
}

// ── Trilinear velocity interpolation ────────────────

glm::vec3 ParticleRenderer::interpolateVelocity(
    const float* ux, const float* uy, const float* uz,
    int nx, int ny, int nz, glm::vec3 pos)
{
    float fx = std::max(0.0f, std::min(pos.x, (float)(nx - 2)));
    float fy = std::max(0.0f, std::min(pos.y, (float)(ny - 2)));
    float fz = std::max(0.0f, std::min(pos.z, (float)(nz - 2)));

    int x0 = (int)fx, y0 = (int)fy, z0 = (int)fz;
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
    float xd = fx - x0, yd = fy - y0, zd = fz - z0;

    auto idx = [&](int x, int y, int z) -> int {
        return z * nx * ny + y * nx + x;
    };

    auto lerp3D = [&](const float* field) -> float {
        float c000 = field[idx(x0, y0, z0)];
        float c100 = field[idx(x1, y0, z0)];
        float c010 = field[idx(x0, y1, z0)];
        float c110 = field[idx(x1, y1, z0)];
        float c001 = field[idx(x0, y0, z1)];
        float c101 = field[idx(x1, y0, z1)];
        float c011 = field[idx(x0, y1, z1)];
        float c111 = field[idx(x1, y1, z1)];

        float c00 = c000 * (1 - xd) + c100 * xd;
        float c01 = c001 * (1 - xd) + c101 * xd;
        float c10 = c010 * (1 - xd) + c110 * xd;
        float c11 = c011 * (1 - xd) + c111 * xd;

        float c0 = c00 * (1 - yd) + c10 * yd;
        float c1 = c01 * (1 - yd) + c11 * yd;

        return c0 * (1 - zd) + c1 * zd;
    };

    return glm::vec3(lerp3D(ux), lerp3D(uy), lerp3D(uz));
}

// ── Seed generation ─────────────────────────────────

void ParticleRenderer::generateSeeds(int ny, int nz) {
    jetSeeds.clear();

    // Create a grid of seed points on the Y-Z plane, with some jitter
    int sqrtN = (int)std::ceil(std::sqrt((float)numJets));
    float dy = (float)(ny - 4) / (float)(sqrtN + 1);
    float dz = (float)(nz - 4) / (float)(sqrtN + 1);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> jitter(-0.3f, 0.3f);

    int count = 0;
    for (int iy = 0; iy < sqrtN && count < numJets; iy++) {
        for (int iz = 0; iz < sqrtN && count < numJets; iz++) {
            float seedY = 2.0f + (iy + 1) * dy + jitter(rng);
            float seedZ = 2.0f + (iz + 1) * dz + jitter(rng);
            jetSeeds.push_back(glm::vec2(seedY, seedZ));
            count++;
        }
    }

    seedsGenerated = true;
    lastNy = ny;
    lastNz = nz;
}

// ── Public API ──────────────────────────────────────

bool ParticleRenderer::init() {
    shader = compileShaderProgram(jetVertSrc, jetFragSrc);
    if (!shader) return false;

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    return true;
}

void ParticleRenderer::shutdown() {
    if (shader) glDeleteProgram(shader);
    if (VAO) glDeleteVertexArrays(1, &VAO);
    if (VBO) glDeleteBuffers(1, &VBO);
    shader = 0;
    VAO = 0;
    VBO = 0;
    particles.clear();
    jetSeeds.clear();
    seedsGenerated = false;
}

void ParticleRenderer::update(
    float dt, const float* ux, const float* uy, const float* uz,
    int nx, int ny, int nz, const uint8_t* cellTypes)
{
    if (!ux || !uy || !uz) return;

    cachedNx = nx;
    cachedNy = ny;
    cachedNz = nz;

    // Generate seed points if needed
    if (!seedsGenerated || lastNy != ny || lastNz != nz)
        generateSeeds(ny, nz);

    auto idx = [&](int x, int y, int z) -> int {
        return z * nx * ny + y * nx + x;
    };

    // Advect existing particles
    for (auto& p : particles) {
        glm::vec3 vel = interpolateVelocity(ux, uy, uz, nx, ny, nz, p.pos);
        p.pos += vel;  // one LBM step = one grid unit of advection
        p.velocity = glm::length(vel);
    }

    // Remove out-of-bounds or solid-hitting particles
    particles.erase(
        std::remove_if(particles.begin(), particles.end(),
            [&](const Particle& p) {
                if (p.pos.x < 0 || p.pos.x >= nx - 1 ||
                    p.pos.y < 1 || p.pos.y >= ny - 1 ||
                    p.pos.z < 1 || p.pos.z >= nz - 1)
                    return true;
                int cx = (int)p.pos.x;
                int cy = (int)p.pos.y;
                int cz = (int)p.pos.z;
                if (cellTypes && cellTypes[idx(cx, cy, cz)] == 1)
                    return true;
                return false;
            }),
        particles.end());

    // Inject new particles at each jet seed point (at inlet x=1)
    // Each jet gets a few particles per frame to form a continuous stream
    int maxTotal = numJets * 150; // max ~150 particles per jet
    int available = maxTotal - (int)particles.size();
    if (available <= 0) return;

    int perJet = std::max(1, available / (int)jetSeeds.size());
    perJet = std::min(perJet, 2); // inject 1-2 per jet per frame

    for (int j = 0; j < (int)jetSeeds.size() && (int)particles.size() < maxTotal; j++) {
        for (int k = 0; k < perJet; k++) {
            Particle p;
            p.pos = glm::vec3(1.5f, jetSeeds[j].x, jetSeeds[j].y);
            p.velocity = 0.0f;
            p.jetId = j;
            particles.push_back(p);
        }
    }
}

void ParticleRenderer::render(const glm::mat4& mvp, float maxVel) {
    if (particles.empty() || cachedNx == 0) return;

    // Sort particles by jet ID so we can render each jet as a line strip
    std::sort(particles.begin(), particles.end(),
        [](const Particle& a, const Particle& b) {
            if (a.jetId != b.jetId) return a.jetId < b.jetId;
            return a.pos.x < b.pos.x; // sort along wind direction within jet
        });

    // Build GPU buffer and track jet offsets
    struct GPUVertex {
        glm::vec3 pos;
        float velocity;
    };

    std::vector<GPUVertex> verts;
    std::vector<int> jetOffsets;
    std::vector<int> jetLengths;

    verts.reserve(particles.size());

    int currentJet = -1;
    for (auto& p : particles) {
        if (p.jetId != currentJet) {
            if (currentJet >= 0)
                jetLengths.push_back((int)verts.size() - jetOffsets.back());
            jetOffsets.push_back((int)verts.size());
            currentJet = p.jetId;
        }
        GPUVertex v;
        v.pos = gridToWorld(p.pos.x, p.pos.y, p.pos.z, cachedNx, cachedNy, cachedNz);
        v.velocity = p.velocity;
        verts.push_back(v);
    }
    if (!jetOffsets.empty())
        jetLengths.push_back((int)verts.size() - jetOffsets.back());

    // Upload
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(GPUVertex),
                 verts.data(), GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GPUVertex),
                          (void*)offsetof(GPUVertex, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(GPUVertex),
                          (void*)offsetof(GPUVertex, velocity));

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(shader);
    glUniformMatrix4fv(glGetUniformLocation(shader, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform1f(glGetUniformLocation(shader, "uMaxVel"), maxVel);

    glLineWidth(2.0f);

    // Draw each jet as a line strip
    for (size_t i = 0; i < jetOffsets.size(); i++) {
        if (jetLengths[i] >= 2)
            glDrawArrays(GL_LINE_STRIP, jetOffsets[i], jetLengths[i]);
    }

    glDisable(GL_BLEND);
    glBindVertexArray(0);
}
