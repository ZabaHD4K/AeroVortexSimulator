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
layout(location=1) in float aDeviation;

uniform mat4 uMVP;

out float vT;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_PointSize = 3.0;
    vT = aDeviation;
}
)";

static const char* jetFragSrc = R"(
#version 460 core
in float vT;
out vec4 FragColor;

vec3 flowColor(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.75, 0.88, 1.00);
    vec3 c1 = vec3(0.12, 0.42, 0.93);
    vec3 c2 = vec3(0.00, 0.72, 0.88);
    vec3 c3 = vec3(0.15, 0.85, 0.35);
    vec3 c4 = vec3(0.93, 0.88, 0.12);
    vec3 c5 = vec3(0.95, 0.42, 0.05);
    vec3 c6 = vec3(0.82, 0.08, 0.08);
    vec3 c7 = vec3(0.50, 0.00, 0.35);

    if (t < 0.04) return mix(c0, c1, t / 0.04);
    if (t < 0.12) return mix(c1, c2, (t - 0.04) / 0.08);
    if (t < 0.25) return mix(c2, c3, (t - 0.12) / 0.13);
    if (t < 0.40) return mix(c3, c4, (t - 0.25) / 0.15);
    if (t < 0.55) return mix(c4, c5, (t - 0.40) / 0.15);
    if (t < 0.75) return mix(c5, c6, (t - 0.55) / 0.20);
    return mix(c6, c7, (t - 0.75) / 0.25);
}

void main() {
    FragColor = vec4(flowColor(vT), 0.90);
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

static glm::vec3 gridToWorld(float gx, float gy, float gz, int nx, int ny, int nz, float voxelSize) {
    return glm::vec3(
        (gx - nx * 0.5f) * voxelSize,
        (gy - ny * 0.5f) * voxelSize,
        (gz - nz * 0.5f) * voxelSize
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

// ── Deviation angle from freestream ─────────────────

static float computeDeviation(glm::vec3 vel, glm::vec3 inletDir) {
    float mag = glm::length(vel);
    if (mag < 1e-6f) return 0.5f;
    glm::vec3 dir = vel / mag;
    float cosAngle = glm::clamp(glm::dot(dir, inletDir), -1.0f, 1.0f);
    return std::acos(cosAngle) / 3.14159265f;
}

// ── Solid AABB detection ────────────────────────────

struct GridAABB {
    int mn[3], mx[3];
    bool valid = false;
};

static GridAABB findSolidAABB(const uint8_t* cellTypes, int nx, int ny, int nz) {
    GridAABB bb;
    bb.mn[0] = nx; bb.mn[1] = ny; bb.mn[2] = nz;
    bb.mx[0] = 0;  bb.mx[1] = 0;  bb.mx[2] = 0;
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++)
                if (cellTypes[z * nx * ny + y * nx + x] == 1) {
                    bb.mn[0] = std::min(bb.mn[0], x);
                    bb.mn[1] = std::min(bb.mn[1], y);
                    bb.mn[2] = std::min(bb.mn[2], z);
                    bb.mx[0] = std::max(bb.mx[0], x);
                    bb.mx[1] = std::max(bb.mx[1], y);
                    bb.mx[2] = std::max(bb.mx[2], z);
                    bb.valid = true;
                }
    return bb;
}

// ── Adaptive seed generation ────────────────────────
//
// Seeds are sized to the model: dense grid covering the solid AABB
// (with a small margin), plus a single sparse ring just outside.
// This ensures most jets interact with the object and very few fly
// past without touching it.

void ParticleRenderer::generateSeeds(int nx, int ny, int nz,
                                      const uint8_t* cellTypes,
                                      WindDirection windDir)
{
    jetSeeds.clear();

    int dim[3] = {nx, ny, nz};
    int fixedAxis;
    switch (windDir) {
        case WIND_POS_X: case WIND_NEG_X: fixedAxis = 0; break;
        case WIND_POS_Z: case WIND_NEG_Z: fixedAxis = 2; break;
        case WIND_POS_Y: case WIND_NEG_Y: fixedAxis = 1; break;
    }
    int a1 = (fixedAxis == 0) ? 1 : 0;
    int a2 = (fixedAxis == 2) ? 1 : 2;

    GridAABB solidBB = findSolidAABB(cellTypes, nx, ny, nz);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> jitter(-0.4f, 0.4f);

    if (solidBB.valid) {
        float size1 = (float)(solidBB.mx[a1] - solidBB.mn[a1]);
        float size2 = (float)(solidBB.mx[a2] - solidBB.mn[a2]);

        // Large wind wall: generous margins all around so the wind
        // visually covers the entire model and beyond. Extra above.
        float margin1Lo = std::max(size1 * 0.50f, 5.0f);
        float margin1Hi = std::max(size1 * 0.80f, 8.0f);
        float margin2Lo = std::max(size2 * 0.50f, 5.0f);
        float margin2Hi = std::max(size2 * 0.50f, 5.0f);
        float cMin1 = std::max(2.0f, (float)solidBB.mn[a1] - margin1Lo);
        float cMax1 = std::min((float)(dim[a1] - 2), (float)solidBB.mx[a1] + margin1Hi);
        float cMin2 = std::max(2.0f, (float)solidBB.mn[a2] - margin2Lo);
        float cMax2 = std::min((float)(dim[a2] - 2), (float)solidBB.mx[a2] + margin2Hi);

        // Dense grid: one jet every ~3 cells
        float spacing = 3.0f;
        int n1 = std::max(3, (int)std::round((cMax1 - cMin1) / spacing));
        int n2 = std::max(3, (int)std::round((cMax2 - cMin2) / spacing));

        // Large budget
        int maxJets = numJets;
        while (n1 * n2 > maxJets && spacing < 10.0f) {
            spacing += 0.5f;
            n1 = std::max(3, (int)std::round((cMax1 - cMin1) / spacing));
            n2 = std::max(3, (int)std::round((cMax2 - cMin2) / spacing));
        }

        // Core seeds (dense grid over object + extended upward)
        float d1 = (cMax1 - cMin1) / (float)(n1 + 1);
        float d2 = (cMax2 - cMin2) / (float)(n2 + 1);
        for (int i1 = 1; i1 <= n1; i1++)
            for (int i2 = 1; i2 <= n2; i2++) {
                jetSeeds.push_back(glm::vec2(
                    cMin1 + i1 * d1 + jitter(rng),
                    cMin2 + i2 * d2 + jitter(rng)
                ));
            }

        // Context ring: single layer of jets just outside the core zone
        float ringGap = spacing;
        float rMin1 = std::max(2.0f, cMin1 - ringGap);
        float rMax1 = std::min((float)(dim[a1] - 2), cMax1 + ringGap);
        float rMin2 = std::max(2.0f, cMin2 - ringGap);
        float rMax2 = std::min((float)(dim[a2] - 2), cMax2 + ringGap);

        int rn1 = std::max(2, (int)std::round((rMax1 - rMin1) / (spacing * 1.5f)));
        int rn2 = std::max(2, (int)std::round((rMax2 - rMin2) / (spacing * 1.5f)));
        float rd1 = (rMax1 - rMin1) / (float)(rn1 + 1);
        float rd2 = (rMax2 - rMin2) / (float)(rn2 + 1);

        for (int i1 = 1; i1 <= rn1; i1++)
            for (int i2 = 1; i2 <= rn2; i2++) {
                float p1 = rMin1 + i1 * rd1 + jitter(rng);
                float p2 = rMin2 + i2 * rd2 + jitter(rng);
                // Only add if outside core zone (the ring, not the filled core)
                if (p1 < cMin1 || p1 > cMax1 || p2 < cMin2 || p2 > cMax2) {
                    jetSeeds.push_back(glm::vec2(p1, p2));
                }
            }

    } else {
        // No solid found: sparse uniform grid
        int side = std::max(3, (int)std::sqrt((float)numJets));
        float d1 = (float)(dim[a1] - 4) / (float)(side + 1);
        float d2 = (float)(dim[a2] - 4) / (float)(side + 1);
        for (int i1 = 1; i1 <= side; i1++)
            for (int i2 = 1; i2 <= side; i2++)
                jetSeeds.push_back(glm::vec2(
                    2.0f + i1 * d1 + jitter(rng),
                    2.0f + i2 * d2 + jitter(rng)
                ));
    }

    // Update numJets to reflect actual count
    numJets = (int)jetSeeds.size();

    seedsGenerated = true;
    lastNy = ny;
    lastNz = nz;
    cachedWindDir = windDir;
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
    int nx, int ny, int nz, const uint8_t* cellTypes,
    WindDirection windDir, glm::vec3 inletDir)
{
    if (!ux || !uy || !uz) return;

    cachedNx = nx;
    cachedNy = ny;
    cachedNz = nz;

    // Normalize inlet direction
    float inletMag = glm::length(inletDir);
    if (inletMag > 1e-6f) inletDir /= inletMag;
    else inletDir = glm::vec3(1, 0, 0);

    if (!seedsGenerated || lastNy != ny || lastNz != nz || cachedWindDir != windDir)
        generateSeeds(nx, ny, nz, cellTypes, windDir);

    // Figure out injection axis and slice
    int dim[3] = {nx, ny, nz};
    int fixedAxis, seedSlice;
    switch (windDir) {
        case WIND_POS_X: fixedAxis = 0; seedSlice = 2;        break;
        case WIND_NEG_X: fixedAxis = 0; seedSlice = nx - 3;   break;
        case WIND_POS_Z: fixedAxis = 2; seedSlice = 2;        break;
        case WIND_NEG_Z: fixedAxis = 2; seedSlice = nz - 3;   break;
        case WIND_POS_Y: fixedAxis = 1; seedSlice = 2;        break;
        case WIND_NEG_Y: fixedAxis = 1; seedSlice = ny - 3;   break;
    }
    int a1 = (fixedAxis == 0) ? 1 : 0;
    int a2 = (fixedAxis == 2) ? 1 : 2;

    auto idx = [&](int x, int y, int z) -> int {
        return z * nx * ny + y * nx + x;
    };

    auto inBounds = [&](glm::vec3 p) -> bool {
        return p.x >= 0 && p.x < nx - 1 && p.y >= 1 && p.y < ny - 1 && p.z >= 1 && p.z < nz - 1;
    };

    auto isSolid = [&](glm::vec3 p) -> bool {
        if (!inBounds(p)) return true;
        return cellTypes && cellTypes[idx((int)p.x, (int)p.y, (int)p.z)] == 1;
    };

    // RK2 advection with collision deflection
    for (auto& p : particles) {
        glm::vec3 vel1 = interpolateVelocity(ux, uy, uz, nx, ny, nz, p.pos);

        // RK2 midpoint
        glm::vec3 midPos = p.pos + 0.5f * vel1;
        glm::vec3 vel2 = vel1;
        if (inBounds(midPos) && !isSolid(midPos))
            vel2 = interpolateVelocity(ux, uy, uz, nx, ny, nz, midPos);

        glm::vec3 newPos = p.pos + vel2;
        p.velocity = glm::length(vel2);
        p.deviation = computeDeviation(vel2, inletDir);
        p.age += 1.0f;

        // Collision detection + deflection
        if (isSolid(newPos) && p.bounces < 5) {
            p.bounces++;
            glm::vec3 normal(0.0f);
            int cx = (int)p.pos.x, cy = (int)p.pos.y, cz = (int)p.pos.z;
            int nx_ = (int)newPos.x, ny_ = (int)newPos.y, nz_ = (int)newPos.z;

            if (nx_ != cx && inBounds(glm::vec3((float)nx_, (float)cy, (float)cz)) &&
                cellTypes[idx(nx_, cy, cz)] == 1)
                normal.x = (nx_ > cx) ? -1.0f : 1.0f;
            if (ny_ != cy && inBounds(glm::vec3((float)cx, (float)ny_, (float)cz)) &&
                cellTypes[idx(cx, ny_, cz)] == 1)
                normal.y = (ny_ > cy) ? -1.0f : 1.0f;
            if (nz_ != cz && inBounds(glm::vec3((float)cx, (float)cy, (float)nz_)) &&
                cellTypes[idx(cx, cy, nz_)] == 1)
                normal.z = (nz_ > cz) ? -1.0f : 1.0f;

            float nLen = glm::length(normal);
            if (nLen > 0.001f) {
                normal /= nLen;
                // Slide along surface (tangent component)
                glm::vec3 tangent = vel2 - glm::dot(vel2, normal) * normal;
                glm::vec3 tanPos = p.pos + tangent;
                if (inBounds(tanPos) && !isSolid(tanPos)) {
                    p.pos = tanPos;
                    continue;
                }
                // Try reflected
                glm::vec3 reflected = vel2 - 2.0f * glm::dot(vel2, normal) * normal;
                glm::vec3 refPos = p.pos + reflected;
                if (inBounds(refPos) && !isSolid(refPos)) {
                    p.pos = refPos;
                    continue;
                }
            }
            p.pos = glm::vec3(-1);
        } else {
            p.pos = newPos;
        }
    }

    // Remove out-of-bounds, dead, or solid-stuck particles
    particles.erase(
        std::remove_if(particles.begin(), particles.end(),
            [&](const Particle& p) {
                if (p.pos.x < 0 || p.pos.x >= nx - 1 ||
                    p.pos.y < 1 || p.pos.y >= ny - 1 ||
                    p.pos.z < 1 || p.pos.z >= nz - 1)
                    return true;
                if (isSolid(p.pos))
                    return true;
                return false;
            }),
        particles.end());

    // Inject new particles at inlet
    int maxTotal = numJets * 180;
    int available = maxTotal - (int)particles.size();
    if (available <= 0) return;

    int perJet = std::max(1, std::min(3, available / std::max(1, (int)jetSeeds.size())));

    for (int j = 0; j < (int)jetSeeds.size() && (int)particles.size() < maxTotal; j++) {
        for (int k = 0; k < perJet; k++) {
            Particle p;
            float coords[3];
            coords[fixedAxis] = (float)seedSlice + 0.5f;
            coords[a1] = jetSeeds[j].x;
            coords[a2] = jetSeeds[j].y;
            p.pos = glm::vec3(coords[0], coords[1], coords[2]);
            p.velocity = 0.0f;
            p.deviation = 0.0f;
            p.jetId = j;
            p.bounces = 0;
            p.age = 0.0f;
            particles.push_back(p);
        }
    }
}

void ParticleRenderer::render(const glm::mat4& mvp, float maxVel, float voxelSize) {
    if (particles.empty() || cachedNx == 0) return;

    // Sort by jet ID for line strip rendering
    std::sort(particles.begin(), particles.end(),
        [](const Particle& a, const Particle& b) {
            if (a.jetId != b.jetId) return a.jetId < b.jetId;
            return a.age < b.age;
        });

    struct GPUVertex {
        glm::vec3 pos;
        float deviation;
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
        v.pos = gridToWorld(p.pos.x, p.pos.y, p.pos.z, cachedNx, cachedNy, cachedNz, voxelSize);
        v.deviation = p.deviation;
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
                          (void*)offsetof(GPUVertex, deviation));

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(shader);
    glUniformMatrix4fv(glGetUniformLocation(shader, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));

    glLineWidth(2.0f);

    for (size_t i = 0; i < jetOffsets.size(); i++) {
        if (jetLengths[i] >= 2)
            glDrawArrays(GL_LINE_STRIP, jetOffsets[i], jetLengths[i]);
    }

    glDisable(GL_BLEND);
    glBindVertexArray(0);
}
