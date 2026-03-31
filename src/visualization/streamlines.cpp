#include "streamlines.h"
#include <glad/gl.h>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>

// ── Shaders ─────────────────────────────────────────

static const char* vertSrc = R"(
#version 460 core
layout(location=0) in vec3 aPos;
layout(location=1) in float aDeviation;
layout(location=2) in float aAlpha;

uniform mat4 uMVP;

out float vT;
out float vAlpha;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vT = aDeviation;
    vAlpha = aAlpha;
}
)";

static const char* fragSrc = R"(
#version 460 core
in float vT;
in float vAlpha;
out vec4 FragColor;

// CFD-style colormap: shows flow deviation from freestream
// Aligned flow = cool colors, deviating/vortex = warm colors
vec3 flowColor(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.75, 0.88, 1.00);  // ice-white   (perfect freestream)
    vec3 c1 = vec3(0.12, 0.42, 0.93);  // blue        (very slight deviation)
    vec3 c2 = vec3(0.00, 0.72, 0.88);  // cyan        (mild)
    vec3 c3 = vec3(0.15, 0.85, 0.35);  // green       (moderate)
    vec3 c4 = vec3(0.93, 0.88, 0.12);  // yellow      (significant)
    vec3 c5 = vec3(0.95, 0.42, 0.05);  // orange      (strong separation)
    vec3 c6 = vec3(0.82, 0.08, 0.08);  // red         (vortex core)
    vec3 c7 = vec3(0.50, 0.00, 0.35);  // dark purple (reverse flow)

    if (t < 0.04) return mix(c0, c1, t / 0.04);
    if (t < 0.12) return mix(c1, c2, (t - 0.04) / 0.08);
    if (t < 0.25) return mix(c2, c3, (t - 0.12) / 0.13);
    if (t < 0.40) return mix(c3, c4, (t - 0.25) / 0.15);
    if (t < 0.55) return mix(c4, c5, (t - 0.40) / 0.15);
    if (t < 0.75) return mix(c5, c6, (t - 0.55) / 0.20);
    return mix(c6, c7, (t - 0.75) / 0.25);
}

void main() {
    vec3 col = flowColor(vT);
    float a = vAlpha * 0.92;
    FragColor = vec4(col, a);
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
            std::cerr << "[StreamlineShader] Compile error:\n" << log << std::endl;
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
        std::cerr << "[StreamlineShader] Link error:\n" << log << std::endl;
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

glm::vec3 StreamlineRenderer::interpolateVelocity(
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
            for (int x = 0; x < nx; x++) {
                if (cellTypes[z * nx * ny + y * nx + x] == 1) {
                    bb.mn[0] = std::min(bb.mn[0], x);
                    bb.mn[1] = std::min(bb.mn[1], y);
                    bb.mn[2] = std::min(bb.mn[2], z);
                    bb.mx[0] = std::max(bb.mx[0], x);
                    bb.mx[1] = std::max(bb.mx[1], y);
                    bb.mx[2] = std::max(bb.mx[2], z);
                    bb.valid = true;
                }
            }
    return bb;
}

// ── Deviation angle from freestream direction ───────

static float computeDeviation(glm::vec3 vel, glm::vec3 inletDir) {
    float mag = glm::length(vel);
    if (mag < 1e-6f) return 0.5f; // stagnant → mid-range

    glm::vec3 dir = vel / mag;
    float cosAngle = glm::dot(dir, inletDir);
    cosAngle = std::max(-1.0f, std::min(1.0f, cosAngle));

    // Map: 1 (aligned) → 0, 0 (perpendicular) → 0.5, -1 (reverse) → 1
    return std::acos(cosAngle) / 3.14159265f;
}

// ── Public API ──────────────────────────────────────

bool StreamlineRenderer::init() {
    shader = compileShaderProgram(vertSrc, fragSrc);
    if (!shader) return false;

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    return true;
}

void StreamlineRenderer::shutdown() {
    if (shader) glDeleteProgram(shader);
    if (VAO) glDeleteVertexArrays(1, &VAO);
    if (VBO) glDeleteBuffers(1, &VBO);
    shader = 0;
    VAO = 0;
    VBO = 0;
}

void StreamlineRenderer::generate(
    const float* ux, const float* uy, const float* uz,
    int nx, int ny, int nz, const uint8_t* cellTypes,
    int numLines, int maxSteps, WindDirection windDir,
    float voxelSize, glm::vec3 inletDir)
{
    std::vector<StreamVertex> allVerts;
    lineOffsets.clear();
    lineLengths.clear();
    lineCount = 0;

    // Normalize inlet direction
    float inletMag = glm::length(inletDir);
    if (inletMag > 1e-6f) inletDir /= inletMag;
    else inletDir = glm::vec3(1, 0, 0);

    // Find solid bounding box for adaptive seeding
    GridAABB solidBB = findSolidAABB(cellTypes, nx, ny, nz);

    int dim[3] = {nx, ny, nz};
    int fixedAxis, seedSlice;
    switch (windDir) {
        case WIND_POS_X: fixedAxis = 0; seedSlice = 1;        break;
        case WIND_NEG_X: fixedAxis = 0; seedSlice = nx - 2;   break;
        case WIND_POS_Z: fixedAxis = 2; seedSlice = 1;        break;
        case WIND_NEG_Z: fixedAxis = 2; seedSlice = nz - 2;   break;
        case WIND_POS_Y: fixedAxis = 1; seedSlice = 1;        break;
        case WIND_NEG_Y: fixedAxis = 1; seedSlice = ny - 2;   break;
    }

    int a1 = (fixedAxis == 0) ? 1 : 0;
    int a2 = (fixedAxis == 2) ? 1 : 2;

    // ── Adaptive seeding: sized to the model ──────────────────
    // Dense grid covering the solid AABB + small margin, plus a
    // single ring of context lines just outside.

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> jitter(-0.4f, 0.4f);

    struct Seed { float v1, v2; bool isSurface = false; float sx = 0, sy = 0, sz = 0; };
    std::vector<Seed> seeds;

    if (solidBB.valid) {
        float size1 = (float)(solidBB.mx[a1] - solidBB.mn[a1]);
        float size2 = (float)(solidBB.mx[a2] - solidBB.mn[a2]);

        // Large wind wall: generous margins all around so the wind
        // visually covers the entire model and beyond. Extra above.
        float margin1Lo = std::max(size1 * 0.50f, 5.0f);  // below/left
        float margin1Hi = std::max(size1 * 0.80f, 8.0f);  // above/right (bigger)
        float margin2Lo = std::max(size2 * 0.50f, 5.0f);
        float margin2Hi = std::max(size2 * 0.50f, 5.0f);
        float cMin1 = std::max(2.0f, (float)solidBB.mn[a1] - margin1Lo);
        float cMax1 = std::min((float)(dim[a1] - 2), (float)solidBB.mx[a1] + margin1Hi);
        float cMin2 = std::max(2.0f, (float)solidBB.mn[a2] - margin2Lo);
        float cMax2 = std::min((float)(dim[a2] - 2), (float)solidBB.mx[a2] + margin2Hi);

        // Dense grid: one line every ~2.5 cells
        float spacing = 2.5f;
        int n1 = std::max(3, (int)std::round((cMax1 - cMin1) / spacing));
        int n2 = std::max(3, (int)std::round((cMax2 - cMin2) / spacing));

        // Large budget
        int maxLines = numLines;
        while (n1 * n2 > maxLines && spacing < 10.0f) {
            spacing += 0.5f;
            n1 = std::max(3, (int)std::round((cMax1 - cMin1) / spacing));
            n2 = std::max(3, (int)std::round((cMax2 - cMin2) / spacing));
        }

        // Core seeds (dense grid over object + extended upward)
        float d1 = (cMax1 - cMin1) / (float)(n1 + 1);
        float d2 = (cMax2 - cMin2) / (float)(n2 + 1);
        for (int i1 = 1; i1 <= n1; i1++)
            for (int i2 = 1; i2 <= n2; i2++)
                seeds.push_back({
                    cMin1 + i1 * d1 + jitter(rng),
                    cMin2 + i2 * d2 + jitter(rng)
                });

        // Context ring: thin layer just outside core (one spacing gap)
        float ringGap = spacing * 1.5f;
        float rMin1 = std::max(2.0f, cMin1 - ringGap);
        float rMax1 = std::min((float)(dim[a1] - 2), cMax1 + ringGap);
        float rMin2 = std::max(2.0f, cMin2 - ringGap);
        float rMax2 = std::min((float)(dim[a2] - 2), cMax2 + ringGap);

        int rn1 = std::max(2, (int)std::round((rMax1 - rMin1) / (spacing * 2.0f)));
        int rn2 = std::max(2, (int)std::round((rMax2 - rMin2) / (spacing * 2.0f)));
        float rd1 = (rMax1 - rMin1) / (float)(rn1 + 1);
        float rd2 = (rMax2 - rMin2) / (float)(rn2 + 1);

        for (int i1 = 1; i1 <= rn1; i1++)
            for (int i2 = 1; i2 <= rn2; i2++) {
                float p1 = rMin1 + i1 * rd1 + jitter(rng);
                float p2 = rMin2 + i2 * rd2 + jitter(rng);
                if (p1 < cMin1 || p1 > cMax1 || p2 < cMin2 || p2 > cMax2)
                    seeds.push_back({p1, p2});
            }
        // ── Surface-hugging seeds: placed 1 cell upstream of solid faces ──
        // These create streamlines that start right next to the body and
        // visually graze its surface before deflecting.
        {
            int streamAxis = fixedAxis;  // wind travels along this axis
            int streamSign = 1;
            switch (windDir) {
                case WIND_NEG_X: case WIND_NEG_Y: case WIND_NEG_Z: streamSign = -1; break;
                default: break;
            }

            // Scan the upstream face of each solid cell: if the upstream
            // neighbor is fluid, place a seed there.
            float surfSpacing = 3.0f;
            std::vector<Seed> surfSeeds;
            for (int z = 1; z < nz - 1; z++)
                for (int y = 1; y < ny - 1; y++)
                    for (int x = 1; x < nx - 1; x++) {
                        if (cellTypes[z * nx * ny + y * nx + x] != 1) continue;
                        // Check upstream neighbor
                        int ux = x, uy = y, uz = z;
                        if (streamAxis == 0) ux -= streamSign;
                        else if (streamAxis == 1) uy -= streamSign;
                        else uz -= streamSign;
                        if (ux < 0 || ux >= nx || uy < 0 || uy >= ny || uz < 0 || uz >= nz) continue;
                        if (cellTypes[uz * nx * ny + uy * nx + ux] != 0) continue;
                        // This is a fluid cell just upstream of solid
                        Seed s;
                        s.v1 = 0; s.v2 = 0;
                        s.isSurface = true;
                        s.sx = (float)ux; s.sy = (float)uy; s.sz = (float)uz;
                        surfSeeds.push_back(s);
                    }
            // Subsample to avoid too many (keep every Nth)
            int maxSurf = numLines / 3;
            if ((int)surfSeeds.size() > maxSurf) {
                int step = (int)surfSeeds.size() / maxSurf;
                std::vector<Seed> filtered;
                for (int i = 0; i < (int)surfSeeds.size(); i += step)
                    filtered.push_back(surfSeeds[i]);
                surfSeeds = filtered;
            }
            for (auto& s : surfSeeds)
                seeds.push_back(s);
        }

    } else {
        // No solid: sparse uniform grid
        int side = std::max(3, (int)std::sqrt((float)numLines));
        float d1 = (float)(dim[a1] - 4) / (float)(side + 1);
        float d2 = (float)(dim[a2] - 4) / (float)(side + 1);
        for (int i1 = 1; i1 <= side; i1++)
            for (int i2 = 1; i2 <= side; i2++)
                seeds.push_back({
                    2.0f + i1 * d1 + jitter(rng),
                    2.0f + i2 * d2 + jitter(rng)
                });
    }

    float ds = 0.5f; // arc-length step in grid cells

    glm::vec3 coastDir(0.0f);
    switch (windDir) {
        case WIND_POS_X: coastDir = glm::vec3( 1, 0, 0); break;
        case WIND_NEG_X: coastDir = glm::vec3(-1, 0, 0); break;
        case WIND_POS_Z: coastDir = glm::vec3( 0, 0, 1); break;
        case WIND_NEG_Z: coastDir = glm::vec3( 0, 0,-1); break;
        case WIND_POS_Y: coastDir = glm::vec3( 0, 1, 0); break;
        case WIND_NEG_Y: coastDir = glm::vec3( 0,-1, 0); break;
    }

    auto idx = [&](int x, int y, int z) -> int {
        return z * nx * ny + y * nx + x;
    };

    auto inBounds = [&](glm::vec3 p) -> bool {
        return p.x >= 0 && p.x < nx - 1 && p.y >= 1 && p.y < ny - 1 && p.z >= 1 && p.z < nz - 1;
    };

    auto isSolid = [&](glm::vec3 p) -> bool {
        if (!inBounds(p)) return true;
        int cx = (int)p.x, cy = (int)p.y, cz = (int)p.z;
        return cellTypes[idx(cx, cy, cz)] == 1;
    };

    // ── Integrate each streamline ────────────────────────
    for (auto& seed : seeds) {
        glm::vec3 pos;
        if (seed.isSurface) {
            // Surface seed: start at the exact 3D position near the body
            pos = glm::vec3(seed.sx, seed.sy, seed.sz);
        } else {
            float coords[3];
            coords[fixedAxis] = (float)seedSlice;
            coords[a1] = seed.v1;
            coords[a2] = seed.v2;
            pos = glm::vec3(coords[0], coords[1], coords[2]);
        }

        std::vector<StreamVertex> lineVerts;
        int solidBounces = 0;
        const int maxBounces = 60;  // many bounces so lines slide along surfaces

        for (int step = 0; step < maxSteps; step++) {
            if (!inBounds(pos)) break;
            if (isSolid(pos)) break;

            glm::vec3 vel = interpolateVelocity(ux, uy, uz, nx, ny, nz, pos);
            float mag = glm::length(vel);

            // Stagnant region: coast in wind direction
            if (mag < 1e-4f) {
                glm::vec3 nextPos = pos + coastDir * 0.5f;
                if (isSolid(nextPos)) {
                    // Try perpendicular nudge to escape stagnation at surface
                    bool escaped = false;
                    for (int d = 0; d < 6 && !escaped; d++) {
                        int ddx[] = {1,-1,0,0,0,0};
                        int ddy[] = {0,0,1,-1,0,0};
                        int ddz[] = {0,0,0,0,1,-1};
                        glm::vec3 tryPos = pos + 0.5f * glm::vec3((float)ddx[d], (float)ddy[d], (float)ddz[d]);
                        if (inBounds(tryPos) && !isSolid(tryPos)) {
                            pos = tryPos;
                            escaped = true;
                        }
                    }
                    if (!escaped) break;
                } else {
                    pos = nextPos;
                }
                continue;
            }

            float dev = computeDeviation(vel, inletDir);
            float arcStep = ds / mag;
            glm::vec3 k1 = arcStep * vel;

            // Check solid collision — slide along surface instead of stopping
            glm::vec3 nextPos = pos + k1;
            if (isSolid(nextPos) && solidBounces < maxBounces) {
                solidBounces++;
                glm::vec3 normal(0.0f);
                int nx0 = (int)pos.x, ny0 = (int)pos.y, nz0 = (int)pos.z;
                int nx1 = (int)nextPos.x, ny1 = (int)nextPos.y, nz1 = (int)nextPos.z;
                if (nx1 != nx0 && inBounds(glm::vec3((float)nx1, (float)ny0, (float)nz0)) &&
                    cellTypes[idx(nx1, ny0, nz0)] == 1)
                    normal.x = (nx1 > nx0) ? -1.0f : 1.0f;
                if (ny1 != ny0 && inBounds(glm::vec3((float)nx0, (float)ny1, (float)nz0)) &&
                    cellTypes[idx(nx0, ny1, nz0)] == 1)
                    normal.y = (ny1 > ny0) ? -1.0f : 1.0f;
                if (nz1 != nz0 && inBounds(glm::vec3((float)nx0, (float)ny0, (float)nz1)) &&
                    cellTypes[idx(nx0, ny0, nz1)] == 1)
                    normal.z = (nz1 > nz0) ? -1.0f : 1.0f;

                float nLen = glm::length(normal);
                if (nLen > 0.001f) {
                    normal /= nLen;
                    glm::vec3 tangent = vel - glm::dot(vel, normal) * normal;
                    float tMag = glm::length(tangent);
                    if (tMag > 1e-6f) {
                        glm::vec3 slidePos = pos + (ds / tMag) * tangent;
                        if (inBounds(slidePos) && !isSolid(slidePos)) {
                            glm::vec3 wp = gridToWorld(pos.x, pos.y, pos.z, nx, ny, nz, voxelSize);
                            lineVerts.push_back({wp, dev, 1.0f});
                            pos = slidePos;
                            continue;
                        }
                    }
                }
                // Failed to slide — try coasting downstream instead of stopping
                glm::vec3 coastPos = pos + coastDir * 0.5f;
                if (inBounds(coastPos) && !isSolid(coastPos)) {
                    glm::vec3 wp = gridToWorld(pos.x, pos.y, pos.z, nx, ny, nz, voxelSize);
                    lineVerts.push_back({wp, dev, 1.0f});
                    pos = coastPos;
                    continue;
                }
                glm::vec3 wp = gridToWorld(pos.x, pos.y, pos.z, nx, ny, nz, voxelSize);
                lineVerts.push_back({wp, dev, 1.0f});
                break;
            }

            // Full RK4
            glm::vec3 p2 = pos + 0.5f * k1;
            if (!inBounds(p2) || isSolid(p2)) {
                glm::vec3 wp = gridToWorld(pos.x, pos.y, pos.z, nx, ny, nz, voxelSize);
                lineVerts.push_back({wp, dev, 1.0f});
                pos += k1; continue;
            }
            glm::vec3 v2 = interpolateVelocity(ux, uy, uz, nx, ny, nz, p2);
            float m2 = glm::length(v2);
            glm::vec3 k2 = (m2 > 1e-6f) ? (ds / m2) * v2 : k1;

            glm::vec3 p3 = pos + 0.5f * k2;
            if (!inBounds(p3) || isSolid(p3)) {
                glm::vec3 wp = gridToWorld(pos.x, pos.y, pos.z, nx, ny, nz, voxelSize);
                lineVerts.push_back({wp, dev, 1.0f});
                pos += k1; continue;
            }
            glm::vec3 v3 = interpolateVelocity(ux, uy, uz, nx, ny, nz, p3);
            float m3 = glm::length(v3);
            glm::vec3 k3 = (m3 > 1e-6f) ? (ds / m3) * v3 : k1;

            glm::vec3 p4 = pos + k3;
            if (!inBounds(p4) || isSolid(p4)) {
                glm::vec3 wp = gridToWorld(pos.x, pos.y, pos.z, nx, ny, nz, voxelSize);
                lineVerts.push_back({wp, dev, 1.0f});
                pos += k1; continue;
            }
            glm::vec3 v4 = interpolateVelocity(ux, uy, uz, nx, ny, nz, p4);
            float m4 = glm::length(v4);
            glm::vec3 k4 = (m4 > 1e-6f) ? (ds / m4) * v4 : k1;

            glm::vec3 wp = gridToWorld(pos.x, pos.y, pos.z, nx, ny, nz, voxelSize);
            lineVerts.push_back({wp, dev, 1.0f});

            pos += (k1 + 2.0f * k2 + 2.0f * k3 + k4) / 6.0f;
        }

        // Keep lines with at least 3 vertices
        if (lineVerts.size() >= 3) {
            int len = (int)lineVerts.size();
            for (int i = 0; i < len; i++) {
                float t = (float)i / (float)(len - 1);
                float fadeIn  = std::min(1.0f, t / 0.05f);
                float fadeOut = std::min(1.0f, (1.0f - t) / 0.15f);
                lineVerts[i].alpha = fadeIn * fadeOut;
            }

            lineOffsets.push_back((int)allVerts.size());
            lineLengths.push_back(len);
            allVerts.insert(allVerts.end(), lineVerts.begin(), lineVerts.end());
            lineCount++;
        }
    }

    vertexCount = (int)allVerts.size();

    // Upload to GPU
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, allVerts.size() * sizeof(StreamVertex),
                 allVerts.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(StreamVertex),
                          (void*)offsetof(StreamVertex, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(StreamVertex),
                          (void*)offsetof(StreamVertex, deviation));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(StreamVertex),
                          (void*)offsetof(StreamVertex, alpha));

    glBindVertexArray(0);
}

void StreamlineRenderer::render(const glm::mat4& mvp) {
    if (vertexCount == 0 || lineCount == 0) return;

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(shader);
    glUniformMatrix4fv(glGetUniformLocation(shader, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));

    glBindVertexArray(VAO);
    glLineWidth(2.0f);

    for (int i = 0; i < lineCount; i++) {
        glDrawArrays(GL_LINE_STRIP, lineOffsets[i], lineLengths[i]);
    }

    glBindVertexArray(0);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}
