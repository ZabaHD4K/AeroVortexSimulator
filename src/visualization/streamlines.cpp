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
layout(location=1) in float aVelocity;
layout(location=2) in float aAlpha;

uniform mat4 uMVP;
uniform float uMaxVel;      // freestream velocity magnitude
uniform float uMaxPerturb;  // max perturbation for normalization

out float vT;
out float vAlpha;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    // Color by perturbation: deviation from freestream velocity
    float perturbation = abs(aVelocity - uMaxVel);
    vT = clamp(perturbation / max(uMaxPerturb, 1e-6), 0.0, 1.0);
    vAlpha = aAlpha;
}
)";

static const char* fragSrc = R"(
#version 460 core
in float vT;
in float vAlpha;
out vec4 FragColor;

// Perturbation colormap: blue (undisturbed) -> cyan -> green -> yellow -> red (high perturbation)
// Like real CFD wind tunnels: freestream is blue, disturbed flow near model is red/yellow
vec3 perturbColor(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.05, 0.15, 0.60);  // deep blue   (undisturbed freestream)
    vec3 c1 = vec3(0.10, 0.55, 0.85);  // cyan-blue   (very low perturbation)
    vec3 c2 = vec3(0.10, 0.80, 0.50);  // green        (mild perturbation)
    vec3 c3 = vec3(0.95, 0.85, 0.10);  // yellow       (moderate perturbation)
    vec3 c4 = vec3(0.90, 0.20, 0.05);  // red          (high perturbation / wake)

    if (t < 0.25) return mix(c0, c1, t / 0.25);
    if (t < 0.50) return mix(c1, c2, (t - 0.25) / 0.25);
    if (t < 0.75) return mix(c2, c3, (t - 0.50) / 0.25);
    return mix(c3, c4, (t - 0.75) / 0.25);
}

void main() {
    vec3 col = perturbColor(vT);
    float a = vAlpha * 0.90;
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

static glm::vec3 gridToWorld(float gx, float gy, float gz, int nx, int ny, int nz) {
    float maxDim = (float)std::max({nx, ny, nz});
    return glm::vec3(
        (gx - nx * 0.5f) / (maxDim * 0.5f),
        (gy - ny * 0.5f) / (maxDim * 0.5f),
        (gz - nz * 0.5f) / (maxDim * 0.5f)
    );
}

// ── Trilinear velocity interpolation ────────────────

glm::vec3 StreamlineRenderer::interpolateVelocity(
    const float* ux, const float* uy, const float* uz,
    int nx, int ny, int nz, glm::vec3 pos)
{
    // Clamp to valid range
    float fx = std::max(0.0f, std::min(pos.x, (float)(nx - 2)));
    float fy = std::max(0.0f, std::min(pos.y, (float)(ny - 2)));
    float fz = std::max(0.0f, std::min(pos.z, (float)(nz - 2)));

    int x0 = (int)fx;
    int y0 = (int)fy;
    int z0 = (int)fz;
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    float xd = fx - x0;
    float yd = fy - y0;
    float zd = fz - z0;

    auto idx = [&](int x, int y, int z) -> int {
        return z * nx * ny + y * nx + x;
    };

    // Trilinear interpolation for each component
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
    int numLines, int maxSteps, WindDirection windDir)
{
    std::vector<StreamVertex> allVerts;
    lineOffsets.clear();
    lineLengths.clear();
    lineCount = 0;

    // Seed streamlines at inlet face, distributed evenly across the two
    // transverse axes depending on wind direction.
    int sqrtN = (int)std::ceil(std::sqrt((float)numLines));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> jitter(-0.4f, 0.4f);

    // Determine seed axes: (fixedAxis, fixedVal, axis1 range, axis2 range)
    // fixedAxis: 0=X, 1=Y, 2=Z — the axis perpendicular to the inlet face
    // Seed at the very edge of the domain so wind starts far from the model
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
    // The two free axes
    int a1 = (fixedAxis == 0) ? 1 : 0;
    int a2 = (fixedAxis == 2) ? 1 : 2;
    float d1 = (float)(dim[a1] - 4) / (float)(sqrtN + 1);
    float d2 = (float)(dim[a2] - 4) / (float)(sqrtN + 1);

    // Arc-length step: each RK4 step advances exactly ds grid cells
    // regardless of local velocity magnitude. Prevents angular artifacts
    // in high-velocity regions (near surfaces) and slow far-field.
    float ds = 0.5f; // grid cells per step

    // Coast direction for stagnant / not-yet-developed zones:
    // when velocity is near-zero, advance this many cells per step
    // in the dominant wind direction so lines can reach the aircraft.
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

    // Check if position is inside domain bounds
    auto inBounds = [&](glm::vec3 p) -> bool {
        return p.x >= 0 && p.x < nx - 1 && p.y >= 1 && p.y < ny - 1 && p.z >= 1 && p.z < nz - 1;
    };

    // Check if cell at position is solid
    auto isSolid = [&](glm::vec3 p) -> bool {
        if (!inBounds(p)) return true;
        int cx = (int)p.x, cy = (int)p.y, cz = (int)p.z;
        return cellTypes[idx(cx, cy, cz)] == 1;
    };

    // Compute freestream velocity magnitude from the seed plane for perturbation reference
    float freeStreamMag = 0.0f;
    int sampleCount = 0;
    {
        int step1 = std::max(1, dim[a1] / 10);
        int step2 = std::max(1, dim[a2] / 10);
        for (int s1 = 2; s1 < dim[a1] - 2; s1 += step1) {
            for (int s2 = 2; s2 < dim[a2] - 2; s2 += step2) {
                int c[3];
                c[fixedAxis] = seedSlice;
                c[a1] = s1;
                c[a2] = s2;
                int si = idx(c[0], c[1], c[2]);
                float vmag = std::sqrt(ux[si]*ux[si] + uy[si]*uy[si] + uz[si]*uz[si]);
                freeStreamMag += vmag;
                sampleCount++;
            }
        }
        if (sampleCount > 0) freeStreamMag /= (float)sampleCount;
    }

    maxPerturbation = 0.01f; // reset, will be updated during integration

    int generated = 0;
    for (int i1 = 0; i1 < sqrtN && generated < numLines; i1++) {
        for (int i2 = 0; i2 < sqrtN && generated < numLines; i2++) {
            float v1 = 2.0f + (i1 + 1) * d1 + jitter(rng);
            float v2 = 2.0f + (i2 + 1) * d2 + jitter(rng);
            glm::vec3 pos(0.0f);
            float coords[3];
            coords[fixedAxis] = (float)seedSlice;
            coords[a1] = v1;
            coords[a2] = v2;
            pos = glm::vec3(coords[0], coords[1], coords[2]);

            std::vector<StreamVertex> lineVerts;
            int solidBounces = 0;
            const int maxBounces = 5; // max collision deflections per line

            for (int step = 0; step < maxSteps; step++) {
                if (!inBounds(pos)) break;
                if (isSolid(pos)) break; // shouldn't happen, but safety

                glm::vec3 vel = interpolateVelocity(ux, uy, uz, nx, ny, nz, pos);
                float mag = glm::length(vel);

                // Stagnant region: coast in wind direction
                if (mag < 1e-4f) {
                    glm::vec3 nextPos = pos + coastDir;
                    if (isSolid(nextPos)) break; // stuck against solid
                    pos = nextPos;
                    continue;
                }

                // Arc-length RK4
                float arcStep = ds / mag;
                glm::vec3 k1 = arcStep * vel;

                // Check if next position hits a solid — deflect along surface
                glm::vec3 nextPos = pos + k1;
                if (isSolid(nextPos) && solidBounces < maxBounces) {
                    solidBounces++;
                    // Find surface normal by checking which axis crossed into solid
                    glm::vec3 normal(0.0f);
                    int nx0 = (int)pos.x, ny0 = (int)pos.y, nz0 = (int)pos.z;
                    int nx1 = (int)nextPos.x, ny1 = (int)nextPos.y, nz1 = (int)nextPos.z;
                    // Approximate normal from solid neighbor check
                    if (nx1 != nx0 && inBounds(glm::vec3(nx1, ny0, nz0)) &&
                        cellTypes[idx(nx1, ny0, nz0)] == 1)
                        normal.x = (nx1 > nx0) ? -1.0f : 1.0f;
                    if (ny1 != ny0 && inBounds(glm::vec3(nx0, ny1, nz0)) &&
                        cellTypes[idx(nx0, ny1, nz0)] == 1)
                        normal.y = (ny1 > ny0) ? -1.0f : 1.0f;
                    if (nz1 != nz0 && inBounds(glm::vec3(nx0, ny0, nz1)) &&
                        cellTypes[idx(nx0, ny0, nz1)] == 1)
                        normal.z = (nz1 > nz0) ? -1.0f : 1.0f;

                    float nLen = glm::length(normal);
                    if (nLen > 0.001f) {
                        normal /= nLen;
                        // Reflect velocity: v' = v - 2*(v·n)*n  then slide along surface
                        glm::vec3 reflected = vel - 2.0f * glm::dot(vel, normal) * normal;
                        float rMag = glm::length(reflected);
                        if (rMag > 1e-6f) {
                            glm::vec3 slidePos = pos + (ds / rMag) * reflected;
                            if (inBounds(slidePos) && !isSolid(slidePos)) {
                                // Emit vertex at current position before deflecting
                                glm::vec3 worldPos = gridToWorld(pos.x, pos.y, pos.z, nx, ny, nz);
                                lineVerts.push_back({worldPos, mag, 1.0f});
                                maxPerturbation = std::max(maxPerturbation, std::abs(mag - freeStreamMag));
                                pos = slidePos;
                                continue;
                            }
                        }
                    }
                    // If deflection failed, just stop at the surface
                    glm::vec3 worldPos = gridToWorld(pos.x, pos.y, pos.z, nx, ny, nz);
                    lineVerts.push_back({worldPos, mag, 1.0f});
                    maxPerturbation = std::max(maxPerturbation, std::abs(mag - freeStreamMag));
                    break;
                }

                // Full RK4 with solid checks at intermediate points
                glm::vec3 p2 = pos + 0.5f * k1;
                if (!inBounds(p2) || isSolid(p2)) {
                    glm::vec3 worldPos = gridToWorld(pos.x, pos.y, pos.z, nx, ny, nz);
                    lineVerts.push_back({worldPos, mag, 1.0f});
                    maxPerturbation = std::max(maxPerturbation, std::abs(mag - freeStreamMag));
                    pos += k1; continue;
                }
                glm::vec3 v2 = interpolateVelocity(ux, uy, uz, nx, ny, nz, p2);
                float m2 = glm::length(v2);
                glm::vec3 k2 = (m2 > 1e-6f) ? (ds / m2) * v2 : k1;
                glm::vec3 p3 = pos + 0.5f * k2;
                if (!inBounds(p3) || isSolid(p3)) {
                    glm::vec3 worldPos = gridToWorld(pos.x, pos.y, pos.z, nx, ny, nz);
                    lineVerts.push_back({worldPos, mag, 1.0f});
                    maxPerturbation = std::max(maxPerturbation, std::abs(mag - freeStreamMag));
                    pos += k1; continue;
                }
                glm::vec3 v3 = interpolateVelocity(ux, uy, uz, nx, ny, nz, p3);
                float m3 = glm::length(v3);
                glm::vec3 k3 = (m3 > 1e-6f) ? (ds / m3) * v3 : k1;
                glm::vec3 p4 = pos + k3;
                if (!inBounds(p4) || isSolid(p4)) {
                    glm::vec3 worldPos = gridToWorld(pos.x, pos.y, pos.z, nx, ny, nz);
                    lineVerts.push_back({worldPos, mag, 1.0f});
                    maxPerturbation = std::max(maxPerturbation, std::abs(mag - freeStreamMag));
                    pos += k1; continue;
                }
                glm::vec3 v4 = interpolateVelocity(ux, uy, uz, nx, ny, nz, p4);
                float m4 = glm::length(v4);
                glm::vec3 k4 = (m4 > 1e-6f) ? (ds / m4) * v4 : k1;

                glm::vec3 worldPos = gridToWorld(pos.x, pos.y, pos.z, nx, ny, nz);
                lineVerts.push_back({worldPos, mag, 1.0f});
                maxPerturbation = std::max(maxPerturbation, std::abs(mag - freeStreamMag));

                pos += (k1 + 2.0f * k2 + 2.0f * k3 + k4) / 6.0f;
            }

            // Only keep lines with at least 3 vertices
            if (lineVerts.size() >= 3) {
                // Set alpha: fade in first 5%, full opacity in middle, fade out last 15%
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
            generated++;
        }
    }

    vertexCount = (int)allVerts.size();

    // Upload to GPU
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, allVerts.size() * sizeof(StreamVertex),
                 allVerts.data(), GL_STATIC_DRAW);

    // Position: location 0
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(StreamVertex),
                          (void*)offsetof(StreamVertex, pos));

    // Velocity: location 1
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(StreamVertex),
                          (void*)offsetof(StreamVertex, velocity));

    // Alpha: location 2
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(StreamVertex),
                          (void*)offsetof(StreamVertex, alpha));

    glBindVertexArray(0);
}

void StreamlineRenderer::render(const glm::mat4& mvp, float freeStreamVel) {
    if (vertexCount == 0 || lineCount == 0) return;

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(shader);
    glUniformMatrix4fv(glGetUniformLocation(shader, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform1f(glGetUniformLocation(shader, "uMaxVel"), freeStreamVel);
    glUniform1f(glGetUniformLocation(shader, "uMaxPerturb"), maxPerturbation);

    glBindVertexArray(VAO);

    glLineWidth(2.0f);

    // Draw each streamline as a separate GL_LINE_STRIP
    for (int i = 0; i < lineCount; i++) {
        glDrawArrays(GL_LINE_STRIP, lineOffsets[i], lineLengths[i]);
    }

    glBindVertexArray(0);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}
