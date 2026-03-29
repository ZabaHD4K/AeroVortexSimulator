#include "surface_pressure.h"
#include <glad/gl.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>

// ── Shaders ─────────────────────────────────────────

static const char* spVertSrc = R"(
#version 460 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in float aPressure;

uniform mat4 uMVP;
uniform mat4 uModel;
uniform mat3 uNormalMat;

out vec3 vNormal;
out vec3 vWorldPos;
out float vPressure;

void main() {
    vWorldPos  = vec3(uModel * vec4(aPos, 1.0));
    vNormal    = normalize(uNormalMat * aNormal);
    vPressure  = aPressure;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)";

static const char* spFragSrc = R"(
#version 460 core
in vec3 vNormal;
in vec3 vWorldPos;
in float vPressure;

uniform vec3 uCamPos;
uniform vec3 uLightDir;
uniform float uMinP;
uniform float uMaxP;

out vec4 FragColor;

vec3 coolwarm(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 cool = vec3(0.230, 0.299, 0.754);  // blue
    vec3 mid  = vec3(0.865, 0.865, 0.865);  // white
    vec3 warm = vec3(0.706, 0.016, 0.150);  // red
    if (t < 0.5) return mix(cool, mid, t * 2.0);
    return mix(mid, warm, (t - 0.5) * 2.0);
}

void main() {
    // Normalize pressure to [0,1]
    float range = uMaxP - uMinP;
    float t = (range > 1e-8) ? clamp((vPressure - uMinP) / range, 0.0, 1.0) : 0.5;

    vec3 baseColor = coolwarm(t);

    // Blinn-Phong lighting
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightDir);
    vec3 V = normalize(uCamPos - vWorldPos);
    vec3 H = normalize(L + V);

    float ambient  = 0.15;
    float diffuse  = max(dot(N, L), 0.0) * 0.60;
    float specular = pow(max(dot(N, H), 0.0), 64.0) * 0.25;

    // Back-face gets dimmer lighting
    float backDiffuse = max(dot(-N, L), 0.0) * 0.20;

    vec3 lit = baseColor * (ambient + diffuse + backDiffuse) + vec3(specular);
    FragColor = vec4(lit, 1.0);
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
            std::cerr << "[SurfacePressureShader] Compile error:\n" << log << std::endl;
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
        std::cerr << "[SurfacePressureShader] Link error:\n" << log << std::endl;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

// ── World-to-grid mapping (inverse of gridToWorld) ──

static glm::vec3 worldToGrid(glm::vec3 wp, int nx, int ny, int nz) {
    float maxDim = (float)std::max({nx, ny, nz});
    return glm::vec3(
        wp.x * (maxDim * 0.5f) + nx * 0.5f,
        wp.y * (maxDim * 0.5f) + ny * 0.5f,
        wp.z * (maxDim * 0.5f) + nz * 0.5f
    );
}

// ── Trilinear interpolation of a scalar field ───────

static float sampleField(const float* field, int nx, int ny, int nz, glm::vec3 gpos) {
    float fx = std::max(0.0f, std::min(gpos.x, (float)(nx - 2)));
    float fy = std::max(0.0f, std::min(gpos.y, (float)(ny - 2)));
    float fz = std::max(0.0f, std::min(gpos.z, (float)(nz - 2)));

    int x0 = (int)fx, y0 = (int)fy, z0 = (int)fz;
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
    float xd = fx - x0, yd = fy - y0, zd = fz - z0;

    auto idx = [&](int x, int y, int z) -> int {
        return z * nx * ny + y * nx + x;
    };

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
}

// ── Public API ──────────────────────────────────────

bool SurfacePressureRenderer::init() {
    shader = compileShaderProgram(spVertSrc, spFragSrc);
    return shader != 0;
}

void SurfacePressureRenderer::shutdown() {
    if (shader) glDeleteProgram(shader);
    if (customVAO) glDeleteVertexArrays(1, &customVAO);
    if (posVBO) glDeleteBuffers(1, &posVBO);
    if (normalVBO) glDeleteBuffers(1, &normalVBO);
    if (pressureAttribVBO) glDeleteBuffers(1, &pressureAttribVBO);
    if (EBO) glDeleteBuffers(1, &EBO);
    shader = 0;
    customVAO = 0;
    posVBO = 0;
    normalVBO = 0;
    pressureAttribVBO = 0;
    EBO = 0;
    cachedModel = nullptr;
}

void SurfacePressureRenderer::render(
    const Model& model, const float* pressure,
    int nx, int ny, int nz, float domainScale,
    const glm::mat4& mvp, const glm::vec3& camPos,
    float minP, float maxP)
{
    if (!pressure || model.meshes.empty()) return;

    // Rebuild geometry buffers if model changed
    if (&model != cachedModel) {
        // Clean up old buffers
        if (customVAO) glDeleteVertexArrays(1, &customVAO);
        if (posVBO) glDeleteBuffers(1, &posVBO);
        if (normalVBO) glDeleteBuffers(1, &normalVBO);
        if (pressureAttribVBO) glDeleteBuffers(1, &pressureAttribVBO);
        if (EBO) glDeleteBuffers(1, &EBO);

        // Collect all vertices and indices across meshes
        std::vector<glm::vec3> allPositions;
        std::vector<glm::vec3> allNormals;
        std::vector<unsigned int> allIndices;

        unsigned int indexOffset = 0;
        for (const auto& mesh : model.meshes) {
            for (const auto& v : mesh.vertices) {
                allPositions.push_back(v.position);
                allNormals.push_back(v.normal);
            }
            for (unsigned int idx : mesh.indices) {
                allIndices.push_back(idx + indexOffset);
            }
            indexOffset += (unsigned int)mesh.vertices.size();
        }

        indexCount = (int)allIndices.size();
        vertexPressures.resize(allPositions.size(), 0.0f);

        // Create VAO
        glGenVertexArrays(1, &customVAO);
        glBindVertexArray(customVAO);

        // Position VBO (location 0)
        glGenBuffers(1, &posVBO);
        glBindBuffer(GL_ARRAY_BUFFER, posVBO);
        glBufferData(GL_ARRAY_BUFFER, allPositions.size() * sizeof(glm::vec3),
                     allPositions.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

        // Normal VBO (location 1)
        glGenBuffers(1, &normalVBO);
        glBindBuffer(GL_ARRAY_BUFFER, normalVBO);
        glBufferData(GL_ARRAY_BUFFER, allNormals.size() * sizeof(glm::vec3),
                     allNormals.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

        // Pressure VBO (location 2) - will be updated each frame
        glGenBuffers(1, &pressureAttribVBO);
        glBindBuffer(GL_ARRAY_BUFFER, pressureAttribVBO);
        glBufferData(GL_ARRAY_BUFFER, vertexPressures.size() * sizeof(float),
                     nullptr, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);

        // Index buffer
        glGenBuffers(1, &EBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, allIndices.size() * sizeof(unsigned int),
                     allIndices.data(), GL_STATIC_DRAW);

        glBindVertexArray(0);
        cachedModel = &model;
    }

    // Sample pressure at each vertex
    int vertIdx = 0;
    for (const auto& mesh : model.meshes) {
        for (const auto& v : mesh.vertices) {
            // Model vertex positions are in world space (identity model matrix)
            glm::vec3 gpos = worldToGrid(v.position, nx, ny, nz);
            vertexPressures[vertIdx++] = sampleField(pressure, nx, ny, nz, gpos);
        }
    }

    // Update pressure VBO
    glBindBuffer(GL_ARRAY_BUFFER, pressureAttribVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0,
                    vertexPressures.size() * sizeof(float), vertexPressures.data());

    // Draw
    glm::mat4 modelMat = glm::mat4(1.0f);
    glm::mat3 normalMat = glm::transpose(glm::inverse(glm::mat3(modelMat)));

    glUseProgram(shader);
    glUniformMatrix4fv(glGetUniformLocation(shader, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniformMatrix4fv(glGetUniformLocation(shader, "uModel"), 1, GL_FALSE, glm::value_ptr(modelMat));
    glUniformMatrix3fv(glGetUniformLocation(shader, "uNormalMat"), 1, GL_FALSE, glm::value_ptr(normalMat));
    glUniform3fv(glGetUniformLocation(shader, "uCamPos"), 1, glm::value_ptr(camPos));
    glUniform3f(glGetUniformLocation(shader, "uLightDir"), 0.4f, 0.8f, 0.4f);
    glUniform1f(glGetUniformLocation(shader, "uMinP"), minP);
    glUniform1f(glGetUniformLocation(shader, "uMaxP"), maxP);

    glBindVertexArray(customVAO);
    glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}
