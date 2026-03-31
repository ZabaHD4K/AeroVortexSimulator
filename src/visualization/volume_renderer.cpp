#include "volume_renderer.h"
#include <glad/gl.h>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>

// ── Shaders ─────────────────────────────────────────

static const char* volVertSrc = R"(
#version 460 core
layout(location=0) in vec3 aPos;   // unit cube [0,1]^3

uniform mat4 uMVP;
uniform vec3 uBoxMin;
uniform vec3 uBoxMax;

out vec3 vWorldPos;

void main() {
    // Map unit cube to volume AABB
    vec3 worldPos = uBoxMin + aPos * (uBoxMax - uBoxMin);
    vWorldPos = worldPos;
    gl_Position = uMVP * vec4(worldPos, 1.0);
}
)";

static const char* volFragSrc = R"(
#version 460 core
in vec3 vWorldPos;

uniform sampler3D uVolume;
uniform vec3 uCamPos;
uniform vec3 uBoxMin;
uniform vec3 uBoxMax;
uniform float uMinVal;
uniform float uMaxVal;
uniform float uOpacity;
uniform float uDensityScale;
uniform float uStepSize;
uniform int   uColormap;  // 0=viridis, 1=coolwarm, 2=inferno

out vec4 FragColor;

// ── Colormaps ──────────────────────────────────────

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

vec3 coolwarm(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 cool = vec3(0.230, 0.299, 0.754);
    vec3 mid  = vec3(0.865, 0.865, 0.865);
    vec3 warm = vec3(0.706, 0.016, 0.150);
    if (t < 0.5) return mix(cool, mid, t * 2.0);
    return mix(mid, warm, (t - 0.5) * 2.0);
}

vec3 inferno(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.001, 0.000, 0.014);
    vec3 c1 = vec3(0.258, 0.039, 0.406);
    vec3 c2 = vec3(0.578, 0.148, 0.404);
    vec3 c3 = vec3(0.865, 0.317, 0.226);
    vec3 c4 = vec3(0.987, 0.645, 0.039);
    vec3 c5 = vec3(0.988, 0.998, 0.645);
    if (t < 0.2) return mix(c0, c1, t / 0.2);
    if (t < 0.4) return mix(c1, c2, (t - 0.2) / 0.2);
    if (t < 0.6) return mix(c2, c3, (t - 0.4) / 0.2);
    if (t < 0.8) return mix(c3, c4, (t - 0.6) / 0.2);
    return mix(c4, c5, (t - 0.8) / 0.2);
}

vec3 applyColormap(float t) {
    if (uColormap == 1) return coolwarm(t);
    if (uColormap == 2) return inferno(t);
    return viridis(t);
}

// ── Ray-AABB intersection ──────────────────────────

vec2 intersectAABB(vec3 rayOrig, vec3 rayDir, vec3 bmin, vec3 bmax) {
    vec3 invDir = 1.0 / rayDir;
    vec3 t0s = (bmin - rayOrig) * invDir;
    vec3 t1s = (bmax - rayOrig) * invDir;
    vec3 tmin = min(t0s, t1s);
    vec3 tmax = max(t0s, t1s);
    float tNear = max(max(tmin.x, tmin.y), tmin.z);
    float tFar  = min(min(tmax.x, tmax.y), tmax.z);
    return vec2(tNear, tFar);
}

void main() {
    vec3 rayDir = normalize(vWorldPos - uCamPos);

    vec2 tHit = intersectAABB(uCamPos, rayDir, uBoxMin, uBoxMax);
    float tNear = max(tHit.x, 0.0);
    float tFar  = tHit.y;

    if (tNear >= tFar) discard;

    vec3 boxSize = uBoxMax - uBoxMin;

    // Front-to-back compositing
    vec4 accum = vec4(0.0);
    float t = tNear;

    for (int i = 0; i < 512; i++) {
        if (t >= tFar || accum.a > 0.95) break;

        vec3 pos = uCamPos + rayDir * t;

        // Convert to texture coordinates [0,1]^3
        vec3 tc = (pos - uBoxMin) / boxSize;
        tc = clamp(tc, 0.001, 0.999);

        float val = texture(uVolume, tc).r;
        float range = uMaxVal - uMinVal;
        float normalized = (range > 1e-8) ? clamp((val - uMinVal) / range, 0.0, 1.0) : 0.0;

        // Skip near-zero values (fluid at rest, solid cells)
        if (normalized > 0.01) {
            vec3 color = applyColormap(normalized);
            float alpha = normalized * uOpacity * uDensityScale * uStepSize;
            alpha = clamp(alpha, 0.0, 1.0);

            // Front-to-back compositing
            accum.rgb += (1.0 - accum.a) * color * alpha;
            accum.a   += (1.0 - accum.a) * alpha;
        }

        t += uStepSize;
    }

    if (accum.a < 0.005) discard;

    FragColor = vec4(accum.rgb, accum.a);
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
            std::cerr << "[VolumeShader] Compile error:\n" << log << std::endl;
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
        std::cerr << "[VolumeShader] Link error:\n" << log << std::endl;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

// ── Grid-to-world (same convention as other renderers) ──

static glm::vec3 gridToWorld(float gx, float gy, float gz, int nx, int ny, int nz, float voxelSize) {
    return glm::vec3(
        (gx - nx * 0.5f) * voxelSize,
        (gy - ny * 0.5f) * voxelSize,
        (gz - nz * 0.5f) * voxelSize
    );
}

// ── Unit cube geometry ──────────────────────────────

static const float cubeVerts[] = {
    0,0,0,  1,0,0,  1,1,0,  0,1,0,
    0,0,1,  1,0,1,  1,1,1,  0,1,1
};
static const unsigned int cubeIndices[] = {
    // front
    0,1,2, 2,3,0,
    // back
    5,4,7, 7,6,5,
    // left
    4,0,3, 3,7,4,
    // right
    1,5,6, 6,2,1,
    // bottom
    4,5,1, 1,0,4,
    // top
    3,2,6, 6,7,3
};

// ── Public API ──────────────────────────────────────

bool VolumeRenderer::init() {
    shader = compileShaderProgram(volVertSrc, volFragSrc);
    if (!shader) return false;

    // Cube geometry
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVerts), cubeVerts, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cubeIndices), cubeIndices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glBindVertexArray(0);

    // 3D texture
    glGenTextures(1, &tex3D);
    glBindTexture(GL_TEXTURE_3D, tex3D);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    return true;
}

void VolumeRenderer::shutdown() {
    if (shader) glDeleteProgram(shader);
    if (VAO) glDeleteVertexArrays(1, &VAO);
    if (VBO) glDeleteBuffers(1, &VBO);
    if (EBO) glDeleteBuffers(1, &EBO);
    if (tex3D) glDeleteTextures(1, &tex3D);
    shader = 0; VAO = 0; VBO = 0; EBO = 0; tex3D = 0;
}

void VolumeRenderer::uploadField(const float* data, int nx, int ny, int nz, float voxelSize) {
    if (!data) return;

    texNx = nx; texNy = ny; texNz = nz;

    // Compute volume AABB in world space
    boxMin = gridToWorld(0, 0, 0, nx, ny, nz, voxelSize);
    boxMax = gridToWorld((float)nx, (float)ny, (float)nz, nx, ny, nz, voxelSize);

    glBindTexture(GL_TEXTURE_3D, tex3D);
    // Data layout: idx = z * nx * ny + y * nx + x
    // OpenGL 3D tex: (width=nx, height=ny, depth=nz), X varies fastest → matches
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, nx, ny, nz, 0,
                 GL_RED, GL_FLOAT, data);
}

void VolumeRenderer::render(const glm::mat4& view, const glm::mat4& proj,
                             const glm::vec3& camPos,
                             float minVal, float maxVal) {
    if (texNx == 0) return;

    glm::mat4 mvp = proj * view;

    // Render back faces of the proxy cube with blending
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);  // draw back faces so every fragment has an entry point
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    glUseProgram(shader);
    glUniformMatrix4fv(glGetUniformLocation(shader, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform3fv(glGetUniformLocation(shader, "uCamPos"), 1, glm::value_ptr(camPos));
    glUniform3fv(glGetUniformLocation(shader, "uBoxMin"), 1, glm::value_ptr(boxMin));
    glUniform3fv(glGetUniformLocation(shader, "uBoxMax"), 1, glm::value_ptr(boxMax));
    glUniform1f(glGetUniformLocation(shader, "uMinVal"), minVal);
    glUniform1f(glGetUniformLocation(shader, "uMaxVal"), maxVal);
    glUniform1f(glGetUniformLocation(shader, "uOpacity"), opacity);
    glUniform1f(glGetUniformLocation(shader, "uDensityScale"), densityScale);
    glUniform1f(glGetUniformLocation(shader, "uStepSize"), stepSize);
    glUniform1i(glGetUniformLocation(shader, "uColormap"), (int)colormap);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, tex3D);
    glUniform1i(glGetUniformLocation(shader, "uVolume"), 0);

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glCullFace(GL_BACK);
    glDisable(GL_CULL_FACE);
}
