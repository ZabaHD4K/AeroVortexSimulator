#include "slice_plane.h"
#include <glad/gl.h>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>

// ── Shaders ─────────────────────────────────────────

static const char* sliceVertSrc = R"(
#version 460 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aUV;

uniform mat4 uMVP;

out vec2 vUV;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vUV = aUV;
}
)";

static const char* sliceFragSrc = R"(
#version 460 core
in vec2 vUV;

uniform sampler2D uField;
uniform float uMinVal;
uniform float uMaxVal;
uniform float uOpacity;

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
    float val = texture(uField, vUV).r;
    // Discard solid cells (sentinel value)
    if (val < -1e20) discard;
    float range = uMaxVal - uMinVal;
    float t = (range > 1e-8) ? clamp((val - uMinVal) / range, 0.0, 1.0) : 0.5;
    vec3 color = viridis(t);
    FragColor = vec4(color, uOpacity);
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
            std::cerr << "[SlicePlaneShader] Compile error:\n" << log << std::endl;
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
        std::cerr << "[SlicePlaneShader] Link error:\n" << log << std::endl;
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

// ── Public API ──────────────────────────────────────

bool SlicePlaneRenderer::init() {
    shader = compileShaderProgram(sliceVertSrc, sliceFragSrc);
    if (!shader) return false;

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenTextures(1, &texture);

    // Configure texture
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    return true;
}

void SlicePlaneRenderer::shutdown() {
    if (shader) glDeleteProgram(shader);
    if (VAO) glDeleteVertexArrays(1, &VAO);
    if (VBO) glDeleteBuffers(1, &VBO);
    if (texture) glDeleteTextures(1, &texture);
    shader = 0;
    VAO = 0;
    VBO = 0;
    texture = 0;
}

void SlicePlaneRenderer::render(
    const float* field3D, int nx, int ny, int nz,
    SliceAxis axis, int sliceIndex,
    const glm::mat4& mvp, float minVal, float maxVal,
    const uint8_t* cellTypes)
{
    if (!field3D) return;

    // Determine slice dimensions and extract 2D data
    int w = 0, h = 0;
    sliceIndex = std::max(0, sliceIndex);

    switch (axis) {
        case SLICE_X:
            sliceIndex = std::min(sliceIndex, nx - 1);
            w = ny; h = nz;
            break;
        case SLICE_Y:
            sliceIndex = std::min(sliceIndex, ny - 1);
            w = nx; h = nz;
            break;
        case SLICE_Z:
            sliceIndex = std::min(sliceIndex, nz - 1);
            w = nx; h = ny;
            break;
    }

    // Extract 2D slice (mark solid cells with sentinel value)
    const float SENTINEL = -1e30f;
    std::vector<float> sliceData(w * h);
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            int x = 0, y = 0, z = 0;
            switch (axis) {
                case SLICE_X: x = sliceIndex; y = i; z = j; break;
                case SLICE_Y: x = i; y = sliceIndex; z = j; break;
                case SLICE_Z: x = i; y = j; z = sliceIndex; break;
            }
            int cellIdx = z * nx * ny + y * nx + x;
            // If cell is solid, use sentinel so shader can discard it
            if (cellTypes && cellTypes[cellIdx] == 1) {
                sliceData[j * w + i] = SENTINEL;
            } else {
                sliceData[j * w + i] = field3D[cellIdx];
            }
        }
    }

    // Upload texture
    if (w != texW || h != texH) {
        texW = w;
        texH = h;
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, sliceData.data());
    } else {
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_FLOAT, sliceData.data());
    }

    // Build quad vertices in world space
    // The quad spans the full extent of the two non-sliced axes, positioned at the slice location
    glm::vec3 corners[4];

    switch (axis) {
        case SLICE_X: {
            float sx = (float)sliceIndex;
            corners[0] = gridToWorld(sx, 0,         0,         nx, ny, nz);
            corners[1] = gridToWorld(sx, (float)ny,  0,         nx, ny, nz);
            corners[2] = gridToWorld(sx, (float)ny,  (float)nz, nx, ny, nz);
            corners[3] = gridToWorld(sx, 0,         (float)nz, nx, ny, nz);
            break;
        }
        case SLICE_Y: {
            float sy = (float)sliceIndex;
            corners[0] = gridToWorld(0,         sy, 0,         nx, ny, nz);
            corners[1] = gridToWorld((float)nx, sy, 0,         nx, ny, nz);
            corners[2] = gridToWorld((float)nx, sy, (float)nz, nx, ny, nz);
            corners[3] = gridToWorld(0,         sy, (float)nz, nx, ny, nz);
            break;
        }
        case SLICE_Z: {
            float sz = (float)sliceIndex;
            corners[0] = gridToWorld(0,         0,         sz, nx, ny, nz);
            corners[1] = gridToWorld((float)nx, 0,         sz, nx, ny, nz);
            corners[2] = gridToWorld((float)nx, (float)ny, sz, nx, ny, nz);
            corners[3] = gridToWorld(0,         (float)ny, sz, nx, ny, nz);
            break;
        }
    }

    // pos(3) + uv(2) = 5 floats per vertex, 6 vertices (two triangles)
    float quadVerts[] = {
        corners[0].x, corners[0].y, corners[0].z, 0.0f, 0.0f,
        corners[1].x, corners[1].y, corners[1].z, 1.0f, 0.0f,
        corners[2].x, corners[2].y, corners[2].z, 1.0f, 1.0f,

        corners[0].x, corners[0].y, corners[0].z, 0.0f, 0.0f,
        corners[2].x, corners[2].y, corners[2].z, 1.0f, 1.0f,
        corners[3].x, corners[3].y, corners[3].z, 0.0f, 1.0f,
    };

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

    // Draw
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(shader);
    glUniformMatrix4fv(glGetUniformLocation(shader, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform1f(glGetUniformLocation(shader, "uMinVal"), minVal);
    glUniform1f(glGetUniformLocation(shader, "uMaxVal"), maxVal);
    glUniform1f(glGetUniformLocation(shader, "uOpacity"), opacity);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glUniform1i(glGetUniformLocation(shader, "uField"), 0);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glDisable(GL_BLEND);
    glBindVertexArray(0);
}
