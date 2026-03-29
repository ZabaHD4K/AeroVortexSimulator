#include "field2d.h"
#include <glad/gl.h>
#include <cstring>
#include <iostream>
#include <cmath>
#include <algorithm>

static const char* quadVert = R"(
#version 460 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vUV = aUV;
}
)";

// Viridis-inspired colormap in shader
static const char* quadFrag = R"(
#version 460 core
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uField;
uniform float uMinVal;
uniform float uMaxVal;

vec3 viridis(float t) {
    // Simplified viridis colormap
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.267, 0.004, 0.329);  // dark purple
    vec3 c1 = vec3(0.282, 0.140, 0.458);  // purple
    vec3 c2 = vec3(0.212, 0.359, 0.551);  // blue
    vec3 c3 = vec3(0.127, 0.566, 0.551);  // teal
    vec3 c4 = vec3(0.267, 0.749, 0.440);  // green
    vec3 c5 = vec3(0.741, 0.873, 0.150);  // yellow-green
    vec3 c6 = vec3(0.993, 0.906, 0.144);  // yellow

    if (t < 0.167) return mix(c0, c1, t / 0.167);
    if (t < 0.333) return mix(c1, c2, (t - 0.167) / 0.167);
    if (t < 0.500) return mix(c2, c3, (t - 0.333) / 0.167);
    if (t < 0.667) return mix(c3, c4, (t - 0.500) / 0.167);
    if (t < 0.833) return mix(c4, c5, (t - 0.667) / 0.167);
    return mix(c5, c6, (t - 0.833) / 0.167);
}

void main() {
    float val = texture(uField, vUV).r;
    float t = (val - uMinVal) / max(uMaxVal - uMinVal, 1e-6);
    FragColor = vec4(viridis(t), 1.0);
}
)";

unsigned int Field2DRenderer::compileShader(const char* vertSrc, const char* fragSrc) {
    auto compile = [](GLenum type, const char* src) -> unsigned int {
        unsigned int s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        int ok;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[512];
            glGetShaderInfoLog(s, 512, nullptr, log);
            std::cerr << "[Field2D Shader] " << log << std::endl;
        }
        return s;
    };

    unsigned int vs = compile(GL_VERTEX_SHADER, vertSrc);
    unsigned int fs = compile(GL_FRAGMENT_SHADER, fragSrc);
    unsigned int prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

bool Field2DRenderer::init() {
    shader = compileShader(quadVert, quadFrag);

    // Fullscreen quad
    float quad[] = {
        // pos      uv
        -1, -1,   0, 0,
         1, -1,   1, 0,
         1,  1,   1, 1,
        -1, -1,   0, 0,
         1,  1,   1, 1,
        -1,  1,   0, 1,
    };

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glBindVertexArray(0);

    // Create texture (will be resized on first render)
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    return shader != 0;
}

void Field2DRenderer::shutdown() {
    if (shader) glDeleteProgram(shader);
    if (VAO) glDeleteVertexArrays(1, &VAO);
    if (VBO) glDeleteBuffers(1, &VBO);
    if (texture) glDeleteTextures(1, &texture);
}

void Field2DRenderer::render(const float* data, int nx, int ny, float minVal, float maxVal) {
    if (!data) return;

    // Upload field data as R32F texture
    glBindTexture(GL_TEXTURE_2D, texture);
    if (nx != texW || ny != texH) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, nx, ny, 0, GL_RED, GL_FLOAT, data);
        texW = nx;
        texH = ny;
    } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx, ny, GL_RED, GL_FLOAT, data);
    }

    // Render fullscreen quad
    glDisable(GL_DEPTH_TEST);
    glUseProgram(shader);
    glUniform1f(glGetUniformLocation(shader, "uMinVal"), minVal);
    glUniform1f(glGetUniformLocation(shader, "uMaxVal"), maxVal);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glUniform1i(glGetUniformLocation(shader, "uField"), 0);

    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glEnable(GL_DEPTH_TEST);
}
