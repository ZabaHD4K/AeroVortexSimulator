#include "renderer.h"
#include <glad/gl.h>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>

// ── Mesh shader ──────────────────────────────────────

static const char* meshVertSrc = R"(
#version 460 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;

uniform mat4 uMVP;
uniform mat4 uModel;
uniform mat3 uNormalMat;

out vec3 vNormal;
out vec3 vWorldPos;

void main() {
    vWorldPos = vec3(uModel * vec4(aPos, 1.0));
    vNormal   = normalize(uNormalMat * aNormal);
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)";

static const char* meshFragSrc = R"(
#version 460 core
in vec3 vNormal;
in vec3 vWorldPos;

uniform vec3 uLightDir;
uniform vec3 uColor;
uniform vec3 uCamPos;

out vec4 FragColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightDir);
    vec3 V = normalize(uCamPos - vWorldPos);
    vec3 H = normalize(L + V);

    float ambient  = 0.15;
    float diffuse  = max(dot(N, L), 0.0) * 0.65;
    float specular = pow(max(dot(N, H), 0.0), 64.0) * 0.3;

    // Back-face gets dimmer lighting
    float backDiffuse = max(dot(-N, L), 0.0) * 0.25;

    vec3 lit = uColor * (ambient + diffuse + backDiffuse) + vec3(specular);
    FragColor = vec4(lit, 1.0);
}
)";

// ── Grid shader ──────────────────────────────────────

static const char* gridVertSrc = R"(
#version 460 core
layout(location=0) in vec3 aPos;

uniform mat4 uMVP;

out float vDist;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vDist = length(aPos.xz);
}
)";

static const char* gridFragSrc = R"(
#version 460 core
in float vDist;
out vec4 FragColor;

void main() {
    float alpha = smoothstep(8.0, 2.0, vDist);
    FragColor = vec4(0.4, 0.4, 0.4, alpha * 0.5);
}
)";

// ── Implementation ───────────────────────────────────

unsigned int Renderer::compileShader(const char* vertSrc, const char* fragSrc) {
    auto compile = [](GLenum type, const char* src) -> unsigned int {
        unsigned int s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        int ok;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[512];
            glGetShaderInfoLog(s, 512, nullptr, log);
            std::cerr << "[Shader] Compile error:\n" << log << std::endl;
        }
        return s;
    };

    unsigned int vs = compile(GL_VERTEX_SHADER, vertSrc);
    unsigned int fs = compile(GL_FRAGMENT_SHADER, fragSrc);
    unsigned int prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    int ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(prog, 512, nullptr, log);
        std::cerr << "[Shader] Link error:\n" << log << std::endl;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

void Renderer::buildGrid() {
    std::vector<float> verts;
    int N = 20;
    float half = (float)N;
    for (int i = -N; i <= N; i++) {
        float f = (float)i;
        // line along X
        verts.insert(verts.end(), {f, 0, -half});
        verts.insert(verts.end(), {f, 0,  half});
        // line along Z
        verts.insert(verts.end(), {-half, 0, f});
        verts.insert(verts.end(), { half, 0, f});
    }
    gridVertCount = (int)verts.size() / 3;

    glGenVertexArrays(1, &gridVAO);
    glGenBuffers(1, &gridVBO);
    glBindVertexArray(gridVAO);
    glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glBindVertexArray(0);
}

bool Renderer::init() {
    meshShader = compileShader(meshVertSrc, meshFragSrc);
    gridShader = compileShader(gridVertSrc, gridFragSrc);
    buildGrid();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    return meshShader != 0 && gridShader != 0;
}

void Renderer::shutdown() {
    if (meshShader) glDeleteProgram(meshShader);
    if (gridShader) glDeleteProgram(gridShader);
    if (gridVAO) glDeleteVertexArrays(1, &gridVAO);
    if (gridVBO) glDeleteBuffers(1, &gridVBO);
}

void Renderer::renderModel(const Model& model, const Camera& cam, float aspect) {
    glm::mat4 view = cam.getViewMatrix();
    glm::mat4 proj = cam.getProjectionMatrix(aspect);
    glm::mat4 modelMat = glm::mat4(1.0f);
    glm::mat4 mvp = proj * view * modelMat;
    glm::mat3 normalMat = glm::transpose(glm::inverse(glm::mat3(modelMat)));

    if (wireframe)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glUseProgram(meshShader);
    glUniformMatrix4fv(glGetUniformLocation(meshShader, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniformMatrix4fv(glGetUniformLocation(meshShader, "uModel"), 1, GL_FALSE, glm::value_ptr(modelMat));
    glUniformMatrix3fv(glGetUniformLocation(meshShader, "uNormalMat"), 1, GL_FALSE, glm::value_ptr(normalMat));
    glUniform3fv(glGetUniformLocation(meshShader, "uLightDir"), 1, glm::value_ptr(glm::normalize(lightDir)));
    glUniform3fv(glGetUniformLocation(meshShader, "uColor"), 1, glm::value_ptr(modelColor));
    glUniform3fv(glGetUniformLocation(meshShader, "uCamPos"), 1, glm::value_ptr(cam.getPosition()));

    for (auto& mesh : model.meshes) {
        glBindVertexArray(mesh.VAO);
        glDrawElements(GL_TRIANGLES, (GLsizei)mesh.indices.size(), GL_UNSIGNED_INT, nullptr);
    }

    if (wireframe)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void Renderer::renderGrid(const Camera& cam, float aspect) {
    if (!showGrid) return;

    glm::mat4 mvp = cam.getProjectionMatrix(aspect) * cam.getViewMatrix();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(gridShader);
    glUniformMatrix4fv(glGetUniformLocation(gridShader, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));

    glBindVertexArray(gridVAO);
    glDrawArrays(GL_LINES, 0, gridVertCount);

    glDisable(GL_BLEND);
}
