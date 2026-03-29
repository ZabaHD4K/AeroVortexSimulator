#pragma once
#include <vector>
#include <string>
#include <glm/glm.hpp>

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    unsigned int VAO = 0, VBO = 0, EBO = 0;
};

struct Model {
    std::vector<Mesh> meshes;
    std::string name;
    glm::vec3 center{0.0f};
    float radius = 1.0f; // bounding sphere radius
};
