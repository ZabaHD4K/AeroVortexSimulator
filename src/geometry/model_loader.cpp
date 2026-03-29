#include "model_loader.h"
#include <glad/gl.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <filesystem>

static Mesh processMesh(aiMesh* aiM) {
    Mesh mesh;
    mesh.vertices.reserve(aiM->mNumVertices);
    for (unsigned int i = 0; i < aiM->mNumVertices; i++) {
        Vertex v;
        v.position = {aiM->mVertices[i].x, aiM->mVertices[i].y, aiM->mVertices[i].z};
        if (aiM->HasNormals())
            v.normal = {aiM->mNormals[i].x, aiM->mNormals[i].y, aiM->mNormals[i].z};
        else
            v.normal = {0.0f, 1.0f, 0.0f};
        mesh.vertices.push_back(v);
    }
    for (unsigned int i = 0; i < aiM->mNumFaces; i++) {
        aiFace& face = aiM->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; j++)
            mesh.indices.push_back(face.mIndices[j]);
    }
    return mesh;
}

static void processNode(aiNode* node, const aiScene* scene, std::vector<Mesh>& meshes) {
    for (unsigned int i = 0; i < node->mNumMeshes; i++)
        meshes.push_back(processMesh(scene->mMeshes[node->mMeshes[i]]));
    for (unsigned int i = 0; i < node->mNumChildren; i++)
        processNode(node->mChildren[i], scene, meshes);
}

std::optional<Model> loadModel(const std::string& path) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path,
        aiProcess_Triangulate |
        aiProcess_GenSmoothNormals |
        aiProcess_JoinIdenticalVertices |
        aiProcess_FixInfacingNormals
    );

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "[Assimp] Error: " << importer.GetErrorString() << std::endl;
        return std::nullopt;
    }

    Model model;
    model.name = std::filesystem::path(path).stem().string();
    processNode(scene->mRootNode, scene, model.meshes);

    // Calculate bounding sphere (center + radius)
    glm::vec3 minB(FLT_MAX), maxB(-FLT_MAX);
    for (auto& mesh : model.meshes) {
        for (auto& v : mesh.vertices) {
            minB = glm::min(minB, v.position);
            maxB = glm::max(maxB, v.position);
        }
    }
    model.center = (minB + maxB) * 0.5f;
    model.radius = glm::length(maxB - minB) * 0.5f;
    if (model.radius < 1e-6f) model.radius = 1.0f;

    // Normalize: center at origin, fit in unit sphere
    float scale = 1.0f / model.radius;
    for (auto& mesh : model.meshes) {
        for (auto& v : mesh.vertices) {
            v.position = (v.position - model.center) * scale;
        }
    }
    model.center = glm::vec3(0.0f);
    model.radius = 1.0f;

    // Upload to GPU
    for (auto& mesh : model.meshes)
        uploadMeshToGPU(mesh);

    std::cout << "[Model] Loaded '" << model.name << "' — "
              << model.meshes.size() << " mesh(es)" << std::endl;
    return model;
}

void uploadMeshToGPU(Mesh& mesh) {
    glGenVertexArrays(1, &mesh.VAO);
    glGenBuffers(1, &mesh.VBO);
    glGenBuffers(1, &mesh.EBO);

    glBindVertexArray(mesh.VAO);

    glBindBuffer(GL_ARRAY_BUFFER, mesh.VBO);
    glBufferData(GL_ARRAY_BUFFER, mesh.vertices.size() * sizeof(Vertex),
                 mesh.vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size() * sizeof(unsigned int),
                 mesh.indices.data(), GL_STATIC_DRAW);

    // position
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    // normal
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          (void*)offsetof(Vertex, normal));

    glBindVertexArray(0);
}

void freeMeshGPU(Mesh& mesh) {
    if (mesh.VAO) glDeleteVertexArrays(1, &mesh.VAO);
    if (mesh.VBO) glDeleteBuffers(1, &mesh.VBO);
    if (mesh.EBO) glDeleteBuffers(1, &mesh.EBO);
    mesh.VAO = mesh.VBO = mesh.EBO = 0;
}

void freeModelGPU(Model& model) {
    for (auto& mesh : model.meshes)
        freeMeshGPU(mesh);
}
