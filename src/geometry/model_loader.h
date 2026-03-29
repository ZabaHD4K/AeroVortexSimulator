#pragma once
#include "mesh.h"
#include <string>
#include <optional>

std::optional<Model> loadModel(const std::string& path);
void uploadMeshToGPU(Mesh& mesh);
void freeMeshGPU(Mesh& mesh);
void freeModelGPU(Model& model);
