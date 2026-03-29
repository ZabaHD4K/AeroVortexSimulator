#pragma once
#include "geometry/mesh.h"
#include "core/lbm3d.cuh"
#include <vector>
#include <cstdint>

struct VoxelGrid {
    std::vector<uint8_t> cells; // nx * ny * nz
    int nx, ny, nz;
    float voxelSize; // size of each voxel in model space

    uint8_t& at(int x, int y, int z) { return cells[z * nx * ny + y * nx + x]; }
    const uint8_t& at(int x, int y, int z) const { return cells[z * nx * ny + y * nx + x]; }
};

// Voxelize a model into the given grid dimensions.
//
// The model is assumed to be normalized (centered at origin, radius ~1).
// domainScale controls how much of the domain the model occupies
// (0.3 = model fills 30% of the domain width).
//
// The resulting grid will have:
//   - CELL_SOLID  where the model surface intersects or encloses interior volume
//   - CELL_INLET  on the X=1 face
//   - CELL_OUTLET on the X=nx-2 face
//   - CELL_FLUID  everywhere else
//   - Y and Z boundaries are handled by the LBM solver as walls
VoxelGrid voxelizeModel(const Model& model, int nx, int ny, int nz,
                        float domainScale = 0.3f, WindDirection windDir = WIND_POS_X);
