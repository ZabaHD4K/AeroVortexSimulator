#include "core/voxelizer.h"
#include <glm/glm.hpp>
#include <queue>
#include <algorithm>
#include <cmath>
#include <cstdio>

// ============================================================================
// Triangle-AABB intersection using the Separating Axis Theorem (SAT)
//
// Tests 13 potential separating axes:
//   - 3 box face normals  (x, y, z axes)
//   - 1 triangle normal
//   - 9 cross products    (3 box edges x 3 triangle edges)
//
// If no separating axis is found, the triangle and AABB overlap.
// ============================================================================

namespace {

// Project triangle vertices onto an axis and check for overlap with box half-extents
// Returns true if projections are separated (i.e. this IS a separating axis)
bool isSeparatingAxis(const glm::vec3& axis,
                      const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
                      const glm::vec3& halfExtents)
{
    // Avoid degenerate axes (from parallel edge cross products)
    float len2 = glm::dot(axis, axis);
    if (len2 < 1e-12f) return false; // not a valid axis, cannot separate

    // Project triangle vertices onto axis
    float p0 = glm::dot(axis, v0);
    float p1 = glm::dot(axis, v1);
    float p2 = glm::dot(axis, v2);

    float triMin = std::min({p0, p1, p2});
    float triMax = std::max({p0, p1, p2});

    // Project box onto axis. The box is centered at origin with given half-extents.
    // The box projection radius is the sum of |axis . each box axis| * halfExtent
    float boxRadius = halfExtents.x * std::abs(axis.x)
                    + halfExtents.y * std::abs(axis.y)
                    + halfExtents.z * std::abs(axis.z);

    // Check separation: box projects to [-boxRadius, +boxRadius]
    return (triMin > boxRadius) || (triMax < -boxRadius);
}

// Test if a triangle (v0, v1, v2) intersects an AABB centered at boxCenter with given halfExtents
bool triangleIntersectsAABB(const glm::vec3& v0_world, const glm::vec3& v1_world, const glm::vec3& v2_world,
                            const glm::vec3& boxCenter, const glm::vec3& halfExtents)
{
    // Translate triangle so that the box center is at origin
    glm::vec3 v0 = v0_world - boxCenter;
    glm::vec3 v1 = v1_world - boxCenter;
    glm::vec3 v2 = v2_world - boxCenter;

    // Triangle edges
    glm::vec3 e0 = v1 - v0;
    glm::vec3 e1 = v2 - v1;
    glm::vec3 e2 = v0 - v2;

    // Box face normals (unit axes)
    glm::vec3 bx(1, 0, 0), by(0, 1, 0), bz(0, 0, 1);

    // --- Test 3 box face normals ---
    if (isSeparatingAxis(bx, v0, v1, v2, halfExtents)) return false;
    if (isSeparatingAxis(by, v0, v1, v2, halfExtents)) return false;
    if (isSeparatingAxis(bz, v0, v1, v2, halfExtents)) return false;

    // --- Test triangle normal ---
    glm::vec3 triNormal = glm::cross(e0, e1);
    if (isSeparatingAxis(triNormal, v0, v1, v2, halfExtents)) return false;

    // --- Test 9 cross products: box edge x triangle edge ---
    if (isSeparatingAxis(glm::cross(bx, e0), v0, v1, v2, halfExtents)) return false;
    if (isSeparatingAxis(glm::cross(bx, e1), v0, v1, v2, halfExtents)) return false;
    if (isSeparatingAxis(glm::cross(bx, e2), v0, v1, v2, halfExtents)) return false;

    if (isSeparatingAxis(glm::cross(by, e0), v0, v1, v2, halfExtents)) return false;
    if (isSeparatingAxis(glm::cross(by, e1), v0, v1, v2, halfExtents)) return false;
    if (isSeparatingAxis(glm::cross(by, e2), v0, v1, v2, halfExtents)) return false;

    if (isSeparatingAxis(glm::cross(bz, e0), v0, v1, v2, halfExtents)) return false;
    if (isSeparatingAxis(glm::cross(bz, e1), v0, v1, v2, halfExtents)) return false;
    if (isSeparatingAxis(glm::cross(bz, e2), v0, v1, v2, halfExtents)) return false;

    // No separating axis found => triangle and AABB overlap
    return true;
}

} // anonymous namespace

// ============================================================================
// Voxelization pipeline
// ============================================================================

VoxelGrid voxelizeModel(const Model& model, int nx, int ny, int nz, float domainScale, WindDirection windDir)
{
    VoxelGrid grid;
    grid.nx = nx;
    grid.ny = ny;
    grid.nz = nz;
    grid.cells.resize(static_cast<size_t>(nx) * ny * nz, CELL_FLUID);

    // -----------------------------------------------------------------------
    // Step 1: Map grid to physical domain
    //
    // The domain spans [-domainExtent, +domainExtent] in each axis,
    // scaled so the model (radius ~1) occupies domainScale fraction of the
    // smallest grid dimension.
    // -----------------------------------------------------------------------
    int minDim = std::min({nx, ny, nz});
    float domainExtent = model.radius / domainScale; // half-size of domain in model units
    grid.voxelSize = (2.0f * domainExtent) / static_cast<float>(minDim);

    // Domain origin: the world-space position of voxel (0,0,0)'s corner
    glm::vec3 domainOrigin = model.center - glm::vec3(
        grid.voxelSize * nx * 0.5f,
        grid.voxelSize * ny * 0.5f,
        grid.voxelSize * nz * 0.5f
    );

    glm::vec3 halfVoxel(grid.voxelSize * 0.5f);

    printf("[Voxelizer] Grid: %d x %d x %d  |  voxelSize: %.6f  |  domainScale: %.2f\n",
           nx, ny, nz, grid.voxelSize, domainScale);

    // -----------------------------------------------------------------------
    // Step 2: Rasterize each triangle into the voxel grid
    //
    // For each triangle, compute its AABB in grid coordinates, then test
    // every candidate voxel with the full SAT intersection test.
    // -----------------------------------------------------------------------
    int totalTriangles = 0;

    for (const Mesh& mesh : model.meshes) {
        const auto& verts = mesh.vertices;
        const auto& idx   = mesh.indices;
        int triCount = static_cast<int>(idx.size()) / 3;
        totalTriangles += triCount;

        for (int t = 0; t < triCount; ++t) {
            // Get triangle vertices in world space
            glm::vec3 v0 = verts[idx[t * 3 + 0]].position;
            glm::vec3 v1 = verts[idx[t * 3 + 1]].position;
            glm::vec3 v2 = verts[idx[t * 3 + 2]].position;

            // Compute triangle AABB in world space
            glm::vec3 triMin = glm::min(glm::min(v0, v1), v2);
            glm::vec3 triMax = glm::max(glm::max(v0, v1), v2);

            // Convert to grid coordinates (which voxels could this triangle touch?)
            int ix0 = std::max(0, static_cast<int>(std::floor((triMin.x - domainOrigin.x) / grid.voxelSize)));
            int iy0 = std::max(0, static_cast<int>(std::floor((triMin.y - domainOrigin.y) / grid.voxelSize)));
            int iz0 = std::max(0, static_cast<int>(std::floor((triMin.z - domainOrigin.z) / grid.voxelSize)));

            int ix1 = std::min(nx - 1, static_cast<int>(std::floor((triMax.x - domainOrigin.x) / grid.voxelSize)));
            int iy1 = std::min(ny - 1, static_cast<int>(std::floor((triMax.y - domainOrigin.y) / grid.voxelSize)));
            int iz1 = std::min(nz - 1, static_cast<int>(std::floor((triMax.z - domainOrigin.z) / grid.voxelSize)));

            // Test each candidate voxel with the precise SAT test
            for (int iz = iz0; iz <= iz1; ++iz) {
                for (int iy = iy0; iy <= iy1; ++iy) {
                    for (int ix = ix0; ix <= ix1; ++ix) {
                        // Voxel center in world space
                        glm::vec3 voxelCenter = domainOrigin + glm::vec3(
                            (ix + 0.5f) * grid.voxelSize,
                            (iy + 0.5f) * grid.voxelSize,
                            (iz + 0.5f) * grid.voxelSize
                        );

                        if (triangleIntersectsAABB(v0, v1, v2, voxelCenter, halfVoxel)) {
                            grid.at(ix, iy, iz) = CELL_SOLID;
                        }
                    }
                }
            }
        }
    }

    printf("[Voxelizer] Rasterized %d triangles\n", totalTriangles);

    // -----------------------------------------------------------------------
    // Step 2.5: Dilate solid cells to thicken thin-shell models
    //
    // Many 3D models (especially game-ready ones) are thin shells with no
    // watertight interior. A single-voxel surface is invisible to the LBM
    // solver. We dilate solid cells by 2 voxels in each direction so the
    // model acts as a proper obstacle.
    // -----------------------------------------------------------------------
    {
        const int ddx[] = {1, -1, 0, 0, 0, 0};
        const int ddy[] = {0, 0, 1, -1, 0, 0};
        const int ddz[] = {0, 0, 0, 0, 1, -1};
        int dilateRadius = 2;

        for (int pass = 0; pass < dilateRadius; pass++) {
            std::vector<uint8_t> prev = grid.cells;
            for (int iz = 1; iz < nz - 1; ++iz) {
                for (int iy = 1; iy < ny - 1; ++iy) {
                    for (int ix = 1; ix < nx - 1; ++ix) {
                        if (prev[iz * nx * ny + iy * nx + ix] == CELL_SOLID) {
                            for (int d = 0; d < 6; ++d) {
                                int ax = ix + ddx[d], ay = iy + ddy[d], az = iz + ddz[d];
                                if (ax >= 1 && ax < nx - 1 && ay >= 1 && ay < ny - 1 && az >= 1 && az < nz - 1) {
                                    grid.at(ax, ay, az) = CELL_SOLID;
                                }
                            }
                        }
                    }
                }
            }
        }

        int solidAfterDilate = 0;
        for (auto c : grid.cells) if (c == CELL_SOLID) solidAfterDilate++;
        printf("[Voxelizer] After dilation (%d passes): %d solid cells\n", dilateRadius, solidAfterDilate);
    }

    // -----------------------------------------------------------------------
    // Step 3: Flood fill from exterior to distinguish inside from outside
    //
    // BFS from all 8 corners of the grid. Any FLUID cell reachable from a
    // corner is truly exterior. Unreachable FLUID cells are enclosed by the
    // solid surface and should be marked SOLID.
    //
    // We use a temporary "visited" marker (value 255) during BFS, then
    // convert back: visited -> FLUID, unvisited FLUID -> SOLID.
    // -----------------------------------------------------------------------
    constexpr uint8_t VISITED = 255;

    std::queue<glm::ivec3> bfsQueue;

    // Seed BFS from 8 corners (if they are fluid)
    int corners[][3] = {
        {0, 0, 0}, {nx-1, 0, 0}, {0, ny-1, 0}, {0, 0, nz-1},
        {nx-1, ny-1, 0}, {nx-1, 0, nz-1}, {0, ny-1, nz-1}, {nx-1, ny-1, nz-1}
    };

    for (auto& c : corners) {
        if (grid.at(c[0], c[1], c[2]) == CELL_FLUID) {
            grid.at(c[0], c[1], c[2]) = VISITED;
            bfsQueue.push(glm::ivec3(c[0], c[1], c[2]));
        }
    }

    // 6-connected BFS
    const int dx[] = {1, -1, 0, 0, 0, 0};
    const int dy[] = {0, 0, 1, -1, 0, 0};
    const int dz[] = {0, 0, 0, 0, 1, -1};

    while (!bfsQueue.empty()) {
        glm::ivec3 p = bfsQueue.front();
        bfsQueue.pop();

        for (int d = 0; d < 6; ++d) {
            int ax = p.x + dx[d];
            int ay = p.y + dy[d];
            int az = p.z + dz[d];

            if (ax < 0 || ax >= nx || ay < 0 || ay >= ny || az < 0 || az >= nz)
                continue;

            if (grid.at(ax, ay, az) == CELL_FLUID) {
                grid.at(ax, ay, az) = VISITED;
                bfsQueue.push(glm::ivec3(ax, ay, az));
            }
        }
    }

    // Convert: VISITED -> FLUID, remaining FLUID (unreachable interior) -> SOLID
    for (size_t i = 0; i < grid.cells.size(); ++i) {
        if (grid.cells[i] == VISITED) {
            grid.cells[i] = CELL_FLUID;
        } else if (grid.cells[i] == CELL_FLUID) {
            grid.cells[i] = CELL_SOLID; // enclosed interior
        }
    }

    // -----------------------------------------------------------------------
    // Step 4: Set boundary conditions based on wind direction
    //
    // Inlet and outlet are placed on opposite faces of the grid along the
    // wind axis. We leave the outermost slice (0 or max-1) as wall and
    // use slice 1 / max-2 for inlet/outlet.
    // -----------------------------------------------------------------------
    auto setFace = [&](int axis, int slice, uint8_t cellVal) {
        // axis: 0=X, 1=Y, 2=Z
        // slice: which index on that axis
        if (axis == 0) {
            for (int iz = 0; iz < nz; ++iz)
                for (int iy = 0; iy < ny; ++iy)
                    if (grid.at(slice, iy, iz) == CELL_FLUID)
                        grid.at(slice, iy, iz) = cellVal;
        } else if (axis == 1) {
            for (int iz = 0; iz < nz; ++iz)
                for (int ix = 0; ix < nx; ++ix)
                    if (grid.at(ix, slice, iz) == CELL_FLUID)
                        grid.at(ix, slice, iz) = cellVal;
        } else {
            for (int iy = 0; iy < ny; ++iy)
                for (int ix = 0; ix < nx; ++ix)
                    if (grid.at(ix, iy, slice) == CELL_FLUID)
                        grid.at(ix, iy, slice) = cellVal;
        }
    };

    switch (windDir) {
        case WIND_POS_X: setFace(0, 1,      CELL_INLET); setFace(0, nx-2, CELL_OUTLET); break;
        case WIND_NEG_X: setFace(0, nx-2,   CELL_INLET); setFace(0, 1,    CELL_OUTLET); break;
        case WIND_POS_Z: setFace(2, 1,      CELL_INLET); setFace(2, nz-2, CELL_OUTLET); break;
        case WIND_NEG_Z: setFace(2, nz-2,   CELL_INLET); setFace(2, 1,    CELL_OUTLET); break;
        case WIND_POS_Y: setFace(1, 1,      CELL_INLET); setFace(1, ny-2, CELL_OUTLET); break;
        case WIND_NEG_Y: setFace(1, ny-2,   CELL_INLET); setFace(1, 1,    CELL_OUTLET); break;
    }

    // -----------------------------------------------------------------------
    // Step 5: Print statistics
    // -----------------------------------------------------------------------
    int nSolid = 0, nFluid = 0, nInlet = 0, nOutlet = 0;
    for (size_t i = 0; i < grid.cells.size(); ++i) {
        switch (grid.cells[i]) {
            case CELL_SOLID:  ++nSolid;  break;
            case CELL_FLUID:  ++nFluid;  break;
            case CELL_INLET:  ++nInlet;  break;
            case CELL_OUTLET: ++nOutlet; break;
        }
    }

    int total = nx * ny * nz;
    printf("[Voxelizer] Cell statistics:\n");
    printf("  Solid:  %7d  (%.1f%%)\n", nSolid,  100.0f * nSolid  / total);
    printf("  Fluid:  %7d  (%.1f%%)\n", nFluid,  100.0f * nFluid  / total);
    printf("  Inlet:  %7d  (%.1f%%)\n", nInlet,  100.0f * nInlet  / total);
    printf("  Outlet: %7d  (%.1f%%)\n", nOutlet, 100.0f * nOutlet / total);

    return grid;
}
