#include "aero_forces.h"
#include <cmath>
#include <set>
#include <algorithm>

enum : uint8_t {
    CELL_FLUID = 0,
    CELL_SOLID = 1,
    CELL_INLET = 2,
    CELL_OUTLET = 3
};

AeroCoefficients calculateAeroCoefficients(
    const float* pressure,
    const uint8_t* cellTypes,
    int nx, int ny, int nz,
    float inletVelocity,
    float domainScale)
{
    AeroCoefficients c;
    if (!pressure || !cellTypes) return c;

    auto idx = [&](int x, int y, int z) { return z * nx * ny + y * nx + x; };

    float pRef = 1.0f / 3.0f;
    float fx = 0, fy = 0, fz = 0;

    int ddx[] = {1, -1, 0, 0, 0, 0};
    int ddy[] = {0, 0, 1, -1, 0, 0};
    int ddz[] = {0, 0, 0, 0, 1, -1};

    for (int z = 1; z < nz - 1; z++) {
        for (int y = 1; y < ny - 1; y++) {
            for (int x = 1; x < nx - 1; x++) {
                if (cellTypes[idx(x, y, z)] != CELL_SOLID) continue;

                for (int d = 0; d < 6; d++) {
                    int x2 = x + ddx[d], y2 = y + ddy[d], z2 = z + ddz[d];
                    if (x2 < 0 || x2 >= nx || y2 < 0 || y2 >= ny || z2 < 0 || z2 >= nz) continue;

                    if (cellTypes[idx(x2, y2, z2)] == CELL_FLUID) {
                        float p = pressure[idx(x2, y2, z2)] - pRef;
                        fx -= p * (float)ddx[d];
                        fy -= p * (float)ddy[d];
                        fz -= p * (float)ddz[d];
                    }
                }
            }
        }
    }

    c.Fx = fx;
    c.Fy = fy;
    c.Fz = fz;

    // Frontal area: project solid cells onto YZ plane
    std::set<int> projected;
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++)
                if (cellTypes[idx(x, y, z)] == CELL_SOLID)
                    projected.insert(y * nz + z);

    float frontalArea = std::max((float)projected.size(), 1.0f);
    float q = 0.5f * 1.0f * inletVelocity * inletVelocity;

    if (q > 1e-10f) {
        c.Cd = fx / (q * frontalArea);
        c.Cl = fy / (q * frontalArea);
        c.Cs = fz / (q * frontalArea);
    }

    return c;
}
