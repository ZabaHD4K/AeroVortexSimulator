// NOTE: This pressure-based force calculation is kept as an alternative/fallback.
// The primary force computation uses GPU momentum exchange (LBM3D::computeForces).

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

    // Use actual average far-field pressure as reference instead of theoretical 1/3.
    // This accounts for global pressure shifts during LBM transients.
    // Sample from inlet cells (which should be at freestream conditions).
    double pRefSum = 0;
    int pRefCount = 0;
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++) {
                if (cellTypes[idx(x, y, z)] == CELL_INLET) {
                    pRefSum += pressure[idx(x, y, z)];
                    pRefCount++;
                }
            }
    float pRef = (pRefCount > 0) ? (float)(pRefSum / pRefCount) : (1.0f / 3.0f);

    // Accumulate forces using double precision to reduce round-off
    double fx = 0, fy = 0, fz = 0;
    int boundaryFaces = 0;

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
                        double p = (double)pressure[idx(x2, y2, z2)] - (double)pRef;
                        fx -= p * (double)ddx[d];
                        fy -= p * (double)ddy[d];
                        fz -= p * (double)ddz[d];
                        boundaryFaces++;
                    }
                }
            }
        }
    }

    c.Fx = (float)fx;
    c.Fy = (float)fy;
    c.Fz = (float)fz;

    // Frontal area: project solid cells onto YZ plane
    std::set<int> projected;
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++)
                if (cellTypes[idx(x, y, z)] == CELL_SOLID)
                    projected.insert(y * nz + z);

    float frontalArea = std::max((float)projected.size(), 1.0f);

    // Dynamic pressure: q = 0.5 * rho * U^2
    // Use cs^2=1/3, so p_dynamic = 0.5 * rho_ref * U^2
    float q = 0.5f * 1.0f * inletVelocity * inletVelocity;

    if (q > 1e-10f && frontalArea > 0) {
        c.Cd = (float)(fx / (double)(q * frontalArea));
        c.Cl = (float)(fy / (double)(q * frontalArea));
        c.Cs = (float)(fz / (double)(q * frontalArea));
    }

    return c;
}
