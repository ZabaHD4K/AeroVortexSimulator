#pragma once
#include <cstdint>

struct AeroCoefficients {
    float Cd = 0;  // drag coefficient
    float Cl = 0;  // lift coefficient
    float Cs = 0;  // side force coefficient
    float Fx = 0;  // force in x (drag direction)
    float Fy = 0;  // force in y (lift direction)
    float Fz = 0;  // force in z (side direction)
};

// Calculate aerodynamic forces from pressure field
// Integrates pressure difference on solid boundary cells
AeroCoefficients calculateAeroCoefficients(
    const float* pressure,      // 3D pressure field (host)
    const uint8_t* cellTypes,   // cell type grid (host)
    int nx, int ny, int nz,
    float inletVelocity,
    float domainScale
);
