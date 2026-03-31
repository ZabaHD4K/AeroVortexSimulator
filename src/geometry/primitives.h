#pragma once
#include "mesh.h"
#include <string>

// Known reference values for validation
struct ValidationRef {
    float expectedCd;
    float expectedCl;
    float reRange[2];  // Reynolds number range where values are valid
    const char* source;
};

// Procedural test geometries (already normalized to unit bounding sphere)
Model generateSphere(int segments = 32);
Model generateCylinder(int segments = 32, float lengthRatio = 3.0f);
Model generateNACA0012(int chordPoints = 80, float span = 1.5f);

// Reference data
ValidationRef getSphereRef();
ValidationRef getCylinderRef();
ValidationRef getNACA0012Ref();
