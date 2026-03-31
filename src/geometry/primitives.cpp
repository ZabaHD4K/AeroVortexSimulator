#include "primitives.h"
#include "model_loader.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Sphere ──────────────────────────────────────────────

Model generateSphere(int segments) {
    Model model;
    model.name = "Sphere (validation)";

    Mesh mesh;
    int rings = segments;
    int sectors = segments;

    for (int r = 0; r <= rings; r++) {
        float phi = (float)M_PI * r / rings;
        float y = std::cos(phi);
        float sinPhi = std::sin(phi);

        for (int s = 0; s <= sectors; s++) {
            float theta = 2.0f * (float)M_PI * s / sectors;
            float x = sinPhi * std::cos(theta);
            float z = sinPhi * std::sin(theta);

            Vertex v;
            v.position = glm::vec3(x, y, z) * 0.5f; // radius 0.5 -> diameter 1 -> fits unit sphere
            v.normal = glm::vec3(x, y, z);
            mesh.vertices.push_back(v);
        }
    }

    for (int r = 0; r < rings; r++) {
        for (int s = 0; s < sectors; s++) {
            int cur = r * (sectors + 1) + s;
            int nxt = cur + sectors + 1;
            mesh.indices.push_back(cur);
            mesh.indices.push_back(nxt);
            mesh.indices.push_back(cur + 1);
            mesh.indices.push_back(cur + 1);
            mesh.indices.push_back(nxt);
            mesh.indices.push_back(nxt + 1);
        }
    }

    uploadMeshToGPU(mesh);
    model.meshes.push_back(std::move(mesh));
    model.center = glm::vec3(0.0f);
    model.radius = 1.0f;
    return model;
}

// ── Cylinder (along X axis) ─────────────────────────────

Model generateCylinder(int segments, float lengthRatio) {
    Model model;
    model.name = "Cylinder (validation)";

    Mesh mesh;
    float halfLen = lengthRatio * 0.5f;
    float radius = 0.5f;

    // Normalize so bounding sphere radius = 1
    float bboxDiag = std::sqrt(halfLen * halfLen + radius * radius) * 2.0f;
    float scale = 2.0f / bboxDiag;
    halfLen *= scale;
    radius *= scale;

    // Side vertices
    for (int s = 0; s <= segments; s++) {
        float theta = 2.0f * (float)M_PI * s / segments;
        float y = radius * std::cos(theta);
        float z = radius * std::sin(theta);
        glm::vec3 normal(0.0f, std::cos(theta), std::sin(theta));

        Vertex v0, v1;
        v0.position = glm::vec3(-halfLen, y, z);
        v0.normal = normal;
        v1.position = glm::vec3(halfLen, y, z);
        v1.normal = normal;
        mesh.vertices.push_back(v0);
        mesh.vertices.push_back(v1);
    }

    // Side indices
    for (int s = 0; s < segments; s++) {
        int i = s * 2;
        mesh.indices.push_back(i);
        mesh.indices.push_back(i + 2);
        mesh.indices.push_back(i + 1);
        mesh.indices.push_back(i + 1);
        mesh.indices.push_back(i + 2);
        mesh.indices.push_back(i + 3);
    }

    // End caps
    int centerLeft = (int)mesh.vertices.size();
    Vertex cl;
    cl.position = glm::vec3(-halfLen, 0, 0);
    cl.normal = glm::vec3(-1, 0, 0);
    mesh.vertices.push_back(cl);

    int centerRight = (int)mesh.vertices.size();
    Vertex cr;
    cr.position = glm::vec3(halfLen, 0, 0);
    cr.normal = glm::vec3(1, 0, 0);
    mesh.vertices.push_back(cr);

    int capBase = (int)mesh.vertices.size();
    for (int s = 0; s <= segments; s++) {
        float theta = 2.0f * (float)M_PI * s / segments;
        float y = radius * std::cos(theta);
        float z = radius * std::sin(theta);

        Vertex vl, vr;
        vl.position = glm::vec3(-halfLen, y, z);
        vl.normal = glm::vec3(-1, 0, 0);
        vr.position = glm::vec3(halfLen, y, z);
        vr.normal = glm::vec3(1, 0, 0);
        mesh.vertices.push_back(vl);
        mesh.vertices.push_back(vr);
    }

    for (int s = 0; s < segments; s++) {
        int i = capBase + s * 2;
        // Left cap
        mesh.indices.push_back(centerLeft);
        mesh.indices.push_back(i + 2);
        mesh.indices.push_back(i);
        // Right cap
        mesh.indices.push_back(centerRight);
        mesh.indices.push_back(i + 1);
        mesh.indices.push_back(i + 3);
    }

    uploadMeshToGPU(mesh);
    model.meshes.push_back(std::move(mesh));
    model.center = glm::vec3(0.0f);
    model.radius = 1.0f;
    return model;
}

// ── NACA 0012 airfoil ───────────────────────────────────

static float naca0012_yt(float x) {
    // NACA 0012 half-thickness at position x (0..1)
    return 0.12f / 0.2f * (
        0.2969f * std::sqrt(x)
        - 0.1260f * x
        - 0.3516f * x * x
        + 0.2843f * x * x * x
        - 0.1015f * x * x * x * x
    );
}

Model generateNACA0012(int chordPoints, float span) {
    Model model;
    model.name = "NACA 0012 (validation)";

    Mesh mesh;

    // Generate airfoil profile points
    std::vector<glm::vec2> upper, lower;
    for (int i = 0; i <= chordPoints; i++) {
        // Cosine spacing for better leading edge resolution
        float t = (float)i / chordPoints;
        float x = 0.5f * (1.0f - std::cos((float)M_PI * t));
        float yt = naca0012_yt(x);
        upper.push_back(glm::vec2(x, yt));
        lower.push_back(glm::vec2(x, -yt));
    }

    // Extrude into 3D wing (along Z axis)
    float halfSpan = span * 0.5f;
    int spanSegments = 2;  // Simple extrusion

    // Normalize to fit unit bounding sphere
    float maxExtent = std::max(1.0f, span);
    float scale = 1.0f / maxExtent;

    auto addWingSurface = [&](const std::vector<glm::vec2>& profile, float nSign) {
        int base = (int)mesh.vertices.size();
        for (int zs = 0; zs <= spanSegments; zs++) {
            float z = (-halfSpan + span * zs / spanSegments) * scale;
            for (int i = 0; i <= chordPoints; i++) {
                Vertex v;
                // Center chord at origin (x offset by -0.5 to center)
                v.position = glm::vec3(
                    (profile[i].x - 0.5f) * scale,
                    profile[i].y * scale,
                    z
                );
                // Approximate normal from profile tangent
                float dx = 0, dy = 0;
                if (i > 0 && i < chordPoints) {
                    dx = profile[i+1].x - profile[i-1].x;
                    dy = profile[i+1].y - profile[i-1].y;
                } else if (i == 0) {
                    dx = profile[1].x - profile[0].x;
                    dy = profile[1].y - profile[0].y;
                } else {
                    dx = profile[i].x - profile[i-1].x;
                    dy = profile[i].y - profile[i-1].y;
                }
                float len = std::sqrt(dx*dx + dy*dy);
                if (len > 1e-6f) {
                    v.normal = glm::vec3(-dy / len * nSign, dx / len * nSign, 0.0f);
                } else {
                    v.normal = glm::vec3(0.0f, nSign, 0.0f);
                }
                mesh.vertices.push_back(v);
            }
        }

        // Indices
        int stride = chordPoints + 1;
        for (int zs = 0; zs < spanSegments; zs++) {
            for (int i = 0; i < chordPoints; i++) {
                int a = base + zs * stride + i;
                int b = a + 1;
                int c = a + stride;
                int d = c + 1;
                if (nSign > 0) {
                    mesh.indices.push_back(a);
                    mesh.indices.push_back(c);
                    mesh.indices.push_back(b);
                    mesh.indices.push_back(b);
                    mesh.indices.push_back(c);
                    mesh.indices.push_back(d);
                } else {
                    mesh.indices.push_back(a);
                    mesh.indices.push_back(b);
                    mesh.indices.push_back(c);
                    mesh.indices.push_back(b);
                    mesh.indices.push_back(d);
                    mesh.indices.push_back(c);
                }
            }
        }
    };

    addWingSurface(upper, 1.0f);
    addWingSurface(lower, -1.0f);

    // Trailing edge cap (close the wing at trailing edge)
    // Wingtip caps
    for (int side = 0; side < 2; side++) {
        float z = (side == 0 ? -halfSpan : halfSpan) * scale;
        glm::vec3 tipNormal = glm::vec3(0, 0, side == 0 ? -1.0f : 1.0f);

        int capBase = (int)mesh.vertices.size();
        for (int i = 0; i <= chordPoints; i++) {
            Vertex vu, vl;
            vu.position = glm::vec3((upper[i].x - 0.5f) * scale, upper[i].y * scale, z);
            vu.normal = tipNormal;
            vl.position = glm::vec3((lower[i].x - 0.5f) * scale, lower[i].y * scale, z);
            vl.normal = tipNormal;
            mesh.vertices.push_back(vu);
            mesh.vertices.push_back(vl);
        }
        for (int i = 0; i < chordPoints; i++) {
            int a = capBase + i * 2;
            int b = a + 1;
            int c = a + 2;
            int d = a + 3;
            if (side == 0) {
                mesh.indices.push_back(a); mesh.indices.push_back(b); mesh.indices.push_back(c);
                mesh.indices.push_back(b); mesh.indices.push_back(d); mesh.indices.push_back(c);
            } else {
                mesh.indices.push_back(a); mesh.indices.push_back(c); mesh.indices.push_back(b);
                mesh.indices.push_back(b); mesh.indices.push_back(c); mesh.indices.push_back(d);
            }
        }
    }

    uploadMeshToGPU(mesh);
    model.meshes.push_back(std::move(mesh));
    model.center = glm::vec3(0.0f);
    model.radius = 1.0f;
    return model;
}

// ── Reference data ──────────────────────────────────────

ValidationRef getSphereRef() {
    return { 0.47f, 0.0f, {100.0f, 10000.0f}, "Schlichting (1979)" };
}

ValidationRef getCylinderRef() {
    return { 1.17f, 0.0f, {100.0f, 10000.0f}, "Roshko (1961)" };
}

ValidationRef getNACA0012Ref() {
    // At 0 angle of attack, symmetric airfoil
    return { 0.012f, 0.0f, {1000.0f, 100000.0f}, "Abbott & Von Doenhoff (1959)" };
}
