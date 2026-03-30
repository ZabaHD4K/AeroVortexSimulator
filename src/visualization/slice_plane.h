#pragma once
#include <glm/glm.hpp>
#include <cstdint>

enum SliceAxis { SLICE_X, SLICE_Y, SLICE_Z };
enum SliceField { FIELD_VELOCITY, FIELD_PRESSURE, FIELD_VORTICITY };

class SlicePlaneRenderer {
public:
    bool init();
    void shutdown();

    // Extract a slice from a 3D field and render it
    // cellTypes: optional, if provided solid cells are rendered transparent
    void render(const float* field3D, int nx, int ny, int nz,
                SliceAxis axis, int sliceIndex,
                const glm::mat4& mvp, float minVal, float maxVal,
                const uint8_t* cellTypes = nullptr);

    SliceAxis axis = SLICE_X;
    int sliceIndex = 50;
    float opacity = 0.7f;

private:
    unsigned int shader = 0;
    unsigned int VAO = 0, VBO = 0;
    unsigned int texture = 0;
    int texW = 0, texH = 0;
};
