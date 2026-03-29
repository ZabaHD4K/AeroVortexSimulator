#pragma once

// Renders a 2D scalar field as a colored texture on a fullscreen quad
class Field2DRenderer {
public:
    bool init();
    void shutdown();

    // Upload new field data and render
    // data: float array of nx*ny values
    // minVal/maxVal: range for colormap
    void render(const float* data, int nx, int ny, float minVal, float maxVal);

private:
    unsigned int shader = 0;
    unsigned int VAO = 0, VBO = 0;
    unsigned int texture = 0;
    int texW = 0, texH = 0;

    unsigned int compileShader(const char* vert, const char* frag);
};
