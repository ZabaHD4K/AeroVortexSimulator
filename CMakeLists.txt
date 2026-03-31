cmake_minimum_required(VERSION 3.24)
project(aero3d_simulator LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# MSVC needs /EHsc (only for C/CXX, not CUDA)
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/EHsc>)
endif()

include(FetchContent)

# ── GLFW ──────────────────────────────────────────────
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# ── ImGui ─────────────────────────────────────────────
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.8
)
FetchContent_MakeAvailable(imgui)

# ── GLM ───────────────────────────────────────────────
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
)
FetchContent_MakeAvailable(glm)

# ── Assimp ────────────────────────────────────────────
FetchContent_Declare(
    assimp
    GIT_REPOSITORY https://github.com/assimp/assimp.git
    GIT_TAG        v5.4.3
)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(assimp)

# ── GLAD (local) ──────────────────────────────────────
add_library(glad STATIC extern/glad/src/gl.c)
target_include_directories(glad PUBLIC extern/glad/include)

# ── ImGui library ─────────────────────────────────────
add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC glfw glad)

# ── LBM CUDA library ─────────────────────────────────
add_library(lbm_cuda STATIC
    src/core/lbm2d.cu
    src/core/lbm3d.cu
)
target_include_directories(lbm_cuda PUBLIC src)
set_target_properties(lbm_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES native
)
target_compile_options(lbm_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc>)

# ── Main executable ───────────────────────────────────
add_executable(aero3d_simulator
    src/main.cpp
    src/app.cpp
    src/geometry/model_loader.cpp
    src/geometry/primitives.cpp
    src/core/voxelizer.cpp
    src/core/aero_forces.cpp
    src/visualization/renderer.cpp
    src/visualization/field2d.cpp
    src/visualization/streamlines.cpp
    src/visualization/slice_plane.cpp
    src/visualization/surface_pressure.cpp
    src/visualization/particles.cpp
    src/visualization/volume_renderer.cpp
    src/export/data_export.cpp
    src/ui/gui.cpp
)

target_include_directories(aero3d_simulator PRIVATE
    src
    ${assimp_SOURCE_DIR}/include
    ${assimp_BINARY_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(aero3d_simulator PRIVATE
    glfw
    glad
    imgui_lib
    glm::glm
    assimp
    lbm_cuda
    opengl32
    comdlg32
)
