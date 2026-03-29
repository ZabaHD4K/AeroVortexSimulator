cmake_minimum_required(VERSION 3.24)
project(aero3d_simulator LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC needs /EHsc for exception handling (required by Assimp)
if(MSVC)
    add_compile_options(/EHsc)
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

# ── Native File Dialog (simple implementation) ────────
# We use Windows native dialog via Win32 API directly

# ── Main executable ───────────────────────────────────
add_executable(aero3d_simulator
    src/main.cpp
    src/app.cpp
    src/geometry/model_loader.cpp
    src/visualization/renderer.cpp
    src/ui/gui.cpp
)

target_include_directories(aero3d_simulator PRIVATE
    src
    ${assimp_SOURCE_DIR}/include
    ${assimp_BINARY_DIR}/include
)

target_link_libraries(aero3d_simulator PRIVATE
    glfw
    glad
    imgui_lib
    glm::glm
    assimp
    opengl32
    comdlg32
)
