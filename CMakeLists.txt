cmake_minimum_required(VERSION 3.24)
project(cursdar LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture - target Turing+ for best perf
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# ── Dependencies ──────────────────────────────────────────────
include(FetchContent)

# GLFW
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# Dear ImGui
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.8-docking
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(imgui)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw)
# Let ImGui use its built-in GL loader

# bzip2 - try system first, then fetch
find_package(BZip2 QUIET)
if(NOT BZip2_FOUND)
    FetchContent_Declare(
        bzip2_src
        URL      https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz
        URL_HASH SHA256=ab5a03176ee106d3f0fa90e381da478ddae405918153cca248e682cd0c4a2269
    )
    FetchContent_MakeAvailable(bzip2_src)
    add_library(bzip2 STATIC
        ${bzip2_src_SOURCE_DIR}/blocksort.c
        ${bzip2_src_SOURCE_DIR}/huffman.c
        ${bzip2_src_SOURCE_DIR}/crctable.c
        ${bzip2_src_SOURCE_DIR}/randtable.c
        ${bzip2_src_SOURCE_DIR}/compress.c
        ${bzip2_src_SOURCE_DIR}/decompress.c
        ${bzip2_src_SOURCE_DIR}/bzlib.c
    )
    target_include_directories(bzip2 PUBLIC ${bzip2_src_SOURCE_DIR})
    add_library(BZip2::BZip2 ALIAS bzip2)
endif()

# ── cursdar executable ────────────────────────────────────────
add_executable(cursdar
    src/main.cpp
    src/app.cpp
    src/nexrad/level2_parser.cpp
    src/net/downloader.cpp
    src/render/gl_interop.cpp
    src/ui/ui.cpp
    src/cuda/renderer.cu
    src/cuda/gpu_pipeline.cu
    src/cuda/volume3d.cu
)

target_include_directories(cursdar PRIVATE src ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(WIN32)
    target_compile_definitions(cursdar PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ── Platform-specific linking ────────────────────────────────
target_link_libraries(cursdar PRIVATE
    imgui
    glfw
    BZip2::BZip2
    CUDA::cudart
)

if(WIN32)
    target_link_libraries(cursdar PRIVATE opengl32 winhttp)
else()
    find_package(CURL REQUIRED)
    target_link_libraries(cursdar PRIVATE GL CURL::libcurl dl pthread)
endif()

# CUDA settings for max performance
set_target_properties(cursdar PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(cursdar PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --expt-relaxed-constexpr
        -Xptxas=-v
    >
)

find_package(CUDAToolkit REQUIRED)
