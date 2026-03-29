# cursdar

CUDA-native NEXRAD Level 2 radar viewer for NVIDIA GPUs.

`cursdar` pulls live WSR-88D volumes from AWS, parses Archive II data, and renders radar products with custom CUDA kernels instead of pre-rendered map tiles. It can load all 162 US NEXRAD sites, switch instantly between stations, build a national mosaic, and render 3D storm volumes and vertical cross-sections in real time.

This is an experimental graphics and data-pipeline project. The core viewer is now in good shape, but it is still not an operational forecasting tool.

## Why it is interesting

- Live ingest from the public Unidata NEXRAD S3 bucket
- 162-station national radar loading with GPU-backed rendering
- Single-station mode, all-site mosaic mode, 3D volume view, and vertical cross-section view
- Seven Level 2 products: `REF`, `VEL`, `SW`, `ZDR`, `CC`, `KDP`, `PHI`
- Storm-relative velocity mode plus experimental detection overlays for debris, hail, and mesocyclone signatures
- Historic tornado playback and one-click archive snapshot loading
- NWS warning polygon overlay

## Current state

What is solid:

- The 2D radar rendering now looks good in normal use and feels like a real viewer instead of an early prototype
- The live data path from download through parse, GPU upload, and rendering
- Multi-station orchestration and fast station switching
- Mosaic rendering, 3D volume rendering, and cross-sections
- Historic event playback and archive snapshot loading

What is still rough:

- The project is still experimental and not operationally validated
- Tilt selection, product presentation, and some UI flows still need refinement
- Detection overlays are still heuristic and should be treated as experimental

If you want the short version: the engine is real, the 2D rendering is finally in a good place, and the remaining work is mostly polish and validation.

## Features

- Downloads current Level 2 data for all 162 NEXRAD sites
- Uses parallel HTTP fetches and BZip2 decompression before GPU upload
- Renders live radar directly from raw gate data with sharp native-resolution 2D output
- Supports national mosaic mode with a CUDA-built spatial acceleration grid
- Supports 3D volumetric storm rendering with ray marching
- Supports draggable vertical cross-sections through storms
- Includes storm-relative velocity controls
- Includes experimental `TDS`, hail, and mesocyclone overlay markers
- Includes historic tornado event playback
- Includes an all-site archive snapshot for March 30, 2025 at 5 PM ET
- Draws active NWS severe warning polygons on top of the radar view

## Architecture

```text
[AWS S3 / NEXRAD Level 2]
        -> [parallel download manager]
        -> [BZ2 decompression]
        -> [Level 2 parse]
        -> [GPU upload + transposition]
        -> [CUDA spatial grid]
        -> [single-station render or national mosaic render]
        -> [optional 3D volume build + ray march]
        -> [optional vertical cross-section]
        -> [CUDA/OpenGL interop texture]
        -> [Dear ImGui UI + overlays]
```

Most of the heavy lifting happens in CUDA. Network I/O and decompression stay on CPU.

## Controls

| Input | Action |
| --- | --- |
| Mouse move | Select nearest radar station |
| Left drag | Pan viewport / move cross-section line |
| Right drag | Move cross-section endpoint / orbit 3D camera |
| Scroll | Zoom |
| `1-7` | Select radar product |
| Left / Right | Cycle products |
| Up / Down | Cycle tilts |
| `A` | Toggle all-site mosaic |
| `V` | Toggle 3D volume mode |
| `X` | Toggle cross-section mode |
| `S` | Toggle storm-relative velocity mode |
| `R` | Refresh live data |
| `Home` | Reset viewport to CONUS |

Most other controls are exposed in the ImGui side panels.

## Build

### Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit
- CMake 3.24+
- Ninja
- Visual Studio 2022 Build Tools on Windows

Windows is the main tested path right now.

### Windows

```bat
build.bat
```

Manual build:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
mkdir build
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

Output:

```text
build/cursdar.exe
```

### Linux

```bash
chmod +x build.sh
./build.sh
```

On Linux you will also need the usual OpenGL/CMake toolchain and a working CUDA install with `nvcc` on `PATH`.

### Third-party dependencies

CMake fetches these automatically:

- [GLFW](https://www.glfw.org/)
- [Dear ImGui](https://github.com/ocornut/imgui)
- [bzip2](https://sourceware.org/bzip2/)

Platform libraries:

- Windows: `WinHTTP`, `OpenGL`
- Linux: `libcurl`, `OpenGL`, `pthread`, `dl`

## Data sources

- NEXRAD Level 2 Archive II data from the public Unidata S3 bucket
- Active severe warning polygons from `api.weather.gov`

## CUDA kernels

| Kernel | Purpose |
| --- | --- |
| `parseMsg31Kernel` | Extract radials from raw Level 2 bytes |
| `transposeKernel` | Convert radar data into GPU-friendly gate-major layout |
| `buildGridKernel` | Build the spatial lookup grid for national mosaic rendering |
| `singleStationKernel` | Render a single radar directly from raw gate data |
| `nativeRenderKernel` | Render the all-site mosaic |
| `buildVolumeKernel` | Build the 3D volume from multiple tilts |
| `rayMarchKernel` | Render the 3D storm volume |
| `crossSectionKernel` | Render a vertical cut through the storm |

## Project layout

```text
src/
  main.cpp
  app.h / app.cpp
  historic.h / historic.cpp
  cuda/
    renderer.cuh / renderer.cu
    gpu_pipeline.cuh / gpu_pipeline.cu
    volume3d.cuh / volume3d.cu
  net/
    downloader.h / downloader.cpp
    warnings.h / warnings.cpp
    aws_nexrad.h
  nexrad/
    level2.h
    level2_parser.h / level2_parser.cpp
    products.h
    stations.h
  render/
    gl_interop.h / gl_interop.cpp
    projection.h
  ui/
    ui.h / ui.cpp
```

## License

MIT
