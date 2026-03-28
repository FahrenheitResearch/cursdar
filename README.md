# cursdar

Real-time NEXRAD radar viewer built entirely on custom CUDA kernels. Downloads all 162 US weather radar stations from AWS S3, parses Level 2 Archive II data, and renders at native resolution using GPU-accelerated pipelines on NVIDIA GPUs.

## What it does

- Downloads latest NEXRAD Level 2 data from all 162 WSR-88D stations simultaneously (48 parallel HTTP connections via WinHTTP)
- BZ2 decompression with parallel block processing
- GPU-accelerated Level 2 binary parsing (custom CUDA kernel extracts MSG31 radials)
- GPU data transposition for coalesced memory access patterns
- Native-resolution rendering: each viewport pixel directly samples raw gate data at its geographic location. Resolution scales infinitely with zoom.
- Single-station mode with instant mouse-follow switching (all data pre-loaded on GPU)
- National mosaic mode compositing all stations with spatial grid acceleration
- 3D volumetric storm rendering with ray marching, shadow rays, gradient-based surface lighting
- Real-time vertical cross-section through storms (draggable cut line)
- Hardware-interpolated color tables via CUDA texture objects
- 7 radar products: REF, VEL, SW, ZDR, CC, KDP, PHI
- Multi-tilt support with split-cut aware sweep organization

## Architecture

```
[AWS S3] --> [WinHTTP 48-thread download]
         --> [CPU: BZ2 parallel decompression]
         --> [GPU: MSG31 parsing kernel]
         --> [GPU: gate data transposition kernel]
         --> [GPU: spatial grid construction kernel]
         --> [GPU: native-res render kernel (single-station or mosaic)]
         --> [GPU: 3D volume build + ray march kernels]
         --> [GPU: cross-section kernel]
         --> [CUDA-GL interop: zero-copy to OpenGL texture]
         --> [Dear ImGui overlay]
```

Everything that can be GPU, is GPU. Only the network I/O and BZ2 decompression (inherently sequential algorithm) remain on CPU.

## Controls

| Key | Action |
|-----|--------|
| Mouse move | Select nearest radar station |
| Left drag | Pan viewport / Move cross-section line |
| Right drag | Reposition cross-section endpoint / Orbit 3D camera |
| Scroll | Zoom |
| 1-7 | Select product (REF, VEL, SW, ZDR, CC, KDP, PHI) |
| Left/Right | Cycle products |
| Up/Down | Cycle elevation tilts |
| A | Toggle national mosaic (all stations) |
| V | Toggle 3D volumetric view |
| X | Toggle cross-section mode |
| R | Refresh data from AWS |
| Home | Reset viewport to CONUS |

## Building

### Requirements

- NVIDIA GPU with CUDA support (tested on RTX 5090)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 11.0+
- Visual Studio 2022 Build Tools (or full VS2022)
- CMake 3.24+
- Ninja build system

### Build

```bat
build.bat
```

Or manually:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

Binary output: `build/cursdar.exe`

### Dependencies (auto-fetched by CMake)

- [GLFW](https://www.glfw.org/) - windowing
- [Dear ImGui](https://github.com/ocornut/imgui) (docking branch) - UI
- [bzip2](https://sourceware.org/bzip2/) - NEXRAD decompression

No other dependencies. WinHTTP and OpenGL are built into Windows.

## Data Source

NEXRAD Level 2 Archive II data from the [Unidata NEXRAD S3 bucket](https://s3.amazonaws.com/unidata-nexrad-level2/) (public, no authentication required).

## CUDA Kernels

| Kernel | Purpose | Key optimization |
|--------|---------|-----------------|
| `parseMsg31Kernel` | Extract radials from raw Level 2 bytes | Parallel message scanning |
| `transposeKernel` | Radial-major to gate-major layout | One thread per (gate, radial) |
| `buildGridKernel` | Spatial acceleration grid | Atomic cell insertion |
| `singleStationKernel` | Native-res single radar render | Shared-memory azimuth binary search |
| `nativeRenderKernel` | National mosaic render | Spatial grid + per-pixel station sampling |
| `buildVolumeKernel` | 3D voxel grid from multi-tilt data | Elevation interpolation between sweeps |
| `rayMarchKernel` | Volumetric storm rendering | Shadow rays, gradient normals, HW trilinear texture |
| `crossSectionKernel` | Vertical atmosphere slice | Direct 3D texture sampling |

## Project Structure

```
src/
  main.cpp              Entry point, GLFW window, main loop
  app.h/cpp             Application state, download orchestration
  cuda/
    cuda_common.cuh     Constants, error macros
    renderer.cuh/cu     Native-res render + compositor + spatial grid
    gpu_pipeline.cuh/cu GPU parsing + transposition kernels
    volume3d.cuh/cu     3D volume build + ray march + cross-section
  nexrad/
    level2.h            NEXRAD binary format structures
    level2_parser.h/cpp BZ2 decompression + message parsing
    products.h          Product enums and metadata
    stations.h          All 162 NEXRAD station coordinates
  net/
    downloader.h/cpp    WinHTTP async download manager
    aws_nexrad.h        S3 URL construction and XML parsing
  render/
    gl_interop.h/cpp    CUDA-OpenGL texture interop
    projection.h        Geographic projection math
  ui/
    ui.h/cpp            ImGui panels and overlays
```

## License

MIT
