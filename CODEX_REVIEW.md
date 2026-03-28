# cursdar - Code Review Handoff

## What This Is

A real-time NEXRAD (weather radar) viewer built from scratch in C++17/CUDA. Downloads live Level 2 radar data from AWS S3, parses the binary format, and renders everything on the GPU using custom CUDA kernels. Built in ~2 days as a rapid prototype by a human + Claude (Anthropic). Runs on Windows (primary) and Linux.

**This is a rough tech demo, not production software.** Expect sharp edges.

## Tech Stack

- **C++17** with **CUDA 13** (targets SM 7.5+ / Turing+)
- **Dear ImGui** (docking branch) + **GLFW** + **OpenGL 3** for UI
- **bzip2** for NEXRAD decompression
- **WinHTTP** (Windows) / **libcurl** (Linux) for HTTP
- CMake + Ninja build system, FetchContent for deps
- Target hardware: RTX 5090 (32GB VRAM, 170 SMs) but runs on any CUDA GPU

## Architecture Overview

```
AWS S3 (unidata-nexrad-level2)
    |
    v
Downloader (48 threads, WinHTTP/curl)
    |
    v
Level2 Parser (BZ2 decompress -> MSG31 parse -> split-cut grouping)
    |
    v
PrecomputedSweep (gate-major transposed data, sorted azimuths)
    |
    v
GPU Upload (pinned memory staging -> async DMA)
    |
    v
CUDA Render Kernels:
  - forwardRenderKernel: 1 thread/gate, polar quad rasterization
  - singleStationKernel: inverse mapping, shared-mem azimuth search
  - nativeRenderKernel: multi-station mosaic with spatial grid
  - rayMarchKernel: 3D volumetric with shadow rays + Blinn-Phong
  - crossSectionKernel: flat 2D vertical slice with beam refraction
    |
    v
CUDA-GL interop -> ImGui background texture -> display
```

## File Map (~5200 LOC excluding data headers)

| File | Lines | Role |
|------|-------|------|
| `src/app.cpp` | 1156 | App orchestration: download pipeline, GPU upload, tilt/product management, detection, dealiasing |
| `src/cuda/renderer.cu` | 972 | All 2D render kernels, color tables, GPU station management, spatial grid |
| `src/ui/ui.cpp` | 625 | ImGui UI: controls, overlays (boundaries, warnings, detection markers, range rings) |
| `src/cuda/volume3d.cu` | 545 | 3D volume build, ray march, cross-section kernels |
| `src/nexrad/level2_parser.cpp` | 522 | BZ2 decompression + NEXRAD MSG31 binary parsing |
| `src/cuda/gpu_pipeline.cu` | 458 | GPU-side parser + transpose kernels (experimental, not primary path) |
| `src/historic.cpp` | 440 | Historic tornado event loader (multi-file download + timeline) |
| `src/net/downloader.cpp` | 253 | HTTP client (WinHTTP on Windows, libcurl on Linux) |
| `src/main.cpp` | 204 | GLFW window, input dispatch, main loop |
| `src/data/us_boundaries.h` | 51K | Embedded state boundary + city data (constexpr, auto-generated) |

## Key Design Decisions

1. **Native-res rendering** - No intermediate textures. Each viewport pixel directly samples raw gate data. Resolution = viewport resolution at any zoom.

2. **Forward render as primary path** - One CUDA thread per gate cell. Computes screen-space quad corners, fills pixels via edge functions. Skips 60-80% of empty gates immediately (branch coherent because weather is sparse). Recently fixed: winding order was wrong, azimuth wraparound created degenerate quads.

3. **Gate-major memory layout** - Data stored as `[gate][radial]` for coalesced GPU reads (adjacent threads in a warp read adjacent radials at the same gate distance).

4. **Split-cut aware parsing** - NEXRAD sends surveillance (REF) and Doppler (VEL/ZDR/CC) at the same elevation with different gate counts. Parser groups by gate count to separate them. `getBestSweeps()` deduplicates by elevation, keeps the sweep with most gates for the active product.

5. **Pinned memory transfers** - `cudaMallocHost` staging buffers for true async DMA (without pinned memory, `cudaMemcpyAsync` silently blocks).

6. **Pre-baked animation cache** - Historic playback caches rendered RGBA frames in VRAM. Invalidated on pan/zoom/product change.

## Known Issues & Smells (What To Look For)

### Correctness Concerns

- **Detection algorithms are naive** - TDS/hail/meso detection in `app.cpp:computeDetection()` is a first pass. The cross-product matching between surveillance (REF) and Doppler (CC/ZDR) sweeps uses approximate range-based indexing that may misalign at certain ranges.

- **Velocity dealiasing is simplistic** - `app.cpp:dealias()` does two-pass spatial consistency with a hardcoded 30 m/s Nyquist. Real dealiasing needs VAD wind profile estimation and the actual Nyquist from the volume header.

- **Parser boundary handling** - `level2_parser.cpp` concatenates BZ2 blocks then steps at 2432-byte boundaries. Works for ~99% of files but VCP transitions may lose radials. The `gpu_pipeline.cu` GPU parser exists but isn't used in the main path.

- **Historic cross-section** - Cross-section in historic mode requires pressing X after the frame loads, and the volume rebuild path is fragile (shared temp GPU slots 200+).

### Architecture Concerns

- **`app.cpp` is doing too much** - Download orchestration, GPU upload, tilt management, detection computation, dealiasing, volume building, animation caching, input handling... all in one 1156-line file. The detection and dealiasing code especially should be extracted.

- **`goto` statements in `ui.cpp`** - `goto skip_station_list` and `goto skip_station_markers` are used to hide UI sections in historic mode. Should be restructured.

- **Temp GPU slots** - Volume building uses station slots 200-230 as scratch space for multi-tilt upload. This is a hack that assumes MAX_STATIONS > 230 and that nothing else uses those slots.

- **Thread safety** - `processDownload` runs on downloader threads and accesses `m_stations[idx]` under `m_stationMutex`, but `computeDetection()` and `dealias()` are called inside that lock and do heavy work. Could block the download threads.

- **No error recovery** - GPU errors call `exit(EXIT_FAILURE)` via `CUDA_CHECK`. A single CUDA error kills the whole app.

### Performance Concerns

- **Forward render pixel fill** - Each gate thread does a nested loop filling its quad's bounding box. At high zoom, a single gate can cover hundreds of pixels. The 10000-pixel cap is a band-aid. A proper scanline rasterizer or atomic min/max approach would be better.

- **Detection runs on CPU** - `computeDetection()` scans gates with nested loops on CPU. For 162 stations this adds measurable parse-time latency. Could be a CUDA kernel.

- **`uploadStation` on tilt change re-uploads ALL stations** - `setTilt()` loops through all parsed stations and re-uploads each one. With 162 stations this is a lot of GPU traffic. The `swapStationPointers` API exists but isn't wired up to a real VRAM cache yet.

- **Spatial grid rebuilt on CPU then uploaded** - `buildSpatialGridGpu` name is misleading, it's partially CPU. Grid construction could be fully GPU-side.

### Memory & Resource Management

- **Pre-baked frames never freed** - `m_cachedFrames[]` are allocated with `cudaMalloc` but only "invalidated" by resetting `m_cachedFrameCount` to 0. The GPU memory is never freed until process exit.

- **Temp volume slots (200+) never freed** - After `toggle3D()` builds the volume, the temp station slots remain allocated.

- **Pinned memory churn** - `uploadStationData` allocates and frees pinned staging buffers every call. Should use a persistent pool.

## Build & Run

```bash
# Windows (requires CUDA Toolkit, VS2022 BuildTools, CMake, Ninja)
build.bat

# Output: build/cursdar.exe
```

## Commit History Context

This was built iteratively over ~2 days:
1. Initial parser + inverse-mapping renderer
2. Native-res rendering + AWIPS color tables
3. Historic events + cross-section + tilt filtering
4. Forward render + pinned memory + overlays (v0.2.0)
5. Detection + SRV + dealiasing + range rings (v0.2.1)
6. Bug fixes: forward render winding, historic pan, 3D historic (v0.2.2)

The pace was fast and there was no formal review pass. Code quality varies - the CUDA kernels are reasonably tight, the app orchestration layer grew organically and shows it.

## What Would Be Most Useful

1. **Correctness issues** - Anything that could produce wrong radar data or crash
2. **Race conditions** - The download threads + main thread interaction
3. **Resource leaks** - GPU memory, GL resources, handles
4. **Architectural suggestions** - How to decompose `app.cpp` and clean up the render pipeline
5. **CUDA-specific issues** - Occupancy problems, bank conflicts, unnecessary syncs
