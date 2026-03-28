# cursdar - Development Context

## Project Overview
Real-time NEXRAD radar viewer with custom CUDA kernels. Everything that can run on GPU does run on GPU.

## Build
Run `build.bat` from the project root. Requires CUDA Toolkit, VS2022 BuildTools, CMake, Ninja.

## Architecture Decisions
- **Native-res rendering**: No intermediate textures. Each viewport pixel directly samples raw gate data. Resolution is always the viewport resolution regardless of zoom.
- **Single-station mode** as default: Mouse-follow selects nearest station. All station data pre-loaded on GPU so switching is a pointer swap (zero latency).
- **Split-cut aware parser**: NEXRAD surveillance (REF) and Doppler (VEL) scans at the same elevation are separated into distinct sweeps by grouping on gate count. Eliminates ring artifacts.
- **Gate-major data layout**: Gate data transposed to `[gate][radial]` order for coalesced GPU memory access. The render kernel reads adjacent gates for adjacent radials in a warp.
- **Pre-computed sweep data**: All sweep transposition done at parse time (on download thread), cached in `PrecomputedSweep`. Tilt switching just does GPU memcpy.
- **3D volume**: Built from all tilts using elevation interpolation between sweeps. Stored as CUDA 3D texture for hardware trilinear filtering. Ray marcher uses shadow rays and gradient-based surface normals.

## Data Source
Unidata NEXRAD S3 bucket: `unidata-nexrad-level2.s3.amazonaws.com`
Public, no auth. Files are BZ2-compressed Level 2 Archive II format.

## Key Files
- `src/cuda/renderer.cu` - Main render kernels (single-station, mosaic, spatial grid)
- `src/cuda/volume3d.cu` - 3D volume build, ray march, cross-section
- `src/nexrad/level2_parser.cpp` - BZ2 decompression + MSG31 parsing
- `src/app.cpp` - Download orchestration, GPU upload, tilt/product management

## Known Issues
- Parser: concatenates all BZ2 blocks then steps through at 2432-byte boundaries. Gets ~700 radials/sweep which is correct for super-res, but some edge cases in VCP transitions may lose a few radials.
- 3D mode requires pressing V to build volume, which uploads all tilts to temp GPU slots. Could be optimized to keep persistent multi-tilt data.
- Cross-section line dragging: left-drag moves the whole line, right-drag repositions endpoint. Could add individual endpoint handles.
