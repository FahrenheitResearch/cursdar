#pragma once
#include "cuda_common.cuh"
#include "renderer.cuh"
#include <cuda_runtime.h>

// 3D volumetric storm renderer
// Builds a voxel grid from multi-tilt radar data, then ray-marches through it.

constexpr int VOL_XY = 256;
constexpr int VOL_Z  = 96;
constexpr float VOL_RANGE_KM = 230.0f;  // horizontal extent ±230km
constexpr float VOL_HEIGHT_KM = 22.0f;  // real height 0-22km (catches tall storms)
constexpr float VOL_Z_EXAGGERATION = 10.0f; // softer vertical exaggeration for a less bowl-like volume
constexpr float VOL_DISPLAY_HEIGHT = VOL_HEIGHT_KM * VOL_Z_EXAGGERATION;

struct Camera3D {
    float orbit_angle;   // horizontal orbit (degrees, 0=north)
    float tilt_angle;    // vertical tilt (degrees, 0=horizon, 90=top-down)
    float distance;      // distance from center (km)
    float target_z;      // look-at altitude (km)
};

namespace gpu {

// Build 3D voxel volume from multi-tilt station data
// All tilts must be uploaded for the active station
void buildVolume(int station_idx, int product,
                 const GpuStationInfo* sweep_infos, int num_sweeps,
                 const float* const* d_azimuths_per_sweep,
                 const uint16_t* const* d_gates_per_sweep);

// Ray-march the volume and render to output buffer
void renderVolume(const Camera3D& cam, int width, int height,
                  int product, float dbz_min,
                  uint32_t* d_output);

// Cross-section: vertical slice through atmosphere along a line
void renderCrossSection(
    int station_idx, int product, float dbz_min,
    float start_lat, float start_lon, float end_lat, float end_lon,
    float station_lat, float station_lon,
    int width, int height,
    uint32_t* d_output);

void initVolume();
void freeVolume();

} // namespace gpu
