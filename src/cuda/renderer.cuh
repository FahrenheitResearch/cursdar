#pragma once
#include "cuda_common.cuh"
#include <cuda_runtime.h>

// ── GPU-side data structures ────────────────────────────────

struct GpuStationInfo {
    float lat, lon;
    float elevation_angle;
    int   num_radials;
    int   num_gates[NUM_PRODUCTS];
    float first_gate_km[NUM_PRODUCTS];
    float gate_spacing_km[NUM_PRODUCTS];
    float scale[NUM_PRODUCTS];
    float offset[NUM_PRODUCTS];
    bool  has_product[NUM_PRODUCTS];
};

// Device pointers for one station's data
struct GpuStationPtrs {
    float*    azimuths;
    uint16_t* gates[NUM_PRODUCTS];
};

struct GpuStationBuffers {
    float*    d_azimuths = nullptr;
    uint16_t* d_gates[NUM_PRODUCTS] = {};
    bool      allocated = false;
    cudaStream_t stream = nullptr;
    int       azimuth_capacity = 0;
    size_t    gate_capacity_bytes[NUM_PRODUCTS] = {};
    float*    h_azimuths = nullptr;
    size_t    h_azimuth_capacity_bytes = 0;
    uint16_t* h_gates[NUM_PRODUCTS] = {};
    size_t    h_gate_capacity_bytes[NUM_PRODUCTS] = {};
};

struct GpuViewport {
    float center_lat, center_lon;
    float deg_per_pixel_x, deg_per_pixel_y;
    int   width, height;
};

// Spatial grid cell for fast station lookup
constexpr int SPATIAL_GRID_W = 128;
constexpr int SPATIAL_GRID_H = 64;
constexpr int MAX_STATIONS_PER_CELL = 32;

struct SpatialGrid {
    int cells[SPATIAL_GRID_H][SPATIAL_GRID_W][MAX_STATIONS_PER_CELL];
    int counts[SPATIAL_GRID_H][SPATIAL_GRID_W];
    float min_lat, max_lat, min_lon, max_lon;
};

// ── API ─────────────────────────────────────────────────────

namespace gpu {

void init();
void shutdown();

void allocateStation(int station_idx, const GpuStationInfo& info);
void freeStation(int station_idx);

void uploadStationData(int station_idx, const GpuStationInfo& info,
                       const float* azimuths,
                       const uint16_t* gate_data[NUM_PRODUCTS]);

// Native-res mosaic (all stations)
void renderNative(const GpuViewport& vp,
                  const GpuStationInfo* stations, int num_stations,
                  const SpatialGrid& grid,
                  int product, float dbz_min_threshold,
                  uint32_t* d_output);

// Single-station native-res render (inverse mapping)
void renderSingleStation(const GpuViewport& vp,
                          int station_idx,
                          int product, float dbz_min_threshold,
                          uint32_t* d_output,
                          float srv_speed = 0.0f,
                          float srv_dir = 0.0f);

// Forward render: one thread per gate, rasterizes polar sectors directly.
// Internally falls back to inverse mapping when zoom/pathology makes that
// more correct than brute-force forward writes.
// srv_speed/srv_dir: Storm-Relative Velocity params (0 = disabled)
void forwardRenderStation(const GpuViewport& vp,
                           int station_idx,
                           int product, float dbz_min_threshold,
                           uint32_t* d_output,
                           float srv_speed = 0.0f, float srv_dir = 0.0f);

void syncStation(int station_idx);

// Swap device pointers for a station (for instant tilt switching from VRAM cache)
void swapStationPointers(int station_idx, const GpuStationInfo& info,
                          float* d_azimuths, uint16_t* d_gates[NUM_PRODUCTS]);

// Get device pointers for a station's data
float*    getStationAzimuths(int station_idx);
uint16_t* getStationGates(int station_idx, int product);

// GPU spatial grid construction
void buildSpatialGridGpu(const GpuStationInfo* h_stations, int num_stations,
                          SpatialGrid* h_grid_out);

} // namespace gpu
