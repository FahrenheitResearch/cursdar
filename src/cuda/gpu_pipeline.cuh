#pragma once
#include "cuda_common.cuh"
#include <cuda_runtime.h>

// ── GPU-accelerated data pipeline ───────────────────────────
// Kernels for parsing, transposing, and processing radar data
// entirely on the GPU after raw bytes are uploaded.

namespace gpu_pipeline {

// ── 1. GPU Level 2 Parser ───────────────────────────────────
// Uploads raw decompressed bytes to GPU, parses MSG31 headers,
// extracts radial metadata and gate data in parallel.

// Parsed radial info (output of GPU parser)
struct GpuParsedRadial {
    float    azimuth;
    float    elevation;
    uint8_t  radial_status;
    uint8_t  elevation_number;
    int      moment_offsets[NUM_PRODUCTS]; // byte offset to gate data, -1 if absent
    int      num_gates[NUM_PRODUCTS];
    int      gate_spacing[NUM_PRODUCTS];   // meters
    int      first_gate[NUM_PRODUCTS];     // meters
    float    scale[NUM_PRODUCTS];
    float    offset[NUM_PRODUCTS];
    uint8_t  data_word_size[NUM_PRODUCTS]; // 8 or 16
};

// Parse raw decompressed Level 2 data on GPU.
// Input: raw bytes uploaded to d_raw_data (size raw_size)
// Output: array of GpuParsedRadial on device
// Returns: number of radials found
int parseOnGpu(const uint8_t* d_raw_data, size_t raw_size,
               GpuParsedRadial* d_radials_out, int max_radials,
               cudaStream_t stream = 0);

// ── 2. GPU Transposition Kernel ─────────────────────────────
// Transpose gate data from radial-major to gate-major layout
// for coalesced GPU memory access in the render kernel.

// Transpose: input[radial][gate] → output[gate][radial]
// Handles both 8-bit and 16-bit gate data
void transposeGatesGpu(
    const uint8_t* d_raw_data,        // raw file bytes on GPU
    const GpuParsedRadial* d_radials, // parsed radial info
    int num_radials,
    int product,                       // which product to transpose
    uint16_t* d_output,               // output: gate-major [num_gates][num_radials]
    int out_num_gates,                // output gate count (uniform)
    float* d_azimuths_out,            // output: sorted azimuth array
    cudaStream_t stream = 0);

// ── 3. Full GPU ingest pipeline ─────────────────────────────
// Upload raw bytes → parse → transpose → ready for rendering.
// All on GPU, returns device pointers ready for the render kernel.

struct GpuIngestResult {
    float*    d_azimuths;
    uint16_t* d_gates[NUM_PRODUCTS];
    int       num_radials;
    int       num_gates[NUM_PRODUCTS];
    float     first_gate_km[NUM_PRODUCTS];
    float     gate_spacing_km[NUM_PRODUCTS];
    float     scale[NUM_PRODUCTS];
    float     offset[NUM_PRODUCTS];
    bool      has_product[NUM_PRODUCTS];
    float     elevation_angle;
    float     station_lat, station_lon;
};

// Run full GPU ingest for one sweep's worth of raw data
GpuIngestResult ingestSweepGpu(
    const uint8_t* h_raw_data, size_t raw_size,
    cudaStream_t stream = 0);

void freeIngestResult(GpuIngestResult& result);

} // namespace gpu_pipeline
