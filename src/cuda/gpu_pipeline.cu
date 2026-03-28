#include "gpu_pipeline.cuh"
#include "../nexrad/level2.h"
#include <cstdio>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

namespace gpu_pipeline {

// ── GPU Parser Kernel ───────────────────────────────────────
// Each thread scans from a potential message start position.
// We check every 2432-byte boundary AND variable offsets.

__device__ bool isValidMsg31(const uint8_t* data, size_t pos, size_t size) {
    if (pos + 12 + 16 + 32 > size) return false;
    // CTM at pos, MessageHeader at pos+12
    uint8_t msg_type = data[pos + 12 + 3]; // message_type byte
    if (msg_type != 31) return false;
    // Check message size is sane
    uint16_t msize = ((uint16_t)data[pos + 12] << 8) | data[pos + 12 + 1];
    if (msize < 20 || msize > 30000) return false;
    return true;
}

// Byte-swap helpers (big-endian to little-endian)
__device__ uint16_t d_bswap16(uint16_t v) {
    return (v >> 8) | (v << 8);
}
__device__ uint32_t d_bswap32(uint32_t v) {
    return ((v >> 24) & 0xFF) | ((v >> 8) & 0xFF00) |
           ((v << 8) & 0xFF0000) | ((v << 24) & 0xFF000000);
}
__device__ float d_bswapf(float v) {
    uint32_t i;
    memcpy(&i, &v, 4);
    i = d_bswap32(i);
    float r;
    memcpy(&r, &i, 4);
    return r;
}

// Product code matching on GPU
__device__ int d_productFromCode(const char* code) {
    // REF=0, VEL=1, SW=2, ZDR=3, RHO=4, KDP=5, PHI=6
    if (code[0] == 'R' && code[1] == 'E' && code[2] == 'F') return 0;
    if (code[0] == 'V' && code[1] == 'E' && code[2] == 'L') return 1;
    if (code[0] == 'S' && code[1] == 'W')                   return 2;
    if (code[0] == 'Z' && code[1] == 'D' && code[2] == 'R') return 3;
    if (code[0] == 'R' && code[1] == 'H' && code[2] == 'O') return 4;
    if (code[0] == 'K' && code[1] == 'D' && code[2] == 'P') return 5;
    if (code[0] == 'P' && code[1] == 'H' && code[2] == 'I') return 6;
    return -1;
}

__global__ void parseMsg31Kernel(
    const uint8_t* __restrict__ raw,
    size_t raw_size,
    const int* __restrict__ msg_offsets, // pre-found message positions
    int num_messages,
    GpuParsedRadial* __restrict__ radials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_messages) return;

    int pos = msg_offsets[idx];
    if (pos < 0 || (size_t)pos + 12 + 16 + 32 > raw_size) return;

    // Skip CTM(12) + MessageHeader(16) → Msg31Header starts at pos+28
    const uint8_t* msg = raw + pos + 12 + 16;
    size_t msg_remain = raw_size - (pos + 12 + 16);
    if (msg_remain < 32) return;

    GpuParsedRadial& r = radials[idx];
    memset(&r, 0, sizeof(r));
    for (int p = 0; p < NUM_PRODUCTS; p++) r.moment_offsets[p] = -1;

    // Parse Msg31 header
    // Bytes 12-15: azimuth angle (float32 BE)
    // Bytes 24-27: elevation angle (float32 BE)
    r.azimuth = d_bswapf(*(const float*)(msg + 12));
    r.elevation = d_bswapf(*(const float*)(msg + 24));
    r.radial_status = msg[21];
    r.elevation_number = msg[22];

    // Validate
    if (r.azimuth < 0.0f || r.azimuth >= 360.0f) { r.azimuth = -1; return; }
    if (r.elevation < -2.0f || r.elevation > 90.0f) { r.azimuth = -1; return; }

    // Data block count at bytes 30-31
    uint16_t block_count = d_bswap16(*(const uint16_t*)(msg + 30));
    if (block_count < 1 || block_count > 20) return;

    // Data block pointers start at byte 32
    const uint32_t* ptrs = (const uint32_t*)(msg + 32);

    for (int b = 0; b < block_count && b < 10; b++) {
        uint32_t bptr = d_bswap32(ptrs[b]);
        if (bptr == 0 || bptr >= msg_remain) continue;

        const uint8_t* block = msg + bptr;
        char btype = (char)block[0];
        char bname[3] = {(char)block[1], (char)block[2], (char)block[3]};

        if (btype == 'D') {
            int p = d_productFromCode(bname);
            if (p < 0 || p >= NUM_PRODUCTS) continue;
            if (bptr + 28 > msg_remain) continue;

            r.num_gates[p] = d_bswap16(*(const uint16_t*)(block + 8));
            r.first_gate[p] = d_bswap16(*(const uint16_t*)(block + 10));
            r.gate_spacing[p] = d_bswap16(*(const uint16_t*)(block + 12));
            r.data_word_size[p] = block[19];
            r.scale[p] = d_bswapf(*(const float*)(block + 20));
            r.offset[p] = d_bswapf(*(const float*)(block + 24));

            // Store the absolute byte offset to gate data within raw buffer
            r.moment_offsets[p] = (int)((msg + bptr + 28) - raw);

            if (r.num_gates[p] == 0 || r.num_gates[p] > 2000) {
                r.moment_offsets[p] = -1;
                r.num_gates[p] = 0;
            }
            if (r.scale[p] == 0.0f) {
                r.moment_offsets[p] = -1;
            }
        }
    }
}

// ── Message offset finder kernel ────────────────────────────
// Scans raw data for valid MSG31 positions at 2432-byte boundaries
// and variable offsets.

__global__ void findMessageOffsetsKernel(
    const uint8_t* __restrict__ raw,
    size_t raw_size,
    int* __restrict__ offsets_out,
    int* __restrict__ count_out,
    int max_messages)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread checks one potential position
    // Try 2432-byte aligned positions AND positions based on message size
    size_t pos = (size_t)tid * 2432;
    if (pos + 28 > raw_size) return;

    if (isValidMsg31(raw, pos, raw_size)) {
        int slot = atomicAdd(count_out, 1);
        if (slot < max_messages) {
            offsets_out[slot] = (int)pos;
        }
    }
}

// Second pass: find messages at variable offsets (between 2432-byte boundaries)
__global__ void findVariableOffsetsKernel(
    const uint8_t* __restrict__ raw,
    size_t raw_size,
    const int* __restrict__ aligned_offsets,
    int num_aligned,
    int* __restrict__ offsets_out,
    int* __restrict__ count_out,
    int max_messages)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_aligned) return;

    int base = aligned_offsets[idx];
    if (base < 0 || (size_t)base + 28 > raw_size) return;

    // Read message size and compute next message position
    uint16_t msize = ((uint16_t)raw[base + 12] << 8) | raw[base + 12 + 1];
    if (msize < 20 || msize > 30000) return;

    size_t next_pos = (size_t)base + (size_t)msize * 2 + 12;
    // Check if there's a valid message at the variable-offset position
    // that we wouldn't find at a 2432-byte boundary
    if (next_pos % 2432 != 0 && next_pos + 28 < raw_size) {
        if (isValidMsg31(raw, next_pos, raw_size)) {
            int slot = atomicAdd(count_out, 1);
            if (slot < max_messages) {
                offsets_out[slot] = (int)next_pos;
            }
        }
    }
}

// ── Transposition Kernel ────────────────────────────────────
// One thread per (gate, radial) pair. Reads from raw buffer at
// the offset stored in the parsed radial info, writes to
// gate-major output buffer.

__global__ void transposeKernel(
    const uint8_t* __restrict__ raw_data,
    const GpuParsedRadial* __restrict__ radials,
    const int* __restrict__ radial_indices, // sorted indices by azimuth
    int num_radials,
    int product,
    uint16_t* __restrict__ output,          // [num_gates][num_radials]
    int out_num_gates)
{
    int gate = blockIdx.x * blockDim.x + threadIdx.x;
    int radial = blockIdx.y * blockDim.y + threadIdx.y;
    if (gate >= out_num_gates || radial >= num_radials) return;

    int sorted_idx = radial_indices[radial];
    const GpuParsedRadial& r = radials[sorted_idx];

    uint16_t value = 0;
    int offset = r.moment_offsets[product];
    if (offset >= 0 && gate < r.num_gates[product]) {
        if (r.data_word_size[product] == 16) {
            // 16-bit big-endian
            const uint8_t* p = raw_data + offset + gate * 2;
            value = ((uint16_t)p[0] << 8) | p[1];
        } else {
            // 8-bit
            value = raw_data[offset + gate];
        }
    }

    // gate-major layout: output[gate * num_radials + radial]
    output[gate * num_radials + radial] = value;
}

// Extract sorted azimuths kernel
__global__ void extractAzimuthsKernel(
    const GpuParsedRadial* __restrict__ radials,
    const int* __restrict__ radial_indices,
    int num_radials,
    float* __restrict__ azimuths_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_radials) return;
    azimuths_out[i] = radials[radial_indices[i]].azimuth;
}

// ── API Implementation ──────────────────────────────────────

int parseOnGpu(const uint8_t* d_raw_data, size_t raw_size,
               GpuParsedRadial* d_radials_out, int max_radials,
               cudaStream_t stream) {
    // Allocate offset arrays
    int* d_offsets;
    int* d_count;
    CUDA_CHECK(cudaMallocAsync(&d_offsets, max_radials * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&d_count, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(int), stream));

    // Pass 1: find messages at 2432-byte boundaries
    int num_potential = (int)(raw_size / 2432) + 1;
    int threads = 256;
    int blocks = (num_potential + threads - 1) / threads;
    findMessageOffsetsKernel<<<blocks, threads, 0, stream>>>(
        d_raw_data, raw_size, d_offsets, d_count, max_radials);

    // Get count from pass 1
    int h_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_count, d_count, sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (h_count > 0) {
        // Pass 2: find variable-offset messages
        findVariableOffsetsKernel<<<(h_count + 255) / 256, 256, 0, stream>>>(
            d_raw_data, raw_size, d_offsets, h_count,
            d_offsets, d_count, max_radials);

        CUDA_CHECK(cudaMemcpyAsync(&h_count, d_count, sizeof(int),
                                    cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (h_count <= 0) {
        cudaFreeAsync(d_offsets, stream);
        cudaFreeAsync(d_count, stream);
        return 0;
    }

    h_count = (h_count > max_radials) ? max_radials : h_count;

    // Sort offsets
    thrust::device_ptr<int> d_off_ptr(d_offsets);
    thrust::sort(d_off_ptr, d_off_ptr + h_count);

    // Parse all found messages
    parseMsg31Kernel<<<(h_count + 255) / 256, 256, 0, stream>>>(
        d_raw_data, raw_size, d_offsets, h_count, d_radials_out);

    cudaFreeAsync(d_offsets, stream);
    cudaFreeAsync(d_count, stream);
    return h_count;
}

// Sort helper for radials by azimuth
struct AzimuthSortKey {
    float azimuth;
    int   index;
};

void transposeGatesGpu(
    const uint8_t* d_raw_data,
    const GpuParsedRadial* d_radials,
    int num_radials,
    int product,
    uint16_t* d_output,
    int out_num_gates,
    float* d_azimuths_out,
    cudaStream_t stream)
{
    // Build sort indices (sort radials by azimuth)
    // Copy radials to host, sort, upload indices
    std::vector<GpuParsedRadial> h_radials(num_radials);
    CUDA_CHECK(cudaMemcpy(h_radials.data(), d_radials,
                           num_radials * sizeof(GpuParsedRadial),
                           cudaMemcpyDeviceToHost));

    // Filter valid radials and sort by azimuth
    std::vector<int> indices;
    indices.reserve(num_radials);
    for (int i = 0; i < num_radials; i++) {
        if (h_radials[i].azimuth >= 0.0f && h_radials[i].azimuth < 360.0f)
            indices.push_back(i);
    }
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return h_radials[a].azimuth < h_radials[b].azimuth;
    });

    int valid_count = (int)indices.size();
    if (valid_count == 0) return;

    // Upload sorted indices
    int* d_indices;
    CUDA_CHECK(cudaMalloc(&d_indices, valid_count * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_indices, indices.data(),
                           valid_count * sizeof(int), cudaMemcpyHostToDevice));

    // Clear output
    CUDA_CHECK(cudaMemset(d_output, 0,
                           (size_t)out_num_gates * valid_count * sizeof(uint16_t)));

    // Launch transpose kernel
    dim3 block(32, 8);
    dim3 grid((out_num_gates + 31) / 32, (valid_count + 7) / 8);
    transposeKernel<<<grid, block, 0, stream>>>(
        d_raw_data, d_radials, d_indices, valid_count,
        product, d_output, out_num_gates);

    // Extract sorted azimuths
    extractAzimuthsKernel<<<(valid_count + 255) / 256, 256, 0, stream>>>(
        d_radials, d_indices, valid_count, d_azimuths_out);

    cudaFree(d_indices);
}

// ── Full ingest pipeline ────────────────────────────────────

GpuIngestResult ingestSweepGpu(const uint8_t* h_raw_data, size_t raw_size,
                                cudaStream_t stream) {
    GpuIngestResult result = {};

    // Upload raw data to GPU
    uint8_t* d_raw;
    CUDA_CHECK(cudaMallocAsync(&d_raw, raw_size, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_raw, h_raw_data, raw_size,
                                cudaMemcpyHostToDevice, stream));

    // Parse on GPU
    constexpr int MAX_PARSED = 8192;
    GpuParsedRadial* d_radials;
    CUDA_CHECK(cudaMallocAsync(&d_radials, MAX_PARSED * sizeof(GpuParsedRadial), stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    int num_parsed = parseOnGpu(d_raw, raw_size, d_radials, MAX_PARSED, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (num_parsed <= 0) {
        cudaFreeAsync(d_raw, stream);
        cudaFreeAsync(d_radials, stream);
        return result;
    }

    // Read back parsed radials to determine sweep structure and gate params
    std::vector<GpuParsedRadial> h_radials(num_parsed);
    CUDA_CHECK(cudaMemcpy(h_radials.data(), d_radials,
                           num_parsed * sizeof(GpuParsedRadial),
                           cudaMemcpyDeviceToHost));

    // Find the lowest elevation sweep's radials
    // Group by elevation_number, pick the lowest
    float lowest_elev = 999.0f;
    int lowest_elev_num = -1;
    for (auto& r : h_radials) {
        if (r.azimuth < 0) continue;
        if (r.elevation < lowest_elev) {
            lowest_elev = r.elevation;
            lowest_elev_num = r.elevation_number;
        }
    }

    // Count radials in lowest sweep
    int sweep_count = 0;
    for (auto& r : h_radials) {
        if (r.azimuth >= 0 && r.elevation_number == lowest_elev_num)
            sweep_count++;
    }

    result.elevation_angle = lowest_elev;
    result.num_radials = sweep_count;

    // Determine gate params from first valid radial in lowest sweep
    for (auto& r : h_radials) {
        if (r.azimuth < 0 || r.elevation_number != lowest_elev_num) continue;
        for (int p = 0; p < NUM_PRODUCTS; p++) {
            if (r.moment_offsets[p] >= 0 && r.num_gates[p] > 0) {
                result.has_product[p] = true;
                result.num_gates[p] = r.num_gates[p];
                result.first_gate_km[p] = r.first_gate[p] / 1000.0f;
                result.gate_spacing_km[p] = r.gate_spacing[p] / 1000.0f;
                result.scale[p] = r.scale[p];
                result.offset[p] = r.offset[p];
            }
        }
        break;
    }

    // Allocate output buffers
    CUDA_CHECK(cudaMalloc(&result.d_azimuths, sweep_count * sizeof(float)));

    for (int p = 0; p < NUM_PRODUCTS; p++) {
        if (result.has_product[p]) {
            size_t sz = (size_t)result.num_gates[p] * sweep_count * sizeof(uint16_t);
            CUDA_CHECK(cudaMalloc(&result.d_gates[p], sz));

            // GPU transpose
            transposeGatesGpu(d_raw, d_radials, num_parsed, p,
                              result.d_gates[p], result.num_gates[p],
                              result.d_azimuths, stream);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Cleanup temp buffers
    cudaFreeAsync(d_raw, stream);
    cudaFreeAsync(d_radials, stream);

    return result;
}

void freeIngestResult(GpuIngestResult& result) {
    if (result.d_azimuths) { cudaFree(result.d_azimuths); result.d_azimuths = nullptr; }
    for (int p = 0; p < NUM_PRODUCTS; p++) {
        if (result.d_gates[p]) { cudaFree(result.d_gates[p]); result.d_gates[p] = nullptr; }
    }
}

} // namespace gpu_pipeline
