#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define KERNEL_CHECK() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA kernel error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Warp-level constants
constexpr int WARP_SIZE = 32;

// Maximum stations we support
constexpr int MAX_STATIONS = 256;

// Maximum radials per sweep (super-res = 720, normal = 360)
constexpr int MAX_RADIALS = 720;

// Maximum gates per radial
constexpr int MAX_GATES = 1840;

// Number of products
constexpr int NUM_PRODUCTS = 7;

// Product indices
constexpr int PROD_REF = 0;
constexpr int PROD_VEL = 1;
constexpr int PROD_SW  = 2;
constexpr int PROD_ZDR = 3;
constexpr int PROD_CC  = 4;
constexpr int PROD_KDP = 5;
constexpr int PROD_PHI = 6;

// Per-station render texture size (2048 for sharp zoomed-in detail)
constexpr int STATION_TEX_SIZE = 2048;

// Null gate value (below threshold / no data)
constexpr uint16_t GATE_BELOW_THRESHOLD = 0;
constexpr uint16_t GATE_RANGE_FOLDED = 1;
