#pragma once
#include "../cuda/cuda_common.cuh"
#include <vector>
#include <cstdint>

// Pre-computed GPU-ready sweep data (computed once at parse time)
struct PrecomputedSweep {
    float elevation_angle = 0;
    int   num_radials = 0;
    std::vector<float> azimuths;
    struct ProductData {
        bool has_data = false;
        int  num_gates = 0;
        float first_gate_km = 0;
        float gate_spacing_km = 0;
        float scale = 0, offset = 0;
        std::vector<uint16_t> gates; // gate-major: [gate][radial]
    };
    ProductData products[NUM_PRODUCTS];
};
