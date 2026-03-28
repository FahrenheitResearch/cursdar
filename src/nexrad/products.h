#pragma once
#include <cstdint>
#include <string>

enum class Product : int {
    REF = 0,  // Reflectivity (dBZ)
    VEL = 1,  // Radial Velocity (m/s)
    SW  = 2,  // Spectrum Width (m/s)
    ZDR = 3,  // Differential Reflectivity (dB)
    CC  = 4,  // Correlation Coefficient (unitless)
    KDP = 5,  // Specific Differential Phase (deg/km)
    PHI = 6,  // Differential Phase (deg)
    COUNT = 7
};

struct ProductInfo {
    const char* name;
    const char* code;     // 3-char NEXRAD code
    const char* units;
    float min_value;
    float max_value;
};

inline constexpr ProductInfo PRODUCT_INFO[] = {
    {"Reflectivity",              "REF", "dBZ",    -30.0f, 75.0f},
    {"Radial Velocity",           "VEL", "m/s",    -64.0f, 64.0f},
    {"Spectrum Width",            "SW ", "m/s",      0.0f, 30.0f},
    {"Differential Reflectivity", "ZDR", "dB",      -8.0f,  8.0f},
    {"Correlation Coefficient",   "RHO", "",         0.2f,  1.05f},
    {"Specific Diff. Phase",      "KDP", "deg/km", -10.0f, 15.0f},
    {"Differential Phase",        "PHI", "deg",      0.0f, 360.0f},
};

// Map 3-char NEXRAD code to product index, returns -1 if unknown
inline int productFromCode(const char* code) {
    for (int i = 0; i < (int)Product::COUNT; i++) {
        if (code[0] == PRODUCT_INFO[i].code[0] &&
            code[1] == PRODUCT_INFO[i].code[1] &&
            code[2] == PRODUCT_INFO[i].code[2])
            return i;
    }
    return -1;
}
