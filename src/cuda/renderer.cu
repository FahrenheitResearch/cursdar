#include "renderer.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Static state ────────────────────────────────────────────

static GpuStationInfo    s_stationInfo[MAX_STATIONS];
static GpuStationBuffers s_stationBufs[MAX_STATIONS];
static int               s_numStations = 0;

// Color tables: constant memory array + texture objects for hw interpolation
__constant__ uint32_t c_colorTable[NUM_PRODUCTS][256];

// CUDA texture objects for hardware-interpolated color lookups (1D, float4)
static cudaTextureObject_t s_colorTextures[NUM_PRODUCTS] = {};
static cudaArray_t         s_colorArrays[NUM_PRODUCTS] = {};
static bool                s_colorTexturesCreated = false;

// Spatial grid in device memory
static SpatialGrid* d_spatialGrid = nullptr;

// Persistent device buffers (avoid per-frame malloc)
static GpuStationInfo*  d_stationInfoBuf = nullptr;
static GpuStationPtrs*  d_stationPtrsBuf = nullptr;
static int              d_bufSize = 0;

// Persistent buffers for grid construction
static GpuStationInfo* d_gridStationsBuf = nullptr;
static uint8_t*        d_gridActiveBuf = nullptr;
static SpatialGrid*    d_gridBuildBuf = nullptr;
static int             d_gridBuildCapacity = 0;

// Deterministic forward-render accumulation buffer
static uint64_t* d_forwardAccumBuf = nullptr;
static size_t    d_forwardAccumCapacity = 0;

// Host-side pointer tracking
static GpuStationPtrs h_stationPtrs[MAX_STATIONS] = {};

static void initializeSpatialGrid(SpatialGrid* grid) {
    if (!grid) return;

    memset(grid, 0, sizeof(SpatialGrid));
    grid->min_lat = 15.0f;
    grid->max_lat = 72.0f;
    grid->min_lon = -180.0f;
    grid->max_lon = -60.0f;
    for (int gy = 0; gy < SPATIAL_GRID_H; gy++)
        for (int gx = 0; gx < SPATIAL_GRID_W; gx++)
            for (int s = 0; s < MAX_STATIONS_PER_CELL; s++)
                grid->cells[gy][gx][s] = -1;
}

// ── Color table generation ──────────────────────────────────

__device__ __host__ static uint32_t makeRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
    return (uint32_t)r | ((uint32_t)g << 8) | ((uint32_t)b << 16) | ((uint32_t)a << 24);
}

constexpr uint32_t kBackgroundColor = 0xFF140F0Fu;
constexpr uint64_t kEmptyForwardPixel = ~0ull;

__device__ __host__ static float angleDiffDeg(float a, float b) {
    float d = fabsf(a - b);
    return (d > 180.0f) ? (360.0f - d) : d;
}

__device__ __host__ static float positiveAngleDeltaDeg(float from, float to) {
    float d = to - from;
    if (d < 0.0f) d += 360.0f;
    return d;
}

__device__ __host__ static float wrapAngleDeg(float angle) {
    while (angle < 0.0f) angle += 360.0f;
    while (angle >= 360.0f) angle -= 360.0f;
    return angle;
}

__device__ __host__ static float productThreshold(int product, float dbz_min) {
    if (product == PROD_VEL || product == PROD_ZDR || product == PROD_KDP || product == PROD_PHI)
        return -999.0f;
    if (product == PROD_CC) return 0.3f;
    if (product == PROD_SW) return 0.5f;
    return dbz_min;
}

__device__ __host__ static bool passesThreshold(int product, float value, float threshold) {
    if (value <= -998.0f) return false;
    if (product == PROD_VEL)
        return fabsf(value) >= fmaxf(threshold, 0.0f);
    return value >= productThreshold(product, threshold);
}

__device__ __host__ static void productColorRange(int product, float& min_val, float& max_val) {
    switch (product) {
        case PROD_REF: min_val = -30.0f; max_val = 75.0f; break;
        case PROD_VEL: min_val = -64.0f; max_val = 64.0f; break;
        case PROD_SW:  min_val = 0.0f;   max_val = 30.0f; break;
        case PROD_ZDR: min_val = -8.0f;  max_val = 8.0f; break;
        case PROD_CC:  min_val = 0.2f;   max_val = 1.05f; break;
        case PROD_KDP: min_val = -10.0f; max_val = 15.0f; break;
        default:       min_val = 0.0f;   max_val = 360.0f; break;
    }
}

__device__ __host__ static float normalizedColorCoord(float value, int product) {
    float min_val, max_val;
    productColorRange(product, min_val, max_val);
    float norm = (value - min_val) / (max_val - min_val);
    norm = fminf(fmaxf(norm, 0.0f), 1.0f);
    return (norm * 254.0f + 1.0f) / 256.0f;
}

__device__ static uint64_t forwardDepthKey(float range_km, uint32_t rgba) {
    uint32_t depth_m = (uint32_t)fminf(fmaxf(range_km * 1000.0f, 0.0f), 4294967294.0f);
    return (uint64_t(depth_m) << 32) | uint64_t(rgba);
}

__device__ static void atomicMin64(uint64_t* addr, uint64_t value) {
    unsigned long long* ptr = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *ptr;
    while (value < old) {
        unsigned long long assumed = old;
        old = atomicCAS(ptr, assumed, value);
        if (old == assumed) break;
    }
}

// Map a physical value to a color table index [0..255]
static int valToIdx(float val, float min_val, float max_val) {
    int idx = (int)((val - min_val) / (max_val - min_val) * 255.0f);
    return (idx < 0) ? 0 : (idx > 255) ? 255 : idx;
}

// Fill a range of indices with one color (stepped, no gradient)
static void fillRange(uint32_t* table, float v0, float v1, float vmin, float vmax,
                      uint8_t r, uint8_t g, uint8_t b) {
    int i0 = valToIdx(v0, vmin, vmax);
    int i1 = valToIdx(v1, vmin, vmax);
    for (int i = i0; i < i1 && i < 256; i++)
        table[i] = makeRGBA(r, g, b);
}

// ── AWIPS Standard Reflectivity (exact NWS RGB values) ──────
static void generateRefColorTable(uint32_t* table) {
    memset(table, 0, 256 * sizeof(uint32_t));
    const float mn = -30, mx = 75;
    fillRange(table,  5, 10, mn, mx,   0, 131, 174); // teal
    fillRange(table, 10, 15, mn, mx,  65,  90, 160); // slate blue
    fillRange(table, 15, 20, mn, mx,  62, 169, 214); // sky blue
    fillRange(table, 20, 25, mn, mx,   0, 220, 183); // cyan-green
    fillRange(table, 25, 30, mn, mx,  15, 195,  21); // bright green
    fillRange(table, 30, 35, mn, mx,  11, 147,  22); // medium green
    fillRange(table, 35, 40, mn, mx,  10,  95,  19); // dark green
    fillRange(table, 40, 45, mn, mx, 255, 245,   5); // yellow
    fillRange(table, 45, 50, mn, mx, 255, 190,   0); // orange
    fillRange(table, 50, 55, mn, mx, 255,   0,   0); // red
    fillRange(table, 55, 60, mn, mx, 120,   0,   0); // dark red
    fillRange(table, 60, 65, mn, mx, 255, 255, 255); // white
    fillRange(table, 65, 70, mn, mx, 201, 161, 255); // lavender
    fillRange(table, 70, 75, mn, mx, 174,   0, 255); // purple
    fillRange(table, 75, 76, mn, mx,   5, 221, 225); // bright cyan
}

static void interpolateColor(uint32_t* table, int i0, int i1,
                              uint8_t r0, uint8_t g0, uint8_t b0,
                              uint8_t r1, uint8_t g1, uint8_t b1) {
    if (i1 <= i0) return;
    for (int i = i0; i <= i1; i++) {
        float t = (float)(i - i0) / (float)(i1 - i0);
        table[i] = makeRGBA((uint8_t)(r0 + t * (r1 - r0)),
                             (uint8_t)(g0 + t * (g1 - g0)),
                             (uint8_t)(b0 + t * (b1 - b0)));
    }
}

// ── AWIPS Enhanced Base Velocity (exact NWS values) ─────────
static void generateVelColorTable(uint32_t* table) {
    memset(table, 0, 256 * sizeof(uint32_t));
    const float mn = -64, mx = 64;
    // Approaching (negative = green/blue)
    fillRange(table, -64, -50, mn, mx,   0,   0, 100); // dark blue
    fillRange(table, -50, -40, mn, mx, 100, 255, 255); // cyan
    fillRange(table, -40, -30, mn, mx,   0, 255,   0); // bright green
    fillRange(table, -30, -20, mn, mx,   0, 209,   0); // green
    fillRange(table, -20, -10, mn, mx,   0, 163,   0); // med green
    fillRange(table, -10,  -5, mn, mx,   0, 116,   0); // dark green
    fillRange(table,  -5,   0, mn, mx,   0,  70,   0); // very dark green
    // Near zero
    fillRange(table,   0,   5, mn, mx, 120, 120, 120); // gray
    // Receding (positive = red/orange)
    fillRange(table,   5,  10, mn, mx,  70,   0,   0); // very dark red
    fillRange(table,  10,  20, mn, mx, 116,   0,   0); // dark red
    fillRange(table,  20,  30, mn, mx, 209,   0,   0); // red
    fillRange(table,  30,  40, mn, mx, 255,   0,   0); // bright red
    fillRange(table,  40,  50, mn, mx, 255, 129, 125); // pink
    fillRange(table,  50,  60, mn, mx, 255, 140,  70); // orange
    fillRange(table,  60,  64, mn, mx, 255, 255,   0); // yellow
}

static void uploadColorTables() {
    uint32_t tables[NUM_PRODUCTS][256];
    generateRefColorTable(tables[PROD_REF]);
    generateVelColorTable(tables[PROD_VEL]);

    // ── AWIPS Spectrum Width ────────────────────────────────
    memset(tables[PROD_SW], 0, 256*4);
    {
        const float mn = 0, mx = 30;
        fillRange(tables[PROD_SW],  0,  3, mn, mx,  45,  45,  45);
        fillRange(tables[PROD_SW],  3,  5, mn, mx, 117, 117, 117);
        fillRange(tables[PROD_SW],  5,  7, mn, mx, 200, 200, 200);
        fillRange(tables[PROD_SW],  7,  9, mn, mx, 255, 230,   0);
        fillRange(tables[PROD_SW],  9, 12, mn, mx, 255, 195,   0);
        fillRange(tables[PROD_SW], 12, 15, mn, mx, 255, 110,   0);
        fillRange(tables[PROD_SW], 15, 18, mn, mx, 255,  10,   0);
        fillRange(tables[PROD_SW], 18, 22, mn, mx, 255,   5, 100);
        fillRange(tables[PROD_SW], 22, 26, mn, mx, 255,   0, 200);
        fillRange(tables[PROD_SW], 26, 30, mn, mx, 255, 159, 234);
    }

    // ── AWIPS Differential Reflectivity (ZDR) ───────────────
    memset(tables[PROD_ZDR], 0, 256*4);
    {
        const float mn = -8, mx = 8;
        fillRange(tables[PROD_ZDR], -8, -3, mn, mx,  55,  55,  55);
        fillRange(tables[PROD_ZDR], -3, -1, mn, mx, 138, 138, 138);
        fillRange(tables[PROD_ZDR], -1,  0, mn, mx, 148, 132, 177);
        fillRange(tables[PROD_ZDR],  0,  0.5f, mn, mx,  29,  89, 174);
        fillRange(tables[PROD_ZDR],  0.5f, 1, mn, mx,  49, 169, 193);
        fillRange(tables[PROD_ZDR],  1, 1.5f, mn, mx,  68, 248, 212);
        fillRange(tables[PROD_ZDR],  1.5f, 2, mn, mx,  90, 221,  98);
        fillRange(tables[PROD_ZDR],  2, 2.5f, mn, mx, 255, 255, 100);
        fillRange(tables[PROD_ZDR],  2.5f, 3, mn, mx, 238, 133,  53);
        fillRange(tables[PROD_ZDR],  3, 4, mn, mx, 220,  10,   5);
        fillRange(tables[PROD_ZDR],  4, 5, mn, mx, 208,  60,  90);
        fillRange(tables[PROD_ZDR],  5, 6, mn, mx, 240, 120, 180);
        fillRange(tables[PROD_ZDR],  6, 7, mn, mx, 255, 255, 255);
        fillRange(tables[PROD_ZDR],  7, 8, mn, mx, 200, 150, 203);
    }

    // ── AWIPS Correlation Coefficient (CC/RhoHV) ────────────
    memset(tables[PROD_CC], 0, 256*4);
    {
        const float mn = 0.2f, mx = 1.05f;
        fillRange(tables[PROD_CC], 0.20f, 0.45f, mn, mx,  20,   0,  50);
        fillRange(tables[PROD_CC], 0.45f, 0.60f, mn, mx,   0,   0, 110);
        fillRange(tables[PROD_CC], 0.60f, 0.70f, mn, mx,   0,   0, 150);
        fillRange(tables[PROD_CC], 0.70f, 0.75f, mn, mx,   0,   0, 170);
        fillRange(tables[PROD_CC], 0.75f, 0.80f, mn, mx,   0,   0, 255);
        fillRange(tables[PROD_CC], 0.80f, 0.85f, mn, mx, 125, 125, 255);
        fillRange(tables[PROD_CC], 0.85f, 0.90f, mn, mx,  85, 255,  85);
        fillRange(tables[PROD_CC], 0.90f, 0.92f, mn, mx, 255, 255,   0);
        fillRange(tables[PROD_CC], 0.92f, 0.95f, mn, mx, 255, 110,   0);
        fillRange(tables[PROD_CC], 0.95f, 0.97f, mn, mx, 255,  55,   0);
        fillRange(tables[PROD_CC], 0.97f, 1.00f, mn, mx, 255,   0,   0);
        fillRange(tables[PROD_CC], 1.00f, 1.05f, mn, mx, 145,   0, 135);
    }

    // ── AWIPS Specific Differential Phase (KDP) ─────────────
    memset(tables[PROD_KDP], 0, 256*4);
    {
        const float mn = -10, mx = 15;
        fillRange(tables[PROD_KDP], -10, -1, mn, mx, 101, 101, 101);
        fillRange(tables[PROD_KDP],  -1,  0, mn, mx, 166,  10,  50);
        fillRange(tables[PROD_KDP],   0,  1, mn, mx, 228, 105, 161);
        fillRange(tables[PROD_KDP],   1,  2, mn, mx, 166, 125, 185);
        fillRange(tables[PROD_KDP],   2,  3, mn, mx,  90, 255, 255);
        fillRange(tables[PROD_KDP],   3,  4, mn, mx,  20, 246,  20);
        fillRange(tables[PROD_KDP],   4,  5, mn, mx, 255, 251,   3);
        fillRange(tables[PROD_KDP],   5,  6, mn, mx, 255, 129,  21);
        fillRange(tables[PROD_KDP],   6,  8, mn, mx, 255, 162,  75);
        fillRange(tables[PROD_KDP],   8, 15, mn, mx, 145,  37, 125);
    }

    // ── Differential Phase (PHI) ────────────────────────────
    memset(tables[PROD_PHI], 0, 256*4);
    interpolateColor(tables[PROD_PHI], 1, 64, 0,0,200, 0,200,255);
    interpolateColor(tables[PROD_PHI], 64, 128, 0,200,255, 0,255,0);
    interpolateColor(tables[PROD_PHI], 128, 192, 0,255,0, 255,255,0);
    interpolateColor(tables[PROD_PHI], 192, 255, 255,255,0, 255,0,0);

    CUDA_CHECK(cudaMemcpyToSymbol(c_colorTable, tables, sizeof(tables)));

    // Create CUDA texture objects for hardware-interpolated color lookups
    for (int p = 0; p < NUM_PRODUCTS; p++) {
        // Convert RGBA uint32 to float4 for texture
        float4 texData[256];
        for (int i = 0; i < 256; i++) {
            uint32_t c = tables[p][i];
            texData[i] = make_float4(
                (float)(c & 0xFF) / 255.0f,
                (float)((c >> 8) & 0xFF) / 255.0f,
                (float)((c >> 16) & 0xFF) / 255.0f,
                (float)((c >> 24) & 0xFF) / 255.0f);
        }

        // Create CUDA array
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
        CUDA_CHECK(cudaMallocArray(&s_colorArrays[p], &desc, 256));
        CUDA_CHECK(cudaMemcpy2DToArray(s_colorArrays[p], 0, 0, texData,
                                        256 * sizeof(float4), 256 * sizeof(float4), 1,
                                        cudaMemcpyHostToDevice));

        // Create texture object
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = s_colorArrays[p];

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;  // HW interpolation!
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;               // 0.0-1.0 addressing

        CUDA_CHECK(cudaCreateTextureObject(&s_colorTextures[p], &resDesc, &texDesc, nullptr));
    }
    s_colorTexturesCreated = true;
}

// ── Native-res kernel ───────────────────────────────────────
// One pass: viewport pixel → lat/lon → for each nearby station:
//   az/range → binary search radials → interpolate gates → color → composite
// No intermediate textures. Resolution = viewport resolution at any zoom.

__device__ int bsearchAz(const float* az, int n, float target) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (az[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ float sampleStation(
    const GpuStationInfo& info, const GpuStationPtrs& ptrs,
    float az, float range_km, int product, float dbz_min)
{
    if (!info.has_product[product] || !ptrs.gates[product]) return -999.0f;

    int ng = info.num_gates[product];
    int nr = info.num_radials;
    float fgkm = info.first_gate_km[product];
    float gskm = info.gate_spacing_km[product];
    if (ng <= 0 || nr <= 0 || gskm <= 0.0f) return -999.0f;

    float max_range = fgkm + ng * gskm;
    if (range_km < fgkm || range_km > max_range) return -999.0f;

    // Nearest radial with fallback (fills beam width)
    int idx_hi = bsearchAz(ptrs.azimuths, nr, az);
    int idx_lo = (idx_hi == 0) ? nr - 1 : idx_hi - 1;
    if (idx_hi >= nr) idx_hi = 0;

    float d_lo = angleDiffDeg(az, ptrs.azimuths[idx_lo]);
    float d_hi = angleDiffDeg(az, ptrs.azimuths[idx_hi]);

    int gi = (int)((range_km - fgkm) / gskm);
    if (gi < 0 || gi >= ng) return -999.0f;

    const uint16_t* gd = ptrs.gates[product];
    int ri_first = (d_lo <= d_hi) ? idx_lo : idx_hi;
    int ri_second = (d_lo <= d_hi) ? idx_hi : idx_lo;
    uint16_t raw = __ldg(&gd[gi * nr + ri_first]);
    if (raw <= 1) raw = __ldg(&gd[gi * nr + ri_second]);
    if (raw <= 1) return -999.0f;

    float sc = info.scale[product], off = info.offset[product];
    float value = ((float)raw - off) / sc;

    if (!passesThreshold(product, value, dbz_min)) return -999.0f;

    return value;
}

__global__ void nativeRenderKernel(
    const GpuViewport vp,
    const GpuStationInfo* __restrict__ stations,
    const GpuStationPtrs* __restrict__ ptrs,
    int num_stations,
    const SpatialGrid* __restrict__ grid,
    int product,
    float dbz_min,
    cudaTextureObject_t colorTex,  // HW-interpolated color texture
    uint32_t* __restrict__ output)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= vp.width || py >= vp.height) return;

    float lon = vp.center_lon + (px - vp.width * 0.5f) * vp.deg_per_pixel_x;
    float lat = vp.center_lat - (py - vp.height * 0.5f) * vp.deg_per_pixel_y;

    // Background
    uint32_t result = makeRGBA(15, 15, 20, 255);

    // Spatial grid lookup
    float gfx = (lon - grid->min_lon) / (grid->max_lon - grid->min_lon) * SPATIAL_GRID_W;
    float gfy = (lat - grid->min_lat) / (grid->max_lat - grid->min_lat) * SPATIAL_GRID_H;
    int gx = (int)gfx, gy = (int)gfy;

    if (gx < 0 || gx >= SPATIAL_GRID_W || gy < 0 || gy >= SPATIAL_GRID_H) {
        output[py * vp.width + px] = result;
        return;
    }

    int count = grid->counts[gy][gx];
    if (count > MAX_STATIONS_PER_CELL) count = MAX_STATIONS_PER_CELL;
    float best_value = -999.0f;
    float best_range = 1e9f;

    // Check each station in this cell
    for (int ci = 0; ci < count; ci++) {
        int si = grid->cells[gy][gx][ci];
        if (si < 0 || si >= num_stations) continue;

        const auto& info = stations[si];

        // Distance in km (flat earth approx, good for <500km)
        float dlat_km = (lat - info.lat) * 111.0f;
        float dlon_km = (lon - info.lon) * 111.0f * cosf(info.lat * (float)M_PI / 180.0f);
        float range_km = sqrtf(dlat_km * dlat_km + dlon_km * dlon_km);

        if (range_km > 460.0f) continue;

        float az = atan2f(dlon_km, dlat_km) * (180.0f / (float)M_PI);
        if (az < 0.0f) az += 360.0f;

        float val = sampleStation(info, ptrs[si], az, range_km, product, dbz_min);
        if (val > -998.0f && range_km < best_range) {
            best_value = val;
            best_range = range_km;
        }
    }

    if (best_value <= -998.0f) {
        output[py * vp.width + px] = result;
        return;
    }

    // Map value to color
    // Hardware-interpolated texture lookup (float4 RGBA, normalized coords)
    // Offset to skip index 0 (transparent) and use indices 1-255
    float tex_coord = normalizedColorCoord(best_value, product);
    float4 tc = tex1D<float4>(colorTex, tex_coord);

    if (tc.w < 0.01f) {
        output[py * vp.width + px] = result;
        return;
    }

    // Blend over background using texture alpha
    uint8_t br = result & 0xFF, bg = (result >> 8) & 0xFF, bb = (result >> 16) & 0xFF;
    result = makeRGBA(
        (uint8_t)(br * (1-tc.w) + tc.x * 255.0f * tc.w),
        (uint8_t)(bg * (1-tc.w) + tc.y * 255.0f * tc.w),
        (uint8_t)(bb * (1-tc.w) + tc.z * 255.0f * tc.w), 255);

    output[py * vp.width + px] = result;
}

// ── API ─────────────────────────────────────────────────────

static void ensureStationDeviceCapacity(GpuStationBuffers& buf, const GpuStationInfo& info) {
    if (!buf.allocated) {
        CUDA_CHECK(cudaStreamCreate(&buf.stream));
        buf.allocated = true;
    }

    if (info.num_radials > buf.azimuth_capacity) {
        if (buf.d_azimuths) CUDA_CHECK(cudaFree(buf.d_azimuths));
        CUDA_CHECK(cudaMalloc(&buf.d_azimuths, size_t(info.num_radials) * sizeof(float)));
        buf.azimuth_capacity = info.num_radials;
    }

    for (int p = 0; p < NUM_PRODUCTS; p++) {
        size_t needed = 0;
        if (info.has_product[p] && info.num_gates[p] > 0 && info.num_radials > 0)
            needed = size_t(info.num_gates[p]) * size_t(info.num_radials) * sizeof(uint16_t);
        if (needed > buf.gate_capacity_bytes[p]) {
            if (buf.d_gates[p]) CUDA_CHECK(cudaFree(buf.d_gates[p]));
            CUDA_CHECK(cudaMalloc(&buf.d_gates[p], needed));
            buf.gate_capacity_bytes[p] = needed;
        }
    }
}

static void ensureStationPinnedCapacity(GpuStationBuffers& buf, size_t azimuth_bytes,
                                        const GpuStationInfo& info) {
    if (azimuth_bytes > buf.h_azimuth_capacity_bytes) {
        if (buf.h_azimuths) CUDA_CHECK(cudaFreeHost(buf.h_azimuths));
        CUDA_CHECK(cudaMallocHost(&buf.h_azimuths, azimuth_bytes));
        buf.h_azimuth_capacity_bytes = azimuth_bytes;
    }

    for (int p = 0; p < NUM_PRODUCTS; p++) {
        size_t needed = 0;
        if (info.has_product[p] && info.num_gates[p] > 0 && info.num_radials > 0)
            needed = size_t(info.num_gates[p]) * size_t(info.num_radials) * sizeof(uint16_t);
        if (needed > buf.h_gate_capacity_bytes[p]) {
            if (buf.h_gates[p]) CUDA_CHECK(cudaFreeHost(buf.h_gates[p]));
            CUDA_CHECK(cudaMallocHost(&buf.h_gates[p], needed));
            buf.h_gate_capacity_bytes[p] = needed;
        }
    }
}

static void ensureGridBuildCapacity(int num_stations) {
    if (num_stations <= d_gridBuildCapacity) return;

    if (d_gridStationsBuf) CUDA_CHECK(cudaFree(d_gridStationsBuf));
    if (d_gridActiveBuf) CUDA_CHECK(cudaFree(d_gridActiveBuf));

    CUDA_CHECK(cudaMalloc(&d_gridStationsBuf, size_t(num_stations) * sizeof(GpuStationInfo)));
    CUDA_CHECK(cudaMalloc(&d_gridActiveBuf, size_t(num_stations) * sizeof(uint8_t)));
    d_gridBuildCapacity = num_stations;
}

static void ensureForwardAccumCapacity(size_t pixel_count) {
    if (pixel_count <= d_forwardAccumCapacity) return;
    if (d_forwardAccumBuf) CUDA_CHECK(cudaFree(d_forwardAccumBuf));
    CUDA_CHECK(cudaMalloc(&d_forwardAccumBuf, pixel_count * sizeof(uint64_t)));
    d_forwardAccumCapacity = pixel_count;
}

static bool shouldUseInverseFallback(const GpuViewport& vp,
                                     const GpuStationInfo& info,
                                     int product) {
    if (product < 0 || product >= NUM_PRODUCTS || info.num_radials <= 0)
        return true;

    float gskm = info.gate_spacing_km[product];
    if (gskm <= 0.0f) return true;

    float cos_lat = cosf(info.lat * (float)M_PI / 180.0f);
    cos_lat = fmaxf(cos_lat, 0.1f);

    float km_per_px_x = fabsf(vp.deg_per_pixel_x) * 111.0f * cos_lat;
    float km_per_px_y = fabsf(vp.deg_per_pixel_y) * 111.0f;
    if (km_per_px_x <= 0.0f || km_per_px_y <= 0.0f) return false;

    float px_per_km = 1.0f / fminf(km_per_px_x, km_per_px_y);
    float gate_depth_px = gskm * px_per_km;
    float nominal_span_rad = (2.0f * (float)M_PI) / fmaxf((float)info.num_radials, 1.0f);
    float sample_range_km = fmaxf(info.first_gate_km[product] + 8.0f * gskm, 20.0f);
    float beam_width_px = sample_range_km * nominal_span_rad * px_per_km;

    return gate_depth_px > 48.0f ||
           beam_width_px > 48.0f ||
           (gate_depth_px * fmaxf(beam_width_px, 1.0f)) > 2048.0f;
}

namespace gpu {

void init() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("CUDA Device: %s (SM %d.%d, %d SMs, %.1f GB)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    uploadColorTables();

    CUDA_CHECK(cudaMalloc(&d_spatialGrid, sizeof(SpatialGrid)));
    memset(h_stationPtrs, 0, sizeof(h_stationPtrs));
    memset(s_stationInfo, 0, sizeof(s_stationInfo));
    memset(s_stationBufs, 0, sizeof(s_stationBufs));
    s_numStations = 0;
    printf("GPU renderer initialized (native-res mode).\n");
}

void shutdown() {
    for (int i = 0; i < MAX_STATIONS; i++) freeStation(i);
    if (d_spatialGrid) { cudaFree(d_spatialGrid); d_spatialGrid = nullptr; }
    if (d_stationInfoBuf) { cudaFree(d_stationInfoBuf); d_stationInfoBuf = nullptr; }
    if (d_stationPtrsBuf) { cudaFree(d_stationPtrsBuf); d_stationPtrsBuf = nullptr; }
    if (d_gridStationsBuf) { cudaFree(d_gridStationsBuf); d_gridStationsBuf = nullptr; }
    if (d_gridActiveBuf) { cudaFree(d_gridActiveBuf); d_gridActiveBuf = nullptr; }
    if (d_gridBuildBuf) { cudaFree(d_gridBuildBuf); d_gridBuildBuf = nullptr; }
    if (d_forwardAccumBuf) { cudaFree(d_forwardAccumBuf); d_forwardAccumBuf = nullptr; }
    d_bufSize = 0;
    d_gridBuildCapacity = 0;
    d_forwardAccumCapacity = 0;
    // Destroy color texture objects
    if (s_colorTexturesCreated) {
        for (int p = 0; p < NUM_PRODUCTS; p++) {
            if (s_colorTextures[p]) cudaDestroyTextureObject(s_colorTextures[p]);
            if (s_colorArrays[p]) cudaFreeArray(s_colorArrays[p]);
        }
        s_colorTexturesCreated = false;
    }
}

void allocateStation(int idx, const GpuStationInfo& info) {
    if (idx < 0 || idx >= MAX_STATIONS) return;
    auto& buf = s_stationBufs[idx];
    s_stationInfo[idx] = info;
    ensureStationDeviceCapacity(buf, info);

    // Track device pointers
    h_stationPtrs[idx].azimuths = buf.d_azimuths;
    for (int p = 0; p < NUM_PRODUCTS; p++)
        h_stationPtrs[idx].gates[p] = buf.d_gates[p];

    if (idx >= s_numStations) s_numStations = idx + 1;
}

void freeStation(int idx) {
    if (idx < 0 || idx >= MAX_STATIONS) return;
    auto& buf = s_stationBufs[idx];
    if (!buf.allocated) return;

    CUDA_CHECK(cudaStreamSynchronize(buf.stream));
    CUDA_CHECK(cudaStreamDestroy(buf.stream));
    if (buf.d_azimuths) CUDA_CHECK(cudaFree(buf.d_azimuths));
    for (int p = 0; p < NUM_PRODUCTS; p++)
        if (buf.d_gates[p]) CUDA_CHECK(cudaFree(buf.d_gates[p]));
    if (buf.h_azimuths) CUDA_CHECK(cudaFreeHost(buf.h_azimuths));
    for (int p = 0; p < NUM_PRODUCTS; p++)
        if (buf.h_gates[p]) CUDA_CHECK(cudaFreeHost(buf.h_gates[p]));

    memset(&h_stationPtrs[idx], 0, sizeof(GpuStationPtrs));
    memset(&buf, 0, sizeof(buf));
    memset(&s_stationInfo[idx], 0, sizeof(GpuStationInfo));
}

void uploadStationData(int idx, const GpuStationInfo& info,
                       const float* azimuths,
                       const uint16_t* gate_data[NUM_PRODUCTS]) {
    if (idx < 0 || idx >= MAX_STATIONS) return;
    auto& buf = s_stationBufs[idx];
    if (!buf.allocated) return;

    s_stationInfo[idx] = info;
    ensureStationDeviceCapacity(buf, info);

    size_t az_size = info.num_radials * sizeof(float);
    ensureStationPinnedCapacity(buf, az_size, info);

    if (az_size > 0) {
        memcpy(buf.h_azimuths, azimuths, az_size);
        CUDA_CHECK(cudaMemcpyAsync(buf.d_azimuths, buf.h_azimuths, az_size,
                                    cudaMemcpyHostToDevice, buf.stream));
    }

    for (int p = 0; p < NUM_PRODUCTS; p++) {
        if (info.has_product[p] && gate_data[p] && buf.d_gates[p]) {
            size_t sz = (size_t)info.num_gates[p] * info.num_radials * sizeof(uint16_t);
            memcpy(buf.h_gates[p], gate_data[p], sz);
            CUDA_CHECK(cudaMemcpyAsync(buf.d_gates[p], buf.h_gates[p], sz,
                                        cudaMemcpyHostToDevice, buf.stream));
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(buf.stream));

    h_stationPtrs[idx].azimuths = buf.d_azimuths;
    for (int p = 0; p < NUM_PRODUCTS; p++)
        h_stationPtrs[idx].gates[p] = buf.d_gates[p];
}

void renderNative(const GpuViewport& vp,
                  const GpuStationInfo* stations, int num_stations,
                  const SpatialGrid& grid,
                  int product, float dbz_min,
                  uint32_t* d_output) {
    // Upload spatial grid
    CUDA_CHECK(cudaMemcpy(d_spatialGrid, &grid, sizeof(SpatialGrid), cudaMemcpyHostToDevice));

    // Resize persistent buffers if needed
    if (num_stations > d_bufSize) {
        if (d_stationInfoBuf) cudaFree(d_stationInfoBuf);
        if (d_stationPtrsBuf) cudaFree(d_stationPtrsBuf);
        CUDA_CHECK(cudaMalloc(&d_stationInfoBuf, num_stations * sizeof(GpuStationInfo)));
        CUDA_CHECK(cudaMalloc(&d_stationPtrsBuf, num_stations * sizeof(GpuStationPtrs)));
        d_bufSize = num_stations;
    }
    CUDA_CHECK(cudaMemcpy(d_stationInfoBuf, stations,
                           num_stations * sizeof(GpuStationInfo), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_stationPtrsBuf, h_stationPtrs,
                           num_stations * sizeof(GpuStationPtrs), cudaMemcpyHostToDevice));

    dim3 block(32, 8);
    dim3 grid_dim((vp.width + 31) / 32, (vp.height + 7) / 8);

    nativeRenderKernel<<<grid_dim, block>>>(
        vp, d_stationInfoBuf, d_stationPtrsBuf, num_stations,
        d_spatialGrid, product, dbz_min,
        s_colorTextures[product],  // HW-interpolated color texture
        d_output);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "Native render kernel error: %s\n", cudaGetErrorString(err));
}

// ── Single-station kernel (FAST) ─────────────────────────────
// No spatial grid, no station loop. One station, direct sampling.
// This is the hot path for mouse-follow mode.

__global__ void singleStationKernel(
    const GpuViewport vp,
    const GpuStationInfo info,
    const float* __restrict__ azimuths,
    const uint16_t* __restrict__ gates, // for active product
    int product,
    float dbz_min,
    cudaTextureObject_t colorTex,
    float srv_speed,
    float srv_dir_rad,
    uint32_t* __restrict__ output)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= vp.width || py >= vp.height) return;

    float lon = vp.center_lon + (px - vp.width * 0.5f) * vp.deg_per_pixel_x;
    float lat = vp.center_lat - (py - vp.height * 0.5f) * vp.deg_per_pixel_y;

    uint32_t bg = kBackgroundColor;

    if (!gates) { output[py * vp.width + px] = bg; return; }

    int ng = info.num_gates[product];
    int nr = info.num_radials;
    float fgkm = info.first_gate_km[product];
    float gskm = info.gate_spacing_km[product];
    if (ng <= 0 || nr <= 0 || gskm <= 0.0f) { output[py * vp.width + px] = bg; return; }

    // Distance from station
    float dlat_km = (lat - info.lat) * 111.0f;
    float dlon_km = (lon - info.lon) * 111.0f * cosf(info.lat * (float)M_PI / 180.0f);
    float range_km = sqrtf(dlat_km * dlat_km + dlon_km * dlon_km);

    float max_range = fgkm + ng * gskm;
    if (range_km < fgkm || range_km > max_range) {
        output[py * vp.width + px] = bg;
        return;
    }

    // Azimuth
    float az = atan2f(dlon_km, dlat_km) * (180.0f / (float)M_PI);
    if (az < 0.0f) az += 360.0f;

    // Binary search azimuths (shared memory for speed)
    extern __shared__ float s_az[];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;
    for (int i = tid; i < nr; i += block_size)
        s_az[i] = azimuths[i];
    __syncthreads();

    // Nearest radial with fallback to adjacent (fills beam width, no gaps)
    int idx_hi = bsearchAz(s_az, nr, az);
    int idx_lo = (idx_hi == 0) ? nr - 1 : idx_hi - 1;
    if (idx_hi >= nr) idx_hi = 0;

    float d_lo = angleDiffDeg(az, s_az[idx_lo]);
    float d_hi = angleDiffDeg(az, s_az[idx_hi]);

    // Nearest gate
    int gi = (int)((range_km - fgkm) / gskm);
    if (gi < 0 || gi >= ng) { output[py * vp.width + px] = bg; return; }

    // Try nearest radial first, then fallback to the other adjacent one
    int ri_first = (d_lo <= d_hi) ? idx_lo : idx_hi;
    int ri_second = (d_lo <= d_hi) ? idx_hi : idx_lo;

    uint16_t raw = __ldg(&gates[gi * nr + ri_first]);
    if (raw <= 1) raw = __ldg(&gates[gi * nr + ri_second]); // fallback
    if (raw <= 1) { output[py * vp.width + px] = bg; return; }

    float sc = info.scale[product], off = info.offset[product];
    float value = ((float)raw - off) / sc;

    if (srv_speed > 0.0f && product == PROD_VEL) {
        float az_rad = s_az[ri_first] * ((float)M_PI / 180.0f);
        value -= srv_speed * cosf(az_rad - srv_dir_rad);
    }

    if (!passesThreshold(product, value, dbz_min)) { output[py * vp.width + px] = bg; return; }

    // Color via HW texture
    float tc_coord = normalizedColorCoord(value, product);
    float4 tc = tex1D<float4>(colorTex, tc_coord);

    if (tc.w < 0.01f) { output[py * vp.width + px] = bg; return; }

    uint8_t br = bg & 0xFF, bgg = (bg >> 8) & 0xFF, bb = (bg >> 16) & 0xFF;
    output[py * vp.width + px] = makeRGBA(
        (uint8_t)(br*(1-tc.w) + tc.x*255*tc.w),
        (uint8_t)(bgg*(1-tc.w) + tc.y*255*tc.w),
        (uint8_t)(bb*(1-tc.w) + tc.z*255*tc.w), 255);
}

__global__ void clearKernel(uint32_t* output, int width, int height, uint32_t color) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px < width && py < height)
        output[py * width + px] = color;
}

static void clearOutputBuffer(const GpuViewport& vp, uint32_t* d_output) {
    dim3 block(32, 8);
    dim3 grid((vp.width + 31) / 32, (vp.height + 7) / 8);
    clearKernel<<<grid, block>>>(d_output, vp.width, vp.height, kBackgroundColor);
}

static void renderSingleStationInternal(const GpuViewport& vp, int station_idx,
                                        int product, float dbz_min,
                                        uint32_t* d_output,
                                        float srv_speed, float srv_dir_deg) {
    if (station_idx < 0 || station_idx >= MAX_STATIONS) {
        clearOutputBuffer(vp, d_output);
        return;
    }
    auto& buf = s_stationBufs[station_idx];
    auto& info = s_stationInfo[station_idx];
    if (!buf.allocated || !info.has_product[product] || !buf.d_gates[product]) {
        clearOutputBuffer(vp, d_output);
        return;
    }

    dim3 block(16, 16);
    dim3 grid((vp.width + 15) / 16, (vp.height + 15) / 16);
    size_t shared = info.num_radials * sizeof(float);
    // Cap shared memory (48KB typical max)
    if (shared > 48000) shared = 48000;

    singleStationKernel<<<grid, block, shared>>>(
        vp, info, buf.d_azimuths, buf.d_gates[product],
        product, dbz_min, s_colorTextures[product],
        srv_speed, srv_dir_deg * (float)M_PI / 180.0f,
        d_output);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "Single station render error: %s\n", cudaGetErrorString(err));
}

void renderSingleStation(const GpuViewport& vp, int station_idx,
                          int product, float dbz_min, uint32_t* d_output,
                          float srv_speed, float srv_dir) {
    renderSingleStationInternal(vp, station_idx, product, dbz_min, d_output, srv_speed, srv_dir);
}

// ── Forward Render Kernel ────────────────────────────────────
// One thread per gate cell. Computes 4 screen-space corners of the polar
// quad, fills all pixels inside with the gate's color. Empty gates skip
// immediately. Crisp per-gate rendering by construction.

__device__ float2 polarToScreen(float range_km, float az_rad,
                                 float slat, float slon,
                                 const GpuViewport& vp) {
    float cos_lat = cosf(slat * (float)M_PI / 180.0f);
    float east_km = range_km * sinf(az_rad);
    float north_km = range_km * cosf(az_rad);
    float lon_off = east_km / (111.0f * cos_lat);
    float lat_off = north_km / 111.0f;
    float px = ((slon + lon_off) - vp.center_lon) / vp.deg_per_pixel_x + vp.width * 0.5f;
    float py = (vp.center_lat - (slat + lat_off)) / vp.deg_per_pixel_y + vp.height * 0.5f;
    return make_float2(px, py);
}

__device__ float radialBoundaryStartDeg(const float* azimuths, int nr, int ri) {
    float curr = __ldg(&azimuths[ri]);
    float prev = __ldg(&azimuths[(ri + nr - 1) % nr]);
    float nominal = 360.0f / fmaxf((float)nr, 1.0f);
    float min_half = fmaxf(0.25f * nominal, 0.05f);
    float max_half = fmaxf(2.0f * nominal, 1.0f);
    float half_gap = 0.5f * positiveAngleDeltaDeg(prev, curr);
    half_gap = fminf(fmaxf(half_gap, min_half), max_half);
    return wrapAngleDeg(curr - half_gap);
}

__device__ float radialBoundaryEndDeg(const float* azimuths, int nr, int ri) {
    float curr = __ldg(&azimuths[ri]);
    float next = __ldg(&azimuths[(ri + 1) % nr]);
    float nominal = 360.0f / fmaxf((float)nr, 1.0f);
    float min_half = fmaxf(0.25f * nominal, 0.05f);
    float max_half = fmaxf(2.0f * nominal, 1.0f);
    float half_gap = 0.5f * positiveAngleDeltaDeg(curr, next);
    half_gap = fminf(fmaxf(half_gap, min_half), max_half);
    return wrapAngleDeg(curr + half_gap);
}

__device__ bool pointInConvexQuad(const float2 corners[4], float px, float py) {
    bool saw_pos = false;
    bool saw_neg = false;
    for (int e = 0; e < 4; e++) {
        float2 a = corners[e];
        float2 b = corners[(e + 1) & 3];
        float cross = (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x);
        saw_pos |= (cross > 0.01f);
        saw_neg |= (cross < -0.01f);
        if (saw_pos && saw_neg) return false;
    }
    return true;
}

__global__ void forwardRenderKernel(
    const float* __restrict__ azimuths,
    const uint16_t* __restrict__ gates,
    GpuStationInfo info,
    GpuViewport vp,
    int product, float dbz_min,
    cudaTextureObject_t colorTex,
    float srv_speed, float srv_dir_rad, // SRV: storm motion (0 = disabled)
    uint64_t* __restrict__ accum)
{
    int ri = blockIdx.x * blockDim.x + threadIdx.x;
    int gi = blockIdx.y * blockDim.y + threadIdx.y;

    int nr = info.num_radials;
    int ng = info.num_gates[product];
    if (ri >= nr || gi >= ng) return;

    // Early exit: empty gate (60-80% of gates skip here)
    uint16_t raw = __ldg(&gates[gi * nr + ri]);
    if (raw <= 1) return;

    float sc = info.scale[product], off = info.offset[product];
    float value = ((float)raw - off) / sc;

    // SRV: subtract storm motion component from velocity
    if (srv_speed > 0.0f && product == PROD_VEL) {
        float az_rad = __ldg(&azimuths[ri]) * ((float)M_PI / 180.0f);
        value -= srv_speed * cosf(az_rad - srv_dir_rad);
    }

    if (!passesThreshold(product, value, dbz_min)) return;

    // Color lookup
    float tc = normalizedColorCoord(value, product);
    float4 col = tex1D<float4>(colorTex, tc);
    if (col.w < 0.01f) return;
    uint32_t rgba = makeRGBA((uint8_t)(col.x*255), (uint8_t)(col.y*255),
                              (uint8_t)(col.z*255), 255);

    float az0 = radialBoundaryStartDeg(azimuths, nr, ri) * ((float)M_PI / 180.0f);
    float az1 = radialBoundaryEndDeg(azimuths, nr, ri) * ((float)M_PI / 180.0f);
    float gskm = info.gate_spacing_km[product];
    float r0 = info.first_gate_km[product] + gi * gskm;
    float r1 = r0 + gskm;

    float2 c0 = polarToScreen(r0, az0, info.lat, info.lon, vp);
    float2 c1 = polarToScreen(r1, az0, info.lat, info.lon, vp);
    float2 c2 = polarToScreen(r1, az1, info.lat, info.lon, vp);
    float2 c3 = polarToScreen(r0, az1, info.lat, info.lon, vp);

    // Bounding box (clipped to viewport)
    int ix0 = max(0, (int)floorf(fminf(fminf(c0.x, c1.x), fminf(c2.x, c3.x))));
    int ix1 = min(vp.width - 1, (int)ceilf(fmaxf(fmaxf(c0.x, c1.x), fmaxf(c2.x, c3.x))));
    int iy0 = max(0, (int)floorf(fminf(fminf(c0.y, c1.y), fminf(c2.y, c3.y))));
    int iy1 = min(vp.height - 1, (int)ceilf(fmaxf(fmaxf(c0.y, c1.y), fmaxf(c2.y, c3.y))));

    if (ix0 > ix1 || iy0 > iy1) return;

    float2 corners[4] = {c0, c1, c2, c3};
    uint64_t candidate = forwardDepthKey(r0 + 0.5f * gskm, rgba);

    for (int py = iy0; py <= iy1; py++) {
        for (int px = ix0; px <= ix1; px++) {
            float fx = (float)px + 0.5f;
            float fy = (float)py + 0.5f;
            if (pointInConvexQuad(corners, fx, fy))
                atomicMin64(&accum[py * vp.width + px], candidate);
        }
    }
}

__global__ void forwardResolveKernel(const uint64_t* __restrict__ accum,
                                     int width, int height,
                                     uint32_t* __restrict__ output) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    uint64_t packed = accum[py * width + px];
    output[py * width + px] = (packed == kEmptyForwardPixel)
        ? kBackgroundColor
        : uint32_t(packed & 0xFFFFFFFFu);
}

void forwardRenderStation(const GpuViewport& vp, int station_idx,
                           int product, float dbz_min, uint32_t* d_output,
                           float srv_speed, float srv_dir) {
    if (station_idx < 0 || station_idx >= MAX_STATIONS) {
        clearOutputBuffer(vp, d_output);
        return;
    }
    auto& buf = s_stationBufs[station_idx];
    auto& info = s_stationInfo[station_idx];
    if (!buf.allocated || !info.has_product[product] || !buf.d_gates[product]) {
        clearOutputBuffer(vp, d_output);
        return;
    }

    if (shouldUseInverseFallback(vp, info, product)) {
        renderSingleStationInternal(vp, station_idx, product, dbz_min,
                                    d_output, srv_speed, srv_dir);
        return;
    }

    size_t pixel_count = size_t(vp.width) * size_t(vp.height);
    ensureForwardAccumCapacity(pixel_count);
    CUDA_CHECK(cudaMemsetAsync(d_forwardAccumBuf, 0xFF, pixel_count * sizeof(uint64_t)));

    // Forward render: one thread per (radial, gate)
    dim3 block(32, 8); // 256 threads, warp-aligned
    dim3 grid((info.num_radials + 31) / 32,
              (info.num_gates[product] + 7) / 8);

    float srv_dir_rad = srv_dir * (float)M_PI / 180.0f;
    forwardRenderKernel<<<grid, block>>>(
        buf.d_azimuths, buf.d_gates[product],
        info, vp, product, dbz_min,
        s_colorTextures[product],
        srv_speed, srv_dir_rad,
        d_forwardAccumBuf);

    dim3 resolveBlock(32, 8);
    dim3 resolveGrid((vp.width + 31) / 32, (vp.height + 7) / 8);
    forwardResolveKernel<<<resolveGrid, resolveBlock>>>(d_forwardAccumBuf,
                                                        vp.width, vp.height,
                                                        d_output);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "Forward render error: %s\n", cudaGetErrorString(err));
}

void syncStation(int idx) {
    if (idx >= 0 && idx < MAX_STATIONS && s_stationBufs[idx].allocated)
        CUDA_CHECK(cudaStreamSynchronize(s_stationBufs[idx].stream));
}

// Instant tilt switch: swap device pointers without re-uploading data
void swapStationPointers(int idx, const GpuStationInfo& info,
                          float* d_az, uint16_t* d_g[NUM_PRODUCTS]) {
    if (idx < 0 || idx >= MAX_STATIONS) return;
    s_stationInfo[idx] = info;
    if (s_stationBufs[idx].allocated) {
        // Don't free old pointers - they're in the VRAM cache
        s_stationBufs[idx].d_azimuths = d_az;
        for (int p = 0; p < NUM_PRODUCTS; p++)
            s_stationBufs[idx].d_gates[p] = d_g[p];
    }
    h_stationPtrs[idx].azimuths = d_az;
    for (int p = 0; p < NUM_PRODUCTS; p++)
        h_stationPtrs[idx].gates[p] = d_g[p];
}

float* getStationAzimuths(int idx) {
    if (idx >= 0 && idx < MAX_STATIONS && s_stationBufs[idx].allocated)
        return s_stationBufs[idx].d_azimuths;
    return nullptr;
}

uint16_t* getStationGates(int idx, int product) {
    if (idx >= 0 && idx < MAX_STATIONS && s_stationBufs[idx].allocated)
        return s_stationBufs[idx].d_gates[product];
    return nullptr;
}

// ── GPU Spatial Grid Construction ───────────────────────────
// One thread per station. Each station atomically inserts itself
// into all grid cells it covers.

__global__ void buildGridKernel(
    const GpuStationInfo* __restrict__ stations,
    const uint8_t* __restrict__ active,   // which stations have data
    int num_stations,
    SpatialGrid* __restrict__ grid)
{
    int si = blockIdx.x * blockDim.x + threadIdx.x;
    if (si >= num_stations || !active[si]) return;

    float slat = stations[si].lat, slon = stations[si].lon;
    float lat_range = grid->max_lat - grid->min_lat;
    float lon_range = grid->max_lon - grid->min_lon;
    float max_range_deg = 460.0f / 111.0f;

    int gx_min = (int)((slon - max_range_deg - grid->min_lon) / lon_range * SPATIAL_GRID_W);
    int gx_max = (int)((slon + max_range_deg - grid->min_lon) / lon_range * SPATIAL_GRID_W);
    int gy_min = (int)((slat - max_range_deg - grid->min_lat) / lat_range * SPATIAL_GRID_H);
    int gy_max = (int)((slat + max_range_deg - grid->min_lat) / lat_range * SPATIAL_GRID_H);

    gx_min = max(0, gx_min); gx_max = min(SPATIAL_GRID_W - 1, gx_max);
    gy_min = max(0, gy_min); gy_max = min(SPATIAL_GRID_H - 1, gy_max);

    for (int gy = gy_min; gy <= gy_max; gy++) {
        for (int gx = gx_min; gx <= gx_max; gx++) {
            int* count = &grid->counts[gy][gx];
            int slot = atomicAdd(count, 0);
            while (slot < MAX_STATIONS_PER_CELL) {
                int old = atomicCAS(count, slot, slot + 1);
                if (old == slot) {
                    grid->cells[gy][gx][slot] = si;
                    break;
                }
                slot = old;
            }
        }
    }
}

void buildSpatialGridGpu(const GpuStationInfo* h_stations, int num_stations,
                          SpatialGrid* h_grid_out) {
    if (!h_grid_out) return;
    if (num_stations <= 0 || !h_stations) {
        initializeSpatialGrid(h_grid_out);
        return;
    }

    ensureGridBuildCapacity(num_stations);
    if (!d_gridBuildBuf)
        CUDA_CHECK(cudaMalloc(&d_gridBuildBuf, sizeof(SpatialGrid)));

    CUDA_CHECK(cudaMemcpy(d_gridStationsBuf, h_stations,
                           num_stations * sizeof(GpuStationInfo), cudaMemcpyHostToDevice));

    // Build active flags (station has data if num_radials > 0)
    std::vector<uint8_t> h_active(num_stations);
    for (int i = 0; i < num_stations; i++)
        h_active[i] = (h_stations[i].num_radials > 0) ? 1 : 0;
    CUDA_CHECK(cudaMemcpy(d_gridActiveBuf, h_active.data(), num_stations * sizeof(uint8_t),
                           cudaMemcpyHostToDevice));

    auto init_grid = std::make_unique<SpatialGrid>();
    initializeSpatialGrid(init_grid.get());
    CUDA_CHECK(cudaMemcpy(d_gridBuildBuf, init_grid.get(), sizeof(SpatialGrid), cudaMemcpyHostToDevice));

    buildGridKernel<<<(num_stations + 255) / 256, 256>>>(
        d_gridStationsBuf, d_gridActiveBuf, num_stations, d_gridBuildBuf);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_grid_out, d_gridBuildBuf, sizeof(SpatialGrid), cudaMemcpyDeviceToHost));
}

} // namespace gpu
