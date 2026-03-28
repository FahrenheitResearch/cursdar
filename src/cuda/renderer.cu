#include "renderer.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>
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

// Host-side pointer tracking
static GpuStationPtrs h_stationPtrs[MAX_STATIONS] = {};

// ── Color table generation ──────────────────────────────────

__device__ __host__ static uint32_t makeRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
    return (uint32_t)r | ((uint32_t)g << 8) | ((uint32_t)b << 16) | ((uint32_t)a << 24);
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

    float d_lo = fabsf(az - ptrs.azimuths[idx_lo]);
    float d_hi = fabsf(az - ptrs.azimuths[idx_hi]);
    if (d_lo > 180.0f) d_lo = 360.0f - d_lo;
    if (d_hi > 180.0f) d_hi = 360.0f - d_hi;

    int gi = (int)((range_km - fgkm) / gskm);
    if (gi < 0 || gi >= ng) return -999.0f;

    const uint16_t* gd = ptrs.gates[product];
    int ri_first = (d_lo <= d_hi) ? idx_lo : idx_hi;
    int ri_second = (d_lo <= d_hi) ? idx_hi : idx_lo;
    uint16_t raw = gd[gi * nr + ri_first];
    if (raw <= 1) raw = gd[gi * nr + ri_second];
    if (raw <= 1) return -999.0f;

    float sc = info.scale[product], off = info.offset[product];
    float value = ((float)raw - off) / sc;

    if (value <= -998.0f) return -999.0f;

    // Per-product threshold
    float threshold = dbz_min;
    if (product == PROD_VEL || product == PROD_ZDR || product == PROD_KDP || product == PROD_PHI)
        threshold = -999.0f;
    else if (product == PROD_CC) threshold = 0.3f;
    else if (product == PROD_SW) threshold = 0.5f;

    if (value < threshold) return -999.0f;

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
    float min_val, max_val;
    switch (product) {
        case PROD_REF: min_val = -30.0f; max_val = 75.0f; break;
        case PROD_VEL: min_val = -64.0f; max_val = 64.0f; break;
        case PROD_SW:  min_val = 0.0f;   max_val = 30.0f; break;
        case PROD_ZDR: min_val = -8.0f;  max_val = 8.0f;  break;
        case PROD_CC:  min_val = 0.2f;   max_val = 1.05f;  break;
        case PROD_KDP: min_val = -10.0f; max_val = 15.0f; break;
        default:       min_val = 0.0f;   max_val = 360.0f; break;
    }

    float norm = (best_value - min_val) / (max_val - min_val);
    norm = fminf(fmaxf(norm, 0.0f), 1.0f);

    // Hardware-interpolated texture lookup (float4 RGBA, normalized coords)
    // Offset to skip index 0 (transparent) and use indices 1-255
    float tex_coord = (norm * 254.0f + 1.0f) / 256.0f;
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
    s_numStations = 0;
    printf("GPU renderer initialized (native-res mode).\n");
}

void shutdown() {
    for (int i = 0; i < MAX_STATIONS; i++) freeStation(i);
    if (d_spatialGrid) { cudaFree(d_spatialGrid); d_spatialGrid = nullptr; }
    if (d_stationInfoBuf) { cudaFree(d_stationInfoBuf); d_stationInfoBuf = nullptr; }
    if (d_stationPtrsBuf) { cudaFree(d_stationPtrsBuf); d_stationPtrsBuf = nullptr; }
    d_bufSize = 0;
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
    if (buf.allocated) freeStation(idx);

    s_stationInfo[idx] = info;
    CUDA_CHECK(cudaStreamCreate(&buf.stream));
    CUDA_CHECK(cudaMalloc(&buf.d_azimuths, info.num_radials * sizeof(float)));

    for (int p = 0; p < NUM_PRODUCTS; p++) {
        if (info.has_product[p] && info.num_gates[p] > 0) {
            size_t sz = (size_t)info.num_gates[p] * info.num_radials * sizeof(uint16_t);
            CUDA_CHECK(cudaMalloc(&buf.d_gates[p], sz));
        } else {
            buf.d_gates[p] = nullptr;
        }
    }

    buf.allocated = true;

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

    cudaStreamSynchronize(buf.stream);
    cudaStreamDestroy(buf.stream);
    cudaFree(buf.d_azimuths);
    for (int p = 0; p < NUM_PRODUCTS; p++)
        if (buf.d_gates[p]) cudaFree(buf.d_gates[p]);

    memset(&h_stationPtrs[idx], 0, sizeof(GpuStationPtrs));
    memset(&buf, 0, sizeof(buf));
}

void uploadStationData(int idx, const GpuStationInfo& info,
                       const float* azimuths,
                       const uint16_t* gate_data[NUM_PRODUCTS]) {
    if (idx < 0 || idx >= MAX_STATIONS) return;
    auto& buf = s_stationBufs[idx];
    if (!buf.allocated) return;

    s_stationInfo[idx] = info;

    CUDA_CHECK(cudaMemcpyAsync(buf.d_azimuths, azimuths,
                                info.num_radials * sizeof(float),
                                cudaMemcpyHostToDevice, buf.stream));

    for (int p = 0; p < NUM_PRODUCTS; p++) {
        if (info.has_product[p] && gate_data[p] && buf.d_gates[p]) {
            size_t sz = (size_t)info.num_gates[p] * info.num_radials * sizeof(uint16_t);
            CUDA_CHECK(cudaMemcpyAsync(buf.d_gates[p], gate_data[p], sz,
                                        cudaMemcpyHostToDevice, buf.stream));
        }
    }

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
    uint32_t* __restrict__ output)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= vp.width || py >= vp.height) return;

    float lon = vp.center_lon + (px - vp.width * 0.5f) * vp.deg_per_pixel_x;
    float lat = vp.center_lat - (py - vp.height * 0.5f) * vp.deg_per_pixel_y;

    uint32_t bg = makeRGBA(15, 15, 20, 255);

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

    float d_lo = fabsf(az - s_az[idx_lo]);
    float d_hi = fabsf(az - s_az[idx_hi]);
    if (d_lo > 180.0f) d_lo = 360.0f - d_lo;
    if (d_hi > 180.0f) d_hi = 360.0f - d_hi;

    // Nearest gate
    int gi = (int)((range_km - fgkm) / gskm);
    if (gi < 0 || gi >= ng) { output[py * vp.width + px] = bg; return; }

    // Try nearest radial first, then fallback to the other adjacent one
    int ri_first = (d_lo <= d_hi) ? idx_lo : idx_hi;
    int ri_second = (d_lo <= d_hi) ? idx_hi : idx_lo;

    uint16_t raw = gates[gi * nr + ri_first];
    if (raw <= 1) raw = gates[gi * nr + ri_second]; // fallback
    if (raw <= 1) { output[py * vp.width + px] = bg; return; }

    float sc = info.scale[product], off = info.offset[product];
    float value = ((float)raw - off) / sc;

    if (value <= -998.0f) { output[py * vp.width + px] = bg; return; }

    // Threshold
    float threshold = dbz_min;
    if (product == PROD_VEL || product == PROD_ZDR || product == PROD_KDP || product == PROD_PHI)
        threshold = -999.0f;
    else if (product == PROD_CC) threshold = 0.3f;
    else if (product == PROD_SW) threshold = 0.5f;
    if (value < threshold) { output[py * vp.width + px] = bg; return; }

    // Color via HW texture
    float min_val, max_val;
    switch (product) {
        case PROD_REF: min_val=-30;max_val=75; break;
        case PROD_VEL: min_val=-64;max_val=64; break;
        case PROD_SW:  min_val=0;max_val=30; break;
        case PROD_ZDR: min_val=-8;max_val=8; break;
        case PROD_CC:  min_val=0.2f;max_val=1.05f; break;
        case PROD_KDP: min_val=-10;max_val=15; break;
        default:       min_val=0;max_val=360; break;
    }
    float norm = fminf(fmaxf((value - min_val) / (max_val - min_val), 0.0f), 1.0f);
    float tc_coord = (norm * 254.0f + 1.0f) / 256.0f;
    float4 tc = tex1D<float4>(colorTex, tc_coord);

    if (tc.w < 0.01f) { output[py * vp.width + px] = bg; return; }

    uint8_t br = bg & 0xFF, bgg = (bg >> 8) & 0xFF, bb = (bg >> 16) & 0xFF;
    output[py * vp.width + px] = makeRGBA(
        (uint8_t)(br*(1-tc.w) + tc.x*255*tc.w),
        (uint8_t)(bgg*(1-tc.w) + tc.y*255*tc.w),
        (uint8_t)(bb*(1-tc.w) + tc.z*255*tc.w), 255);
}

void renderSingleStation(const GpuViewport& vp, int station_idx,
                          int product, float dbz_min, uint32_t* d_output) {
    if (station_idx < 0 || station_idx >= MAX_STATIONS) return;
    auto& buf = s_stationBufs[station_idx];
    auto& info = s_stationInfo[station_idx];
    if (!buf.allocated || !info.has_product[product] || !buf.d_gates[product]) return;

    dim3 block(16, 16);
    dim3 grid((vp.width + 15) / 16, (vp.height + 15) / 16);
    size_t shared = info.num_radials * sizeof(float);
    // Cap shared memory (48KB typical max)
    if (shared > 48000) shared = 48000;

    singleStationKernel<<<grid, block, shared>>>(
        vp, info, buf.d_azimuths, buf.d_gates[product],
        product, dbz_min, s_colorTextures[product], d_output);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "Single station render error: %s\n", cudaGetErrorString(err));
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

__global__ void forwardRenderKernel(
    const float* __restrict__ azimuths,
    const uint16_t* __restrict__ gates,
    GpuStationInfo info,
    GpuViewport vp,
    int product, float dbz_min,
    cudaTextureObject_t colorTex,
    uint32_t* __restrict__ output)
{
    int ri = blockIdx.x * blockDim.x + threadIdx.x;
    int gi = blockIdx.y * blockDim.y + threadIdx.y;

    int nr = info.num_radials;
    int ng = info.num_gates[product];
    if (ri >= nr || gi >= ng) return;

    // Early exit: empty gate (60-80% of gates skip here)
    uint16_t raw = gates[gi * nr + ri];
    if (raw <= 1) return;

    float sc = info.scale[product], off = info.offset[product];
    float value = ((float)raw - off) / sc;

    // Threshold
    float threshold = dbz_min;
    if (product == PROD_VEL || product == PROD_ZDR || product == PROD_KDP || product == PROD_PHI)
        threshold = -999.0f;
    else if (product == PROD_CC) threshold = 0.3f;
    else if (product == PROD_SW) threshold = 0.5f;
    if (value < threshold) return;

    // Color lookup
    float min_val, max_val;
    switch (product) {
        case PROD_REF: min_val=-30;max_val=75; break;
        case PROD_VEL: min_val=-64;max_val=64; break;
        case PROD_SW:  min_val=0;max_val=30; break;
        case PROD_ZDR: min_val=-8;max_val=8; break;
        case PROD_CC:  min_val=0.2f;max_val=1.05f; break;
        case PROD_KDP: min_val=-10;max_val=15; break;
        default:       min_val=0;max_val=360; break;
    }
    float norm = fminf(fmaxf((value - min_val) / (max_val - min_val), 0.0f), 1.0f);
    float tc = (norm * 254.0f + 1.0f) / 256.0f;
    float4 col = tex1D<float4>(colorTex, tc);
    if (col.w < 0.01f) return;
    uint32_t rgba = makeRGBA((uint8_t)(col.x*255), (uint8_t)(col.y*255),
                              (uint8_t)(col.z*255), 255);

    // Compute 4 corners of this polar quad in screen space
    int ri_next = (ri + 1) % nr;
    float az0 = azimuths[ri] * ((float)M_PI / 180.0f);
    float az1 = azimuths[ri_next] * ((float)M_PI / 180.0f);
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

    // Skip if entirely off-screen
    if (ix0 > ix1 || iy0 > iy1) return;

    // Edge functions for convex quad (CCW winding)
    float2 corners[4] = {c0, c1, c2, c3};
    float enx[4], eny[4], ed[4];
    for (int e = 0; e < 4; e++) {
        float2 v0 = corners[e];
        float2 v1 = corners[(e + 1) & 3];
        enx[e] = v1.y - v0.y;
        eny[e] = -(v1.x - v0.x);
        ed[e] = enx[e] * v0.x + eny[e] * v0.y;
    }

    // Fill all pixels inside the quad
    for (int py = iy0; py <= iy1; py++) {
        for (int px = ix0; px <= ix1; px++) {
            float fx = (float)px + 0.5f;
            float fy = (float)py + 0.5f;
            bool inside = true;
            for (int e = 0; e < 4; e++) {
                if (enx[e] * fx + eny[e] * fy < ed[e]) { inside = false; break; }
            }
            if (inside) {
                output[py * vp.width + px] = rgba;
            }
        }
    }
}

// Clear kernel (fills background)
__global__ void clearKernel(uint32_t* output, int width, int height, uint32_t color) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px < width && py < height)
        output[py * width + px] = color;
}

void forwardRenderStation(const GpuViewport& vp, int station_idx,
                           int product, float dbz_min, uint32_t* d_output) {
    if (station_idx < 0 || station_idx >= MAX_STATIONS) return;
    auto& buf = s_stationBufs[station_idx];
    auto& info = s_stationInfo[station_idx];
    if (!buf.allocated || !info.has_product[product] || !buf.d_gates[product]) return;

    // Clear to background first
    dim3 clearBlock(32, 8);
    dim3 clearGrid((vp.width + 31) / 32, (vp.height + 7) / 8);
    clearKernel<<<clearGrid, clearBlock>>>(d_output, vp.width, vp.height,
                                            makeRGBA(15, 15, 20, 255));

    // Forward render: one thread per (radial, gate)
    dim3 block(32, 8); // 256 threads, warp-aligned
    dim3 grid((info.num_radials + 31) / 32,
              (info.num_gates[product] + 7) / 8);

    forwardRenderKernel<<<grid, block>>>(
        buf.d_azimuths, buf.d_gates[product],
        info, vp, product, dbz_min,
        s_colorTextures[product], d_output);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "Forward render error: %s\n", cudaGetErrorString(err));
}

void syncStation(int idx) {
    if (idx >= 0 && idx < MAX_STATIONS && s_stationBufs[idx].allocated)
        CUDA_CHECK(cudaStreamSynchronize(s_stationBufs[idx].stream));
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
    const bool* __restrict__ active,   // which stations have data
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
            int slot = atomicAdd(&grid->counts[gy][gx], 1);
            if (slot < MAX_STATIONS_PER_CELL) {
                grid->cells[gy][gx][slot] = si;
            }
        }
    }
}

void buildSpatialGridGpu(const GpuStationInfo* h_stations, int num_stations,
                          SpatialGrid* h_grid_out) {
    // Upload station info
    GpuStationInfo* d_stations;
    bool* d_active;
    CUDA_CHECK(cudaMalloc(&d_stations, num_stations * sizeof(GpuStationInfo)));
    CUDA_CHECK(cudaMalloc(&d_active, num_stations * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_stations, h_stations,
                           num_stations * sizeof(GpuStationInfo), cudaMemcpyHostToDevice));

    // Build active flags (station has data if num_radials > 0)
    std::vector<uint8_t> h_active(num_stations);
    for (int i = 0; i < num_stations; i++)
        h_active[i] = (h_stations[i].num_radials > 0) ? 1 : 0;
    CUDA_CHECK(cudaMemcpy(d_active, h_active.data(), num_stations * sizeof(bool),
                           cudaMemcpyHostToDevice));

    // Init grid on device
    SpatialGrid* d_grid;
    CUDA_CHECK(cudaMalloc(&d_grid, sizeof(SpatialGrid)));

    // Set grid bounds and zero counts
    SpatialGrid init_grid = {};
    init_grid.min_lat = 15.0f;  init_grid.max_lat = 72.0f;
    init_grid.min_lon = -180.0f; init_grid.max_lon = -60.0f;
    CUDA_CHECK(cudaMemcpy(d_grid, &init_grid, sizeof(SpatialGrid), cudaMemcpyHostToDevice));

    // Launch kernel
    buildGridKernel<<<(num_stations + 255) / 256, 256>>>(
        d_stations, d_active, num_stations, d_grid);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_grid_out, d_grid, sizeof(SpatialGrid), cudaMemcpyDeviceToHost));

    cudaFree(d_stations);
    cudaFree(d_active);
    cudaFree(d_grid);
}

} // namespace gpu
