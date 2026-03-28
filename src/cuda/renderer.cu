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

static void generateRefColorTable(uint32_t* table) {
    memset(table, 0, 256 * sizeof(uint32_t));
    auto dBZtoIdx = [](float dbz) -> int {
        int idx = (int)((dbz + 30.0f) * 255.0f / 105.0f);
        return (idx < 0) ? 0 : (idx > 255) ? 255 : idx;
    };
    struct { float dbz; uint8_t r, g, b; } steps[] = {
        { 5.0f,   4, 233, 231},  {10.0f,   1, 159, 244},
        {15.0f,   3,   0, 244},  {20.0f,   2, 253,   2},
        {25.0f,   1, 197,   1},  {30.0f,   0, 142,   0},
        {35.0f, 253, 248,   2},  {40.0f, 229, 188,   0},
        {45.0f, 253, 149,   0},  {50.0f, 253,   0,   0},
        {55.0f, 212,   0,   0},  {60.0f, 188,   0,   0},
        {65.0f, 248,   0, 253},  {70.0f, 152,  84, 198},
        {75.0f, 253, 253, 253},
    };
    int nsteps = sizeof(steps) / sizeof(steps[0]);
    for (int s = 0; s < nsteps; s++) {
        int i0 = dBZtoIdx(steps[s].dbz);
        int i1 = (s + 1 < nsteps) ? dBZtoIdx(steps[s + 1].dbz) : 256;
        for (int i = i0; i < i1 && i < 256; i++)
            table[i] = makeRGBA(steps[s].r, steps[s].g, steps[s].b);
    }
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

static void generateVelColorTable(uint32_t* table) {
    memset(table, 0, 256 * sizeof(uint32_t));
    interpolateColor(table, 1, 32,    0, 255, 0,    0, 200, 0);
    interpolateColor(table, 32, 64,   0, 200, 0,    0, 150, 0);
    interpolateColor(table, 64, 96,   0, 150, 0,    0, 100, 50);
    interpolateColor(table, 96, 128,  0, 100, 50,   50, 50, 50);
    interpolateColor(table, 128, 160, 50, 50, 50,   100, 50, 0);
    interpolateColor(table, 160, 192, 100, 50, 0,   200, 100, 0);
    interpolateColor(table, 192, 224, 200, 100, 0,  255, 50, 0);
    interpolateColor(table, 224, 255, 255, 50, 0,   255, 0, 0);
}

static void uploadColorTables() {
    uint32_t tables[NUM_PRODUCTS][256];
    generateRefColorTable(tables[PROD_REF]);
    generateVelColorTable(tables[PROD_VEL]);
    // SW
    memset(tables[PROD_SW], 0, 256*4);
    interpolateColor(tables[PROD_SW], 1, 60, 40,40,40, 0,100,0);
    interpolateColor(tables[PROD_SW], 60, 130, 0,100,0, 0,255,0);
    interpolateColor(tables[PROD_SW], 130, 200, 0,255,0, 255,255,0);
    interpolateColor(tables[PROD_SW], 200, 255, 255,255,0, 255,0,0);
    // ZDR
    memset(tables[PROD_ZDR], 0, 256*4);
    interpolateColor(tables[PROD_ZDR], 1, 128, 0,0,180, 100,100,180);
    interpolateColor(tables[PROD_ZDR], 128, 192, 200,200,200, 255,255,0);
    interpolateColor(tables[PROD_ZDR], 192, 255, 255,255,0, 255,0,0);
    // CC
    memset(tables[PROD_CC], 0, 256*4);
    interpolateColor(tables[PROD_CC], 1, 60, 0,0,100, 128,0,128);
    interpolateColor(tables[PROD_CC], 60, 170, 128,0,128, 255,150,0);
    interpolateColor(tables[PROD_CC], 170, 230, 255,150,0, 0,200,0);
    interpolateColor(tables[PROD_CC], 230, 255, 0,200,0, 255,255,255);
    // KDP
    memset(tables[PROD_KDP], 0, 256*4);
    interpolateColor(tables[PROD_KDP], 1, 80, 0,0,150, 0,150,255);
    interpolateColor(tables[PROD_KDP], 80, 128, 0,150,255, 0,255,0);
    interpolateColor(tables[PROD_KDP], 128, 200, 0,255,0, 255,255,0);
    interpolateColor(tables[PROD_KDP], 200, 255, 255,255,0, 255,0,0);
    // PHI
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

    // Find bracketing radials
    int idx_hi = bsearchAz(ptrs.azimuths, nr, az);
    int idx_lo = (idx_hi == 0) ? nr - 1 : idx_hi - 1;
    if (idx_hi >= nr) idx_hi = 0;

    float az_lo = ptrs.azimuths[idx_lo];
    float az_hi = ptrs.azimuths[idx_hi];
    float daz = az_hi - az_lo;
    if (daz < 0) daz += 360.0f;
    float az_off = az - az_lo;
    if (az_off < 0) az_off += 360.0f;
    float t_az = (daz > 0.001f) ? (az_off / daz) : 0.0f;
    t_az = fminf(fmaxf(t_az, 0.0f), 1.0f);

    // Gate index
    float gate_f = (range_km - fgkm) / gskm;
    int g0 = (int)gate_f;
    int g1 = g0 + 1;
    float t_g = gate_f - (float)g0;
    if (g0 < 0) g0 = 0;
    if (g1 >= ng) g1 = ng - 1;
    if (g0 >= ng) return -999.0f;

    // Read 4 gate values (gate-major: gates[gate * num_radials + radial])
    const uint16_t* gd = ptrs.gates[product];
    uint16_t v00 = gd[g0 * nr + idx_lo];
    uint16_t v01 = gd[g0 * nr + idx_hi];
    uint16_t v10 = gd[g1 * nr + idx_lo];
    uint16_t v11 = gd[g1 * nr + idx_hi];

    float sc = info.scale[product], off = info.offset[product];
    auto decode = [sc, off](uint16_t raw) -> float {
        return (raw <= 1) ? -999.0f : ((float)raw - off) / sc;
    };

    float f00 = decode(v00), f01 = decode(v01);
    float f10 = decode(v10), f11 = decode(v11);

    // Bilinear interpolation with missing-data handling
    float value;
    if (f00 > -998.0f && f01 > -998.0f && f10 > -998.0f && f11 > -998.0f) {
        value = f00 * (1-t_az)*(1-t_g) + f01 * t_az*(1-t_g)
              + f10 * (1-t_az)*t_g     + f11 * t_az*t_g;
    } else {
        // Nearest valid
        float best_w = -1.0f; value = -999.0f;
        auto tryV = [&](float v, float w) { if (v > -998.0f && w > best_w) { best_w = w; value = v; } };
        tryV(f00, (1-t_az)*(1-t_g));
        tryV(f01, t_az*(1-t_g));
        tryV(f10, (1-t_az)*t_g);
        tryV(f11, t_az*t_g);
    }

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
    int   best_product_idx = -1;
    GpuStationInfo best_info;

    // Check each station in this cell
    for (int ci = 0; ci < count; ci++) {
        int si = grid->cells[gy][gx][ci];
        if (si < 0 || si >= num_stations) continue;

        const auto& info = stations[si];
        float slat = info.lat, slon = info.lon;

        // Distance in km (flat earth approx, good for <500km)
        float dlat_km = (lat - slat) * 111.0f;
        float dlon_km = (lon - slon) * 111.0f * cosf(slat * (float)M_PI / 180.0f);
        float range_km = sqrtf(dlat_km * dlat_km + dlon_km * dlon_km);

        if (range_km > 460.0f) continue;

        // Azimuth from station to pixel
        float az = atan2f(dlon_km, dlat_km) * (180.0f / (float)M_PI);
        if (az < 0.0f) az += 360.0f;

        float val = sampleStation(info, ptrs[si], az, range_km, product, dbz_min);
        if (val > -998.0f && range_km < best_range) {
            best_value = val;
            best_range = range_km;
            best_info = info;
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

    dim3 block(16, 16);
    dim3 grid_dim((vp.width + 15) / 16, (vp.height + 15) / 16);

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

    int idx_hi = bsearchAz(s_az, nr, az);
    int idx_lo = (idx_hi == 0) ? nr - 1 : idx_hi - 1;
    if (idx_hi >= nr) idx_hi = 0;

    float az_lo = s_az[idx_lo], az_hi = s_az[idx_hi];
    float daz = az_hi - az_lo;
    if (daz < 0) daz += 360.0f;
    float az_off = az - az_lo;
    if (az_off < 0) az_off += 360.0f;
    float t_az = (daz > 0.001f) ? (az_off / daz) : 0.0f;
    t_az = fminf(fmaxf(t_az, 0.0f), 1.0f);

    // Gate index
    float gate_f = (range_km - fgkm) / gskm;
    int g0 = (int)gate_f, g1 = g0 + 1;
    float t_g = gate_f - (float)g0;
    if (g0 < 0) g0 = 0;
    if (g1 >= ng) g1 = ng - 1;
    if (g0 >= ng) { output[py * vp.width + px] = bg; return; }

    // Bilinear sample (gate-major layout)
    uint16_t v00 = gates[g0 * nr + idx_lo];
    uint16_t v01 = gates[g0 * nr + idx_hi];
    uint16_t v10 = gates[g1 * nr + idx_lo];
    uint16_t v11 = gates[g1 * nr + idx_hi];

    float sc = info.scale[product], off = info.offset[product];
    auto decode = [sc, off](uint16_t raw) -> float {
        return (raw <= 1) ? -999.0f : ((float)raw - off) / sc;
    };

    float f00 = decode(v00), f01 = decode(v01);
    float f10 = decode(v10), f11 = decode(v11);

    float value;
    if (f00 > -998.0f && f01 > -998.0f && f10 > -998.0f && f11 > -998.0f) {
        value = f00*(1-t_az)*(1-t_g) + f01*t_az*(1-t_g) + f10*(1-t_az)*t_g + f11*t_az*t_g;
    } else {
        float best_w = -1.0f; value = -999.0f;
        auto tryV = [&](float v, float w) { if (v > -998.0f && w > best_w) { best_w=w; value=v; } };
        tryV(f00, (1-t_az)*(1-t_g)); tryV(f01, t_az*(1-t_g));
        tryV(f10, (1-t_az)*t_g);     tryV(f11, t_az*t_g);
    }

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
