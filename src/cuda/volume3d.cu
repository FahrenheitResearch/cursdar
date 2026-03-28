#include "volume3d.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float*              d_volume_raw = nullptr;
static cudaArray_t         d_volume_array = nullptr;
static cudaTextureObject_t d_volume_tex = 0;
static bool                s_volumeReady = false;

extern __constant__ uint32_t c_colorTable[NUM_PRODUCTS][256];

namespace {

constexpr float kMissingValue = -999.0f;
constexpr float kRadarEffectiveEarthRadiusKm = 8494.0f;
constexpr float kRadarBeamWidthRad = 0.01745329251994329577f;
constexpr float kHalfRadarBeamWidthRad = kRadarBeamWidthRad * 0.5f;
constexpr float kBeamMatchTolerance = 1.35f;
constexpr float kCrossSectionMaxHeightKm = 15.0f;

struct SweepDesc {
    float elevation_deg = 0.0f;
    int num_radials = 0;
    int num_gates = 0;
    float first_gate_km = 0.0f;
    float gate_spacing_km = 0.0f;
    float scale = 0.0f;
    float offset = 0.0f;
    const float* azimuths = nullptr;
    const uint16_t* gates = nullptr;
};

__constant__ SweepDesc c_sweeps[32];
__constant__ int c_numSweeps;

__device__ __host__ uint32_t mkRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
    return (uint32_t)r | ((uint32_t)g << 8) | ((uint32_t)b << 16) | ((uint32_t)a << 24);
}

__device__ __host__ float clamp01(float v) {
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

__device__ __host__ bool isValidSample(float v) {
    return v > -998.0f;
}

__device__ __host__ void productRange(int product, float& min_val, float& max_val) {
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

__device__ __host__ float productThreshold(int product, float reflectivity_threshold) {
    if (product == PROD_VEL || product == PROD_ZDR || product == PROD_KDP || product == PROD_PHI)
        return kMissingValue;
    if (product == PROD_CC) return 0.3f;
    if (product == PROD_SW) return 0.5f;
    return reflectivity_threshold;
}

__device__ __host__ bool passesThreshold(int product, float value, float reflectivity_threshold) {
    if (!isValidSample(value)) return false;
    return value >= productThreshold(product, reflectivity_threshold);
}

__device__ __host__ float sampleMagnitude(int product, float value) {
    float min_val = 0.0f, max_val = 1.0f;
    productRange(product, min_val, max_val);
    if (product == PROD_VEL || product == PROD_ZDR || product == PROD_KDP) {
        float max_abs = fmaxf(fabsf(min_val), fabsf(max_val));
        return (max_abs > 0.0f) ? clamp01(fabsf(value) / max_abs) : 0.0f;
    }
    return clamp01((value - min_val) / fmaxf(max_val - min_val, 1e-6f));
}

__device__ __host__ int colorIndexForValue(int product, float value) {
    float min_val = 0.0f, max_val = 1.0f;
    productRange(product, min_val, max_val);
    float norm = clamp01((value - min_val) / fmaxf(max_val - min_val, 1e-6f));
    int idx = (int)(norm * 254.0f) + 1;
    if (idx < 1) idx = 1;
    if (idx > 255) idx = 255;
    return idx;
}

__device__ int bsAz(const float* azimuths, int n, float target) {
    int lo = 0;
    int hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (azimuths[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ float decodeRaw(const SweepDesc& sw, uint16_t raw) {
    if (raw <= 1 || sw.scale == 0.0f) return kMissingValue;
    return ((float)raw - sw.offset) / sw.scale;
}

__device__ bool beamGeometryAtRange(const SweepDesc& sw,
                                    float ground_range_km,
                                    float* slant_range_km,
                                    float* beam_height_km,
                                    float* beam_half_width_km) {
    if (!sw.gates || !sw.azimuths || sw.num_radials <= 0 || sw.num_gates <= 0 || sw.gate_spacing_km <= 0.0f)
        return false;

    float elev_rad = sw.elevation_deg * (float)M_PI / 180.0f;
    float cos_e = cosf(elev_rad);
    if (fabsf(cos_e) < 1e-4f) return false;

    float slant = ground_range_km / cos_e;
    float beam_h = slant * sinf(elev_rad) +
                   (ground_range_km * ground_range_km) / (2.0f * kRadarEffectiveEarthRadiusKm);
    float half_width = fmaxf(0.25f, slant * kHalfRadarBeamWidthRad);

    *slant_range_km = slant;
    *beam_height_km = beam_h;
    *beam_half_width_km = half_width;
    return true;
}

__device__ float interpolate4(float v00, float w00,
                              float v01, float w01,
                              float v10, float w10,
                              float v11, float w11) {
    float wsum = w00 + w01 + w10 + w11;
    if (wsum <= 1e-6f) return kMissingValue;
    return (v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11) / wsum;
}

__device__ float sampleSweepValue(const SweepDesc& sw, float azimuth_deg, float slant_range_km) {
    float max_range = sw.first_gate_km + (sw.num_gates - 1) * sw.gate_spacing_km;
    if (slant_range_km < sw.first_gate_km || slant_range_km > max_range)
        return kMissingValue;

    float gate_pos = (slant_range_km - sw.first_gate_km) / sw.gate_spacing_km;
    int gate0 = (int)floorf(gate_pos);
    if (gate0 < 0 || gate0 >= sw.num_gates)
        return kMissingValue;
    int gate1 = (gate0 + 1 < sw.num_gates) ? gate0 + 1 : gate0;
    float gate_t = clamp01(gate_pos - gate0);

    int idx_hi = bsAz(sw.azimuths, sw.num_radials, azimuth_deg);
    if (idx_hi >= sw.num_radials) idx_hi = 0;
    int idx_lo = (idx_hi == 0) ? sw.num_radials - 1 : idx_hi - 1;

    float az_lo = sw.azimuths[idx_lo];
    float az_hi = sw.azimuths[idx_hi];
    float az_span = az_hi - az_lo;
    if (az_span < 0.0f) az_span += 360.0f;
    if (az_span < 0.01f) az_span = 360.0f / fmaxf((float)sw.num_radials, 1.0f);
    float az_off = azimuth_deg - az_lo;
    if (az_off < 0.0f) az_off += 360.0f;
    float az_t = clamp01(az_off / az_span);

    float v00 = decodeRaw(sw, sw.gates[gate0 * sw.num_radials + idx_lo]);
    float v01 = decodeRaw(sw, sw.gates[gate0 * sw.num_radials + idx_hi]);
    float v10 = decodeRaw(sw, sw.gates[gate1 * sw.num_radials + idx_lo]);
    float v11 = decodeRaw(sw, sw.gates[gate1 * sw.num_radials + idx_hi]);

    float w00 = isValidSample(v00) ? (1.0f - gate_t) * (1.0f - az_t) : 0.0f;
    float w01 = isValidSample(v01) ? (1.0f - gate_t) * az_t : 0.0f;
    float w10 = isValidSample(v10) ? gate_t * (1.0f - az_t) : 0.0f;
    float w11 = isValidSample(v11) ? gate_t * az_t : 0.0f;
    return interpolate4(v00, w00, v01, w01, v10, w10, v11, w11);
}

__global__ void buildVolumeKernel(float* __restrict__ volume) {
    int vx = blockIdx.x * blockDim.x + threadIdx.x;
    int vy = blockIdx.y * blockDim.y + threadIdx.y;
    int vz = blockIdx.z;
    if (vx >= VOL_XY || vy >= VOL_XY || vz >= VOL_Z) return;

    float x_km = ((float)vx / VOL_XY - 0.5f) * 2.0f * VOL_RANGE_KM;
    float y_km = ((float)vy / VOL_XY - 0.5f) * 2.0f * VOL_RANGE_KM;
    float z_km = ((float)vz / VOL_Z) * VOL_HEIGHT_KM;

    float ground_range = sqrtf(x_km * x_km + y_km * y_km);
    float azimuth = atan2f(x_km, y_km) * (180.0f / (float)M_PI);
    if (azimuth < 0.0f) azimuth += 360.0f;

    float best_value0 = kMissingValue;
    float best_value1 = kMissingValue;
    float best_score0 = 1e30f;
    float best_score1 = 1e30f;

    for (int s = 0; s < c_numSweeps; s++) {
        const SweepDesc& sw = c_sweeps[s];

        float slant_range = 0.0f;
        float beam_height = 0.0f;
        float beam_half_width = 0.0f;
        if (!beamGeometryAtRange(sw, ground_range, &slant_range, &beam_height, &beam_half_width))
            continue;

        float sample = sampleSweepValue(sw, azimuth, slant_range);
        if (!isValidSample(sample))
            continue;

        float beam_offset = fabsf(beam_height - z_km);
        float score = beam_offset / fmaxf(beam_half_width, 0.1f);
        if (score > kBeamMatchTolerance)
            continue;

        if (score < best_score0) {
            best_score1 = best_score0;
            best_value1 = best_value0;
            best_score0 = score;
            best_value0 = sample;
        } else if (score < best_score1) {
            best_score1 = score;
            best_value1 = sample;
        }
    }

    float value = kMissingValue;
    if (isValidSample(best_value0) && isValidSample(best_value1)) {
        float w0 = 1.0f / fmaxf(best_score0, 0.05f);
        float w1 = 1.0f / fmaxf(best_score1, 0.05f);
        value = (best_value0 * w0 + best_value1 * w1) / (w0 + w1);
    } else if (isValidSample(best_value0)) {
        value = best_value0;
    } else if (isValidSample(best_value1)) {
        value = best_value1;
    }

    volume[(size_t)vz * VOL_XY * VOL_XY + vy * VOL_XY + vx] = value;
}

__global__ void rayMarchKernel(
    cudaTextureObject_t volTex,
    float cam_x, float cam_y, float cam_z,
    float fwd_x, float fwd_y, float fwd_z,
    float right_x, float right_y, float right_z,
    float up_x, float up_y, float up_z,
    float fov_scale,
    int width, int height,
    int product, float dbz_min,
    uint32_t* __restrict__ output) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    float u = ((float)px / width - 0.5f) * 2.0f * fov_scale * ((float)width / height);
    float v = (0.5f - (float)py / height) * 2.0f * fov_scale;

    float dx = fwd_x + right_x * u + up_x * v;
    float dy = fwd_y + right_y * u + up_y * v;
    float dz = fwd_z + right_z * u + up_z * v;
    float inv_dir_len = rsqrtf(dx * dx + dy * dy + dz * dz);
    dx *= inv_dir_len;
    dy *= inv_dir_len;
    dz *= inv_dir_len;

    float bmin = -VOL_RANGE_KM;
    float bmax = VOL_RANGE_KM;
    float bzmax = VOL_DISPLAY_HEIGHT;
    float tmin = -1e9f;
    float tmax = 1e9f;

    if (fabsf(dx) > 1e-6f) {
        float t1 = (bmin - cam_x) / dx;
        float t2 = (bmax - cam_x) / dx;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        tmin = fmaxf(tmin, t1);
        tmax = fminf(tmax, t2);
    }
    if (fabsf(dy) > 1e-6f) {
        float t1 = (bmin - cam_y) / dy;
        float t2 = (bmax - cam_y) / dy;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        tmin = fmaxf(tmin, t1);
        tmax = fminf(tmax, t2);
    }
    if (fabsf(dz) > 1e-6f) {
        float t1 = (0.0f - cam_z) / dz;
        float t2 = (bzmax - cam_z) / dz;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        tmin = fmaxf(tmin, t1);
        tmax = fminf(tmax, t2);
    }

    float sky_t = fmaxf(0.0f, v * 0.3f + 0.3f);
    float3 bg = {0.03f + sky_t * 0.04f, 0.03f + sky_t * 0.06f, 0.06f + sky_t * 0.10f};

    float ground_t = -1.0f;
    if (fabsf(dz) > 1e-6f)
        ground_t = -cam_z / dz;

    bool hit_ground = false;
    float3 ground_color = bg;
    if (ground_t > 0.0f && (tmin > tmax || ground_t < tmin)) {
        float gx = cam_x + dx * ground_t;
        float gy = cam_y + dy * ground_t;
        float gmod_x = fmodf(fabsf(gx), 50.0f);
        float gmod_y = fmodf(fabsf(gy), 50.0f);
        float line_x = fminf(gmod_x, 50.0f - gmod_x);
        float line_y = fminf(gmod_y, 50.0f - gmod_y);
        float grid_line = fminf(line_x, line_y);
        float grid_alpha = fmaxf(0.0f, 1.0f - grid_line * 0.8f) * 0.15f;
        float gdist = sqrtf(gx * gx + gy * gy);
        float gfade = fmaxf(0.0f, 1.0f - gdist / (VOL_RANGE_KM * 1.5f));
        grid_alpha *= gfade;
        ground_color = {bg.x + grid_alpha * 0.3f, bg.y + grid_alpha * 0.4f, bg.z + grid_alpha * 0.5f};
        hit_ground = true;
    }

    if (tmin > tmax || tmax < 0.0f) {
        float3 c = hit_ground ? ground_color : bg;
        output[py * width + px] = mkRGBA((uint8_t)(c.x * 255.0f),
                                         (uint8_t)(c.y * 255.0f),
                                         (uint8_t)(c.z * 255.0f));
        return;
    }

    tmin = fmaxf(tmin, 0.001f);

    float base_step = 0.7f;
    int max_steps = (int)fminf((tmax - tmin) / base_step, 600.0f);

    const float lx = 0.45f;
    const float ly = -0.35f;
    const float lz = 0.75f;
    const float eps = 1.2f / VOL_XY;

    float3 accum = {0.0f, 0.0f, 0.0f};
    float alpha = 0.0f;
    float threshold = productThreshold(product, dbz_min);

    for (int step = 0; step < max_steps && alpha < 0.995f; step++) {
        float t = tmin + (float)step * base_step;
        if (t > tmax) break;

        float sx = cam_x + dx * t;
        float sy = cam_y + dy * t;
        float sz = cam_z + dz * t;
        float tx = sx / VOL_RANGE_KM * 0.5f + 0.5f;
        float ty = sy / VOL_RANGE_KM * 0.5f + 0.5f;
        float tz = (sz / VOL_Z_EXAGGERATION) / VOL_HEIGHT_KM;

        if (tx < 0.002f || tx > 0.998f || ty < 0.002f || ty > 0.998f ||
            tz < 0.002f || tz > 0.998f) {
            continue;
        }

        float val = tex3D<float>(volTex, tx, ty, tz);
        if (!passesThreshold(product, val, dbz_min))
            continue;

        float gnx = tex3D<float>(volTex, tx + eps, ty, tz) - tex3D<float>(volTex, tx - eps, ty, tz);
        float gny = tex3D<float>(volTex, tx, ty + eps, tz) - tex3D<float>(volTex, tx, ty - eps, tz);
        float gnz = tex3D<float>(volTex, tx, ty, tz + eps) - tex3D<float>(volTex, tx, ty, tz - eps);
        float gl = rsqrtf(gnx * gnx + gny * gny + gnz * gnz + 1e-6f);
        float nx = gnx * gl;
        float ny = gny * gl;
        float nz = gnz * gl;

        float ndotl = fmaxf(0.0f, nx * lx + ny * ly + nz * lz);
        float ambient = 0.25f;
        float diffuse = 0.55f * ndotl;

        float shadow = 1.0f;
        float stx = tx;
        float sty = ty;
        float stz = tz;
        float sl_dx = lx * eps * 3.0f;
        float sl_dy = ly * eps * 3.0f;
        float sl_dz = lz * (1.0f / VOL_Z) * 3.0f;
        for (int si = 0; si < 8; si++) {
            stx += sl_dx;
            sty += sl_dy;
            stz += sl_dz;
            if (stx < 0.0f || stx > 1.0f || sty < 0.0f || sty > 1.0f || stz < 0.0f || stz > 1.0f)
                break;
            float sv = tex3D<float>(volTex, stx, sty, stz);
            if (passesThreshold(product, sv, dbz_min))
                shadow -= sampleMagnitude(product, sv) * 0.15f;
        }
        shadow = fmaxf(shadow, 0.15f);

        float hx = lx - dx;
        float hy = ly - dy;
        float hz = lz - dz;
        float hl = rsqrtf(hx * hx + hy * hy + hz * hz + 1e-6f);
        float ndoth = fmaxf(0.0f, nx * hx * hl + ny * hy * hl + nz * hz * hl);
        float specular = powf(ndoth, 64.0f) * 0.7f * shadow;

        float ndotv = fabsf(nx * (-dx) + ny * (-dy) + nz * (-dz));
        float rim = powf(1.0f - ndotv, 3.0f) * 0.4f;
        float lighting = ambient + diffuse * shadow + rim;

        uint32_t color = c_colorTable[product][colorIndexForValue(product, val)];
        float cr = (float)(color & 0xFF) / 255.0f;
        float cg = (float)((color >> 8) & 0xFF) / 255.0f;
        float cb = (float)((color >> 16) & 0xFF) / 255.0f;

        cr = cr * lighting + specular;
        cg = cg * lighting + specular * 0.85f;
        cb = cb * lighting + specular * 0.7f;

        if (product == PROD_REF) {
            float glow = fmaxf(0.0f, (val - 45.0f) / 20.0f);
            cr += glow * glow * 0.5f;
            cg += glow * glow * 0.2f;
        }

        float intensity = sampleMagnitude(product, val);
        float opacity = 0.04f + powf(intensity, 2.5f) * 0.32f;
        if (product == PROD_REF && threshold > kMissingValue) {
            float ref_gate = clamp01((val - threshold) / fmaxf(75.0f - threshold, 1.0f));
            opacity = 0.02f + powf(ref_gate, 2.7f) * 0.48f;
        }

        accum.x += (1.0f - alpha) * fminf(cr, 1.5f) * opacity;
        accum.y += (1.0f - alpha) * fminf(cg, 1.5f) * opacity;
        accum.z += (1.0f - alpha) * fminf(cb, 1.5f) * opacity;
        alpha += (1.0f - alpha) * opacity;
    }

    float3 final_bg = hit_ground ? ground_color : bg;
    float fr = fminf(accum.x + final_bg.x * (1.0f - alpha), 1.0f);
    float fg = fminf(accum.y + final_bg.y * (1.0f - alpha), 1.0f);
    float fb = fminf(accum.z + final_bg.z * (1.0f - alpha), 1.0f);

    output[py * width + px] = mkRGBA((uint8_t)(fr * 255.0f),
                                     (uint8_t)(fg * 255.0f),
                                     (uint8_t)(fb * 255.0f));
}

__global__ void crossSectionKernel(
    float start_x_km, float start_y_km,
    float dir_x, float dir_y,
    float total_dist_km,
    int width, int height,
    int product, float dbz_min,
    uint32_t* __restrict__ output) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    float dist_along = ((float)px / width) * total_dist_km;
    float alt_km = (1.0f - (float)py / height) * kCrossSectionMaxHeightKm;

    float x_km = start_x_km + dir_x * dist_along;
    float y_km = start_y_km + dir_y * dist_along;

    float ground_range = sqrtf(x_km * x_km + y_km * y_km);
    if (ground_range < 1.0f) ground_range = 1.0f;

    float azimuth = atan2f(x_km, y_km) * (180.0f / (float)M_PI);
    if (azimuth < 0.0f) azimuth += 360.0f;

    uint32_t bg = mkRGBA(18, 18, 25);
    float hgrid = fmodf(dist_along, 25.0f);
    float vgrid = fmodf(alt_km, 1.524f);
    if (fminf(hgrid, 25.0f - hgrid) < 0.3f)
        bg = mkRGBA(25, 25, 35);
    if (fminf(vgrid, 1.524f - vgrid) < 0.02f)
        bg = mkRGBA(25, 25, 35);

    float best_val = kMissingValue;
    float best_score = 1e30f;
    float best_dist = 1e30f;

    for (int s = 0; s < c_numSweeps; s++) {
        const SweepDesc& sw = c_sweeps[s];

        float slant_range = 0.0f;
        float beam_height = 0.0f;
        float beam_half_width = 0.0f;
        if (!beamGeometryAtRange(sw, ground_range, &slant_range, &beam_height, &beam_half_width))
            continue;

        float value = sampleSweepValue(sw, azimuth, slant_range);
        if (!passesThreshold(product, value, dbz_min))
            continue;

        float beam_offset = fabsf(beam_height - alt_km);
        float score = beam_offset / fmaxf(beam_half_width, 0.1f);
        if (score > kBeamMatchTolerance)
            continue;

        if (score < best_score ||
            (fabsf(score - best_score) < 1e-3f && beam_offset < best_dist) ||
            (fabsf(score - best_score) < 1e-3f && fabsf(beam_offset - best_dist) < 1e-3f &&
             fabsf(value) > fabsf(best_val))) {
            best_score = score;
            best_dist = beam_offset;
            best_val = value;
        }
    }

    if (!passesThreshold(product, best_val, dbz_min)) {
        output[py * width + px] = bg;
        return;
    }

    uint32_t color = c_colorTable[product][colorIndexForValue(product, best_val)];
    output[py * width + px] = color | 0xFF000000u;
}

} // namespace

namespace gpu {

void initVolume() {
    freeVolume();

    size_t vol_size = (size_t)VOL_XY * VOL_XY * VOL_Z * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_volume_raw, vol_size));

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent(VOL_XY, VOL_XY, VOL_Z);
    CUDA_CHECK(cudaMalloc3DArray(&d_volume_array, &desc, extent));

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = d_volume_array;

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.addressMode[2] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;

    CUDA_CHECK(cudaCreateTextureObject(&d_volume_tex, &res_desc, &tex_desc, nullptr));

    printf("3D volume: %dx%dx%d, HW trilinear texture, %.1f MB\n",
           VOL_XY, VOL_XY, VOL_Z, vol_size / (1024.0f * 1024.0f));
}

void freeVolume() {
    if (d_volume_tex) {
        cudaDestroyTextureObject(d_volume_tex);
        d_volume_tex = 0;
    }
    if (d_volume_array) {
        cudaFreeArray(d_volume_array);
        d_volume_array = nullptr;
    }
    if (d_volume_raw) {
        cudaFree(d_volume_raw);
        d_volume_raw = nullptr;
    }
    s_volumeReady = false;
}

void buildVolume(int station_idx, int product,
                 const GpuStationInfo* sweep_infos, int num_sweeps,
                 const float* const* d_azimuths_per_sweep,
                 const uint16_t* const* d_gates_per_sweep) {
    (void)station_idx;
    s_volumeReady = false;
    if (num_sweeps <= 0 || !d_volume_raw) return;

    std::vector<SweepDesc> h_sweeps;
    h_sweeps.reserve((num_sweeps < 32) ? num_sweeps : 32);

    for (int s = 0; s < num_sweeps && (int)h_sweeps.size() < 32; s++) {
        const GpuStationInfo& info = sweep_infos[s];
        if (!info.has_product[product] ||
            info.num_radials <= 0 ||
            info.num_gates[product] <= 0 ||
            info.gate_spacing_km[product] <= 0.0f ||
            !d_azimuths_per_sweep[s] ||
            !d_gates_per_sweep[s]) {
            continue;
        }

        SweepDesc sw = {};
        sw.elevation_deg = info.elevation_angle;
        sw.num_radials = info.num_radials;
        sw.num_gates = info.num_gates[product];
        sw.first_gate_km = info.first_gate_km[product];
        sw.gate_spacing_km = info.gate_spacing_km[product];
        sw.scale = info.scale[product];
        sw.offset = info.offset[product];
        sw.azimuths = d_azimuths_per_sweep[s];
        sw.gates = d_gates_per_sweep[s];
        h_sweeps.push_back(sw);
    }

    int count = (int)h_sweeps.size();
    if (count <= 0) return;

    CUDA_CHECK(cudaMemcpyToSymbol(c_sweeps, h_sweeps.data(), count * sizeof(SweepDesc)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_numSweeps, &count, sizeof(int)));

    dim3 block(8, 8);
    dim3 grid((VOL_XY + 7) / 8, (VOL_XY + 7) / 8, VOL_Z);
    buildVolumeKernel<<<grid, block>>>(d_volume_raw);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy3DParms copy_params = {};
    copy_params.srcPtr = make_cudaPitchedPtr(d_volume_raw, VOL_XY * sizeof(float), VOL_XY, VOL_XY);
    copy_params.dstArray = d_volume_array;
    copy_params.extent = make_cudaExtent(VOL_XY, VOL_XY, VOL_Z);
    copy_params.kind = cudaMemcpyDeviceToDevice;
    CUDA_CHECK(cudaMemcpy3D(&copy_params));

    s_volumeReady = true;
    printf("3D volume built: %d sweeps, HW trilinear ready\n", count);
}

void renderVolume(const Camera3D& cam, int width, int height,
                  int product, float dbz_min, uint32_t* d_output) {
    if (!s_volumeReady) return;

    float theta = cam.orbit_angle * (float)M_PI / 180.0f;
    float phi = cam.tilt_angle * (float)M_PI / 180.0f;

    float cx = cam.distance * sinf(theta) * cosf(phi);
    float cy = cam.distance * cosf(theta) * cosf(phi);
    float cz = cam.distance * sinf(phi) + cam.target_z;

    float fx = -cx;
    float fy = -cy;
    float fz = cam.target_z - cz;
    float fl = rsqrtf(fx * fx + fy * fy + fz * fz);
    fx *= fl;
    fy *= fl;
    fz *= fl;

    float rx = fy;
    float ry = -fx;
    float rz = 0.0f;
    float rl = rsqrtf(rx * rx + ry * ry + rz * rz + 1e-8f);
    rx *= rl;
    ry *= rl;
    rz *= rl;

    float ux = ry * fz - rz * fy;
    float uy = rz * fx - rx * fz;
    float uz = rx * fy - ry * fx;

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    rayMarchKernel<<<grid, block>>>(
        d_volume_tex,
        cx, cy, cz,
        fx, fy, fz,
        rx, ry, rz,
        ux, uy, uz,
        0.7f, width, height, product, dbz_min, d_output);
    CUDA_CHECK(cudaGetLastError());
}

void renderCrossSection(
    int station_idx, int product, float dbz_min,
    float start_lat, float start_lon, float end_lat, float end_lon,
    float station_lat, float station_lon,
    int width, int height,
    uint32_t* d_output) {
    (void)station_idx;
    if (!s_volumeReady) return;

    float cos_lat = cosf(station_lat * (float)M_PI / 180.0f);
    float sx_km = (start_lon - station_lon) * 111.0f * cos_lat;
    float sy_km = (start_lat - station_lat) * 111.0f;
    float ex_km = (end_lon - station_lon) * 111.0f * cos_lat;
    float ey_km = (end_lat - station_lat) * 111.0f;

    float ddx = ex_km - sx_km;
    float ddy = ey_km - sy_km;
    float total = sqrtf(ddx * ddx + ddy * ddy);
    if (total < 1.0f) return;

    float nx = ddx / total;
    float ny = ddy / total;

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    crossSectionKernel<<<grid, block>>>(
        sx_km, sy_km,
        nx, ny,
        total,
        width, height,
        product, dbz_min,
        d_output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu
