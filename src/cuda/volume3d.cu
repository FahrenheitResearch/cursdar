#include "volume3d.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float2*             d_volume_raw = nullptr;
static float2*             d_volume_scratch = nullptr;
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
constexpr int kVolumeSmoothPasses = 2;

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

__device__ __host__ float lerpFloat(float a, float b, float t) {
    return a + (b - a) * t;
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
    if (product == PROD_VEL)
        return fabsf(value) >= fmaxf(reflectivity_threshold, 0.0f);
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

__device__ float gaussianFalloff(float dist, float sigma) {
    float s = fmaxf(sigma, 1e-3f);
    float q = dist / s;
    return expf(-0.5f * q * q);
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

__device__ float sampleVolumeDensity(int product, float value, float coverage, float threshold) {
    if (coverage <= 0.01f || !isValidSample(value) || !passesThreshold(product, value, threshold))
        return 0.0f;

    float mag = sampleMagnitude(product, value);
    if (product == PROD_REF && threshold > kMissingValue) {
        float gate = clamp01((value - threshold) / fmaxf(75.0f - threshold, 1.0f));
        return powf(gate, 1.55f) * (0.15f + 0.85f * coverage);
    }
    if (product == PROD_CC) {
        float cc_mag = clamp01((value - 0.3f) / 0.75f);
        return powf(cc_mag, 1.4f) * (0.15f + 0.85f * coverage);
    }
    return powf(mag, 1.35f) * (0.14f + 0.86f * coverage);
}

__device__ float sampleVolumeDensityTex(cudaTextureObject_t volTex,
                                        float tx, float ty, float tz,
                                        int product, float threshold) {
    float2 sample = tex3D<float2>(volTex, tx, ty, tz);
    return sampleVolumeDensity(product, sample.x, sample.y, threshold);
}

__global__ void buildVolumeKernel(float2* __restrict__ volume, int product) {
    int vx = blockIdx.x * blockDim.x + threadIdx.x;
    int vy = blockIdx.y * blockDim.y + threadIdx.y;
    int vz = blockIdx.z;
    if (vx >= VOL_XY || vy >= VOL_XY || vz >= VOL_Z) return;

    float x_km = (((float)vx + 0.5f) / VOL_XY - 0.5f) * 2.0f * VOL_RANGE_KM;
    float y_km = (((float)vy + 0.5f) / VOL_XY - 0.5f) * 2.0f * VOL_RANGE_KM;
    float z_km = (((float)vz + 0.5f) / VOL_Z) * VOL_HEIGHT_KM;

    float ground_range = sqrtf(x_km * x_km + y_km * y_km);
    float azimuth = atan2f(x_km, y_km) * (180.0f / (float)M_PI);
    if (azimuth < 0.0f) azimuth += 360.0f;

    float weighted_value = 0.0f;
    float weight_sum = 0.0f;
    float intensity_sum = 0.0f;
    float footprint_sum = 0.0f;
    float below_gap = 1e30f;
    float above_gap = 1e30f;
    int contrib_count = 0;

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

        float beam_offset = beam_height - z_km;
        float sigma_z = fmaxf(beam_half_width * 0.85f, 0.40f);
        float vertical_weight = gaussianFalloff(beam_offset, sigma_z);
        if (vertical_weight < 0.025f)
            continue;

        float mag = sampleMagnitude(product, sample);
        float footprint_weight = 1.0f / (1.0f + beam_half_width * 0.20f);
        float weight = vertical_weight * footprint_weight;
        if (product == PROD_REF)
            weight *= 0.60f + 0.40f * mag;

        weighted_value += sample * weight;
        weight_sum += weight;
        intensity_sum += mag * weight;
        footprint_sum += beam_half_width * weight;
        if (beam_height <= z_km) below_gap = fminf(below_gap, z_km - beam_height);
        else                    above_gap = fminf(above_gap, beam_height - z_km);
        contrib_count++;
    }

    float2 out = make_float2(0.0f, 0.0f);
    if (weight_sum > 0.02f) {
        float value = weighted_value / weight_sum;
        float mean_intensity = intensity_sum / weight_sum;
        float mean_footprint = footprint_sum / weight_sum;

        float bracket = 0.35f;
        if (below_gap < 1e20f)
            bracket += 0.25f * gaussianFalloff(below_gap, fmaxf(mean_footprint, 0.5f));
        if (above_gap < 1e20f)
            bracket += 0.25f * gaussianFalloff(above_gap, fmaxf(mean_footprint, 0.5f));
        if (below_gap < 1e20f && above_gap < 1e20f)
            bracket = fmaxf(bracket, 0.95f);

        float support = clamp01(weight_sum * 0.75f);
        float footprint_conf = 1.0f / (1.0f + fmaxf(mean_footprint - 0.75f, 0.0f) * 0.28f);
        float coverage = (0.20f + mean_intensity * 0.80f) * support * bracket * footprint_conf;
        if (contrib_count == 1)
            coverage *= 0.72f;

        out = make_float2(value, clamp01(coverage));
    }

    volume[(size_t)vz * VOL_XY * VOL_XY + vy * VOL_XY + vx] = out;
}

__global__ void smoothVolumeKernel(const float2* __restrict__ src,
                                   float2* __restrict__ dst,
                                   int product) {
    int vx = blockIdx.x * blockDim.x + threadIdx.x;
    int vy = blockIdx.y * blockDim.y + threadIdx.y;
    int vz = blockIdx.z;
    if (vx >= VOL_XY || vy >= VOL_XY || vz >= VOL_Z) return;

    size_t idx = (size_t)vz * VOL_XY * VOL_XY + vy * VOL_XY + vx;
    float2 center = src[idx];

    float x_km = (((float)vx + 0.5f) / VOL_XY - 0.5f) * 2.0f * VOL_RANGE_KM;
    float y_km = (((float)vy + 0.5f) / VOL_XY - 0.5f) * 2.0f * VOL_RANGE_KM;
    float range_norm = clamp01(sqrtf(x_km * x_km + y_km * y_km) / VOL_RANGE_KM);
    float sigma_xy = 0.85f + range_norm * 1.15f;
    float sigma_z = 0.70f + range_norm * 0.65f;
    float center_intensity =
        (center.y > 0.01f && isValidSample(center.x)) ? sampleMagnitude(product, center.x) : 0.0f;

    float sum_w = 0.0f;
    float sum_val = 0.0f;
    float sum_cov = 0.0f;
    float similarity_scale = (product == PROD_REF) ? 0.07f : 0.035f;

    for (int oz = -1; oz <= 1; ++oz) {
        int nz = vz + oz;
        if (nz < 0 || nz >= VOL_Z) continue;
        for (int oy = -1; oy <= 1; ++oy) {
            int ny = vy + oy;
            if (ny < 0 || ny >= VOL_XY) continue;
            for (int ox = -1; ox <= 1; ++ox) {
                int nx = vx + ox;
                if (nx < 0 || nx >= VOL_XY) continue;

                float2 sample = src[(size_t)nz * VOL_XY * VOL_XY + ny * VOL_XY + nx];
                if (sample.y <= 0.01f || !isValidSample(sample.x))
                    continue;

                float spatial =
                    expf(-0.5f * ((ox * ox + oy * oy) / (sigma_xy * sigma_xy) +
                                  (oz * oz) / (sigma_z * sigma_z)));
                float similarity = 1.0f;
                if (center.y > 0.01f && isValidSample(center.x))
                    similarity = expf(-fabsf(sample.x - center.x) * similarity_scale);

                float weight = spatial * (0.12f + 0.88f * sample.y) * similarity;
                if (ox == 0 && oy == 0 && oz == 0)
                    weight *= 1.35f;

                sum_w += weight;
                sum_val += sample.x * weight;
                sum_cov += sample.y * spatial;
            }
        }
    }

    if (sum_w <= 1e-5f) {
        dst[idx] = make_float2(0.0f, 0.0f);
        return;
    }

    float filtered_val = sum_val / sum_w;
    float filtered_cov = clamp01(sum_cov * 0.19f);
    float smooth_mix = clamp01(0.18f + range_norm * 0.55f);
    smooth_mix *= (1.0f - center_intensity * 0.35f);
    if (center.y <= 0.01f || !isValidSample(center.x))
        smooth_mix = 1.0f;

    float out_val = (center.y > 0.01f && isValidSample(center.x))
        ? lerpFloat(center.x, filtered_val, smooth_mix)
        : filtered_val;
    float out_cov = (center.y > 0.01f)
        ? clamp01(fmaxf(center.y * 0.85f, center.y * 0.55f + filtered_cov * 0.45f))
        : clamp01(filtered_cov * 0.82f);

    if (out_cov < 0.015f || !isValidSample(out_val)) {
        dst[idx] = make_float2(0.0f, 0.0f);
        return;
    }

    dst[idx] = make_float2(out_val, out_cov);
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

    float base_step = 0.55f;
    int max_steps = (int)fminf((tmax - tmin) / base_step, 720.0f);

    const float lx = 0.34f;
    const float ly = -0.22f;
    const float lz = 0.91f;
    const float eps = 1.2f / VOL_XY;

    float3 accum = {0.0f, 0.0f, 0.0f};
    float alpha = 0.0f;
    float threshold = productThreshold(product, dbz_min);
    float t = tmin;
    int step = 0;

    while (t <= tmax && step < max_steps && alpha < 0.995f) {
        step++;

        float sx = cam_x + dx * t;
        float sy = cam_y + dy * t;
        float sz = cam_z + dz * t;
        float tx = sx / VOL_RANGE_KM * 0.5f + 0.5f;
        float ty = sy / VOL_RANGE_KM * 0.5f + 0.5f;
        float tz = (sz / VOL_Z_EXAGGERATION) / VOL_HEIGHT_KM;

        if (tx < 0.002f || tx > 0.998f || ty < 0.002f || ty > 0.998f ||
            tz < 0.002f || tz > 0.998f) {
            t += base_step;
            continue;
        }

        float2 sample = tex3D<float2>(volTex, tx, ty, tz);
        float val = sample.x;
        float coverage = sample.y;
        float density = sampleVolumeDensity(product, val, coverage, threshold);
        if (density < 0.01f) {
            t += base_step * 1.25f;
            continue;
        }

        float gnx = sampleVolumeDensityTex(volTex, tx + eps, ty, tz, product, threshold) -
                    sampleVolumeDensityTex(volTex, tx - eps, ty, tz, product, threshold);
        float gny = sampleVolumeDensityTex(volTex, tx, ty + eps, tz, product, threshold) -
                    sampleVolumeDensityTex(volTex, tx, ty - eps, tz, product, threshold);
        float gnz = sampleVolumeDensityTex(volTex, tx, ty, tz + eps, product, threshold) -
                    sampleVolumeDensityTex(volTex, tx, ty, tz - eps, product, threshold);
        float gl = rsqrtf(gnx * gnx + gny * gny + gnz * gnz + 1e-6f);
        float nx = gnx * gl;
        float ny = gny * gl;
        float nz = gnz * gl;

        float ndotl = fmaxf(0.0f, nx * lx + ny * ly + nz * lz);
        float ambient = 0.18f;

        float shadow = 1.0f;
        float stx = tx;
        float sty = ty;
        float stz = tz;
        float sl_dx = lx * eps * 4.0f;
        float sl_dy = ly * eps * 4.0f;
        float sl_dz = lz * (1.0f / VOL_Z) * 4.0f;
        for (int si = 0; si < 6; si++) {
            stx += sl_dx;
            sty += sl_dy;
            stz += sl_dz;
            if (stx < 0.0f || stx > 1.0f || sty < 0.0f || sty > 1.0f || stz < 0.0f || stz > 1.0f)
                break;
            float shadow_density = sampleVolumeDensityTex(volTex, stx, sty, stz, product, threshold);
            shadow *= expf(-shadow_density * 0.45f);
        }
        shadow = fmaxf(shadow, 0.18f);

        float hx = lx - dx;
        float hy = ly - dy;
        float hz = lz - dz;
        float hl = rsqrtf(hx * hx + hy * hy + hz * hz + 1e-6f);
        float ndoth = fmaxf(0.0f, nx * hx * hl + ny * hy * hl + nz * hz * hl);
        float specular = powf(ndoth, 28.0f) * 0.22f * shadow;

        float ndotv = fabsf(nx * (-dx) + ny * (-dy) + nz * (-dz));
        float rim = powf(1.0f - ndotv, 2.5f) * (0.08f + 0.22f * density);
        float powder = 1.0f - expf(-density * 1.8f);
        float lighting = ambient + 0.42f * ndotl * shadow + 0.25f * powder + rim;

        uint32_t color = c_colorTable[product][colorIndexForValue(product, val)];
        float cr = (float)(color & 0xFF) / 255.0f;
        float cg = (float)((color >> 8) & 0xFF) / 255.0f;
        float cb = (float)((color >> 16) & 0xFF) / 255.0f;

        float luminance = (cr + cg + cb) / 3.0f;
        float saturation = 0.65f + density * 0.35f;
        cr = lerpFloat(luminance, cr, saturation);
        cg = lerpFloat(luminance, cg, saturation);
        cb = lerpFloat(luminance, cb, saturation);

        cr = cr * lighting + specular;
        cg = cg * lighting + specular * 0.90f;
        cb = cb * lighting + specular * 0.80f;

        if (product == PROD_REF) {
            float glow = fmaxf(0.0f, (val - 50.0f) / 20.0f);
            float core = glow * glow;
            cr += core * 0.35f;
            cg += core * 0.10f;
            cb += core * 0.05f;
        }

        float forward = powf(clamp01(lx * (-dx) + ly * (-dy) + lz * (-dz)), 10.0f) * density * 0.12f;
        cr += forward;
        cg += forward * 0.8f;
        cb += forward * 0.6f;

        float extinction = density * ((product == PROD_REF) ? 1.6f : 1.2f);
        float opacity = 1.0f - expf(-extinction * base_step);

        accum.x += (1.0f - alpha) * fminf(cr, 1.5f) * opacity;
        accum.y += (1.0f - alpha) * fminf(cg, 1.5f) * opacity;
        accum.z += (1.0f - alpha) * fminf(cb, 1.5f) * opacity;
        alpha += (1.0f - alpha) * opacity;
        t += base_step;
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

    size_t vol_size = (size_t)VOL_XY * VOL_XY * VOL_Z * sizeof(float2);
    CUDA_CHECK(cudaMalloc(&d_volume_raw, vol_size));
    CUDA_CHECK(cudaMalloc(&d_volume_scratch, vol_size));

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();
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
    if (d_volume_scratch) {
        cudaFree(d_volume_scratch);
        d_volume_scratch = nullptr;
    }
    s_volumeReady = false;
}

void buildVolume(int station_idx, int product,
                 const GpuStationInfo* sweep_infos, int num_sweeps,
                 const float* const* d_azimuths_per_sweep,
                 const uint16_t* const* d_gates_per_sweep) {
    (void)station_idx;
    s_volumeReady = false;
    if (num_sweeps <= 0 || !d_volume_raw || !d_volume_scratch) return;

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
    buildVolumeKernel<<<grid, block>>>(d_volume_raw, product);
    for (int pass = 0; pass < kVolumeSmoothPasses; ++pass) {
        smoothVolumeKernel<<<grid, block>>>(
            (pass & 1) ? d_volume_scratch : d_volume_raw,
            (pass & 1) ? d_volume_raw : d_volume_scratch,
            product);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy3DParms copy_params = {};
    copy_params.srcPtr = make_cudaPitchedPtr(d_volume_raw, VOL_XY * sizeof(float2), VOL_XY, VOL_XY);
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
        0.62f, width, height, product, dbz_min, d_output);
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
