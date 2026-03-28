#include "volume3d.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Volume storage ──────────────────────────────────────────

static float*             d_volume_raw = nullptr;   // linear buffer for building
static cudaArray_t        d_volume_array = nullptr;  // 3D array for texture
static cudaTextureObject_t d_volume_tex = 0;         // HW trilinear texture
static bool               s_volumeReady = false;

extern __constant__ uint32_t c_colorTable[NUM_PRODUCTS][256];

__device__ __host__ static uint32_t mkRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
    return (uint32_t)r | ((uint32_t)g << 8) | ((uint32_t)b << 16) | ((uint32_t)a << 24);
}

// ── Volume build ────────────────────────────────────────────

struct SweepDesc {
    float elevation_deg;
    int   num_radials, num_gates;
    float first_gate_km, gate_spacing_km;
    float scale, offset;
    const float*    azimuths;
    const uint16_t* gates;
};

__constant__ SweepDesc c_sweeps[32];
__constant__ int       c_numSweeps;

__device__ int bsAz(const float* az, int n, float t) {
    int lo = 0, hi = n - 1;
    while (lo < hi) { int m = (lo+hi)>>1; if (az[m] < t) lo = m+1; else hi = m; }
    return lo;
}

__global__ void buildVolumeKernel(float* __restrict__ vol) {
    int vx = blockIdx.x * blockDim.x + threadIdx.x;
    int vy = blockIdx.y * blockDim.y + threadIdx.y;
    int vz = blockIdx.z;
    if (vx >= VOL_XY || vy >= VOL_XY || vz >= VOL_Z) return;

    float x_km = ((float)vx / VOL_XY - 0.5f) * 2.0f * VOL_RANGE_KM;
    float y_km = ((float)vy / VOL_XY - 0.5f) * 2.0f * VOL_RANGE_KM;
    float z_km = ((float)vz / VOL_Z) * VOL_HEIGHT_KM;

    float hr = sqrtf(x_km*x_km + y_km*y_km);
    float az = atan2f(x_km, y_km) * (180.0f / (float)M_PI);
    if (az < 0) az += 360.0f;
    float elev = atan2f(z_km, hr) * (180.0f / (float)M_PI);
    float sr = sqrtf(hr*hr + z_km*z_km);

    // Find two closest sweeps, interpolate between them
    int sw0 = -1, sw1 = -1;
    float d0 = 999, d1 = 999;
    for (int s = 0; s < c_numSweeps; s++) {
        float d = fabsf(c_sweeps[s].elevation_deg - elev);
        if (d < d0) { sw1 = sw0; d1 = d0; sw0 = s; d0 = d; }
        else if (d < d1) { sw1 = s; d1 = d; }
    }

    auto sampleSweep = [&](int si) -> float {
        if (si < 0) return -999.0f;
        const SweepDesc& sw = c_sweeps[si];
        if (!sw.gates || sw.num_radials <= 0 || sw.gate_spacing_km <= 0) return -999.0f;
        float mr = sw.first_gate_km + sw.num_gates * sw.gate_spacing_km;
        if (sr < sw.first_gate_km || sr > mr) return -999.0f;

        int ih = bsAz(sw.azimuths, sw.num_radials, az);
        int il = (ih == 0) ? sw.num_radials - 1 : ih - 1;
        if (ih >= sw.num_radials) ih = 0;

        int gi = (int)((sr - sw.first_gate_km) / sw.gate_spacing_km);
        if (gi < 0 || gi >= sw.num_gates) return -999.0f;

        // Azimuth interpolation
        float azl = sw.azimuths[il], azh = sw.azimuths[ih];
        float daz = azh - azl; if (daz < 0) daz += 360.0f;
        float aoff = az - azl; if (aoff < 0) aoff += 360.0f;
        float ta = (daz > 0.01f) ? aoff / daz : 0.0f;
        ta = fminf(fmaxf(ta, 0.0f), 1.0f);

        uint16_t r0 = sw.gates[gi * sw.num_radials + il];
        uint16_t r1 = sw.gates[gi * sw.num_radials + ih];
        float v0 = (r0 > 1) ? ((float)r0 - sw.offset) / sw.scale : -999.0f;
        float v1 = (r1 > 1) ? ((float)r1 - sw.offset) / sw.scale : -999.0f;

        if (v0 > -998 && v1 > -998) return v0 * (1-ta) + v1 * ta;
        if (v0 > -998) return v0;
        if (v1 > -998) return v1;
        return -999.0f;
    };

    float v0 = sampleSweep(sw0);
    float v1 = sampleSweep(sw1);

    float value = -999.0f;
    if (v0 > -998 && v1 > -998 && d0 + d1 > 0.01f) {
        float w = d0 / (d0 + d1);
        value = v0 * (1.0f - w) + v1 * w;
    } else if (v0 > -998) value = v0;
    else if (v1 > -998) value = v1;

    vol[(size_t)vz * VOL_XY * VOL_XY + vy * VOL_XY + vx] = value;
}

// ── Ray march kernel (SOTA quality) ─────────────────────────

__global__ void rayMarchKernel(
    cudaTextureObject_t volTex,
    float cam_x, float cam_y, float cam_z,
    float fwd_x, float fwd_y, float fwd_z,
    float right_x, float right_y, float right_z,
    float up_x, float up_y, float up_z,
    float fov_scale,
    int width, int height,
    int product, float dbz_min,
    uint32_t* __restrict__ output)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    float u = ((float)px / width - 0.5f) * 2.0f * fov_scale * ((float)width / height);
    float v = (0.5f - (float)py / height) * 2.0f * fov_scale;

    float dx = fwd_x + right_x*u + up_x*v;
    float dy = fwd_y + right_y*u + up_y*v;
    float dz = fwd_z + right_z*u + up_z*v;
    float dl = rsqrtf(dx*dx + dy*dy + dz*dz);
    dx *= dl; dy *= dl; dz *= dl;

    // Ray-box intersection
    float bmin = -VOL_RANGE_KM, bmax = VOL_RANGE_KM, bzmax = VOL_DISPLAY_HEIGHT;
    float tmin = -1e9f, tmax = 1e9f;
    if (fabsf(dx)>1e-6f) { float t1=(bmin-cam_x)/dx, t2=(bmax-cam_x)/dx; if(t1>t2){float t=t1;t1=t2;t2=t;} tmin=fmaxf(tmin,t1); tmax=fminf(tmax,t2); }
    if (fabsf(dy)>1e-6f) { float t1=(bmin-cam_y)/dy, t2=(bmax-cam_y)/dy; if(t1>t2){float t=t1;t1=t2;t2=t;} tmin=fmaxf(tmin,t1); tmax=fminf(tmax,t2); }
    if (fabsf(dz)>1e-6f) { float t1=(0-cam_z)/dz, t2=(bzmax-cam_z)/dz; if(t1>t2){float t=t1;t1=t2;t2=t;} tmin=fmaxf(tmin,t1); tmax=fminf(tmax,t2); }

    // Sky gradient background
    float sky_t = fmaxf(0.0f, v * 0.3f + 0.3f);
    float3 bg = {0.03f + sky_t * 0.04f, 0.03f + sky_t * 0.06f, 0.06f + sky_t * 0.10f};

    // Ground plane hit (z=0 in display space)
    float ground_t = -1.0f;
    if (fabsf(dz) > 1e-6f) {
        ground_t = -cam_z / dz;
    }

    // Draw ground plane with grid
    bool hitGround = false;
    float3 groundColor = bg;
    if (ground_t > 0.0f && (tmin > tmax || ground_t < tmin)) {
        float gx = cam_x + dx * ground_t;
        float gy = cam_y + dy * ground_t;
        // Grid lines every 50km
        float gmod_x = fmodf(fabsf(gx), 50.0f);
        float gmod_y = fmodf(fabsf(gy), 50.0f);
        float line = fminf(gmod_x, 50.0f - gmod_x);
        float line2 = fminf(gmod_y, 50.0f - gmod_y);
        float gridLine = fminf(line, line2);
        float gridAlpha = fmaxf(0.0f, 1.0f - gridLine * 0.8f) * 0.15f;
        // Distance fade
        float gdist = sqrtf(gx*gx + gy*gy);
        float gfade = fmaxf(0.0f, 1.0f - gdist / (VOL_RANGE_KM * 1.5f));
        gridAlpha *= gfade;
        groundColor = {bg.x + gridAlpha * 0.3f, bg.y + gridAlpha * 0.4f, bg.z + gridAlpha * 0.5f};
        hitGround = true;
    }

    if (tmin > tmax || tmax < 0.0f) {
        float3 c = hitGround ? groundColor : bg;
        output[py * width + px] = mkRGBA((uint8_t)(c.x*255), (uint8_t)(c.y*255), (uint8_t)(c.z*255));
        return;
    }
    tmin = fmaxf(tmin, 0.001f);
    if (hitGround && ground_t > 0 && ground_t < tmin) tmin = fmaxf(tmin, 0.001f); // volume is above ground

    // ── Ray march with shadow rays, rim lighting, solid appearance ──
    float base_step = 0.7f;
    int max_steps = (int)fminf((tmax - tmin) / base_step, 600.0f);

    // Light from upper-right
    const float lx = 0.45f, ly = -0.35f, lz = 0.75f;
    const float eps = 1.2f / VOL_XY;

    float3 accum = {0, 0, 0};
    float alpha = 0.0f;

    for (int step = 0; step < max_steps && alpha < 0.995f; step++) {
        float t = tmin + (float)step * base_step;
        if (t > tmax) break;

        float sx = cam_x + dx*t, sy = cam_y + dy*t, sz = cam_z + dz*t;
        float tx = sx / VOL_RANGE_KM * 0.5f + 0.5f;
        float ty = sy / VOL_RANGE_KM * 0.5f + 0.5f;
        float real_z = sz / VOL_Z_EXAGGERATION;
        float tz = real_z / VOL_HEIGHT_KM;

        if (tx < 0.002f || tx > 0.998f || ty < 0.002f || ty > 0.998f ||
            tz < 0.002f || tz > 0.998f) continue;

        float val = tex3D<float>(volTex, tx, ty, tz);
        if (val <= dbz_min) continue;

        // ── Gradient normal ──
        float gnx = tex3D<float>(volTex, tx+eps, ty, tz) - tex3D<float>(volTex, tx-eps, ty, tz);
        float gny = tex3D<float>(volTex, tx, ty+eps, tz) - tex3D<float>(volTex, tx, ty-eps, tz);
        float gnz = tex3D<float>(volTex, tx, ty, tz+eps) - tex3D<float>(volTex, tx, ty, tz-eps);
        float gl = rsqrtf(gnx*gnx + gny*gny + gnz*gnz + 1e-6f);
        float nx = gnx*gl, ny = gny*gl, nz = gnz*gl;

        // ── Lighting ──
        float ndotl = fmaxf(0.0f, nx*lx + ny*ly + nz*lz);
        float ambient = 0.25f;
        float diffuse = 0.55f * ndotl;

        // Shadow ray: march toward light, accumulate occlusion
        float shadow = 1.0f;
        {
            float stx = tx, sty = ty, stz = tz;
            float sl_dx = lx * eps * 3.0f, sl_dy = ly * eps * 3.0f;
            float sl_dz = lz * (1.0f / VOL_Z) * 3.0f;
            for (int si = 0; si < 8; si++) {
                stx += sl_dx; sty += sl_dy; stz += sl_dz;
                if (stx<0||stx>1||sty<0||sty>1||stz<0||stz>1) break;
                float sv = tex3D<float>(volTex, stx, sty, stz);
                if (sv > dbz_min) {
                    float si_norm = fminf((sv - dbz_min) / 30.0f, 1.0f);
                    shadow -= si_norm * 0.15f;
                }
            }
            shadow = fmaxf(shadow, 0.15f);
        }

        // Specular (Blinn-Phong, tighter)
        float hx = lx-dx, hy = ly-dy, hz = lz-dz;
        float hl = rsqrtf(hx*hx+hy*hy+hz*hz+1e-6f);
        float ndoth = fmaxf(0.0f, nx*hx*hl + ny*hy*hl + nz*hz*hl);
        float specular = powf(ndoth, 64.0f) * 0.7f * shadow;

        // Rim/fresnel lighting (edges glow)
        float ndotv = fabsf(nx*(-dx) + ny*(-dy) + nz*(-dz));
        float rim = powf(1.0f - ndotv, 3.0f) * 0.4f;

        float lighting = ambient + (diffuse * shadow) + rim;

        // ── Color ──
        float norm = fminf(fmaxf((val + 30.0f) / 105.0f, 0.0f), 1.0f);
        int cidx = min(max((int)(norm * 254.0f) + 1, 1), 255);
        uint32_t color = c_colorTable[product][cidx];
        float cr = (float)(color & 0xFF) / 255.0f;
        float cg = (float)((color >> 8) & 0xFF) / 255.0f;
        float cb = (float)((color >> 16) & 0xFF) / 255.0f;

        // Apply lighting + specular
        cr = cr * lighting + specular;
        cg = cg * lighting + specular * 0.85f;
        cb = cb * lighting + specular * 0.7f;

        // Core emission (>45 dBZ)
        float glow = fmaxf(0.0f, (val - 45.0f) / 20.0f);
        cr += glow * glow * 0.5f;
        cg += glow * glow * 0.2f;

        // ── Opacity: very steep, nearly isosurface ──
        float intensity = fminf(fmaxf((val - dbz_min) / 20.0f, 0.0f), 1.0f);
        float opacity = intensity * intensity * intensity * intensity * 0.5f;

        // Front-to-back
        accum.x += (1.0f - alpha) * fminf(cr, 1.5f) * opacity;
        accum.y += (1.0f - alpha) * fminf(cg, 1.5f) * opacity;
        accum.z += (1.0f - alpha) * fminf(cb, 1.5f) * opacity;
        alpha += (1.0f - alpha) * opacity;
    }

    float3 final_bg = hitGround ? groundColor : bg;
    float fr = fminf(accum.x + final_bg.x * (1-alpha), 1.0f);
    float fg = fminf(accum.y + final_bg.y * (1-alpha), 1.0f);
    float fb = fminf(accum.z + final_bg.z * (1-alpha), 1.0f);

    output[py * width + px] = mkRGBA((uint8_t)(fr*255), (uint8_t)(fg*255), (uint8_t)(fb*255));
}

// ── Cross-section kernel (flat 2D grid, GR2Analyst style) ───
// NOT a 3D volume slice. For each (distance, height) pixel:
// compute range from station, find the tilt whose beam passes at
// that height at that range, sample the gate. Flat rectangular grid.

constexpr float XS_MAX_HEIGHT_KM = 15.0f;

__global__ void crossSectionKernel(
    float start_x_km, float start_y_km,
    float dir_x, float dir_y,
    float total_dist_km,
    float station_x_km, float station_y_km, // station offset from itself = 0,0
    int width, int height,
    int product, float dbz_min,
    uint32_t* __restrict__ output)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    // Flat grid: X = distance along line, Y = altitude
    float dist_along = ((float)px / width) * total_dist_km;
    float alt_km = (1.0f - (float)py / height) * XS_MAX_HEIGHT_KM;

    // Position along the line in km relative to station
    float x_km = start_x_km + dir_x * dist_along;
    float y_km = start_y_km + dir_y * dist_along;

    // Ground range from station to this point
    float ground_range = sqrtf(x_km * x_km + y_km * y_km);
    if (ground_range < 1.0f) ground_range = 1.0f;

    // Azimuth from station
    float azimuth = atan2f(x_km, y_km) * (180.0f / (float)M_PI);
    if (azimuth < 0) azimuth += 360.0f;

    // Background with grid
    uint32_t bg = mkRGBA(18, 18, 25);
    float hgrid = fmodf(dist_along, 25.0f);
    float vgrid = fmodf(alt_km, 1.524f); // 5kft in km
    if (fminf(hgrid, 25.0f - hgrid) < 0.3f)
        bg = mkRGBA(25, 25, 35);
    if (fminf(vgrid, 1.524f - vgrid) < 0.02f)
        bg = mkRGBA(25, 25, 35);

    // For this (ground_range, alt_km), which tilt's beam is closest?
    // Beam height at range r for elevation e: h = r * sin(e) + r^2/(2*Re)
    // Re = 8494 km (4/3 earth radius for standard refraction)
    const float Re = 8494.0f;

    float best_val = -999.0f;
    float best_dist = 999.0f;

    for (int s = 0; s < c_numSweeps; s++) {
        const SweepDesc& sw = c_sweeps[s];
        if (!sw.gates || sw.num_radials <= 0 || sw.gate_spacing_km <= 0) continue;

        float elev_rad = sw.elevation_deg * (float)M_PI / 180.0f;

        // Beam height at this ground range for this tilt
        float slant_range = ground_range / cosf(elev_rad);
        float beam_h = slant_range * sinf(elev_rad) + (ground_range * ground_range) / (2.0f * Re);

        // How far is this beam from the target altitude?
        float h_diff = fabsf(beam_h - alt_km);

        // Beam width in km at this range (~1 degree)
        float beam_width_km = slant_range * 0.0175f; // ~1 degree in radians
        if (beam_width_km < 0.3f) beam_width_km = 0.3f;

        // Only use if within half beam width
        if (h_diff > beam_width_km * 0.6f) continue;
        if (h_diff >= best_dist) continue;

        // Check range bounds
        float max_r = sw.first_gate_km + sw.num_gates * sw.gate_spacing_km;
        if (slant_range < sw.first_gate_km || slant_range > max_r) continue;

        // Sample: nearest radial, nearest gate
        int ih = bsAz(sw.azimuths, sw.num_radials, azimuth);
        int il = (ih == 0) ? sw.num_radials - 1 : ih - 1;
        if (ih >= sw.num_radials) ih = 0;
        float dl = fabsf(azimuth - sw.azimuths[il]);
        float dh = fabsf(azimuth - sw.azimuths[ih]);
        if (dl > 180) dl = 360 - dl;
        if (dh > 180) dh = 360 - dh;
        int ri = (dl <= dh) ? il : ih;

        int gi = (int)((slant_range - sw.first_gate_km) / sw.gate_spacing_km);
        if (gi < 0 || gi >= sw.num_gates) continue;

        uint16_t raw = sw.gates[gi * sw.num_radials + ri];
        if (raw <= 1) continue;

        float val = ((float)raw - sw.offset) / sw.scale;
        if (val > best_val || h_diff < best_dist) {
            best_val = val;
            best_dist = h_diff;
        }
    }

    if (best_val <= dbz_min) { output[py * width + px] = bg; return; }

    float norm = fminf(fmaxf((best_val + 30.0f) / 105.0f, 0.0f), 1.0f);
    int cidx = min(max((int)(norm * 254.0f) + 1, 1), 255);
    uint32_t color = c_colorTable[product][cidx];
    output[py * width + px] = color | 0xFF000000u;
}

// ── API ─────────────────────────────────────────────────────

namespace gpu {

void initVolume() {
    size_t vol_size = (size_t)VOL_XY * VOL_XY * VOL_Z * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_volume_raw, vol_size));

    // Create 3D CUDA array for texture
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent(VOL_XY, VOL_XY, VOL_Z);
    CUDA_CHECK(cudaMalloc3DArray(&d_volume_array, &desc, extent));

    // Create texture object with trilinear filtering
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_volume_array;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear; // HARDWARE TRILINEAR
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    CUDA_CHECK(cudaCreateTextureObject(&d_volume_tex, &resDesc, &texDesc, nullptr));

    printf("3D volume: %dx%dx%d, HW trilinear texture, %.1f MB\n",
           VOL_XY, VOL_XY, VOL_Z, vol_size / (1024.0f * 1024.0f));
}

void freeVolume() {
    if (d_volume_tex) { cudaDestroyTextureObject(d_volume_tex); d_volume_tex = 0; }
    if (d_volume_array) { cudaFreeArray(d_volume_array); d_volume_array = nullptr; }
    if (d_volume_raw) { cudaFree(d_volume_raw); d_volume_raw = nullptr; }
    s_volumeReady = false;
}

void buildVolume(int station_idx, int product,
                 const GpuStationInfo* sweep_infos, int num_sweeps,
                 const float* const* d_azimuths_per_sweep,
                 const uint16_t* const* d_gates_per_sweep) {
    if (num_sweeps <= 0 || !d_volume_raw) return;

    int count = (num_sweeps > 32) ? 32 : num_sweeps;
    std::vector<SweepDesc> h_sweeps(count);
    for (int s = 0; s < count; s++) {
        h_sweeps[s].elevation_deg = sweep_infos[s].elevation_angle;
        h_sweeps[s].num_radials = sweep_infos[s].num_radials;
        h_sweeps[s].num_gates = sweep_infos[s].num_gates[product];
        h_sweeps[s].first_gate_km = sweep_infos[s].first_gate_km[product];
        h_sweeps[s].gate_spacing_km = sweep_infos[s].gate_spacing_km[product];
        h_sweeps[s].scale = sweep_infos[s].scale[product];
        h_sweeps[s].offset = sweep_infos[s].offset[product];
        h_sweeps[s].azimuths = d_azimuths_per_sweep[s];
        h_sweeps[s].gates = d_gates_per_sweep[s];
    }

    CUDA_CHECK(cudaMemcpyToSymbol(c_sweeps, h_sweeps.data(), count * sizeof(SweepDesc)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_numSweeps, &count, sizeof(int)));

    // Build volume (voxelize)
    dim3 block(8, 8);
    dim3 grid((VOL_XY+7)/8, (VOL_XY+7)/8, VOL_Z);
    buildVolumeKernel<<<grid, block>>>(d_volume_raw);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy to 3D array for texture sampling
    cudaMemcpy3DParms p = {};
    p.srcPtr = make_cudaPitchedPtr(d_volume_raw, VOL_XY * sizeof(float), VOL_XY, VOL_XY);
    p.dstArray = d_volume_array;
    p.extent = make_cudaExtent(VOL_XY, VOL_XY, VOL_Z);
    p.kind = cudaMemcpyDeviceToDevice;
    CUDA_CHECK(cudaMemcpy3D(&p));

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

    float fx = -cx, fy = -cy, fz = cam.target_z - cz;
    float fl = rsqrtf(fx*fx + fy*fy + fz*fz);
    fx *= fl; fy *= fl; fz *= fl;

    float rx = fy*1.0f - fz*0.0f;
    float ry = fz*0.0f - fx*1.0f;
    float rz = fx*0.0f - fy*0.0f;
    float rl = rsqrtf(rx*rx + ry*ry + rz*rz + 1e-8f);
    rx *= rl; ry *= rl; rz *= rl;

    float ux = ry*fz - rz*fy;
    float uy = rz*fx - rx*fz;
    float uz = rx*fy - ry*fx;

    dim3 block(16, 16);
    dim3 grid_dim((width+15)/16, (height+15)/16);

    rayMarchKernel<<<grid_dim, block>>>(
        d_volume_tex,
        cx, cy, cz, fx, fy, fz, rx, ry, rz, ux, uy, uz,
        0.7f, width, height, product, dbz_min, d_output);
    CUDA_CHECK(cudaGetLastError());
}

void renderCrossSection(
    int station_idx, int product, float dbz_min,
    float start_lat, float start_lon, float end_lat, float end_lon,
    float station_lat, float station_lon,
    int width, int height,
    uint32_t* d_output) {
    if (!s_volumeReady) return;

    // Convert lat/lon endpoints to km relative to station
    float cos_lat = cosf(station_lat * (float)M_PI / 180.0f);
    float sx_km = (start_lon - station_lon) * 111.0f * cos_lat;
    float sy_km = (start_lat - station_lat) * 111.0f;
    float ex_km = (end_lon - station_lon) * 111.0f * cos_lat;
    float ey_km = (end_lat - station_lat) * 111.0f;

    float ddx = ex_km - sx_km, ddy = ey_km - sy_km;
    float total = sqrtf(ddx*ddx + ddy*ddy);
    if (total < 1.0f) return;
    float nx = ddx / total, ny = ddy / total;

    dim3 block(16, 16);
    dim3 grid((width+15)/16, (height+15)/16);

    crossSectionKernel<<<grid, block>>>(
        sx_km, sy_km, nx, ny, total,
        0.0f, 0.0f, // station is at origin in its own coordinate system
        width, height, product, dbz_min, d_output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu
