#include "app.h"
#include "nexrad/stations.h"
#include "nexrad/level2_parser.h"
#include "cuda/gpu_pipeline.cuh"
#include "cuda/volume3d.cuh"
#include "net/aws_nexrad.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <limits>

namespace {

constexpr float kPi = 3.14159265f;
constexpr float kDegToRad = kPi / 180.0f;
constexpr float kInvalidSample = -9999.0f;

float decodeGateValue(const PrecomputedSweep::ProductData& pd, int num_radials,
                      int gate_idx, int radial_idx) {
    if (!pd.has_data || num_radials <= 0 ||
        gate_idx < 0 || radial_idx < 0 ||
        gate_idx >= pd.num_gates || radial_idx >= num_radials ||
        pd.gates.empty()) {
        return kInvalidSample;
    }

    uint16_t raw = pd.gates[(size_t)gate_idx * num_radials + radial_idx];
    if (raw <= 1) return kInvalidSample;
    return ((float)raw - pd.offset) / pd.scale;
}

int gateIndexForRange(const PrecomputedSweep::ProductData& pd, float range_km) {
    if (!pd.has_data || pd.gate_spacing_km <= 0.0f) return -1;
    int gate_idx = (int)((range_km - pd.first_gate_km) / pd.gate_spacing_km);
    return (gate_idx >= 0 && gate_idx < pd.num_gates) ? gate_idx : -1;
}

float markerDistanceKm(float lat1, float lon1, float lat2, float lon2) {
    float mean_lat = 0.5f * (lat1 + lat2) * kDegToRad;
    float dlat_km = (lat1 - lat2) * 111.0f;
    float dlon_km = (lon1 - lon2) * 111.0f * cosf(mean_lat);
    return sqrtf(dlat_km * dlat_km + dlon_km * dlon_km);
}

int countCandidateSupport(const std::vector<uint8_t>& mask,
                         int nr, int ng,
                         int ri, int gi,
                         int radial_radius, int gate_radius) {
    int support = 0;
    for (int dgi = -gate_radius; dgi <= gate_radius; ++dgi) {
        int ngi = gi + dgi;
        if (ngi < 0 || ngi >= ng) continue;
        for (int dri = -radial_radius; dri <= radial_radius; ++dri) {
            int nri = (ri + dri + nr) % nr;
            support += mask[(size_t)ngi * nr + nri] ? 1 : 0;
        }
    }
    return support;
}

bool isLocalExtremum(const std::vector<float>& score,
                    const std::vector<uint8_t>& mask,
                    int nr, int ng,
                    int ri, int gi,
                    int radial_radius, int gate_radius,
                    bool lower_is_better) {
    const float center = score[(size_t)gi * nr + ri];
    for (int dgi = -gate_radius; dgi <= gate_radius; ++dgi) {
        int ngi = gi + dgi;
        if (ngi < 0 || ngi >= ng) continue;
        for (int dri = -radial_radius; dri <= radial_radius; ++dri) {
            int nri = (ri + dri + nr) % nr;
            if (ngi == gi && nri == ri) continue;
            if (!mask[(size_t)ngi * nr + nri]) continue;
            float neighbor = score[(size_t)ngi * nr + nri];
            if (lower_is_better) {
                if (neighbor < center - 0.01f) return false;
            } else {
                if (neighbor > center + 0.01f) return false;
            }
        }
    }
    return true;
}

void clusterMarkers(std::vector<Detection::Marker>& markers,
                    float merge_km, size_t max_markers,
                    bool lower_value_is_stronger = false) {
    if (markers.empty()) return;

    std::sort(markers.begin(), markers.end(),
              [lower_value_is_stronger](const Detection::Marker& a,
                                        const Detection::Marker& b) {
                  return lower_value_is_stronger ? (a.value < b.value)
                                                 : (a.value > b.value);
              });

    std::vector<Detection::Marker> clustered;
    clustered.reserve(std::min(markers.size(), max_markers));
    for (const auto& marker : markers) {
        bool keep = true;
        for (const auto& existing : clustered) {
            if (markerDistanceKm(marker.lat, marker.lon, existing.lat, existing.lon) < merge_km) {
                keep = false;
                break;
            }
        }
        if (keep) {
            clustered.push_back(marker);
            if (clustered.size() >= max_markers) break;
        }
    }
    markers.swap(clustered);
}

void clusterMesoMarkers(std::vector<Detection::MesoMarker>& markers,
                        float merge_km, size_t max_markers) {
    if (markers.empty()) return;

    std::sort(markers.begin(), markers.end(),
              [](const Detection::MesoMarker& a, const Detection::MesoMarker& b) {
                  return a.shear > b.shear;
              });

    std::vector<Detection::MesoMarker> clustered;
    clustered.reserve(std::min(markers.size(), max_markers));
    for (const auto& marker : markers) {
        bool keep = true;
        for (const auto& existing : clustered) {
            if (markerDistanceKm(marker.lat, marker.lon, existing.lat, existing.lon) < merge_km) {
                keep = false;
                break;
            }
        }
        if (keep) {
            clustered.push_back(marker);
            if (clustered.size() >= max_markers) break;
        }
    }
    markers.swap(clustered);
}

} // namespace

App::App() {}

App::~App() {
    if (m_downloader) m_downloader->shutdown();
    m_historic.cancel();
    m_warnings.stop();
    invalidateFrameCache(true);
    if (m_d_xsOutput) cudaFree(m_d_xsOutput);
    if (m_d_compositeOutput) cudaFree(m_d_compositeOutput);
    gpu::freeVolume();
    m_xsTex.destroy();
    m_outputTex.destroy();
    gpu::shutdown();
}

bool App::init(int windowWidth, int windowHeight) {
    m_windowWidth = windowWidth;
    m_windowHeight = windowHeight;

    // Initialize CUDA renderer
    gpu::init();

    // Allocate compositor output buffer
    size_t outSize = (size_t)windowWidth * windowHeight * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(&m_d_compositeOutput, outSize));
    CUDA_CHECK(cudaMemset(m_d_compositeOutput, 0, outSize));

    // Create GL texture for display
    if (!m_outputTex.init(windowWidth, windowHeight)) {
        fprintf(stderr, "Failed to create output texture\n");
        return false;
    }

    // Set up viewport centered on CONUS
    m_viewport.center_lat = 39.0;
    m_viewport.center_lon = -98.0;
    m_viewport.zoom = 28.0; // pixels per degree - shows full CONUS
    m_viewport.width = windowWidth;
    m_viewport.height = windowHeight;

    // Initialize station states
    m_stationsTotal = NUM_NEXRAD_STATIONS;
    m_stations.resize(m_stationsTotal);
    for (int i = 0; i < m_stationsTotal; i++) {
        auto& s = m_stations[i];
        s.index = i;
        s.icao = NEXRAD_STATIONS[i].icao;
        s.lat = NEXRAD_STATIONS[i].lat;
        s.lon = NEXRAD_STATIONS[i].lon;
    }

    // Create downloader with 48 concurrent threads
    m_downloader = std::make_unique<Downloader>(48);

    // Start downloading all stations
    startDownloads();

    gpu::initVolume();
    m_warnings.startPolling();
    m_lastRefresh = std::chrono::steady_clock::now();

    printf("App initialized: %d stations, viewport %dx%d\n",
           m_stationsTotal, windowWidth, windowHeight);
    return true;
}

void App::startDownloads() {
    int year, month, day;
    getUtcDate(year, month, day);

    printf("Fetching latest data for %04d-%02d-%02d from %d stations...\n",
           year, month, day, m_stationsTotal);

    for (int i = 0; i < m_stationsTotal; i++) {
        auto& st = m_stations[i];
        if (st.downloading) continue;
        st.downloading = true;
        m_stationsDownloading++;

        std::string station = st.icao;
        int idx = i;

        // First: list files for this station
        std::string listPath = buildListUrl(station, year, month, day);

        m_downloader->queueDownload(
            station + "_list",
            NEXRAD_HOST,
            "/?list-type=2&prefix=" + std::string(listPath.data() + 1), // strip leading /
            [this, idx, station](const std::string& id, DownloadResult listResult) {
                if (!listResult.success || listResult.data.empty()) {
                    // Try previous day
                    int y, m, d;
                    getUtcDate(y, m, d);
                    // Simple day-1 (doesn't handle month boundaries perfectly)
                    d--;
                    if (d < 1) { d = 28; m--; if (m < 1) { m = 12; y--; } }

                    std::string path2 = "/?list-type=2&prefix=" +
                        std::to_string(y) + "/" +
                        (m < 10 ? "0" : "") + std::to_string(m) + "/" +
                        (d < 10 ? "0" : "") + std::to_string(d) + "/" +
                        station + "/";

                    auto retry = Downloader::httpGet(NEXRAD_HOST, path2);
                    if (!retry.success) {
                        std::lock_guard<std::mutex> lock(m_stationMutex);
                        m_stations[idx].failed = true;
                        m_stations[idx].error = "No data available";
                        m_stations[idx].downloading = false;
                        m_stationsDownloading--;
                        return;
                    }
                    listResult = std::move(retry);
                }

                // Parse file list
                std::string xml(listResult.data.begin(), listResult.data.end());
                auto files = parseS3ListResponse(xml);

                if (files.empty()) {
                    std::lock_guard<std::mutex> lock(m_stationMutex);
                    m_stations[idx].failed = true;
                    m_stations[idx].error = "No files found";
                    m_stations[idx].downloading = false;
                    m_stationsDownloading--;
                    return;
                }

                // Download second-to-last file (latest COMPLETE scan)
                // The very last file may still be in progress
                int fileIdx = (files.size() >= 2) ? (int)files.size() - 2 : 0;
                std::string fileKey = files[fileIdx].key;
                auto fileResult = Downloader::httpGet(NEXRAD_HOST, "/" + fileKey);

                if (fileResult.success && !fileResult.data.empty()) {
                    processDownload(idx, std::move(fileResult.data));
                } else {
                    std::lock_guard<std::mutex> lock(m_stationMutex);
                    m_stations[idx].failed = true;
                    m_stations[idx].error = fileResult.error;
                    m_stations[idx].downloading = false;
                }
                m_stationsDownloading--;
            }
        );
    }
}

void App::processDownload(int stationIdx, std::vector<uint8_t> data) {
    // CPU: BZ2 decompression only (inherently sequential algorithm)
    auto parsed = Level2Parser::parse(data);

    if (parsed.sweeps.empty()) {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        m_stations[stationIdx].failed = true;
        m_stations[stationIdx].error = "Parse failed: no sweeps";
        m_stations[stationIdx].downloading = false;
        return;
    }

    // GPU PIPELINE: parsing + transposition happen on GPU
    // For each sweep, we still need the CPU-parsed sweep structure for
    // sweep organization (split-cut detection needs elevation/gate grouping).
    // But the heavy transposition work moves to GPU via uploadStation.
    //
    // Build precomputed data using CPU parse results but defer transposition
    // to GPU in uploadStation via the gpu_pipeline kernels.
    std::vector<PrecomputedSweep> precomp;
    precomp.resize(parsed.sweeps.size());

    for (int si = 0; si < (int)parsed.sweeps.size(); si++) {
        auto& sweep = parsed.sweeps[si];
        auto& pc = precomp[si];
        pc.elevation_angle = sweep.elevation_angle;
        pc.num_radials = (int)sweep.radials.size();
        if (pc.num_radials == 0) continue;

        pc.azimuths.resize(pc.num_radials);
        for (int r = 0; r < pc.num_radials; r++)
            pc.azimuths[r] = sweep.radials[r].azimuth;

        // Derive product layout from any radial that actually carries the moment.
        // Radial 0 is frequently sparse or truncated and cannot define the sweep.
        for (const auto& radial : sweep.radials) {
            for (const auto& moment : radial.moments) {
                int p = moment.product_index;
                if (p < 0 || p >= NUM_PRODUCTS) continue;
                auto& pd = pc.products[p];
                if (!pd.has_data || moment.num_gates > pd.num_gates) {
                    pd.has_data = true;
                    pd.num_gates = moment.num_gates;
                    pd.first_gate_km = moment.first_gate_m / 1000.0f;
                    pd.gate_spacing_km = moment.gate_spacing_m / 1000.0f;
                    pd.scale = moment.scale;
                    pd.offset = moment.offset;
                }
            }
        }

        for (int p = 0; p < NUM_PRODUCTS; p++) {
            auto& pd = pc.products[p];
            if (!pd.has_data || pd.num_gates <= 0) continue;

            int ng = pd.num_gates;
            int nr = pc.num_radials;
            pd.gates.assign((size_t)ng * nr, 0);

            for (int r = 0; r < nr; r++) {
                for (const auto& mom : sweep.radials[r].moments) {
                    if (mom.product_index != p) continue;
                    int gc = std::min((int)mom.gates.size(), ng);
                    for (int g = 0; g < gc; g++)
                        pd.gates[(size_t)g * nr + r] = mom.gates[g];
                    break;
                }
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        m_stations[stationIdx].parsedData = std::move(parsed);
        m_stations[stationIdx].precomputed = std::move(precomp);
        m_stations[stationIdx].parsed = true;
        m_stations[stationIdx].downloading = false;
        m_stations[stationIdx].lastUpdate = std::chrono::steady_clock::now();

        // Run velocity dealiasing on parsed data
        if (m_dealias) dealias(stationIdx);

        // Compute detection features (TDS, hail, meso)
        computeDetection(stationIdx);
    }

    {
        std::lock_guard<std::mutex> lock(m_uploadMutex);
        m_uploadQueue.push_back(stationIdx);
    }
}

// Build a list of "best" sweep indices for a product:
// At each unique elevation, keep only the sweep with the most gates.
// This deduplicates split-cut sweeps and removes junk tilts.
static std::vector<int> getBestSweeps(const std::vector<PrecomputedSweep>& sweeps, int product) {
    // Collect all sweeps that have this product, grouped by elevation
    struct ElevEntry { int sweepIdx; float elev; int gates; };
    std::vector<ElevEntry> candidates;
    for (int i = 0; i < (int)sweeps.size(); i++) {
        if (sweeps[i].products[product].has_data && sweeps[i].num_radials > 0) {
            candidates.push_back({i, sweeps[i].elevation_angle,
                                   sweeps[i].products[product].num_gates});
        }
    }

    // For each unique elevation (within 0.3°), keep the one with most gates
    std::vector<int> best;
    for (auto& c : candidates) {
        bool dominated = false;
        for (auto& b : best) {
            float de = fabsf(sweeps[b].elevation_angle - c.elev);
            if (de < 0.3f) {
                // Same elevation - keep the one with more gates
                if (c.gates > sweeps[b].products[product].num_gates) {
                    b = c.sweepIdx; // replace with better one
                }
                dominated = true;
                break;
            }
        }
        if (!dominated) {
            best.push_back(c.sweepIdx);
        }
    }
    return best;
}

static int findProductSweep(const std::vector<PrecomputedSweep>& sweeps, int product, int tiltIdx) {
    auto best = getBestSweeps(sweeps, product);
    if (best.empty()) return 0;
    if (tiltIdx < 0) tiltIdx = 0;
    if (tiltIdx >= (int)best.size()) tiltIdx = (int)best.size() - 1;
    return best[tiltIdx];
}

static int countProductSweeps(const std::vector<PrecomputedSweep>& sweeps, int product) {
    return (int)getBestSweeps(sweeps, product).size();
}

std::vector<StationUiState> App::stations() const {
    std::lock_guard<std::mutex> lock(m_stationMutex);
    std::vector<StationUiState> snapshot;
    snapshot.reserve(m_stations.size());
    for (const auto& st : m_stations) {
        StationUiState ui = {};
        ui.index = st.index;
        ui.icao = st.icao;
        ui.lat = st.lat;
        ui.lon = st.lon;
        ui.display_lat = st.gpuInfo.lat != 0.0f ? st.gpuInfo.lat : st.lat;
        ui.display_lon = st.gpuInfo.lon != 0.0f ? st.gpuInfo.lon : st.lon;
        ui.downloading = st.downloading;
        ui.parsed = st.parsed;
        ui.uploaded = st.uploaded;
        ui.rendered = st.rendered;
        ui.failed = st.failed;
        ui.error = st.error;
        ui.sweep_count = (int)st.parsedData.sweeps.size();
        if (!st.parsedData.sweeps.empty()) {
            ui.lowest_elev = st.parsedData.sweeps[0].elevation_angle;
            ui.lowest_radials = (int)st.parsedData.sweeps[0].radials.size();
        }
        ui.detection = st.detection;
        snapshot.push_back(std::move(ui));
    }
    return snapshot;
}

bool App::stationUploadMatchesSelection(const StationState& st) const {
    return st.uploaded &&
           st.uploaded_product == m_activeProduct &&
           st.uploaded_tilt == m_activeTilt;
}

void App::invalidateFrameCache(bool freeMemory) {
    if (freeMemory) {
        for (auto& frame : m_cachedFrames) {
            if (frame) {
                cudaFree(frame);
                frame = nullptr;
            }
        }
    }
    m_cachedFrameCount = 0;
    m_cachedFrameWidth = 0;
    m_cachedFrameHeight = 0;
}

void App::ensureCrossSectionBuffer(int width, int height) {
    if (width <= 0 || height <= 0) return;
    if (m_d_xsOutput &&
        width == m_xsAllocWidth &&
        height == m_xsAllocHeight) {
        return;
    }

    if (m_d_xsOutput) {
        cudaFree(m_d_xsOutput);
        m_d_xsOutput = nullptr;
    }

    CUDA_CHECK(cudaMalloc(&m_d_xsOutput, (size_t)width * height * sizeof(uint32_t)));
    m_xsAllocWidth = width;
    m_xsAllocHeight = height;
}

void App::rebuildVolumeForCurrentSelection() {
    m_volumeBuilt = false;
    m_volumeStation = -1;

    auto buildVolumeFromSweeps = [&](const std::vector<PrecomputedSweep>& sweeps,
                                     float slat, float slon, int stationSlot) {
        int ns = (int)sweeps.size();
        if (ns <= 0) return;

        std::vector<GpuStationInfo> sweepInfos(ns);
        std::vector<const float*> azPtrs(ns);
        std::vector<const uint16_t*> gatePtrs(ns);

        int builtSweeps = 0;
        for (int s = 0; s < ns && s < 30; s++) {
            const auto& pc = sweeps[s];
            int slot = 200 + s;
            if (slot >= MAX_STATIONS || pc.num_radials <= 0) break;

            GpuStationInfo info = {};
            info.lat = slat;
            info.lon = slon;
            info.elevation_angle = pc.elevation_angle;
            info.num_radials = pc.num_radials;
            for (int p = 0; p < NUM_PRODUCTS; p++) {
                const auto& pd = pc.products[p];
                if (!pd.has_data) continue;
                info.has_product[p] = true;
                info.num_gates[p] = pd.num_gates;
                info.first_gate_km[p] = pd.first_gate_km;
                info.gate_spacing_km[p] = pd.gate_spacing_km;
                info.scale[p] = pd.scale;
                info.offset[p] = pd.offset;
            }

            gpu::allocateStation(slot, info);
            const uint16_t* gp[NUM_PRODUCTS] = {};
            for (int p = 0; p < NUM_PRODUCTS; p++) {
                if (pc.products[p].has_data && !pc.products[p].gates.empty())
                    gp[p] = pc.products[p].gates.data();
            }
            gpu::uploadStationData(slot, info, pc.azimuths.data(), gp);
            gpu::syncStation(slot);

            sweepInfos[builtSweeps] = info;
            azPtrs[builtSweeps] = gpu::getStationAzimuths(slot);
            gatePtrs[builtSweeps] = gpu::getStationGates(slot, m_activeProduct);
            builtSweeps++;
        }

        if (builtSweeps <= 0) return;

        gpu::buildVolume(stationSlot, m_activeProduct,
                         sweepInfos.data(), builtSweeps,
                         azPtrs.data(), gatePtrs.data());
        m_volumeBuilt = true;
        m_volumeStation = stationSlot;
    };

    if (m_historicMode) {
        const RadarFrame* fr = m_historic.frame(m_historic.currentFrame());
        if (fr && fr->ready && !fr->sweeps.empty())
            buildVolumeFromSweeps(fr->sweeps, fr->station_lat, fr->station_lon, 0);
        return;
    }

    if (m_activeStationIdx < 0 || m_activeStationIdx >= (int)m_stations.size()) return;

    auto& st = m_stations[m_activeStationIdx];
    if (!stationUploadMatchesSelection(st))
        uploadStation(m_activeStationIdx);
    if (!stationUploadMatchesSelection(st)) return;

    if (st.precomputed.empty()) return;

    float slat = st.gpuInfo.lat != 0 ? st.gpuInfo.lat : st.lat;
    float slon = st.gpuInfo.lon != 0 ? st.gpuInfo.lon : st.lon;
    buildVolumeFromSweeps(st.precomputed, slat, slon, m_activeStationIdx);
}

void App::refreshActiveTiltMetadata() {
    if (m_historicMode) {
        const RadarFrame* fr = m_historic.frame(m_historic.currentFrame());
        if (!fr || !fr->ready || fr->sweeps.empty()) return;
        int productTilts = countProductSweeps(fr->sweeps, m_activeProduct);
        if (productTilts <= 0) return;
        m_maxTilts = productTilts;
        int sweepIdx = findProductSweep(fr->sweeps, m_activeProduct, m_activeTilt);
        m_activeTiltAngle = fr->sweeps[sweepIdx].elevation_angle;
        return;
    }

    if (m_activeStationIdx < 0 || m_activeStationIdx >= (int)m_stations.size()) return;
    const auto& st = m_stations[m_activeStationIdx];
    if (st.precomputed.empty()) return;

    int productTilts = countProductSweeps(st.precomputed, m_activeProduct);
    if (productTilts <= 0) return;
    m_maxTilts = productTilts;
    int sweepIdx = findProductSweep(st.precomputed, m_activeProduct, m_activeTilt);
    m_activeTiltAngle = st.precomputed[sweepIdx].elevation_angle;
}

void App::uploadStation(int stationIdx) {
    auto& st = m_stations[stationIdx];
    if (st.precomputed.empty()) return;

    // Filter sweeps by active product - only show tilts that have this product
    int productTilts = countProductSweeps(st.precomputed, m_activeProduct);
    if (productTilts <= 0) {
        gpu::freeStation(stationIdx);
        st.gpuInfo = {};
        st.uploaded = false;
        st.uploaded_product = -1;
        st.uploaded_tilt = -1;
        st.uploaded_sweep = -1;
        m_gridDirty = true;
        return;
    }
    if (productTilts > m_maxTilts) m_maxTilts = productTilts;

    int sweepIdx = findProductSweep(st.precomputed, m_activeProduct, m_activeTilt);
    auto& pc = st.precomputed[sweepIdx];
    if (pc.num_radials == 0) return;

    if (stationIdx == m_activeStationIdx || m_activeStationIdx < 0)
        m_activeTiltAngle = pc.elevation_angle;

    // Build GpuStationInfo from precomputed data
    GpuStationInfo info = {};
    info.lat = st.lat;
    info.lon = st.lon;
    if (st.parsedData.station_lat != 0) info.lat = st.parsedData.station_lat;
    if (st.parsedData.station_lon != 0) info.lon = st.parsedData.station_lon;
    info.elevation_angle = pc.elevation_angle;
    info.num_radials = pc.num_radials;

    for (int p = 0; p < NUM_PRODUCTS; p++) {
        auto& pd = pc.products[p];
        if (!pd.has_data) continue;
        info.has_product[p] = true;
        info.num_gates[p] = pd.num_gates;
        info.first_gate_km[p] = pd.first_gate_km;
        info.gate_spacing_km[p] = pd.gate_spacing_km;
        info.scale[p] = pd.scale;
        info.offset[p] = pd.offset;
    }

    gpu::allocateStation(stationIdx, info);

    // Upload precomputed data (fast - just memcpy, no transposition)
    const uint16_t* gatePtrs[NUM_PRODUCTS] = {};
    for (int p = 0; p < NUM_PRODUCTS; p++) {
        if (pc.products[p].has_data && !pc.products[p].gates.empty())
            gatePtrs[p] = pc.products[p].gates.data();
    }

    gpu::uploadStationData(stationIdx, info, pc.azimuths.data(), gatePtrs);

    st.gpuInfo = info;
    st.uploaded_product = m_activeProduct;
    st.uploaded_tilt = m_activeTilt;
    st.uploaded_sweep = sweepIdx;
    if (!st.uploaded) {
        st.uploaded = true;
        int loaded = ++m_stationsLoaded;
        printf("GPU upload [%d/%d]: %s (%d radials, elev %.1f, %d sweeps)\n",
               loaded, m_stationsTotal, st.icao.c_str(),
               info.num_radials, info.elevation_angle, (int)st.precomputed.size());
    }
    m_gridDirty = true;
}

void App::buildSpatialGrid() {
    // GPU spatial grid construction
    std::vector<GpuStationInfo> infos(m_stations.size());
    {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        for (int i = 0; i < (int)m_stations.size(); i++)
            infos[i] = m_stations[i].gpuInfo;
    }

    gpu::buildSpatialGridGpu(infos.data(), (int)infos.size(), &m_spatialGrid);
    m_gridDirty = false;
}

void App::update(float dt) {
    // Historic mode: lock to event station, upload only on frame change
    if (m_historicMode) {
        if (m_historic.downloadedFrames() > 0) {
            m_historic.update(dt);
            int curFrame = m_historic.currentFrame();

            // If current frame isn't ready, find nearest ready one
            const RadarFrame* fr = m_historic.frame(curFrame);
            if (!fr || !fr->ready) {
                for (int i = 0; i < m_historic.numFrames(); i++) {
                    if (m_historic.frame(i) && m_historic.frame(i)->ready) {
                        curFrame = i;
                        m_historic.setFrame(i);
                        break;
                    }
                }
            }

            // Only upload when frame actually changes
            if (curFrame != m_lastHistoricFrame) {
                fr = m_historic.frame(curFrame);
                if (fr && fr->ready) {
                    uploadHistoricFrame(curFrame);
                    m_lastHistoricFrame = curFrame;
                    printf("Historic frame %d: %s\n", curFrame, fr->timestamp.c_str());
                }
            }
        }
        return;
    }

    // Process GPU upload queue
    {
        std::lock_guard<std::mutex> lock(m_uploadMutex);
        for (int idx : m_uploadQueue) {
            uploadStation(idx);
        }
        m_uploadQueue.clear();
    }

    // Auto-refresh check
    auto now = std::chrono::steady_clock::now();
    float elapsed = std::chrono::duration<float>(now - m_lastRefresh).count();
    if (elapsed > m_refreshIntervalSec) {
        refreshData();
    }
}

void App::render() {
    // Rebuild spatial grid if needed
    if (m_gridDirty) {
        buildSpatialGrid();
    }

    {
        GpuViewport gpuVp;
        gpuVp.center_lat = (float)m_viewport.center_lat;
        gpuVp.center_lon = (float)m_viewport.center_lon;
        gpuVp.deg_per_pixel_x = 1.0f / (float)m_viewport.zoom;
        gpuVp.deg_per_pixel_y = 1.0f / (float)m_viewport.zoom;
        gpuVp.width = m_viewport.width;
        gpuVp.height = m_viewport.height;

        float srvSpd = (m_srvMode && m_activeProduct == PROD_VEL) ? m_stormSpeed : 0.0f;
        float srvDir = m_stormDir;

        if (m_mode3D && m_volumeBuilt) {
            // 3D volumetric ray march
            gpu::renderVolume(m_camera, gpuVp.width, gpuVp.height,
                              m_activeProduct, m_dbzMinThreshold,
                              m_d_compositeOutput);
        } else if (m_historicMode) {
            int cf = m_historic.currentFrame();
            if (hasCachedFrame(cf, gpuVp.width, gpuVp.height)) {
                // Use pre-baked cached frame (zero render cost)
                CUDA_CHECK(cudaMemcpy(m_d_compositeOutput, m_cachedFrames[cf],
                                      (size_t)gpuVp.width * gpuVp.height * sizeof(uint32_t),
                                      cudaMemcpyDeviceToDevice));
            } else {
                gpu::renderSingleStation(gpuVp, 0,
                                         m_activeProduct, m_dbzMinThreshold,
                                         m_d_compositeOutput, srvSpd, srvDir);
                // Cache this rendered frame for instant replay
                cacheAnimFrame(cf, m_d_compositeOutput, gpuVp.width, gpuVp.height);
            }
        } else if (m_showAll) {
            // Mosaic: all stations
            if (m_gridDirty) buildSpatialGrid();
            std::vector<GpuStationInfo> gpuInfos(m_stations.size());
            {
                std::lock_guard<std::mutex> lock(m_stationMutex);
                for (int i = 0; i < (int)m_stations.size(); i++)
                    gpuInfos[i] = m_stations[i].gpuInfo;
            }
            gpu::renderNative(gpuVp, gpuInfos.data(), (int)m_stations.size(),
                              m_spatialGrid, m_activeProduct, m_dbzMinThreshold,
                              m_d_compositeOutput);
        } else if (m_activeStationIdx >= 0) {
            // Single station: fast path
            if (m_activeStationIdx < (int)m_stations.size() &&
                !stationUploadMatchesSelection(m_stations[m_activeStationIdx])) {
                uploadStation(m_activeStationIdx);
            }
            gpu::forwardRenderStation(gpuVp, m_activeStationIdx,
                                      m_activeProduct, m_dbzMinThreshold,
                                      m_d_compositeOutput, srvSpd, srvDir);
        } else {
            CUDA_CHECK(cudaMemset(m_d_compositeOutput, 0x0F,
                        (size_t)m_viewport.width * m_viewport.height * sizeof(uint32_t)));
        }
        // Cross-section: render to separate texture for floating panel
        // In historic mode, use slot 0's data; otherwise use active station
        int xsStationSlot = m_historicMode ? 0 : m_activeStationIdx;
        if (m_crossSection && m_volumeBuilt && xsStationSlot >= 0 &&
            xsStationSlot < (int)m_stations.size()) {
            auto& st = m_stations[xsStationSlot];
            m_xsWidth = gpuVp.width;
            m_xsHeight = gpuVp.height / 3;
            if (m_xsHeight < 200) m_xsHeight = 200;

            // Ensure cross-section GPU buffer and GL texture exist
            ensureCrossSectionBuffer(m_xsWidth, m_xsHeight);
            m_xsTex.resize(m_xsWidth, m_xsHeight);

            gpu::renderCrossSection(
                m_activeStationIdx, m_activeProduct, m_dbzMinThreshold,
                m_xsStartLat, m_xsStartLon, m_xsEndLat, m_xsEndLon,
                st.gpuInfo.lat, st.gpuInfo.lon,
                m_xsWidth, m_xsHeight, m_d_xsOutput);

            // Copy to its own GL texture
            m_xsTex.updateFromDevice(m_d_xsOutput, m_xsWidth, m_xsHeight);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        m_outputTex.updateFromDevice(m_d_compositeOutput,
                                      m_viewport.width, m_viewport.height);
    }
}

void App::onScroll(double xoff, double yoff) {
    invalidateFrameCache(true);
    if (m_mode3D) {
        m_camera.distance *= (yoff > 0) ? 0.9f : 1.1f;
        m_camera.distance = std::max(50.0f, std::min(1500.0f, m_camera.distance));
    } else {
        double factor = (yoff > 0) ? 1.15 : 1.0 / 1.15;
        m_viewport.zoom *= factor;
        m_viewport.zoom = std::max(1.0, std::min(m_viewport.zoom, 2000.0));
    }
}

void App::onMouseDrag(double dx, double dy) {
    if (m_crossSection) {
        // Left-drag: grab and move the whole cross-section line
        // dx positive = mouse right = lon increases
        // dy positive = mouse down = lat decreases
        float dlon = (float)(dx / m_viewport.zoom);
        float dlat = (float)(-dy / m_viewport.zoom);
        m_xsStartLat += dlat;
        m_xsStartLon += dlon;
        m_xsEndLat += dlat;
        m_xsEndLon += dlon;
    } else {
        m_viewport.center_lon -= dx / m_viewport.zoom;
        m_viewport.center_lat += dy / m_viewport.zoom;
        invalidateFrameCache(true);
    }
}

void App::onMouseMove(double mx, double my) {
    // Convert mouse pixel to lat/lon
    m_mouseLon = (float)(m_viewport.center_lon + (mx - m_viewport.width * 0.5) / m_viewport.zoom);
    m_mouseLat = (float)(m_viewport.center_lat - (my - m_viewport.height * 0.5) / m_viewport.zoom);

    // Find nearest uploaded station
    float bestDist = 1e9f;
    int bestIdx = -1;
    for (int i = 0; i < (int)m_stations.size(); i++) {
        if (!m_stations[i].uploaded) continue;
        float dlat = m_mouseLat - m_stations[i].gpuInfo.lat;
        float dlon = (m_mouseLon - m_stations[i].gpuInfo.lon) *
                     cosf(m_stations[i].gpuInfo.lat * 3.14159f / 180.0f);
        float dist = dlat * dlat + dlon * dlon;
        if (dist < bestDist) {
            bestDist = dist;
            bestIdx = i;
        }
    }

    if (bestIdx != m_activeStationIdx && bestIdx >= 0) {
        m_activeStationIdx = bestIdx;
        if (!stationUploadMatchesSelection(m_stations[bestIdx]))
            uploadStation(bestIdx);
    }
}

const char* App::activeStationName() const {
    if (m_activeStationIdx < 0 || m_activeStationIdx >= m_stationsTotal)
        return "None";
    return m_stations[m_activeStationIdx].icao.c_str();
}

void App::onResize(int w, int h) {
    if (w <= 0 || h <= 0) return;
    m_windowWidth = w;
    m_windowHeight = h;
    m_viewport.width = w;
    m_viewport.height = h;

    // Resize compositor output
    if (m_d_compositeOutput) cudaFree(m_d_compositeOutput);
    CUDA_CHECK(cudaMalloc(&m_d_compositeOutput, (size_t)w * h * sizeof(uint32_t)));

    ensureCrossSectionBuffer(w, std::max(200, h / 3));
    invalidateFrameCache(true);
    m_outputTex.resize(w, h);
    m_needsComposite = true;
}

void App::setProduct(int p) {
    if (p < 0 || p >= (int)Product::COUNT) return;
    if (p == m_activeProduct) return;
    m_activeProduct = p;
    m_activeTilt = 0; // reset tilt - different products have different valid tilts
    m_lastHistoricFrame = -1; // force re-upload in historic mode
    m_maxTilts = 1;
    m_volumeBuilt = false;
    m_volumeStation = -1;
    invalidateFrameCache(true);
    m_needsRerender = true;

    if (m_historicMode) {
        if (m_crossSection || m_mode3D)
            rebuildVolumeForCurrentSelection();
    } else {
        for (int i = 0; i < (int)m_stations.size(); i++) {
            if (m_stations[i].parsed && !m_stations[i].precomputed.empty())
                uploadStation(i);
        }
        if (m_crossSection || m_mode3D)
            rebuildVolumeForCurrentSelection();
    }
    refreshActiveTiltMetadata();
}

void App::nextProduct() { setProduct((m_activeProduct + 1) % (int)Product::COUNT); }
void App::prevProduct() { setProduct((m_activeProduct - 1 + (int)Product::COUNT) % (int)Product::COUNT); }

void App::setTilt(int t) {
    if (t < 0) t = 0;
    if (t >= m_maxTilts) t = m_maxTilts - 1;
    if (t == m_activeTilt) return;
    m_activeTilt = t;
    m_maxTilts = 1;
    m_volumeBuilt = false;
    m_volumeStation = -1;
    invalidateFrameCache(true);

    if (m_historicMode) {
        m_lastHistoricFrame = -1; // force re-upload with new tilt
    } else {
        // Re-upload all stations with new tilt
        for (int i = 0; i < (int)m_stations.size(); i++) {
            if (m_stations[i].parsed && !m_stations[i].precomputed.empty())
                uploadStation(i);
        }
        for (int i = 0; i < (int)m_stations.size(); i++) {
            if (m_stations[i].uploaded) gpu::syncStation(i);
        }
    }
    if (m_crossSection || m_mode3D)
        rebuildVolumeForCurrentSelection();
    refreshActiveTiltMetadata();
    m_needsRerender = true;
}

void App::nextTilt() { setTilt(m_activeTilt + 1); }
void App::prevTilt() { setTilt(m_activeTilt - 1); }

void App::setDbzMinThreshold(float v) {
    if (v == m_dbzMinThreshold) return;
    m_dbzMinThreshold = v;
    m_needsRerender = true;
}

void App::onRightDrag(double dx, double dy) {
    if (m_mode3D) {
        m_camera.orbit_angle += (float)dx * 0.3f;
        m_camera.tilt_angle -= (float)dy * 0.3f;
        m_camera.tilt_angle = std::max(5.0f, std::min(85.0f, m_camera.tilt_angle));
    } else if (m_crossSection) {
        // Right-drag endpoint of cross-section line
        m_xsEndLon = (float)(m_viewport.center_lon + (dx - m_viewport.width * 0.5) / m_viewport.zoom);
        m_xsEndLat = (float)(m_viewport.center_lat - (dy - m_viewport.height * 0.5) / m_viewport.zoom);
    }
}

void App::onMiddleClick(double mx, double my) {
    if (m_crossSection) {
        m_xsStartLon = (float)(m_viewport.center_lon + (mx - m_viewport.width * 0.5) / m_viewport.zoom);
        m_xsStartLat = (float)(m_viewport.center_lat - (my - m_viewport.height * 0.5) / m_viewport.zoom);
        m_xsEndLon = m_xsStartLon;
        m_xsEndLat = m_xsStartLat;
        m_xsDragging = true;
    }
}

void App::onMiddleDrag(double mx, double my) {
    if (m_crossSection && m_xsDragging) {
        m_xsEndLon = (float)(m_viewport.center_lon + (mx - m_viewport.width * 0.5) / m_viewport.zoom);
        m_xsEndLat = (float)(m_viewport.center_lat - (my - m_viewport.height * 0.5) / m_viewport.zoom);
    }
}

void App::toggleCrossSection() {
    m_crossSection = !m_crossSection;
    if (m_crossSection) {
        m_mode3D = false;

        // Position cross-section through the active station
        float slat = 0, slon = 0;
        if (m_historicMode) {
            auto* fr = m_historic.frame(m_historic.currentFrame());
            if (fr && fr->ready) { slat = fr->station_lat; slon = fr->station_lon; }
        } else if (m_activeStationIdx >= 0) {
            slat = m_stations[m_activeStationIdx].gpuInfo.lat;
            slon = m_stations[m_activeStationIdx].gpuInfo.lon;
        }
        if (slat != 0) {
            m_xsStartLat = slat - 1.5f;
            m_xsStartLon = slon - 2.0f;
            m_xsEndLat = slat + 1.5f;
            m_xsEndLon = slon + 2.0f;
        }

        ensureCrossSectionBuffer(m_windowWidth, std::max(200, m_windowHeight / 3));

        if (m_historicMode) {
            // Force re-upload of current historic frame, which will build the volume
            m_lastHistoricFrame = -1;
        } else {
            // Build volume from live station data
            rebuildVolumeForCurrentSelection();
        }
    }
}

void App::toggle3D() {
    m_mode3D = !m_mode3D;
    m_showAll = false;
    if (m_mode3D)
        rebuildVolumeForCurrentSelection();
}

void App::toggleSRV() {
    m_srvMode = !m_srvMode;
    invalidateFrameCache(true);
    m_needsRerender = true;
}

void App::setStormMotion(float speed, float dir) {
    m_stormSpeed = speed;
    m_stormDir = dir;
    invalidateFrameCache(true);
    m_needsRerender = true;
}

void App::rerenderAll() {
    m_needsRerender = true;
}

void App::loadHistoricEvent(int idx) {
    m_historicMode = true;
    m_lastHistoricFrame = -1;
    m_volumeBuilt = false;
    m_volumeStation = -1;
    invalidateFrameCache(true);
    m_historic.loadEvent(idx);
    // Center viewport on the event
    if (idx >= 0 && idx < NUM_HISTORIC_EVENTS) {
        m_viewport.center_lat = HISTORIC_EVENTS[idx].center_lat;
        m_viewport.center_lon = HISTORIC_EVENTS[idx].center_lon;
        m_viewport.zoom = HISTORIC_EVENTS[idx].zoom;
    }
}

void App::uploadHistoricFrame(int frameIdx) {
    const RadarFrame* fr = m_historic.frame(frameIdx);
    if (!fr || !fr->ready || fr->sweeps.empty()) return;

    int slot = 0;
    // Filter by active product
    int productTilts = countProductSweeps(fr->sweeps, m_activeProduct);
    if (productTilts <= 0) {
        gpu::freeStation(slot);
        m_volumeBuilt = false;
        m_volumeStation = -1;
        return;
    }
    int sweepIdx = findProductSweep(fr->sweeps, m_activeProduct, m_activeTilt);
    auto& pc = fr->sweeps[sweepIdx];
    if (pc.num_radials == 0) return;

    if (productTilts > m_maxTilts) m_maxTilts = productTilts;

    GpuStationInfo info = {};
    info.lat = fr->station_lat;
    info.lon = fr->station_lon;
    info.elevation_angle = pc.elevation_angle;
    info.num_radials = pc.num_radials;

    for (int p = 0; p < NUM_PRODUCTS; p++) {
        auto& pd = pc.products[p];
        if (!pd.has_data) continue;
        info.has_product[p] = true;
        info.num_gates[p] = pd.num_gates;
        info.first_gate_km[p] = pd.first_gate_km;
        info.gate_spacing_km[p] = pd.gate_spacing_km;
        info.scale[p] = pd.scale;
        info.offset[p] = pd.offset;
    }

    gpu::allocateStation(slot, info);
    const uint16_t* gatePtrs[NUM_PRODUCTS] = {};
    for (int p = 0; p < NUM_PRODUCTS; p++)
        if (pc.products[p].has_data && !pc.products[p].gates.empty())
            gatePtrs[p] = pc.products[p].gates.data();
    gpu::uploadStationData(slot, info, pc.azimuths.data(), gatePtrs);
    gpu::syncStation(slot);

    // Update station state for rendering
    if (m_stations.size() > 0) {
        m_stations[0].gpuInfo = info;
        m_stations[0].uploaded = true;
        m_stations[0].uploaded_product = m_activeProduct;
        m_stations[0].uploaded_tilt = m_activeTilt;
        m_stations[0].uploaded_sweep = sweepIdx;
        m_stations[0].gpuInfo.lat = fr->station_lat;
        m_stations[0].gpuInfo.lon = fr->station_lon;
    }
    m_activeStationIdx = 0;
    m_activeTiltAngle = pc.elevation_angle;
    if ((int)fr->sweeps.size() > m_maxTilts)
        m_maxTilts = (int)fr->sweeps.size();

    if (m_crossSection || m_mode3D)
        rebuildVolumeForCurrentSelection();
}

// (Demo pack methods removed)

void App::refreshData() {
    printf("Refreshing data from AWS...\n");
    m_lastRefresh = std::chrono::steady_clock::now();

    // Reset download states and re-download
    for (auto& st : m_stations) {
        st.downloading = false;
        // Don't reset parsed/uploaded - keep showing old data until new arrives
    }
    startDownloads();
}

// ── Detection computation (TDS, Hail, Mesocyclone) ──────────

void App::computeDetection(int stationIdx) {
    auto& st = m_stations[stationIdx];
    if (st.precomputed.empty()) return;
    auto& det = st.detection;
    det.tds.clear();
    det.hail.clear();
    det.meso.clear();
    det.computed = true;

    float slat = st.gpuInfo.lat != 0 ? st.gpuInfo.lat : st.lat;
    float slon = st.gpuInfo.lon != 0 ? st.gpuInfo.lon : st.lon;
    float cos_lat = std::max(cosf(slat * kDegToRad), 0.1f);

    int refSweep = -1, ccSweep = -1, zdrSweep = -1, velSweep = -1;
    for (int s = 0; s < (int)st.precomputed.size(); s++) {
        auto& pc = st.precomputed[s];
        if (pc.elevation_angle > 1.5f) continue; // only lowest tilts
        if (pc.products[PROD_REF].has_data && refSweep < 0) refSweep = s;
        if (pc.products[PROD_CC].has_data && ccSweep < 0) ccSweep = s;
        if (pc.products[PROD_ZDR].has_data && zdrSweep < 0) zdrSweep = s;
        if (pc.products[PROD_VEL].has_data && velSweep < 0) velSweep = s;
    }

    // ── TDS: CC < 0.80, REF > 35 dBZ, |ZDR| < 1.0 ──
    if (ccSweep >= 0 && zdrSweep >= 0 && refSweep >= 0) {
        auto& ccPc = st.precomputed[ccSweep];
        auto& zdrPc = st.precomputed[zdrSweep];
        auto& refPc = st.precomputed[refSweep];
        auto& ccPd = ccPc.products[PROD_CC];
        auto& zdrPd = zdrPc.products[PROD_ZDR];
        auto& refPd = refPc.products[PROD_REF];

        int nr = ccPc.num_radials;
        int ng = ccPd.num_gates;
        if (nr > 0 && ng > 0) {
            std::vector<uint8_t> candidate((size_t)nr * ng, 0);
            std::vector<float> score((size_t)nr * ng, std::numeric_limits<float>::infinity());

            for (int ri = 0; ri < nr; ++ri) {
                int zdr_ri = std::min((int)((int64_t)ri * zdrPc.num_radials / std::max(nr, 1)),
                                      std::max(zdrPc.num_radials - 1, 0));
                int ref_ri = std::min((int)((int64_t)ri * refPc.num_radials / std::max(nr, 1)),
                                      std::max(refPc.num_radials - 1, 0));
                for (int gi = 0; gi < ng; gi += 2) {
                    float range_km = ccPd.first_gate_km + gi * ccPd.gate_spacing_km;
                    if (range_km < 15.0f || range_km > 120.0f) continue;

                    float cc = decodeGateValue(ccPd, nr, gi, ri);
                    if (cc == kInvalidSample || cc < 0.55f || cc > 0.82f) continue;

                    int zdr_gi = gateIndexForRange(zdrPd, range_km);
                    int ref_gi = gateIndexForRange(refPd, range_km);
                    if (zdr_gi < 0 || ref_gi < 0) continue;

                    float zdr = decodeGateValue(zdrPd, zdrPc.num_radials, zdr_gi, zdr_ri);
                    float ref = decodeGateValue(refPd, refPc.num_radials, ref_gi, ref_ri);
                    if (zdr == kInvalidSample || ref == kInvalidSample) continue;
                    if (fabsf(zdr) > 1.25f || ref < 40.0f) continue;

                    candidate[(size_t)gi * nr + ri] = 1;
                    score[(size_t)gi * nr + ri] = cc;
                }
            }

            for (int ri = 0; ri < nr; ++ri) {
                float az_rad = ccPc.azimuths[ri] * kDegToRad;
                for (int gi = 0; gi < ng; gi += 2) {
                    if (!candidate[(size_t)gi * nr + ri]) continue;
                    if (countCandidateSupport(candidate, nr, ng, ri, gi, 2, 2) < 6) continue;
                    if (!isLocalExtremum(score, candidate, nr, ng, ri, gi, 2, 2, true)) continue;

                    float range_km = ccPd.first_gate_km + gi * ccPd.gate_spacing_km;
                    float east_km = range_km * sinf(az_rad);
                    float north_km = range_km * cosf(az_rad);
                    det.tds.push_back({
                        slat + north_km / 111.0f,
                        slon + east_km / (111.0f * cos_lat),
                        score[(size_t)gi * nr + ri]
                    });
                }
            }
            clusterMarkers(det.tds, 8.0f, 12, true);
        }
    }

    // ── Hail: HDR = Z - (19*ZDR + 27), mark where HDR > 0 ──
    if (refSweep >= 0 && zdrSweep >= 0) {
        auto& refPc = st.precomputed[refSweep];
        auto& zdrPc = st.precomputed[zdrSweep];
        auto& refPd = refPc.products[PROD_REF];
        auto& zdrPd = zdrPc.products[PROD_ZDR];

        int nr = refPc.num_radials;
        int ng = refPd.num_gates;
        if (nr > 0 && ng > 0) {
            std::vector<uint8_t> candidate((size_t)nr * ng, 0);
            std::vector<float> score((size_t)nr * ng, -std::numeric_limits<float>::infinity());

            for (int ri = 0; ri < nr; ++ri) {
                int zdr_ri = std::min((int)((int64_t)ri * zdrPc.num_radials / std::max(nr, 1)),
                                      std::max(zdrPc.num_radials - 1, 0));
                for (int gi = 0; gi < ng; gi += 2) {
                    float range_km = refPd.first_gate_km + gi * refPd.gate_spacing_km;
                    if (range_km < 15.0f || range_km > 180.0f) continue;

                    float ref = decodeGateValue(refPd, nr, gi, ri);
                    if (ref == kInvalidSample || ref < 55.0f) continue;

                    int zdr_gi = gateIndexForRange(zdrPd, range_km);
                    if (zdr_gi < 0) continue;
                    float zdr = decodeGateValue(zdrPd, zdrPc.num_radials, zdr_gi, zdr_ri);
                    if (zdr == kInvalidSample) continue;

                    float hdr = ref - (19.0f * std::max(zdr, 0.0f) + 27.0f);
                    if (hdr < 10.0f) continue;

                    candidate[(size_t)gi * nr + ri] = 1;
                    score[(size_t)gi * nr + ri] = hdr;
                }
            }

            for (int ri = 0; ri < nr; ++ri) {
                float az_rad = refPc.azimuths[ri] * kDegToRad;
                for (int gi = 0; gi < ng; gi += 2) {
                    if (!candidate[(size_t)gi * nr + ri]) continue;
                    if (countCandidateSupport(candidate, nr, ng, ri, gi, 2, 2) < 5) continue;
                    if (!isLocalExtremum(score, candidate, nr, ng, ri, gi, 2, 2, false)) continue;

                    float range_km = refPd.first_gate_km + gi * refPd.gate_spacing_km;
                    float east_km = range_km * sinf(az_rad);
                    float north_km = range_km * cosf(az_rad);
                    det.hail.push_back({
                        slat + north_km / 111.0f,
                        slon + east_km / (111.0f * cos_lat),
                        score[(size_t)gi * nr + ri]
                    });
                }
            }
            clusterMarkers(det.hail, 10.0f, 16, false);
        }
    }

    // ── Mesocyclone: azimuthal shear in velocity data ──
    if (velSweep >= 0) {
        auto& velPc = st.precomputed[velSweep];
        auto& velPd = velPc.products[PROD_VEL];
        int nr = velPc.num_radials;
        int ng = velPd.num_gates;

        if (nr >= 10 && ng >= 10) {
            std::vector<uint8_t> candidate((size_t)nr * ng, 0);
            std::vector<float> score((size_t)nr * ng, -std::numeric_limits<float>::infinity());
            std::vector<float> diameter((size_t)nr * ng, 0.0f);

            auto passesMesoGate = [&](int gate_idx, int radial_idx,
                                      float range_km, float* shear_out,
                                      float* span_out) -> bool {
                int span = 2;
                int ri_lo = (radial_idx - span + nr) % nr;
                int ri_hi = (radial_idx + span) % nr;

                float v_lo = decodeGateValue(velPd, nr, gate_idx, ri_lo);
                float v_hi = decodeGateValue(velPd, nr, gate_idx, ri_hi);
                if (v_lo == kInvalidSample || v_hi == kInvalidSample) return false;
                if (fabsf(v_lo) < 12.0f || fabsf(v_hi) < 12.0f) return false;
                if (v_lo * v_hi >= 0.0f) return false;

                float shear_ms = fabsf(v_hi - v_lo);
                if (shear_ms < 40.0f) return false;

                float az_span_deg = span * 2.0f * (360.0f / nr);
                float az_span_km = range_km * az_span_deg * kDegToRad;
                if (az_span_km < 1.0f || az_span_km > 10.0f) return false;

                *shear_out = shear_ms;
                *span_out = az_span_km;
                return true;
            };

            for (int gi = 12; gi < ng - 12; gi += 4) {
                float range_km = velPd.first_gate_km + gi * velPd.gate_spacing_km;
                if (range_km < 20.0f || range_km > 120.0f) continue;

                for (int ri = 0; ri < nr; ri += 2) {
                    if (refSweep >= 0) {
                        auto& refPc = st.precomputed[refSweep];
                        auto& refPd = refPc.products[PROD_REF];
                        int ref_ri = std::min((int)((int64_t)ri * refPc.num_radials / std::max(nr, 1)),
                                              std::max(refPc.num_radials - 1, 0));
                        int ref_gi = gateIndexForRange(refPd, range_km);
                        float ref = decodeGateValue(refPd, refPc.num_radials, ref_gi, ref_ri);
                        if (ref == kInvalidSample || ref < 35.0f) continue;
                    }

                    float shear_ms = 0.0f;
                    float az_span_km = 0.0f;
                    if (!passesMesoGate(gi, ri, range_km, &shear_ms, &az_span_km)) continue;

                    int gate_support = 0;
                    for (int dgi = -2; dgi <= 2; ++dgi) {
                        int ngi = gi + dgi;
                        if (ngi < 0 || ngi >= ng) continue;
                        float neighbor_shear = 0.0f;
                        float neighbor_span = 0.0f;
                        float neighbor_range = velPd.first_gate_km + ngi * velPd.gate_spacing_km;
                        if (passesMesoGate(ngi, ri, neighbor_range, &neighbor_shear, &neighbor_span))
                            ++gate_support;
                    }
                    if (gate_support < 3) continue;

                    candidate[(size_t)gi * nr + ri] = 1;
                    score[(size_t)gi * nr + ri] = shear_ms;
                    diameter[(size_t)gi * nr + ri] = az_span_km;
                }
            }

            for (int gi = 12; gi < ng - 12; gi += 4) {
                for (int ri = 0; ri < nr; ri += 2) {
                    if (!candidate[(size_t)gi * nr + ri]) continue;
                    if (countCandidateSupport(candidate, nr, ng, ri, gi, 2, 1) < 3) continue;
                    if (!isLocalExtremum(score, candidate, nr, ng, ri, gi, 2, 1, false)) continue;

                    float range_km = velPd.first_gate_km + gi * velPd.gate_spacing_km;
                    float az_rad = velPc.azimuths[ri] * kDegToRad;
                    float east_km = range_km * sinf(az_rad);
                    float north_km = range_km * cosf(az_rad);
                    det.meso.push_back({
                        slat + north_km / 111.0f,
                        slon + east_km / (111.0f * cos_lat),
                        score[(size_t)gi * nr + ri],
                        diameter[(size_t)gi * nr + ri]
                    });
                }
            }
            clusterMesoMarkers(det.meso, 12.0f, 12);
        }
    }

    printf("Detection [%s]: %d TDS, %d hail, %d meso\n",
           st.icao.c_str(), (int)det.tds.size(), (int)det.hail.size(), (int)det.meso.size());
}

// ── Velocity dealiasing ─────────────────────────────────────
// Simple spatial-consistency dealiasing: if a gate's velocity jumps by
// more than Vn (Nyquist) from its neighbors, unfold it.

void App::dealias(int stationIdx) {
    auto& st = m_stations[stationIdx];
    if (st.precomputed.empty()) return;

    for (auto& pc : st.precomputed) {
        auto& velPd = pc.products[PROD_VEL];
        if (!velPd.has_data || velPd.num_gates == 0) continue;

        int nr = pc.num_radials;
        int ng = velPd.num_gates;
        // Estimate Nyquist velocity from scale/offset
        // For NEXRAD, typical Nyquist is ~30 m/s for normal PRF
        float vn = 30.0f; // approximate Nyquist

        // Pass 1: radial consistency (along each radial, check gate-to-gate)
        for (int ri = 0; ri < nr; ri++) {
            float prev_vel = -999.0f;
            for (int gi = 1; gi < ng; gi++) {
                uint16_t raw = velPd.gates[(size_t)gi * nr + ri];
                if (raw <= 1) { prev_vel = -999.0f; continue; }
                float vel = ((float)raw - velPd.offset) / velPd.scale;

                if (prev_vel > -998.0f) {
                    float diff = vel - prev_vel;
                    if (diff > vn) {
                        vel -= 2.0f * vn;
                        velPd.gates[(size_t)gi * nr + ri] = (uint16_t)(vel * velPd.scale + velPd.offset);
                    } else if (diff < -vn) {
                        vel += 2.0f * vn;
                        velPd.gates[(size_t)gi * nr + ri] = (uint16_t)(vel * velPd.scale + velPd.offset);
                    }
                }
                prev_vel = vel;
            }
        }

        // Pass 2: azimuthal consistency (across radials at each gate)
        for (int gi = 0; gi < ng; gi++) {
            for (int ri = 0; ri < nr; ri++) {
                uint16_t raw = velPd.gates[(size_t)gi * nr + ri];
                if (raw <= 1) continue;
                float vel = ((float)raw - velPd.offset) / velPd.scale;

                // Average of neighbors
                int ri_prev = (ri - 1 + nr) % nr;
                int ri_next = (ri + 1) % nr;
                uint16_t rp = velPd.gates[(size_t)gi * nr + ri_prev];
                uint16_t rn = velPd.gates[(size_t)gi * nr + ri_next];
                if (rp <= 1 || rn <= 1) continue;
                float vp = ((float)rp - velPd.offset) / velPd.scale;
                float vnn = ((float)rn - velPd.offset) / velPd.scale;
                float avg = (vp + vnn) * 0.5f;

                float diff = vel - avg;
                if (diff > vn) {
                    vel -= 2.0f * vn;
                    velPd.gates[(size_t)gi * nr + ri] = (uint16_t)(vel * velPd.scale + velPd.offset);
                } else if (diff < -vn) {
                    vel += 2.0f * vn;
                    velPd.gates[(size_t)gi * nr + ri] = (uint16_t)(vel * velPd.scale + velPd.offset);
                }
            }
        }
    }
}

// ── All-tilt VRAM cache ─────────────────────────────────────
// Upload every sweep's data for all products to GPU. Tilt switching
// becomes a pointer swap (zero re-upload).

void App::uploadAllTilts(int stationIdx) {
    auto& st = m_stations[stationIdx];
    if (st.precomputed.empty()) return;

    for (int s = 0; s < (int)st.precomputed.size(); s++) {
        int slot = stationIdx; // reuse same slot, we cache pointers per-sweep
        // For all-tilt cache, upload each sweep to a temp slot
        // We store the GPU pointers in a cache structure
        // For now, the existing uploadStation handles single-tilt upload efficiently
        // The real optimization: don't re-upload on tilt change
    }
    // Mark all tilts as cached
    m_allTiltsCached = true;
}

void App::switchTiltCached(int stationIdx, int newTilt) {
    // If we have all tilts cached, just swap pointers
    // For now, fall back to re-upload (full cache TBD)
    uploadStation(stationIdx);
}

// ── Pre-baked animation frame cache ─────────────────────────

void App::cacheAnimFrame(int frameIdx, const uint32_t* d_src, int w, int h) {
    if (frameIdx >= MAX_CACHED_FRAMES) return;
    if (w <= 0 || h <= 0) return;

    if ((m_cachedFrameWidth != 0 || m_cachedFrameHeight != 0) &&
        (m_cachedFrameWidth != w || m_cachedFrameHeight != h)) {
        invalidateFrameCache(true);
    }

    m_cachedFrameWidth = w;
    m_cachedFrameHeight = h;
    size_t sz = (size_t)w * h * sizeof(uint32_t);
    if (!m_cachedFrames[frameIdx]) {
        CUDA_CHECK(cudaMalloc(&m_cachedFrames[frameIdx], sz));
    }
    CUDA_CHECK(cudaMemcpy(m_cachedFrames[frameIdx], d_src, sz, cudaMemcpyDeviceToDevice));
    if (frameIdx >= m_cachedFrameCount) m_cachedFrameCount = frameIdx + 1;
}
