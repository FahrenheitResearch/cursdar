#include "app.h"
#include "nexrad/stations.h"
#include "nexrad/level2_parser.h"
#include "cuda/gpu_pipeline.cuh"
#include "cuda/volume3d.cuh"
#include "net/aws_nexrad.h"
#include <cstdio>
#include <cstring>
#include <algorithm>

App::App() {}

App::~App() {
    if (m_downloader) m_downloader->shutdown();
    if (m_d_compositeOutput) cudaFree(m_d_compositeOutput);
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

        // Store gate params but DON'T transpose on CPU - GPU will do it
        for (const auto& moment : sweep.radials[0].moments) {
            int p = moment.product_index;
            if (p < 0 || p >= NUM_PRODUCTS) continue;
            auto& pd = pc.products[p];
            pd.has_data = true;
            pd.num_gates = moment.num_gates;
            pd.first_gate_km = moment.first_gate_m / 1000.0f;
            pd.gate_spacing_km = moment.gate_spacing_m / 1000.0f;
            pd.scale = moment.scale;
            pd.offset = moment.offset;

            // GPU transposition: pack raw gate data in radial-major order
            // (just a flat copy, no transposition - GPU kernel will transpose)
            int ng = pd.num_gates, nr = pc.num_radials;
            pd.gates.resize((size_t)ng * nr, 0);

            // Upload in radial-major temporarily, GPU will transpose
            for (int r = 0; r < nr; r++) {
                for (const auto& mom : sweep.radials[r].moments) {
                    if (mom.product_index != p) continue;
                    int gc = std::min((int)mom.gates.size(), ng);
                    // Store gate-major for now (GPU expects this)
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

void App::uploadStation(int stationIdx) {
    auto& st = m_stations[stationIdx];
    if (st.precomputed.empty()) return;

    // Filter sweeps by active product - only show tilts that have this product
    int productTilts = countProductSweeps(st.precomputed, m_activeProduct);
    if (productTilts > m_maxTilts) m_maxTilts = productTilts;

    int sweepIdx = findProductSweep(st.precomputed, m_activeProduct, m_activeTilt);
    auto& pc = st.precomputed[sweepIdx];
    if (pc.num_radials == 0) return;

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
    for (int i = 0; i < (int)m_stations.size(); i++)
        infos[i] = m_stations[i].gpuInfo;

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

        if (m_historicMode) {
            gpu::forwardRenderStation(gpuVp, 0,
                                      m_activeProduct, m_dbzMinThreshold,
                                      m_d_compositeOutput);
        } else if (m_mode3D && m_volumeBuilt) {
            // 3D volumetric ray march
            gpu::renderVolume(m_camera, gpuVp.width, gpuVp.height,
                              m_activeProduct, m_dbzMinThreshold,
                              m_d_compositeOutput);
        } else if (m_showAll) {
            // Mosaic: all stations
            if (m_gridDirty) buildSpatialGrid();
            std::vector<GpuStationInfo> gpuInfos(m_stations.size());
            for (int i = 0; i < (int)m_stations.size(); i++)
                gpuInfos[i] = m_stations[i].gpuInfo;
            gpu::renderNative(gpuVp, gpuInfos.data(), (int)m_stations.size(),
                              m_spatialGrid, m_activeProduct, m_dbzMinThreshold,
                              m_d_compositeOutput);
        } else if (m_activeStationIdx >= 0) {
            // Single station: fast path
            gpu::forwardRenderStation(gpuVp, m_activeStationIdx,
                                      m_activeProduct, m_dbzMinThreshold,
                                      m_d_compositeOutput);
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
            size_t xsSz = (size_t)m_xsWidth * m_xsHeight * sizeof(uint32_t);
            if (!m_d_xsOutput) {
                CUDA_CHECK(cudaMalloc(&m_d_xsOutput, xsSz));
            }
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

    m_outputTex.resize(w, h);
    m_needsComposite = true;
}

void App::setProduct(int p) {
    if (p < 0 || p >= (int)Product::COUNT) return;
    if (p == m_activeProduct) return;
    m_activeProduct = p;
    m_activeTilt = 0; // reset tilt - different products have different valid tilts
    m_lastHistoricFrame = -1; // force re-upload in historic mode
    m_needsRerender = true;

    // Re-upload current station with new product's sweeps
    if (m_historicMode) {
        // historic re-upload handled by update loop
    } else if (m_activeStationIdx >= 0) {
        uploadStation(m_activeStationIdx);
    }
}

void App::nextProduct() { setProduct((m_activeProduct + 1) % (int)Product::COUNT); }
void App::prevProduct() { setProduct((m_activeProduct - 1 + (int)Product::COUNT) % (int)Product::COUNT); }

void App::setTilt(int t) {
    if (t < 0) t = 0;
    if (t >= m_maxTilts) t = m_maxTilts - 1;
    if (t == m_activeTilt) return;
    m_activeTilt = t;

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

        // Allocate cross-section output buffer
        if (!m_d_xsOutput) {
            CUDA_CHECK(cudaMalloc(&m_d_xsOutput,
                        (size_t)m_windowWidth * (m_windowHeight / 3) * sizeof(uint32_t)));
        }

        if (m_historicMode) {
            // Force re-upload of current historic frame, which will build the volume
            m_lastHistoricFrame = -1;
        } else {
            // Build volume from live station data
            if (!m_volumeBuilt || m_volumeStation != m_activeStationIdx)
                toggle3D();
            m_mode3D = false;
            m_crossSection = true;
        }
    }
}

void App::toggle3D() {
    m_mode3D = !m_mode3D;
    m_showAll = false;
    if (m_mode3D) {
        // Build 3D volume for active station using all tilts
        if (m_activeStationIdx >= 0 && m_activeStationIdx < (int)m_stations.size()) {
            auto& st = m_stations[m_activeStationIdx];
            if (!st.precomputed.empty()) {
                int ns = (int)st.precomputed.size();
                std::vector<GpuStationInfo> sweepInfos(ns);
                std::vector<const float*> azPtrs(ns);
                std::vector<const uint16_t*> gatePtrs(ns);

                // Upload all sweeps for this station
                for (int s = 0; s < ns; s++) {
                    auto& pc = st.precomputed[s];
                    // We need to upload each sweep's data temporarily
                    // Reuse station slots starting at a high index
                    int slot = 200 + s; // temp GPU slots
                    if (slot >= MAX_STATIONS) break;

                    GpuStationInfo info = {};
                    info.lat = st.gpuInfo.lat;
                    info.lon = st.gpuInfo.lon;
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
                    sweepInfos[s] = info;

                    gpu::allocateStation(slot, info);
                    const uint16_t* gp[NUM_PRODUCTS] = {};
                    for (int p = 0; p < NUM_PRODUCTS; p++)
                        if (pc.products[p].has_data && !pc.products[p].gates.empty())
                            gp[p] = pc.products[p].gates.data();
                    gpu::uploadStationData(slot, info, pc.azimuths.data(), gp);
                    gpu::syncStation(slot);

                    azPtrs[s] = gpu::getStationAzimuths(slot);
                    gatePtrs[s] = gpu::getStationGates(slot, m_activeProduct);
                }

                gpu::buildVolume(m_activeStationIdx, m_activeProduct,
                                  sweepInfos.data(), ns,
                                  azPtrs.data(), gatePtrs.data());
                m_volumeBuilt = true;
                m_volumeStation = m_activeStationIdx;
            }
        }
    }
}

void App::rerenderAll() {
    m_needsRerender = true;
}

void App::loadHistoricEvent(int idx) {
    m_historicMode = true;
    m_lastHistoricFrame = -1;
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
    int sweepIdx = findProductSweep(fr->sweeps, m_activeProduct, m_activeTilt);
    auto& pc = fr->sweeps[sweepIdx];
    if (pc.num_radials == 0) return;

    int productTilts = countProductSweeps(fr->sweeps, m_activeProduct);
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
        m_stations[0].gpuInfo.lat = fr->station_lat;
        m_stations[0].gpuInfo.lon = fr->station_lon;
    }
    m_activeStationIdx = 0;
    m_activeTiltAngle = pc.elevation_angle;
    if ((int)fr->sweeps.size() > m_maxTilts)
        m_maxTilts = (int)fr->sweeps.size();

    // If cross-section is active, rebuild 3D volume from this frame's ALL sweeps
    if (m_crossSection) {
        int ns = (int)fr->sweeps.size();
        std::vector<GpuStationInfo> sweepInfos(ns);
        std::vector<const float*> azPtrs(ns);
        std::vector<const uint16_t*> gatePtrs2(ns);

        for (int s = 0; s < ns && s < 30; s++) {
            auto& spc = fr->sweeps[s];
            int tempSlot = 200 + s;
            if (tempSlot >= MAX_STATIONS) break;

            GpuStationInfo si = {};
            si.lat = fr->station_lat;
            si.lon = fr->station_lon;
            si.elevation_angle = spc.elevation_angle;
            si.num_radials = spc.num_radials;
            for (int p = 0; p < NUM_PRODUCTS; p++) {
                auto& pd = spc.products[p];
                if (!pd.has_data) continue;
                si.has_product[p] = true;
                si.num_gates[p] = pd.num_gates;
                si.first_gate_km[p] = pd.first_gate_km;
                si.gate_spacing_km[p] = pd.gate_spacing_km;
                si.scale[p] = pd.scale;
                si.offset[p] = pd.offset;
            }
            sweepInfos[s] = si;

            gpu::allocateStation(tempSlot, si);
            const uint16_t* gp[NUM_PRODUCTS] = {};
            for (int p = 0; p < NUM_PRODUCTS; p++)
                if (spc.products[p].has_data && !spc.products[p].gates.empty())
                    gp[p] = spc.products[p].gates.data();
            gpu::uploadStationData(tempSlot, si, spc.azimuths.data(), gp);
            gpu::syncStation(tempSlot);

            azPtrs[s] = gpu::getStationAzimuths(tempSlot);
            gatePtrs2[s] = gpu::getStationGates(tempSlot, m_activeProduct);
        }

        gpu::buildVolume(0, m_activeProduct,
                          sweepInfos.data(), ns,
                          azPtrs.data(), gatePtrs2.data());
        m_volumeBuilt = true;
    }
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
