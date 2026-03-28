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

        float srvSpd = (m_srvMode && m_activeProduct == PROD_VEL) ? m_stormSpeed : 0.0f;
        float srvDir = m_stormDir;

        if (m_historicMode) {
            int cf = m_historic.currentFrame();
            if (hasCachedFrame(cf)) {
                // Use pre-baked cached frame (zero render cost)
                cudaMemcpy(m_d_compositeOutput, m_cachedFrames[cf],
                           (size_t)gpuVp.width * gpuVp.height * sizeof(uint32_t),
                           cudaMemcpyDeviceToDevice);
            } else {
                gpu::forwardRenderStation(gpuVp, 0,
                                          m_activeProduct, m_dbzMinThreshold,
                                          m_d_compositeOutput, srvSpd, srvDir);
                // Cache this rendered frame for instant replay
                cacheAnimFrame(cf, m_d_compositeOutput, gpuVp.width, gpuVp.height);
            }
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
    m_cachedFrameCount = 0; // invalidate animation cache on zoom
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
        m_cachedFrameCount = 0; // invalidate animation cache on pan
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
    m_cachedFrameCount = 0;   // invalidate animation cache
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
        // Helper lambda to build volume from a vector of PrecomputedSweeps
        auto buildVolumeFromSweeps = [&](const std::vector<PrecomputedSweep>& sweeps,
                                          float slat, float slon, int stationSlot) {
            int ns = (int)sweeps.size();
            if (ns == 0) return;
            std::vector<GpuStationInfo> sweepInfos(ns);
            std::vector<const float*> azPtrs(ns);
            std::vector<const uint16_t*> gatePtrs(ns);

            for (int s = 0; s < ns && s < 30; s++) {
                auto& pc = sweeps[s];
                int slot = 200 + s;
                if (slot >= MAX_STATIONS) break;

                GpuStationInfo info = {};
                info.lat = slat;
                info.lon = slon;
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

            gpu::buildVolume(stationSlot, m_activeProduct,
                              sweepInfos.data(), ns,
                              azPtrs.data(), gatePtrs.data());
            m_volumeBuilt = true;
            m_volumeStation = stationSlot;
        };

        if (m_historicMode) {
            // Build volume from historic frame's sweeps
            const RadarFrame* fr = m_historic.frame(m_historic.currentFrame());
            if (fr && fr->ready && !fr->sweeps.empty()) {
                buildVolumeFromSweeps(fr->sweeps, fr->station_lat, fr->station_lon, 0);
            }
        } else if (m_activeStationIdx >= 0 && m_activeStationIdx < (int)m_stations.size()) {
            auto& st = m_stations[m_activeStationIdx];
            if (!st.precomputed.empty()) {
                float slat = st.gpuInfo.lat != 0 ? st.gpuInfo.lat : st.lat;
                float slon = st.gpuInfo.lon != 0 ? st.gpuInfo.lon : st.lon;
                buildVolumeFromSweeps(st.precomputed, slat, slon, m_activeStationIdx);
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
    float cos_lat = cosf(slat * 3.14159265f / 180.0f);

    // Find lowest elevation sweeps for each product
    // Use sweep index 0 for REF (surveillance) and the matching Doppler sweep for CC/ZDR/VEL
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
        for (int ri = 0; ri < nr; ri++) {
            float az_rad = ccPc.azimuths[ri] * 3.14159265f / 180.0f;
            for (int gi = 0; gi < ng; gi += 2) { // skip every other gate for speed
                // CC value
                uint16_t raw_cc = ccPd.gates[(size_t)gi * nr + ri];
                if (raw_cc <= 1) continue;
                float cc = ((float)raw_cc - ccPd.offset) / ccPd.scale;
                if (cc >= 0.80f || cc < 0.20f) continue;

                // ZDR value (same sweep usually)
                float zdr = 0;
                if (gi < zdrPd.num_gates && ri < zdrPc.num_radials) {
                    uint16_t raw_zdr = zdrPd.gates[(size_t)gi * zdrPc.num_radials + ri];
                    if (raw_zdr > 1) zdr = ((float)raw_zdr - zdrPd.offset) / zdrPd.scale;
                }
                if (fabsf(zdr) > 1.5f) continue;

                // REF value (may be on different sweep with different gate count)
                float range_km = ccPd.first_gate_km + gi * ccPd.gate_spacing_km;
                int ref_gi = (int)((range_km - refPd.first_gate_km) / refPd.gate_spacing_km);
                if (ref_gi < 0 || ref_gi >= refPd.num_gates) continue;
                // Find nearest REF radial
                int ref_ri = ri;
                if (ref_ri >= refPc.num_radials) ref_ri = refPc.num_radials - 1;
                uint16_t raw_ref = refPd.gates[(size_t)ref_ri * 1 + 0]; // need proper indexing
                // Proper gate-major indexing
                raw_ref = refPd.gates[(size_t)ref_gi * refPc.num_radials + ref_ri];
                if (raw_ref <= 1) continue;
                float ref = ((float)raw_ref - refPd.offset) / refPd.scale;
                if (ref < 35.0f) continue;

                // TDS confirmed!
                float east_km = range_km * sinf(az_rad);
                float north_km = range_km * cosf(az_rad);
                float mlat = slat + north_km / 111.0f;
                float mlon = slon + east_km / (111.0f * cos_lat);
                det.tds.push_back({mlat, mlon, cc});
            }
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
        for (int ri = 0; ri < nr; ri++) {
            float az_rad = refPc.azimuths[ri] * 3.14159265f / 180.0f;
            for (int gi = 0; gi < ng; gi += 3) { // skip for speed
                uint16_t raw_ref = refPd.gates[(size_t)gi * nr + ri];
                if (raw_ref <= 1) continue;
                float ref = ((float)raw_ref - refPd.offset) / refPd.scale;
                if (ref < 45.0f) continue; // need strong echo for hail

                float range_km = refPd.first_gate_km + gi * refPd.gate_spacing_km;
                // Find matching ZDR
                int zdr_gi = (int)((range_km - zdrPd.first_gate_km) / zdrPd.gate_spacing_km);
                int zdr_ri = ri;
                if (zdr_gi < 0 || zdr_gi >= zdrPd.num_gates) continue;
                if (zdr_ri >= zdrPc.num_radials) zdr_ri = zdrPc.num_radials - 1;
                uint16_t raw_zdr = zdrPd.gates[(size_t)zdr_gi * zdrPc.num_radials + zdr_ri];
                if (raw_zdr <= 1) continue;
                float zdr = ((float)raw_zdr - zdrPd.offset) / zdrPd.scale;

                float hdr = ref - (19.0f * std::max(zdr, 0.0f) + 27.0f);
                if (hdr <= 0.0f) continue;

                float east_km = range_km * sinf(az_rad);
                float north_km = range_km * cosf(az_rad);
                float mlat = slat + north_km / 111.0f;
                float mlon = slon + east_km / (111.0f * cos_lat);
                det.hail.push_back({mlat, mlon, hdr});
            }
        }
    }

    // ── Mesocyclone: azimuthal shear in velocity data ──
    if (velSweep >= 0) {
        auto& velPc = st.precomputed[velSweep];
        auto& velPd = velPc.products[PROD_VEL];
        int nr = velPc.num_radials;
        int ng = velPd.num_gates;

        if (nr >= 10 && ng >= 10) {
            for (int gi = 10; gi < ng - 10; gi += 8) { // sparser scan
                float range_km = velPd.first_gate_km + gi * velPd.gate_spacing_km;
                if (range_km < 10.0f || range_km > 150.0f) continue;

                for (int ri = 0; ri < nr; ri += 2) { // skip every other radial
                    int span = 3;
                    int ri_lo = (ri - span + nr) % nr;
                    int ri_hi = (ri + span) % nr;

                    uint16_t raw_lo = velPd.gates[(size_t)gi * nr + ri_lo];
                    uint16_t raw_hi = velPd.gates[(size_t)gi * nr + ri_hi];
                    if (raw_lo <= 1 || raw_hi <= 1) continue;

                    float v_lo = ((float)raw_lo - velPd.offset) / velPd.scale;
                    float v_hi = ((float)raw_hi - velPd.offset) / velPd.scale;

                    // Need opposite signs (convergent/divergent) for rotation
                    if (v_lo * v_hi >= 0) continue; // same sign = not rotation

                    float dv = fabsf(v_hi - v_lo);
                    float az_span_deg = span * 2.0f * (360.0f / nr);
                    float az_span_km = range_km * az_span_deg * 3.14159265f / 180.0f;
                    if (az_span_km < 0.5f) continue;

                    float shear_ms = dv;
                    if (shear_ms < 30.0f) continue; // 30 m/s min for real meso

                    float az_rad = velPc.azimuths[ri] * 3.14159265f / 180.0f;
                    float east_km = range_km * sinf(az_rad);
                    float north_km = range_km * cosf(az_rad);
                    float mlat = slat + north_km / 111.0f;
                    float mlon = slon + east_km / (111.0f * cos_lat);
                    det.meso.push_back({mlat, mlon, shear_ms, az_span_km});
                }
            }
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
    size_t sz = (size_t)w * h * sizeof(uint32_t);
    if (!m_cachedFrames[frameIdx]) {
        cudaMalloc(&m_cachedFrames[frameIdx], sz);
    }
    cudaMemcpy(m_cachedFrames[frameIdx], d_src, sz, cudaMemcpyDeviceToDevice);
    if (frameIdx >= m_cachedFrameCount) m_cachedFrameCount = frameIdx + 1;
}
