#pragma once
#include "nexrad/level2.h"
#include "nexrad/products.h"
#include "cuda/renderer.cuh"
#include "render/gl_interop.h"
#include "render/projection.h"
#include "cuda/volume3d.cuh"
#include "net/downloader.h"
#include "net/warnings.h"
#include "historic.h"
#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <memory>
#include <chrono>

#include "nexrad/sweep_data.h"

// Detected meteorological features
struct Detection {
    struct Marker { float lat, lon; float value; };
    struct MesoMarker { float lat, lon; float shear; float diameter_km; };
    std::vector<Marker> tds;   // Tornado Debris Signature
    std::vector<Marker> hail;  // Hail (high HDR)
    std::vector<MesoMarker> meso; // Mesocyclone/TVS
    bool computed = false;
};

// Per-station state
struct StationState {
    int          index;
    std::string  icao;
    float        lat, lon;
    bool         downloading = false;
    bool         parsed = false;
    bool         uploaded = false;
    bool         rendered = false;
    bool         failed = false;
    std::string  error;
    ParsedRadarData  parsedData;
    GpuStationInfo   gpuInfo;
    std::vector<PrecomputedSweep> precomputed; // all sweeps, ready for GPU
    std::chrono::steady_clock::time_point lastUpdate;
    Detection detection;
    int uploaded_product = -1;
    int uploaded_tilt = -1;
    int uploaded_sweep = -1;
};

struct StationUiState {
    int          index = -1;
    std::string  icao;
    float        lat = 0.0f, lon = 0.0f;
    float        display_lat = 0.0f, display_lon = 0.0f;
    bool         downloading = false;
    bool         parsed = false;
    bool         uploaded = false;
    bool         rendered = false;
    bool         failed = false;
    std::string  error;
    int          sweep_count = 0;
    float        lowest_elev = 0.0f;
    int          lowest_radials = 0;
    Detection    detection;
};

class App {
public:
    App();
    ~App();

    // Initialize GPU, start downloads
    bool init(int windowWidth, int windowHeight);

    // Main update loop (called each frame)
    void update(float dt);

    // Render the viewport to the GL texture
    void render();

    // Handle input
    void onScroll(double xoff, double yoff);
    void onMouseDrag(double dx, double dy);
    void onMouseMove(double mx, double my);
    void onResize(int w, int h);

    // Active station (nearest to mouse)
    int  activeStation() const { return m_activeStationIdx; }
    const char* activeStationName() const;
    bool showAll() const { return m_showAll; }
    void toggleShowAll() { m_showAll = !m_showAll; m_mode3D = false; }
    bool mode3D() const { return m_mode3D; }
    void toggle3D();
    void toggleCrossSection();
    bool crossSection() const { return m_crossSection; }
    Camera3D& camera() { return m_camera; }
    void onRightDrag(double dx, double dy);
    void onMiddleClick(double mx, double my);
    void onMiddleDrag(double mx, double my);
    float xsStartLat() const { return m_xsStartLat; }
    float xsStartLon() const { return m_xsStartLon; }
    float xsEndLat() const { return m_xsEndLat; }
    float xsEndLon() const { return m_xsEndLon; }
    GlCudaTexture& xsTexture() { return m_xsTex; }
    int xsWidth() const { return m_xsWidth; }
    int xsHeight() const { return m_xsHeight; }

    // Getters for UI
    Viewport&       viewport() { return m_viewport; }
    int             activeProduct() const { return m_activeProduct; }
    void            setProduct(int p);
    int             activeTilt() const { return m_activeTilt; }
    void            setTilt(int t);
    int             maxTilts() const { return m_maxTilts; }
    float           activeTiltAngle() const { return m_activeTiltAngle; }
    float           dbzMinThreshold() const {
        return (m_activeProduct == PROD_VEL) ? m_velocityMinThreshold : m_dbzMinThreshold;
    }
    void            setDbzMinThreshold(float v);
    int             stationsLoaded() const { return m_stationsLoaded.load(); }
    int             stationsTotal() const { return m_stationsTotal; }
    int             stationsDownloading() const { return m_stationsDownloading.load(); }
    GlCudaTexture&  outputTexture() { return m_outputTex; }

    // Navigation: arrow keys
    void nextProduct();
    void prevProduct();
    void nextTilt();
    void prevTilt();

    // Station info for UI
    std::vector<StationUiState> stations() const;

    // Force re-render all stations (e.g., after product change)
    void rerenderAll();

    // Trigger refresh from AWS
    void refreshData();
    void loadMarch302025Snapshot(bool lowestSweepOnly = false);
    bool snapshotMode() const { return m_snapshotMode; }
    const char* snapshotLabel() const { return m_snapshotLabel.c_str(); }
    bool snapshotLowestSweepOnly() const { return m_snapshotLowestSweepOnly; }

private:
    // Start downloading all active stations
    void startDownloads();

    // Process a completed download
    void processDownload(int stationIdx, std::vector<uint8_t> data);

    // Upload parsed data to GPU
    void uploadStation(int stationIdx);

    // Build spatial grid for compositor
    void buildSpatialGrid();
    void invalidateFrameCache(bool freeMemory = false);
    void ensureCrossSectionBuffer(int width, int height);
    void rebuildVolumeForCurrentSelection();
    bool stationUploadMatchesSelection(const StationState& st) const;
    void refreshActiveTiltMetadata();
    void resetStationsForReload();
    void startDownloadsForTimestamp(int year, int month, int day, int hour, int minute);

    Viewport         m_viewport;
    int              m_activeProduct = 0;
    int              m_activeTilt = 0;       // sweep index
    int              m_maxTilts = 1;
    float            m_activeTiltAngle = 0.5f;
    float            m_dbzMinThreshold = 5.0f;
    float            m_velocityMinThreshold = 12.0f;
    bool             m_snapshotMode = false;
    bool             m_snapshotLowestSweepOnly = false;
    std::string      m_snapshotLabel;
    int              m_windowWidth = 1920;
    int              m_windowHeight = 1080;

    // Station data
    std::vector<StationState> m_stations;
    int m_stationsTotal = 0;
    std::atomic<int> m_stationsLoaded{0};
    std::atomic<int> m_stationsDownloading{0};

    // GPU compositor output
    uint32_t*       m_d_compositeOutput = nullptr;
    GlCudaTexture   m_outputTex;

    // Spatial grid for fast station lookup in compositor
    SpatialGrid     m_spatialGrid;
    bool            m_gridDirty = true;

    // Download manager
    std::unique_ptr<Downloader> m_downloader;

    // Mutex for station state updates from download threads
    mutable std::mutex m_stationMutex;

    // Queue of stations ready to upload to GPU (from download threads)
    std::vector<int> m_uploadQueue;
    std::mutex       m_uploadMutex;

    // Active station tracking
    int   m_activeStationIdx = -1;
    float m_mouseLat = 39.0f, m_mouseLon = -98.0f;
    bool  m_showAll = false;
    bool  m_mode3D = false;
    Camera3D m_camera = {32.0f, 24.0f, 440.0f, 54.0f};
    bool  m_volumeBuilt = false;
    int   m_volumeStation = -1;

    // Cross-section mode
    bool  m_crossSection = false;
    float m_xsStartLat = 0, m_xsStartLon = 0;
    float m_xsEndLat = 0, m_xsEndLon = 0;
    bool  m_xsDragging = false;
    uint32_t* m_d_xsOutput = nullptr;
    GlCudaTexture m_xsTex;          // separate GL texture for cross-section panel
    int m_xsWidth = 0, m_xsHeight = 0;
    int m_xsAllocWidth = 0, m_xsAllocHeight = 0;

    // Re-render flag
    bool m_needsRerender = true;
    bool m_needsComposite = true;

    // Auto-refresh timer
    std::chrono::steady_clock::time_point m_lastRefresh;
    float m_refreshIntervalSec = 300.0f;

public:
    // NWS warning overlay
    WarningFetcher m_warnings;

public:
    // Historic event viewer
    HistoricLoader m_historic;
    bool m_historicMode = false;
    int  m_lastHistoricFrame = -1;
    void loadHistoricEvent(int idx);
    void uploadHistoricFrame(int frameIdx);

    // Storm-Relative Velocity mode
    bool  m_srvMode = false;
    float m_stormSpeed = 15.0f;  // m/s
    float m_stormDir = 225.0f;   // degrees from north
    void toggleSRV();
    bool srvMode() const { return m_srvMode; }
    float stormSpeed() const { return m_stormSpeed; }
    float stormDir() const { return m_stormDir; }
    void setStormMotion(float speed, float dir);

    // Detection overlays
    bool m_showTDS = false;
    bool m_showHail = false;
    bool m_showMeso = false;

    // All-tilt VRAM cache
    void uploadAllTilts(int stationIdx);
    void switchTiltCached(int stationIdx, int newTilt);
    bool m_allTiltsCached = false;

    // Detection computation
    void computeDetection(int stationIdx);

    // Velocity dealiasing
    void dealias(int stationIdx);
    bool m_dealias = true;

    // Pre-baked animation frame cache
    static constexpr int MAX_CACHED_FRAMES = 60;
    uint32_t* m_cachedFrames[MAX_CACHED_FRAMES] = {};
    int m_cachedFrameCount = 0;
    int m_cachedFrameWidth = 0;
    int m_cachedFrameHeight = 0;
    void cacheAnimFrame(int frameIdx, const uint32_t* d_src, int w, int h);
    bool hasCachedFrame(int frameIdx, int w, int h) const {
        return frameIdx >= 0 &&
               frameIdx < m_cachedFrameCount &&
               m_cachedFrames[frameIdx] &&
               m_cachedFrameWidth == w &&
               m_cachedFrameHeight == h;
    }
};
