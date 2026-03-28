#pragma once
#include "nexrad/level2.h"
#include "nexrad/level2_parser.h"
#include "nexrad/sweep_data.h"
#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <functional>
#include <memory>
#include <thread>

// A single radar frame (one volume scan at one time)
struct RadarFrame {
    std::string filename;
    std::string timestamp; // extracted from filename
    std::vector<PrecomputedSweep> sweeps; // precomputed GPU-ready data (from app.h)
    float station_lat = 0, station_lon = 0;
    bool ready = false;
};

// Preset historic tornado events
struct HistoricEvent {
    const char* name;
    const char* station;
    int year, month, day;
    int start_hour, start_min; // UTC
    int end_hour, end_min;     // UTC
    float center_lat, center_lon;
    float zoom;
    const char* description;
};

inline const HistoricEvent HISTORIC_EVENTS[] = {
    {"Moore, OK EF5", "KTLX", 2013, 5, 20, 19, 0, 21, 30,
     35.33f, -97.49f, 150.0f,
     "May 20, 2013 - EF5 tornado devastated Moore, OK. 24 killed."},

    {"El Reno, OK EF3", "KTLX", 2013, 5, 31, 22, 0, 0, 30,
     35.55f, -98.04f, 120.0f,
     "May 31, 2013 - Widest tornado ever recorded (2.6 mi). 8 killed."},

    {"Joplin, MO EF5", "KSGF", 2011, 5, 22, 21, 0, 23, 30,
     37.08f, -94.51f, 150.0f,
     "May 22, 2011 - EF5 tornado destroyed Joplin. 158 killed."},

    {"Tuscaloosa-Birmingham EF4", "KBMX", 2011, 4, 27, 20, 0, 23, 0,
     33.21f, -87.57f, 100.0f,
     "April 27, 2011 - Super Outbreak. EF4 tore through Tuscaloosa/Birmingham."},

    {"Greensburg, KS EF5", "KDDC", 2007, 5, 5, 1, 0, 4, 0,
     37.60f, -99.29f, 120.0f,
     "May 4, 2007 - EF5 destroyed 95% of Greensburg, KS."},

    {"Hattiesburg, MS EF4", "KDGX", 2013, 2, 10, 19, 0, 22, 0,
     31.33f, -89.29f, 120.0f,
     "Feb 10, 2013 - EF4 hit Hattiesburg and USM campus."},

    {"Mayfield, KY EF4", "KPAH", 2021, 12, 11, 3, 0, 7, 0,
     36.74f, -88.64f, 100.0f,
     "Dec 10-11, 2021 - Long-track EF4, 250+ mi path. 57 killed in KY."},

    {"Rolling Fork, MS EF4", "KDGX", 2023, 3, 25, 1, 0, 4, 0,
     32.91f, -90.88f, 120.0f,
     "March 24, 2023 - Nocturnal EF4 devastated Rolling Fork."},
};

inline constexpr int NUM_HISTORIC_EVENTS = sizeof(HISTORIC_EVENTS) / sizeof(HISTORIC_EVENTS[0]);

// (Multi-station demo packs removed - not working)

// Historic event loader - downloads all frames for an event
class HistoricLoader {
public:
    using ProgressCallback = std::function<void(int downloaded, int total)>;
    ~HistoricLoader();

    // Start downloading frames for an event (async)
    void loadEvent(int eventIdx, ProgressCallback cb = nullptr);

    // Cancel current download
    void cancel();

    // State
    bool loading() const { return m_loading.load(); }
    bool loaded() const { return m_loaded.load(); }
    int  totalFrames() const { return m_totalFrames.load(); }
    int  downloadedFrames() const { return m_downloadedFrames.load(); }
    const HistoricEvent* currentEvent() const { return m_event.load(); }

    // Frame access
    int numFrames() const;
    const RadarFrame* frame(int idx) const;

    // Animation state
    int   currentFrame() const { return m_currentFrame; }
    void  setFrame(int f) { m_currentFrame = std::max(0, std::min(f, numFrames()-1)); }
    bool  playing() const { return m_playing; }
    void  togglePlay() { m_playing = !m_playing; }
    void  setSpeed(float fps) { m_fps = fps; }
    float speed() const { return m_fps; }

    // Call each frame to advance animation
    void  update(float dt);

private:
    void joinWorker();

    std::vector<std::shared_ptr<const RadarFrame>> m_frames;
    mutable std::mutex m_framesMutex;
    std::mutex m_workerMutex;
    std::thread m_worker;
    std::shared_ptr<class Downloader> m_downloader;
    std::atomic<const HistoricEvent*> m_event{nullptr};
    std::atomic<int> m_downloadedFrames{0};
    std::atomic<int> m_totalFrames{0};
    std::atomic<bool> m_loading{false};
    std::atomic<bool> m_loaded{false};
    std::atomic<bool> m_cancel{false};

    int   m_currentFrame = 0;
    bool  m_playing = false;
    float m_fps = 4.0f; // frames per second
    float m_accumulator = 0;
};
