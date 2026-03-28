#pragma once
#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <chrono>

struct WarningPolygon {
    std::string event;    // "Tornado Warning", "Severe Thunderstorm Warning", etc.
    std::string headline;
    std::vector<float> lats; // polygon vertices
    std::vector<float> lons;
    uint32_t color;       // RGBA
    float line_width;
};

class WarningFetcher {
public:
    void startPolling();
    void stop();

    std::vector<WarningPolygon> getWarnings() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_warnings;
    }

private:
    void fetchOnce();

    std::vector<WarningPolygon> m_warnings;
    mutable std::mutex m_mutex;
    std::atomic<bool> m_running{false};
};
