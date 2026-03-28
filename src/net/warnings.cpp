#include "warnings.h"
#include "downloader.h"
#include <thread>
#include <cstdio>
#include <cstring>

// Simple JSON coordinate extractor (no JSON library needed)
static bool extractCoordinates(const std::string& json, size_t start,
                                std::vector<float>& lats, std::vector<float>& lons) {
    // Find "coordinates":[[[...
    size_t pos = json.find("\"coordinates\"", start);
    if (pos == std::string::npos) return false;

    // Find the opening [[[
    pos = json.find("[[[", pos);
    if (pos == std::string::npos) {
        pos = json.find("[[", start);
        if (pos == std::string::npos) return false;
        pos += 2;
    } else {
        pos += 3;
    }

    // Parse [lon,lat] pairs
    while (pos < json.size()) {
        if (json[pos] == ']') break;
        if (json[pos] == '[') {
            pos++;
            // Read lon
            char* end;
            float lon = strtof(json.c_str() + pos, &end);
            pos = end - json.c_str();
            if (json[pos] == ',') pos++;
            float lat = strtof(json.c_str() + pos, &end);
            pos = end - json.c_str();
            lons.push_back(lon);
            lats.push_back(lat);
            // Skip to after ]
            while (pos < json.size() && json[pos] != ']') pos++;
            if (pos < json.size()) pos++; // skip ]
            if (pos < json.size() && json[pos] == ',') pos++;
        } else {
            pos++;
        }
    }
    return !lats.empty();
}

static uint32_t warningColor(const std::string& event) {
    if (event.find("Tornado Warning") != std::string::npos)
        return 0xC00000FF; // red, semi-transparent
    if (event.find("Severe Thunderstorm Warning") != std::string::npos)
        return 0xC000A5FF; // orange
    if (event.find("Flash Flood") != std::string::npos)
        return 0xC0008B00; // dark green
    if (event.find("Tornado Watch") != std::string::npos)
        return 0x8000FFFF; // yellow, more transparent
    if (event.find("Severe Thunderstorm Watch") != std::string::npos)
        return 0x80FFBF00; // cyan-ish
    return 0x60B5E4FF; // wheat for other
}

void WarningFetcher::fetchOnce() {
    auto result = Downloader::httpGet(
        "api.weather.gov",
        "/alerts/active?status=actual&message_type=alert,update&event=Tornado%20Warning,Severe%20Thunderstorm%20Warning,Flash%20Flood%20Warning,Tornado%20Watch,Severe%20Thunderstorm%20Watch");

    if (!result.success) return;

    std::string json(result.data.begin(), result.data.end());
    std::vector<WarningPolygon> warnings;

    // Parse each feature
    size_t pos = 0;
    while (true) {
        // Find next "event":"..."
        size_t eventPos = json.find("\"event\"", pos);
        if (eventPos == std::string::npos) break;

        // Extract event name
        size_t nameStart = json.find("\"", eventPos + 8);
        if (nameStart == std::string::npos) break;
        nameStart++;
        size_t nameEnd = json.find("\"", nameStart);
        if (nameEnd == std::string::npos) break;
        std::string event = json.substr(nameStart, nameEnd - nameStart);

        // Find coordinates near this event
        WarningPolygon wp;
        wp.event = event;
        wp.color = warningColor(event);
        wp.line_width = (event.find("Tornado Warning") != std::string::npos) ? 3.0f : 2.0f;

        if (extractCoordinates(json, pos, wp.lats, wp.lons) && wp.lats.size() >= 3) {
            warnings.push_back(std::move(wp));
        }

        pos = nameEnd + 1;
    }

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_warnings = std::move(warnings);
    }
    printf("Fetched %d active warnings\n", (int)m_warnings.size());
}

void WarningFetcher::startPolling() {
    if (m_running) return;
    m_running = true;

    std::thread([this]() {
        while (m_running) {
            fetchOnce();
            // Poll every 60 seconds
            for (int i = 0; i < 60 && m_running; i++)
                std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }).detach();
}

void WarningFetcher::stop() {
    m_running = false;
}
