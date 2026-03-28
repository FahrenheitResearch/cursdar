#include "historic.h"
#include "net/downloader.h"
#include "net/aws_nexrad.h"
#include "nexrad/products.h"
#include "nexrad/stations.h"
#include <algorithm>
#include <thread>
#include <cstdio>
#include <cstring>
#include <cmath>

// ── Download and parse all frames for a historic event ──────

void HistoricLoader::loadEvent(int eventIdx, ProgressCallback cb) {
    if (eventIdx < 0 || eventIdx >= NUM_HISTORIC_EVENTS) return;
    if (m_loading) return;

    m_event = &HISTORIC_EVENTS[eventIdx];
    m_loading = true;
    m_loaded = false;
    m_cancel = false;
    m_frames.clear();
    m_currentFrame = 0;
    m_downloadedFrames = 0;
    m_playing = false;

    // Launch download thread
    std::thread([this, cb]() {
        const auto& ev = *m_event;
        printf("Loading historic event: %s (%s %04d-%02d-%02d)\n",
               ev.name, ev.station, ev.year, ev.month, ev.day);

        // List all files for this station/date
        std::string listPath = "/?list-type=2&prefix=" +
            std::to_string(ev.year) + "/" +
            (ev.month < 10 ? "0" : "") + std::to_string(ev.month) + "/" +
            (ev.day < 10 ? "0" : "") + std::to_string(ev.day) + "/" +
            std::string(ev.station) + "/&max-keys=1000";

        auto listResult = Downloader::httpGet(NEXRAD_HOST, listPath);
        if (!listResult.success) {
            printf("Failed to list files: %s\n", listResult.error.c_str());
            m_loading = false;
            return;
        }

        std::string xml(listResult.data.begin(), listResult.data.end());
        auto files = parseS3ListResponse(xml);

        // Filter by time range
        std::vector<NexradFile> filtered;
        for (auto& f : files) {
            // Extract HHMMSS from filename like KTLX20130520_195431_V06
            // The timestamp is after the date: station + YYYYMMDD + _ + HHMMSS
            size_t us = f.key.rfind('/');
            std::string fname = (us != std::string::npos) ? f.key.substr(us + 1) : f.key;

            // Find the underscore after the date
            size_t dateEnd = fname.find('_');
            if (dateEnd == std::string::npos || dateEnd + 7 > fname.size()) continue;

            std::string timeStr = fname.substr(dateEnd + 1, 6);
            int hh = std::stoi(timeStr.substr(0, 2));
            int mm = std::stoi(timeStr.substr(2, 2));
            int timeMinutes = hh * 60 + mm;

            int startMin = ev.start_hour * 60 + ev.start_min;
            int endMin = ev.end_hour * 60 + ev.end_min;

            // Handle overnight events
            if (endMin < startMin) {
                if (timeMinutes >= startMin || timeMinutes <= endMin)
                    filtered.push_back(f);
            } else {
                if (timeMinutes >= startMin && timeMinutes <= endMin)
                    filtered.push_back(f);
            }
        }

        // Also check next day for overnight events
        if (ev.end_hour < ev.start_hour) {
            int nextDay = ev.day + 1;
            int nextMonth = ev.month;
            // Simple day rollover (not perfect for month boundaries)
            if (nextDay > 28) { nextDay = 1; nextMonth++; }

            std::string listPath2 = "/?list-type=2&prefix=" +
                std::to_string(ev.year) + "/" +
                (nextMonth < 10 ? "0" : "") + std::to_string(nextMonth) + "/" +
                (nextDay < 10 ? "0" : "") + std::to_string(nextDay) + "/" +
                std::string(ev.station) + "/&max-keys=1000";

            auto list2 = Downloader::httpGet(NEXRAD_HOST, listPath2);
            if (list2.success) {
                std::string xml2(list2.data.begin(), list2.data.end());
                auto files2 = parseS3ListResponse(xml2);
                for (auto& f : files2) {
                    size_t us = f.key.rfind('/');
                    std::string fname = (us != std::string::npos) ? f.key.substr(us + 1) : f.key;
                    size_t dateEnd = fname.find('_');
                    if (dateEnd == std::string::npos || dateEnd + 7 > fname.size()) continue;
                    std::string timeStr = fname.substr(dateEnd + 1, 6);
                    int hh = std::stoi(timeStr.substr(0, 2));
                    int mm = std::stoi(timeStr.substr(2, 2));
                    int timeMinutes = hh * 60 + mm;
                    if (timeMinutes <= ev.end_hour * 60 + ev.end_min)
                        filtered.push_back(f);
                }
            }
        }

        if (filtered.empty()) {
            printf("No files found in time range\n");
            m_loading = false;
            return;
        }

        // Sort by key (chronological)
        std::sort(filtered.begin(), filtered.end(),
                  [](const NexradFile& a, const NexradFile& b) { return a.key < b.key; });

        m_totalFrames = (int)filtered.size();
        m_frames.resize(m_totalFrames);
        printf("Found %d frames to download\n", m_totalFrames);

        // Download and parse each frame (parallel with 8 threads)
        Downloader dl(8);
        for (int i = 0; i < m_totalFrames; i++) {
            if (m_cancel) break;

            auto& nf = filtered[i];
            int idx = i;

            dl.queueDownload(nf.key, NEXRAD_HOST, "/" + nf.key,
                [this, idx, cb](const std::string& id, DownloadResult result) {
                    if (!result.success || result.data.empty()) {
                        m_downloadedFrames++;
                        if (cb) cb(m_downloadedFrames.load(), m_totalFrames);
                        return;
                    }

                    // Parse
                    auto parsed = Level2Parser::parse(result.data);
                    if (parsed.sweeps.empty()) {
                        m_downloadedFrames++;
                        if (cb) cb(m_downloadedFrames.load(), m_totalFrames);
                        return;
                    }

                    // Extract timestamp from key
                    size_t us = id.rfind('/');
                    std::string fname = (us != std::string::npos) ? id.substr(us + 1) : id;

                    // Precompute sweep data
                    auto& frame = m_frames[idx];
                    frame.filename = fname;

                    // Extract HH:MM:SS
                    size_t dateEnd = fname.find('_');
                    if (dateEnd != std::string::npos && dateEnd + 7 <= fname.size()) {
                        std::string ts = fname.substr(dateEnd + 1, 6);
                        frame.timestamp = ts.substr(0, 2) + ":" + ts.substr(2, 2) + ":" + ts.substr(4, 2);
                    }

                    frame.station_lat = parsed.station_lat;
                    frame.station_lon = parsed.station_lon;

                    // Precompute all sweeps
                    frame.sweeps.resize(parsed.sweeps.size());
                    for (int si = 0; si < (int)parsed.sweeps.size(); si++) {
                        auto& sweep = parsed.sweeps[si];
                        auto& pc = frame.sweeps[si];
                        pc.elevation_angle = sweep.elevation_angle;
                        pc.num_radials = (int)sweep.radials.size();
                        if (pc.num_radials == 0) continue;

                        pc.azimuths.resize(pc.num_radials);
                        for (int r = 0; r < pc.num_radials; r++)
                            pc.azimuths[r] = sweep.radials[r].azimuth;

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

                            int ng = pd.num_gates, nr = pc.num_radials;
                            pd.gates.resize((size_t)ng * nr, 0);
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

                    frame.ready = true;
                    int done = ++m_downloadedFrames;
                    if (cb) cb(done, m_totalFrames);
                    printf("\rFrames: %d/%d", done, m_totalFrames);
                    fflush(stdout);
                }
            );
        }

        dl.waitAll();
        printf("\nHistoric event loaded: %d frames ready\n", m_downloadedFrames.load());
        m_loaded = true;
        m_loading = false;
    }).detach();
}

void HistoricLoader::update(float dt) {
    if (!m_playing || m_frames.empty()) return;

    m_accumulator += dt;
    float frameDur = 1.0f / m_fps;
    while (m_accumulator >= frameDur) {
        m_accumulator -= frameDur;
        m_currentFrame++;
        // Skip unready frames
        while (m_currentFrame < (int)m_frames.size() && !m_frames[m_currentFrame].ready)
            m_currentFrame++;
        if (m_currentFrame >= (int)m_frames.size())
            m_currentFrame = 0; // loop
    }
}

// ── Helper: precompute one parsed file into a RadarFrame ────
static void precomputeFrame(RadarFrame& frame, ParsedRadarData& parsed, const std::string& fname) {
    frame.filename = fname;
    frame.station_lat = parsed.station_lat;
    frame.station_lon = parsed.station_lon;

    // Extract HH:MM:SS from filename
    size_t us = fname.find('_');
    if (us != std::string::npos && us + 7 <= fname.size()) {
        std::string ts = fname.substr(us + 1, 6);
        if (ts.size() >= 6)
            frame.timestamp = ts.substr(0, 2) + ":" + ts.substr(2, 2) + ":" + ts.substr(4, 2);
    }

    frame.sweeps.resize(parsed.sweeps.size());
    for (int si = 0; si < (int)parsed.sweeps.size(); si++) {
        auto& sweep = parsed.sweeps[si];
        auto& pc = frame.sweeps[si];
        pc.elevation_angle = sweep.elevation_angle;
        pc.num_radials = (int)sweep.radials.size();
        if (pc.num_radials == 0) continue;

        pc.azimuths.resize(pc.num_radials);
        for (int r = 0; r < pc.num_radials; r++)
            pc.azimuths[r] = sweep.radials[r].azimuth;

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
            int ng = pd.num_gates, nr = pc.num_radials;
            pd.gates.resize((size_t)ng * nr, 0);
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
    frame.ready = true;
}

// ── DemoPack Loader ─────────────────────────────────────────

// Extract time in minutes from midnight from a NEXRAD filename
static float extractTimeMinutes(const std::string& fname) {
    size_t us = fname.find('_');
    if (us == std::string::npos || us + 7 > fname.size()) return -1;
    std::string ts = fname.substr(us + 1, 6);
    if (ts.size() < 6) return -1;
    int hh = std::stoi(ts.substr(0, 2));
    int mm = std::stoi(ts.substr(2, 2));
    int ss = std::stoi(ts.substr(4, 2));
    return (float)(hh * 60 + mm) + ss / 60.0f;
}

// (Demo pack code removed)
#if 0
void DemoPackLoader_REMOVED_loadPack(int packIdx) {
    if (packIdx < 0 || packIdx >= NUM_DEMO_PACKS || m_loading) return;

    m_pack = &DEMO_PACKS[packIdx];
    m_loading = true;
    m_loaded = false;
    m_cancel = false;
    m_stationFrames.clear();
    m_downloadedFiles = 0;
    m_currentTime = (float)m_pack->start_hour * 60;
    m_playing = false;

    std::thread([this]() {
        const auto& pk = *m_pack;
        printf("Loading demo pack: %s (%d stations)\n", pk.name, pk.num_stations);

        // Set up per-station frame lists
        m_stationFrames.resize(pk.num_stations);
        for (int s = 0; s < pk.num_stations; s++) {
            m_stationFrames[s].station = pk.stations[s];
            // Find lat/lon from station list
            for (int j = 0; j < NUM_NEXRAD_STATIONS; j++) {
                if (m_stationFrames[s].station == NEXRAD_STATIONS[j].icao) {
                    m_stationFrames[s].lat = NEXRAD_STATIONS[j].lat;
                    m_stationFrames[s].lon = NEXRAD_STATIONS[j].lon;
                    m_stationFrames[s].station_global_idx = j;
                    break;
                }
            }
        }

        // List + download all files for each station
        Downloader dl(12);
        int startMin = pk.start_hour * 60;
        int endMin = pk.end_hour * 60;

        for (int s = 0; s < pk.num_stations; s++) {
            if (m_cancel) break;

            std::string station = pk.stations[s];
            std::string listPath = "/?list-type=2&prefix=" +
                std::to_string(pk.year) + "/" +
                (pk.month < 10 ? "0" : "") + std::to_string(pk.month) + "/" +
                (pk.day < 10 ? "0" : "") + std::to_string(pk.day) + "/" +
                station + "/&max-keys=1000";

            auto listResult = Downloader::httpGet(NEXRAD_HOST, listPath);
            if (!listResult.success) {
                printf("  %s: listing failed\n", station.c_str());
                continue;
            }

            std::string xml(listResult.data.begin(), listResult.data.end());
            auto files = parseS3ListResponse(xml);

            // Filter by time range
            std::vector<NexradFile> filtered;
            for (auto& f : files) {
                size_t us = f.key.rfind('/');
                std::string fname = (us != std::string::npos) ? f.key.substr(us + 1) : f.key;
                float tmin = extractTimeMinutes(fname);
                if (tmin >= 0 && tmin >= startMin && tmin <= endMin)
                    filtered.push_back(f);
            }

            printf("  %s: %d files in time range\n", station.c_str(), (int)filtered.size());
            m_totalFiles += (int)filtered.size();
            m_stationFrames[s].frames.resize(filtered.size());

            for (int fi = 0; fi < (int)filtered.size(); fi++) {
                if (m_cancel) break;
                int stIdx = s;
                int frameIdx = fi;
                auto& nf = filtered[fi];

                dl.queueDownload(nf.key, NEXRAD_HOST, "/" + nf.key,
                    [this, stIdx, frameIdx](const std::string& id, DownloadResult result) {
                        if (!result.success || result.data.empty()) {
                            m_downloadedFiles++;
                            return;
                        }
                        auto parsed = Level2Parser::parse(result.data);
                        if (parsed.sweeps.empty()) {
                            m_downloadedFiles++;
                            return;
                        }

                        size_t us = id.rfind('/');
                        std::string fname = (us != std::string::npos) ? id.substr(us + 1) : id;

                        auto& frame = m_stationFrames[stIdx].frames[frameIdx];
                        precomputeFrame(frame, parsed, fname);

                        int done = ++m_downloadedFiles;
                        if (done % 10 == 0)
                            printf("\r  Demo pack: %d/%d files", done, m_totalFiles);
                    }
                );
            }
        }

        dl.waitAll();
        printf("\nDemo pack loaded: %d files across %d stations\n",
               m_downloadedFiles.load(), pk.num_stations);
        m_loaded = true;
        m_loading = false;
    }).detach();
}

void DemoPackLoader::update(float dt) {
    if (!m_playing) return;
    m_currentTime += dt * m_speed;
    if (m_currentTime > timelineMax())
        m_currentTime = timelineMin(); // loop
}

const RadarFrame* DemoPackLoader::getFrameAtTime(int stationIdx, float timeMinutes) const {
    if (stationIdx < 0 || stationIdx >= (int)m_stationFrames.size()) return nullptr;
    auto& sf = m_stationFrames[stationIdx];

    const RadarFrame* best = nullptr;
    float bestDist = 1e9f;
    for (auto& f : sf.frames) {
        if (!f.ready) continue;
        float tmin = extractTimeMinutes(f.filename);
        if (tmin < 0) continue;
        float dist = fabsf(tmin - timeMinutes);
        if (dist < bestDist) {
            bestDist = dist;
            best = &f;
        }
    }
    return best;
}
#endif
