#include "ui.h"
#include "../app.h"
#include "../nexrad/products.h"
#include "../nexrad/stations.h"
#include "../historic.h"
#include "../net/warnings.h"
#include "../data/us_boundaries.h"
#include <imgui.h>
#include <cstdio>
#include <cmath>
#include <cstring>

namespace ui {

void init() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.WindowBorderSize = 1.0f;
    style.FramePadding = ImVec2(8, 4);
    style.ItemSpacing = ImVec2(8, 6);

    // Dark radar theme
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.08f, 0.08f, 0.10f, 0.92f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.06f, 0.06f, 0.08f, 1.0f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.10f, 0.10f, 0.14f, 1.0f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.12f, 0.12f, 0.15f, 1.0f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.18f, 0.18f, 0.22f, 1.0f);
    colors[ImGuiCol_Button] = ImVec4(0.15f, 0.15f, 0.20f, 1.0f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.25f, 0.25f, 0.35f, 1.0f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.30f, 0.30f, 0.45f, 1.0f);
    colors[ImGuiCol_Header] = ImVec4(0.15f, 0.15f, 0.20f, 1.0f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.20f, 0.20f, 0.30f, 1.0f);
    colors[ImGuiCol_Tab] = ImVec4(0.10f, 0.10f, 0.15f, 1.0f);
    colors[ImGuiCol_TabSelected] = ImVec4(0.20f, 0.20f, 0.30f, 1.0f);
}

void render(App& app) {
    auto& vp = app.viewport();
    const auto stations = app.stations();

    // Background radar image
    auto* drawList = ImGui::GetBackgroundDrawList();
    drawList->AddImage(
        (ImTextureID)(uintptr_t)app.outputTexture().textureId(),
        ImVec2(0, 0),
        ImVec2((float)vp.width, (float)vp.height));

    // ── State boundaries ─────────────────────────────────────
    {
        auto* bdl = ImGui::GetBackgroundDrawList();
        ImU32 lineCol = IM_COL32(50, 50, 70, 140);
        for (int i = 0; i < US_STATE_LINE_COUNT; i++) {
            float lat1 = US_STATE_LINES[i*4+0], lon1 = US_STATE_LINES[i*4+1];
            float lat2 = US_STATE_LINES[i*4+2], lon2 = US_STATE_LINES[i*4+3];
            float sx1 = (float)((lon1 - vp.center_lon) * vp.zoom + vp.width * 0.5);
            float sy1 = (float)((vp.center_lat - lat1) * vp.zoom + vp.height * 0.5);
            float sx2 = (float)((lon2 - vp.center_lon) * vp.zoom + vp.width * 0.5);
            float sy2 = (float)((vp.center_lat - lat2) * vp.zoom + vp.height * 0.5);
            // Coarse viewport cull
            if (sx1 < -50 && sx2 < -50) continue;
            if (sx1 > vp.width+50 && sx2 > vp.width+50) continue;
            if (sy1 < -50 && sy2 < -50) continue;
            if (sy1 > vp.height+50 && sy2 > vp.height+50) continue;
            bdl->AddLine(ImVec2(sx1,sy1), ImVec2(sx2,sy2), lineCol, 1.0f);
        }
    }

    // ── City labels (zoom-dependent) ────────────────────────
    {
        auto* cdl = ImGui::GetBackgroundDrawList();
        // Determine population threshold based on zoom
        int popThreshold = 1000000;  // very zoomed out: only mega cities
        if (vp.zoom > 40) popThreshold = 500000;
        if (vp.zoom > 80) popThreshold = 200000;
        if (vp.zoom > 150) popThreshold = 100000;
        if (vp.zoom > 300) popThreshold = 50000;

        for (int i = 0; i < US_CITY_COUNT; i++) {
            if (US_CITIES[i].population < popThreshold) continue;
            float sx = (float)((US_CITIES[i].lon - vp.center_lon) * vp.zoom + vp.width * 0.5);
            float sy = (float)((vp.center_lat - US_CITIES[i].lat) * vp.zoom + vp.height * 0.5);
            if (sx < -50 || sx > vp.width+50 || sy < -50 || sy > vp.height+50) continue;
            cdl->AddCircleFilled(ImVec2(sx, sy), 2.0f, IM_COL32(200, 200, 220, 180));
            cdl->AddText(ImVec2(sx + 5, sy - 7), IM_COL32(200, 200, 220, 160),
                         US_CITIES[i].name);
        }
    }

    // ── Range rings + azimuth lines ─────────────────────────
    {
        int asi = app.activeStation();
        float slat = 0, slon = 0;
        if (app.m_historicMode) {
            auto* ev = app.m_historic.currentEvent();
            if (ev) {
                // Find station lat/lon from NEXRAD_STATIONS
                for (int i = 0; i < NUM_NEXRAD_STATIONS; i++) {
                    if (strcmp(NEXRAD_STATIONS[i].icao, ev->station) == 0) {
                        slat = NEXRAD_STATIONS[i].lat;
                        slon = NEXRAD_STATIONS[i].lon;
                        break;
                    }
                }
            }
        } else if (asi >= 0 && asi < (int)stations.size()) {
            const auto& st = stations[asi];
            slat = st.display_lat;
            slon = st.display_lon;
        }

        if (slat != 0 && slon != 0 && !app.showAll() && !app.mode3D()) {
            auto* rdl = ImGui::GetBackgroundDrawList();
            float scx = (float)((slon - vp.center_lon) * vp.zoom + vp.width * 0.5);
            float scy = (float)((vp.center_lat - slat) * vp.zoom + vp.height * 0.5);
            float km_per_deg = 111.0f;

            // Range rings at 50km intervals
            ImU32 ringCol = IM_COL32(60, 60, 80, 100);
            for (int r = 50; r <= 450; r += 50) {
                float deg = (float)r / km_per_deg;
                float px_radius = (float)(deg * vp.zoom);
                if (px_radius < 10 || px_radius > vp.width * 3) continue;
                rdl->AddCircle(ImVec2(scx, scy), px_radius, ringCol, 72);
                if (px_radius > 30) {
                    char buf[16];
                    snprintf(buf, sizeof(buf), "%d", r);
                    rdl->AddText(ImVec2(scx + px_radius + 2, scy - 7),
                                 IM_COL32(80, 80, 110, 140), buf);
                }
            }

            // Cardinal + intercardinal azimuth lines
            ImU32 azCol = IM_COL32(50, 50, 70, 70);
            float maxR = 460.0f / km_per_deg * (float)vp.zoom;
            const char* dirs[] = {"N","NE","E","SE","S","SW","W","NW"};
            for (int d = 0; d < 8; d++) {
                float angle = d * 45.0f * 3.14159265f / 180.0f;
                float ex = scx + sinf(angle) * maxR;
                float ey = scy - cosf(angle) * maxR;
                float lw = (d % 2 == 0) ? 1.0f : 0.5f; // cardinals thicker
                rdl->AddLine(ImVec2(scx, scy), ImVec2(ex, ey), azCol, lw);
                float lx = scx + sinf(angle) * fminf(maxR, 60.0f);
                float ly = scy - cosf(angle) * fminf(maxR, 60.0f);
                if (d % 2 == 0) // only label cardinals
                    rdl->AddText(ImVec2(lx - 4, ly - 7),
                                 IM_COL32(100, 100, 140, 180), dirs[d]);
            }
        }
    }

    // ── Status bar (top) ────────────────────────────────────
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2((float)vp.width, 36));
    ImGui::Begin("##statusbar", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                 ImGuiWindowFlags_NoCollapse);

    int loaded = app.stationsLoaded();
    int total = app.stationsTotal();
    int downloading = app.stationsDownloading();

    int asi = app.activeStation();
    if (app.m_historicMode && app.m_historic.currentEvent()) {
        auto* ev = app.m_historic.currentEvent();
        auto* fr = app.m_historic.frame(app.m_historic.currentFrame());
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "%s", ev->station);
        ImGui::SameLine(80);
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "%s", ev->name);
        ImGui::SameLine(350);
        ImGui::Text("%s UTC", fr ? fr->timestamp.c_str() : "--:--");
    } else {
    const char* stName = (asi >= 0 && asi < total) ? NEXRAD_STATIONS[asi].name : "---";
    ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1.0f), "%s", app.activeStationName());
    ImGui::SameLine(80);
    ImGui::Text("%s", stName);
    if (app.snapshotMode()) {
        ImGui::SameLine(350);
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "Snapshot: %s", app.snapshotLabel());
    }
    }
    ImGui::SameLine(280);
    ImGui::Text("%s | Tilt %d/%d (%.1f deg)",
                PRODUCT_INFO[app.activeProduct()].name,
                app.activeTilt() + 1, app.maxTilts(), app.activeTiltAngle());
    ImGui::SameLine(600);
    ImGui::Text("Loaded: %d/%d", loaded, total);
    if (downloading > 0) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.0f, 1.0f), "(%d DL)", downloading);
    }

    ImGui::End();

    // ── Controls panel (left) ───────────────────────────────
    ImGui::SetNextWindowPos(ImVec2(10, 46));
    ImGui::SetNextWindowSize(ImVec2(230, 0));
    ImGui::Begin("Controls", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_AlwaysAutoResize);

    // Product buttons
    ImGui::Text("Product (Left/Right):");
    for (int i = 0; i < (int)Product::COUNT; i++) {
        bool selected = (app.activeProduct() == i);
        if (selected)
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.35f, 0.55f, 1.0f));

        char label[64];
        snprintf(label, sizeof(label), "[%d] %s", i + 1, PRODUCT_INFO[i].name);
        if (ImGui::Button(label, ImVec2(210, 24)))
            app.setProduct(i);

        if (selected) ImGui::PopStyleColor();
    }

    ImGui::Separator();

    // Tilt selector
    ImGui::Text("Tilt / Elevation (Up/Down):");
    char tiltLabel[64];
    snprintf(tiltLabel, sizeof(tiltLabel), "Tilt %d/%d  (%.1f deg)",
             app.activeTilt() + 1, app.maxTilts(), app.activeTiltAngle());
    ImGui::Text("%s", tiltLabel);
    if (app.showAll() || app.snapshotMode())
        ImGui::TextDisabled("Mosaic uses lowest sweep per site");
    if (ImGui::Button("Tilt Up", ImVec2(100, 24))) app.nextTilt();
    ImGui::SameLine();
    if (ImGui::Button("Tilt Down", ImVec2(100, 24))) app.prevTilt();

    ImGui::Separator();

    // Product-aware threshold slider
    bool velocityFilter = (app.activeProduct() == PROD_VEL);
    ImGui::Text("%s", velocityFilter ? "Min |Velocity| Filter:" : "Min dBZ Filter:");
    float threshold = app.dbzMinThreshold();
    bool changed = velocityFilter
        ? ImGui::SliderFloat("##dbz", &threshold, 0.0f, 50.0f, "%.0f m/s")
        : ImGui::SliderFloat("##dbz", &threshold, -30.0f, 40.0f, "%.0f dBZ");
    if (changed) {
        app.setDbzMinThreshold(threshold);
    }

    ImGui::Separator();

    // Show All toggle
    bool showAll = app.showAll();
    if (ImGui::Button(showAll ? "Single Station" : "Show All (A)", ImVec2(210, 24)))
        app.toggleShowAll();

    ImGui::Separator();
    if (ImGui::Button("Refresh Data", ImVec2(210, 24)))
        app.refreshData();

    if (!app.snapshotMode()) {
        if (ImGui::Button("Load Mar 30 2025 5 PM ET", ImVec2(210, 24)))
            app.loadMarch302025Snapshot();
        if (ImGui::Button("Load Mar 30 2025 Lowest Sweep", ImVec2(210, 24)))
            app.loadMarch302025Snapshot(true);
    } else {
        if (ImGui::Button("Back to Live", ImVec2(210, 24)))
            app.refreshData();
    }

    ImGui::Separator();

    // ── SRV mode ────────────────────────────────────────────
    if (app.activeProduct() == PROD_VEL) {
        bool srv = app.srvMode();
        if (ImGui::Checkbox("Storm-Relative (S)", &srv))
            app.toggleSRV();
        if (srv) {
            float spd = app.stormSpeed();
            float dir = app.stormDir();
            ImGui::SetNextItemWidth(100);
            if (ImGui::SliderFloat("##srvSpd", &spd, 0.0f, 40.0f, "%.0f m/s"))
                app.setStormMotion(spd, dir);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100);
            if (ImGui::SliderFloat("##srvDir", &dir, 0.0f, 360.0f, "%.0f deg"))
                app.setStormMotion(spd, dir);
        }
        ImGui::Separator();
    }

    // ── Detection overlays ──────────────────────────────────
    if (ImGui::CollapsingHeader("Detection Overlays", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("TDS (Debris)", &app.m_showTDS);
        ImGui::Checkbox("Hail (HDR)", &app.m_showHail);
        ImGui::Checkbox("Meso/TVS", &app.m_showMeso);
        ImGui::Checkbox("Dealiasing", &app.m_dealias);
    }

    ImGui::Separator();

    // ── Historic Events ─────────────────────────────────────
    if (ImGui::CollapsingHeader("Historic Tornadoes")) {
        for (int i = 0; i < NUM_HISTORIC_EVENTS; i++) {
            auto& ev = HISTORIC_EVENTS[i];
            if (ImGui::Button(ev.name, ImVec2(210, 22))) {
                app.loadHistoricEvent(i);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("%s", ev.description);
                ImGui::Text("Station: %s  |  %04d-%02d-%02d",
                            ev.station, ev.year, ev.month, ev.day);
                ImGui::Text("%02d:%02d - %02d:%02d UTC",
                            ev.start_hour, ev.start_min, ev.end_hour, ev.end_min);
                ImGui::EndTooltip();
            }
        }

        if (app.m_historicMode) {
            ImGui::Separator();
            if (ImGui::Button("Back to Live", ImVec2(210, 24))) {
                app.m_historicMode = false;
            }
        }
    }

    // (Demo packs removed)

    ImGui::End();

    // ── Single-station timeline (historic mode) ─────────────
    if (app.m_historicMode) {
        auto& hist = app.m_historic;
        float barH = 60;
        ImGui::SetNextWindowPos(ImVec2(240, (float)vp.height - barH));
        ImGui::SetNextWindowSize(ImVec2((float)vp.width - 500, barH));
        ImGui::Begin("##timeline", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        if (hist.loading()) {
            ImGui::TextColored(ImVec4(1, 0.8f, 0.2f, 1),
                               "Downloading: %d / %d frames",
                               hist.downloadedFrames(), hist.totalFrames());
            float prog = hist.totalFrames() > 0 ?
                         (float)hist.downloadedFrames() / hist.totalFrames() : 0;
            ImGui::ProgressBar(prog, ImVec2(-1, 14));
        } else if (hist.loaded() && hist.numFrames() > 0) {
            // Event name + current time
            const auto* ev = hist.currentEvent();
            const auto* fr = hist.frame(hist.currentFrame());
            ImGui::Text("%s  |  %s UTC",
                        ev ? ev->name : "???",
                        fr ? fr->timestamp.c_str() : "--:--:--");
            ImGui::SameLine((float)(vp.width - 500) - 200);

            // Play/pause + speed
            if (ImGui::Button(hist.playing() ? "Pause" : "Play", ImVec2(60, 20)))
                hist.togglePlay();
            ImGui::SameLine();
            float spd = hist.speed();
            ImGui::SetNextItemWidth(80);
            if (ImGui::SliderFloat("##spd", &spd, 1.0f, 15.0f, "%.0f fps"))
                hist.setSpeed(spd);

            // Timeline scrubber
            int frame = hist.currentFrame();
            ImGui::SetNextItemWidth(-1);
            if (ImGui::SliderInt("##frame", &frame, 0, hist.numFrames() - 1)) {
                hist.setFrame(frame);
                app.m_lastHistoricFrame = -1; // force re-upload
            }
        }

        ImGui::End();
    }

    // ── Station list (right panel, hide in historic mode) ──
    if (app.m_historicMode) goto skip_station_list;

    ImGui::SetNextWindowPos(ImVec2((float)vp.width - 260, 46));
    ImGui::SetNextWindowSize(ImVec2(250, (float)vp.height - 56));
    ImGui::Begin("Stations", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    ImGui::BeginChild("station_list", ImVec2(0, 0), ImGuiChildFlags_None);

    for (int i = 0; i < (int)stations.size(); i++) {
        const auto& st = stations[i];
        ImVec4 color;
        if (st.rendered)      color = ImVec4(0.3f, 1.0f, 0.3f, 1.0f); // green
        else if (st.uploaded) color = ImVec4(0.8f, 0.8f, 0.3f, 1.0f); // yellow
        else if (st.parsed)   color = ImVec4(0.3f, 0.7f, 1.0f, 1.0f); // blue
        else if (st.downloading) color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f); // gray
        else if (st.failed)   color = ImVec4(1.0f, 0.3f, 0.3f, 0.6f); // red
        else                  color = ImVec4(0.4f, 0.4f, 0.4f, 1.0f); // dim

        ImGui::TextColored(color, "%s", st.icao.c_str());

        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("%s (%s)", NEXRAD_STATIONS[i].name, NEXRAD_STATIONS[i].state);
            ImGui::Text("Lat: %.4f  Lon: %.4f", st.lat, st.lon);
            if (st.failed) ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "Error: %s", st.error.c_str());
            if (st.parsed) {
                ImGui::Text("Sweeps: %d", st.sweep_count);
                if (st.sweep_count > 0) {
                    ImGui::Text("Lowest elev: %.1f deg", st.lowest_elev);
                    ImGui::Text("Radials: %d", st.lowest_radials);
                }
            }
            ImGui::EndTooltip();
        }

        // Click to center viewport on station
        if (ImGui::IsItemClicked()) {
            app.viewport().center_lat = st.lat;
            app.viewport().center_lon = st.lon;
            app.viewport().zoom = 200.0; // zoom in
        }

        if ((i + 1) % 8 != 0) ImGui::SameLine(0, 12);
    }

    ImGui::EndChild();
    ImGui::End();
    skip_station_list:

    // ── Station markers on map (hide in historic mode) ──────
    if (app.m_historicMode) goto skip_station_markers;
    {
        auto* dl = ImGui::GetBackgroundDrawList();
        int activeIdx = app.activeStation();

        for (int i = 0; i < (int)stations.size(); i++) {
            const auto& st = stations[i];
            if (!st.uploaded && !st.parsed) continue;

            // Convert lat/lon to screen pixel
            float px = (float)((st.display_lon - vp.center_lon) * vp.zoom + vp.width * 0.5);
            float py = (float)((vp.center_lat - st.display_lat) * vp.zoom + vp.height * 0.5);

            // Skip if off-screen
            if (px < -50 || px > vp.width + 50 || py < -50 || py > vp.height + 50)
                continue;

            bool isActive = (i == activeIdx);
            float boxW = 36, boxH = 14;

            // Background rectangle
            ImU32 bgCol = isActive ?
                IM_COL32(0, 180, 80, 220) :  // green for active
                IM_COL32(40, 40, 50, 180);    // dark for others
            ImU32 borderCol = isActive ?
                IM_COL32(100, 255, 150, 255) :
                IM_COL32(80, 80, 100, 200);
            ImU32 textCol = isActive ?
                IM_COL32(255, 255, 255, 255) :
                IM_COL32(180, 180, 200, 220);

            ImVec2 tl(px - boxW * 0.5f, py - boxH * 0.5f);
            ImVec2 br(px + boxW * 0.5f, py + boxH * 0.5f);

            dl->AddRectFilled(tl, br, bgCol, 3.0f);
            dl->AddRect(tl, br, borderCol, 3.0f);

            // Station ICAO text
            const char* label = st.icao.c_str();
            ImVec2 textSize = ImGui::CalcTextSize(label);
            dl->AddText(ImVec2(px - textSize.x * 0.5f, py - textSize.y * 0.5f),
                        textCol, label);
        }
    }
    skip_station_markers:

    // ── NWS Warning Polygons ────────────────────────────────
    if (!app.m_historicMode) {
        auto* wdl = ImGui::GetBackgroundDrawList();
        auto warnings = app.m_warnings.getWarnings();
        for (auto& w : warnings) {
            if (w.lats.size() < 3) continue;
            // Convert polygon to screen coordinates and draw
            std::vector<ImVec2> pts;
            pts.reserve(w.lats.size());
            bool anyOnScreen = false;
            for (int i = 0; i < (int)w.lats.size(); i++) {
                float sx = (float)((w.lons[i] - vp.center_lon) * vp.zoom + vp.width * 0.5);
                float sy = (float)((vp.center_lat - w.lats[i]) * vp.zoom + vp.height * 0.5);
                pts.push_back(ImVec2(sx, sy));
                if (sx > -100 && sx < vp.width + 100 && sy > -100 && sy < vp.height + 100)
                    anyOnScreen = true;
            }
            if (!anyOnScreen) continue;

            // Draw filled polygon (semi-transparent)
            uint32_t fillCol = w.color;
            if (pts.size() >= 3) {
                // ImGui convex fill - warnings are typically convex
                wdl->AddConvexPolyFilled(pts.data(), (int)pts.size(), fillCol);
            }
            // Draw outline
            uint32_t outlineCol = fillCol | 0xFF000000; // full alpha
            for (int i = 0; i < (int)pts.size(); i++) {
                int j = (i + 1) % (int)pts.size();
                wdl->AddLine(pts[i], pts[j], outlineCol, w.line_width);
            }
        }
    }

    // ── Detection overlays (TDS, Hail, Meso) ─────────────────
    {
        auto* ddl = ImGui::GetBackgroundDrawList();
        int dsi = app.activeStation();
        if (dsi >= 0 && dsi < (int)stations.size()) {
            const auto& dst = stations[dsi];
            const auto& det = dst.detection;

            // TDS markers: white inverted triangles with red border
            if (app.m_showTDS && !det.tds.empty()) {
                for (auto& t : det.tds) {
                    float sx = (float)((t.lon - vp.center_lon) * vp.zoom + vp.width * 0.5);
                    float sy = (float)((vp.center_lat - t.lat) * vp.zoom + vp.height * 0.5);
                    if (sx < -20 || sx > vp.width+20 || sy < -20 || sy > vp.height+20) continue;
                    float sz = 6.0f;
                    ddl->AddTriangleFilled(
                        ImVec2(sx, sy + sz), ImVec2(sx - sz, sy - sz), ImVec2(sx + sz, sy - sz),
                        IM_COL32(255, 255, 255, 200));
                    ddl->AddTriangle(
                        ImVec2(sx, sy + sz), ImVec2(sx - sz, sy - sz), ImVec2(sx + sz, sy - sz),
                        IM_COL32(255, 0, 0, 255), 2.0f);
                }
            }

            // Hail markers: green/magenta circles with H
            if (app.m_showHail && !det.hail.empty()) {
                for (auto& h : det.hail) {
                    float sx = (float)((h.lon - vp.center_lon) * vp.zoom + vp.width * 0.5);
                    float sy = (float)((vp.center_lat - h.lat) * vp.zoom + vp.height * 0.5);
                    if (sx < -20 || sx > vp.width+20 || sy < -20 || sy > vp.height+20) continue;
                    float r = 5.0f;
                    ImU32 col = h.value > 10.0f ? IM_COL32(255, 50, 255, 220) :
                                                   IM_COL32(0, 255, 100, 200);
                    ddl->AddCircleFilled(ImVec2(sx, sy), r, col);
                    ddl->AddText(ImVec2(sx - 3, sy - 6), IM_COL32(0, 0, 0, 255), "H");
                }
            }

            // Mesocyclone markers: circles with rotation indicator
            if (app.m_showMeso && !det.meso.empty()) {
                for (auto& m : det.meso) {
                    float sx = (float)((m.lon - vp.center_lon) * vp.zoom + vp.width * 0.5);
                    float sy = (float)((vp.center_lat - m.lat) * vp.zoom + vp.height * 0.5);
                    if (sx < -20 || sx > vp.width+20 || sy < -20 || sy > vp.height+20) continue;
                    float r = m.shear > 30.0f ? 10.0f : 7.0f;
                    ImU32 col = m.shear > 30.0f ? IM_COL32(255, 0, 0, 255) :
                                                    IM_COL32(255, 255, 0, 255);
                    ddl->AddCircle(ImVec2(sx, sy), r, col, 12, 2.5f);
                    ddl->AddLine(ImVec2(sx + r, sy), ImVec2(sx + r - 3, sy - 3), col, 2.0f);
                    ddl->AddLine(ImVec2(sx + r, sy), ImVec2(sx + r + 1, sy - 4), col, 2.0f);
                }
            }
        }
    }

    // ── Cross-section line overlay ────────────────────���───────
    if (app.crossSection()) {
        auto* dl2 = ImGui::GetBackgroundDrawList();
        // Draw the cross-section line on the radar view
        float sx = (float)((app.xsStartLon() - vp.center_lon) * vp.zoom + vp.width * 0.5);
        float sy = (float)((vp.center_lat - app.xsStartLat()) * vp.zoom + vp.height * 0.5);
        float ex = (float)((app.xsEndLon() - vp.center_lon) * vp.zoom + vp.width * 0.5);
        float ey = (float)((vp.center_lat - app.xsEndLat()) * vp.zoom + vp.height * 0.5);

        dl2->AddLine(ImVec2(sx, sy), ImVec2(ex, ey), IM_COL32(255, 255, 0, 200), 3.0f);
        dl2->AddCircleFilled(ImVec2(sx, sy), 6, IM_COL32(255, 100, 100, 255));
        dl2->AddCircleFilled(ImVec2(ex, ey), 6, IM_COL32(100, 255, 100, 255));

        // Label
        float xsBottom = (float)(vp.height / 2);
        // Cross-section floating panel (book view)
        if (app.xsTexture().textureId() != 0 && app.xsWidth() > 0) {
            float panelW = (float)vp.width * 0.8f;
            float panelH = (float)app.xsHeight() + 40.0f;
            ImGui::SetNextWindowPos(ImVec2((float)vp.width * 0.1f,
                                           (float)vp.height - panelH - 10), ImGuiCond_Once);
            ImGui::SetNextWindowSize(ImVec2(panelW, panelH), ImGuiCond_Once);
            ImGui::Begin("Cross Section", nullptr,
                         ImGuiWindowFlags_NoCollapse);

            ImVec2 avail = ImGui::GetContentRegionAvail();
            float imgW = avail.x, imgH = avail.y;

            ImGui::Image((ImTextureID)(uintptr_t)app.xsTexture().textureId(),
                         ImVec2(imgW, imgH));

            // Altitude labels (kft like GR2Analyst)
            ImVec2 imgPos = ImGui::GetItemRectMin();
            auto* wdl = ImGui::GetWindowDrawList();
            for (int kft = 0; kft <= 45; kft += 5) {
                float alt_km = (float)kft * 0.3048f; // kft to km
                float frac = alt_km / 15.0f; // 15km max
                if (frac > 1.0f) break;
                float yy = imgPos.y + imgH * (1.0f - frac);
                char altLabel[16];
                snprintf(altLabel, sizeof(altLabel), "%d kft", kft);
                wdl->AddText(ImVec2(imgPos.x + 4, yy - 7),
                             IM_COL32(200, 200, 255, 200), altLabel);
                wdl->AddLine(ImVec2(imgPos.x + 40, yy),
                             ImVec2(imgPos.x + imgW, yy),
                             IM_COL32(100, 100, 140, 60), 1.0f);
            }

            ImGui::End();
        }
    }

    // ── Keyboard shortcuts ──────────────────────────────────
    if (!ImGui::GetIO().WantCaptureKeyboard) {
        // Number keys: direct product select
        for (int i = 0; i < (int)Product::COUNT; i++) {
            if (ImGui::IsKeyPressed((ImGuiKey)(ImGuiKey_1 + i)))
                app.setProduct(i);
        }
        // Arrow keys: left/right = product, up/down = tilt
        if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow))  app.prevProduct();
        if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) app.nextProduct();
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow))    app.nextTilt();
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow))  app.prevTilt();
        // V = 3D volume, X = cross-section, A = toggle show all
        if (ImGui::IsKeyPressed(ImGuiKey_V)) app.toggle3D();
        if (ImGui::IsKeyPressed(ImGuiKey_X)) app.toggleCrossSection();
        if (ImGui::IsKeyPressed(ImGuiKey_A)) app.toggleShowAll();
        if (ImGui::IsKeyPressed(ImGuiKey_R)) app.refreshData();
        if (ImGui::IsKeyPressed(ImGuiKey_S)) app.toggleSRV();
        if (ImGui::IsKeyPressed(ImGuiKey_Home)) {
            app.viewport().center_lat = 39.0;
            app.viewport().center_lon = -98.0;
            app.viewport().zoom = 28.0;
        }
    }
}

void shutdown() {
}

} // namespace ui
