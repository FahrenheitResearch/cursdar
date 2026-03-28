#include "ui.h"
#include "../app.h"
#include "../nexrad/products.h"
#include "../nexrad/stations.h"
#include <imgui.h>
#include <cstdio>

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

    // Background radar image
    auto* drawList = ImGui::GetBackgroundDrawList();
    drawList->AddImage(
        (ImTextureID)(uintptr_t)app.outputTexture().textureId(),
        ImVec2(0, 0),
        ImVec2((float)vp.width, (float)vp.height));

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
    const char* stName = (asi >= 0 && asi < total) ? NEXRAD_STATIONS[asi].name : "---";
    ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1.0f), "%s", app.activeStationName());
    ImGui::SameLine(80);
    ImGui::Text("%s", stName);
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
    if (ImGui::Button("Tilt Up", ImVec2(100, 24))) app.nextTilt();
    ImGui::SameLine();
    if (ImGui::Button("Tilt Down", ImVec2(100, 24))) app.prevTilt();

    ImGui::Separator();

    // dBZ threshold slider
    ImGui::Text("Min dBZ Filter:");
    float threshold = app.dbzMinThreshold();
    if (ImGui::SliderFloat("##dbz", &threshold, -30.0f, 40.0f, "%.0f dBZ")) {
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

    ImGui::End();

    // ── Station list (right panel, collapsible) ─────────────
    ImGui::SetNextWindowPos(ImVec2((float)vp.width - 260, 46));
    ImGui::SetNextWindowSize(ImVec2(250, (float)vp.height - 56));
    ImGui::Begin("Stations", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    ImGui::BeginChild("station_list", ImVec2(0, 0), ImGuiChildFlags_None);

    auto& stations = app.stations();
    for (int i = 0; i < (int)stations.size(); i++) {
        auto& st = stations[i];
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
                ImGui::Text("Sweeps: %d", (int)st.parsedData.sweeps.size());
                if (!st.parsedData.sweeps.empty()) {
                    ImGui::Text("Lowest elev: %.1f deg", st.parsedData.sweeps[0].elevation_angle);
                    ImGui::Text("Radials: %d", (int)st.parsedData.sweeps[0].radials.size());
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

    // ── Station markers on map (RadarScope-style) ───────────
    {
        auto* dl = ImGui::GetBackgroundDrawList();
        auto& stations2 = app.stations();
        int activeIdx = app.activeStation();

        for (int i = 0; i < (int)stations2.size(); i++) {
            auto& st = stations2[i];
            if (!st.uploaded && !st.parsed) continue;

            // Convert lat/lon to screen pixel
            float px = (float)((st.lon - vp.center_lon) * vp.zoom + vp.width * 0.5);
            float py = (float)((vp.center_lat - st.lat) * vp.zoom + vp.height * 0.5);

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

    // ── Cross-section line overlay ────────────────────────────
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

            // Altitude labels as overlay on the image
            ImVec2 imgPos = ImGui::GetItemRectMin();
            auto* wdl = ImGui::GetWindowDrawList();
            for (int alt = 0; alt <= 20; alt += 2) {
                float yy = imgPos.y + imgH - (float)alt / 22.0f * imgH;
                char altLabel[16];
                snprintf(altLabel, sizeof(altLabel), "%dkm", alt);
                wdl->AddText(ImVec2(imgPos.x + 4, yy - 7),
                             IM_COL32(200, 200, 255, 200), altLabel);
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
