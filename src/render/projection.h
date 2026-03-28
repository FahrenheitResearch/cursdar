#pragma once
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

constexpr double EARTH_RADIUS_KM = 6371.0;
constexpr double DEG_TO_RAD = M_PI / 180.0;
constexpr double RAD_TO_DEG = 180.0 / M_PI;
constexpr double NEXRAD_MAX_RANGE_KM = 460.0;

// Viewport: maps screen pixels to geographic coordinates
struct Viewport {
    double center_lat = 39.0;   // Center of CONUS
    double center_lon = -98.0;
    double zoom = 1.0;          // pixels per degree at equator
    int width = 1920;
    int height = 1080;

    // Degrees visible in each direction from center
    double halfExtentLon() const {
        return (width * 0.5) / zoom;
    }
    double halfExtentLat() const {
        return (height * 0.5) / zoom;
    }

    // Screen pixel to lat/lon
    void pixelToLatLon(int px, int py, double& lat, double& lon) const {
        lon = center_lon + (px - width * 0.5) / zoom;
        lat = center_lat - (py - height * 0.5) / zoom;
    }

    // Lat/lon to screen pixel
    void latLonToPixel(double lat, double lon, int& px, int& py) const {
        px = (int)((lon - center_lon) * zoom + width * 0.5);
        py = (int)((center_lat - lat) * zoom + height * 0.5);
    }
};

// Great circle distance between two lat/lon points (km)
inline double haversineKm(double lat1, double lon1, double lat2, double lon2) {
    double dlat = (lat2 - lat1) * DEG_TO_RAD;
    double dlon = (lon2 - lon1) * DEG_TO_RAD;
    double a = sin(dlat / 2) * sin(dlat / 2) +
               cos(lat1 * DEG_TO_RAD) * cos(lat2 * DEG_TO_RAD) *
               sin(dlon / 2) * sin(dlon / 2);
    return EARTH_RADIUS_KM * 2.0 * atan2(sqrt(a), sqrt(1 - a));
}

// Azimuth from point 1 to point 2 (degrees, 0=North, clockwise)
inline double azimuthDeg(double lat1, double lon1, double lat2, double lon2) {
    double dlon = (lon2 - lon1) * DEG_TO_RAD;
    double la1 = lat1 * DEG_TO_RAD;
    double la2 = lat2 * DEG_TO_RAD;
    double y = sin(dlon) * cos(la2);
    double x = cos(la1) * sin(la2) - sin(la1) * cos(la2) * cos(dlon);
    double az = atan2(y, x) * RAD_TO_DEG;
    return fmod(az + 360.0, 360.0);
}

// Offset a lat/lon by km in N/E directions (flat earth approx, good for <500km)
inline void offsetKm(double lat, double lon, double north_km, double east_km,
                     double& out_lat, double& out_lon) {
    out_lat = lat + (north_km / EARTH_RADIUS_KM) * RAD_TO_DEG;
    out_lon = lon + (east_km / (EARTH_RADIUS_KM * cos(lat * DEG_TO_RAD))) * RAD_TO_DEG;
}

// Bounding box of a station's coverage area
struct StationBounds {
    double min_lat, max_lat, min_lon, max_lon;
};

inline StationBounds stationCoverageBounds(double lat, double lon, double range_km = NEXRAD_MAX_RANGE_KM) {
    double dlat = (range_km / EARTH_RADIUS_KM) * RAD_TO_DEG;
    double dlon = dlat / cos(lat * DEG_TO_RAD);
    return {lat - dlat, lat + dlat, lon - dlon, lon + dlon};
}
