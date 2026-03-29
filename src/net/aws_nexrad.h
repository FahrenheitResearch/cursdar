#pragma once
#include <string>
#include <vector>
#include <ctime>
#include <algorithm>

// AWS S3 NEXRAD Level 2 bucket
constexpr const char* NEXRAD_BUCKET = "unidata-nexrad-level2";
constexpr const char* NEXRAD_HOST = "unidata-nexrad-level2.s3.amazonaws.com";

struct NexradFile {
    std::string key;
    std::string url;
    size_t      size;
};

// Build the S3 list URL for a station on a given date
inline std::string buildListUrl(const std::string& station,
                                 int year, int month, int day) {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "/%04d/%02d/%02d/%s/",
             year, month, day, station.c_str());
    return std::string(buf);
}

// Build the download URL for a specific key
inline std::string buildDownloadUrl(const std::string& key) {
    return "/" + key;
}

// Get current UTC date
inline void getUtcDate(int& year, int& month, int& day) {
    time_t t = time(nullptr);
    struct tm utc;
#ifdef _WIN32
    gmtime_s(&utc, &t);
#else
    gmtime_r(&t, &utc);
#endif
    year = utc.tm_year + 1900;
    month = utc.tm_mon + 1;
    day = utc.tm_mday;
}

inline bool isLeapYear(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

inline int daysInMonth(int year, int month) {
    static const int kDaysPerMonth[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month == 2) return isLeapYear(year) ? 29 : 28;
    return kDaysPerMonth[month - 1];
}

inline void shiftDate(int& year, int& month, int& day, int deltaDays) {
    while (deltaDays < 0) {
        day--;
        if (day < 1) {
            month--;
            if (month < 1) {
                month = 12;
                year--;
            }
            day = daysInMonth(year, month);
        }
        deltaDays++;
    }

    while (deltaDays > 0) {
        day++;
        if (day > daysInMonth(year, month)) {
            day = 1;
            month++;
            if (month > 12) {
                month = 1;
                year++;
            }
        }
        deltaDays--;
    }
}

// Parse S3 XML list response to extract file keys
// Simple tag extraction - no XML library needed
inline std::vector<NexradFile> parseS3ListResponse(const std::string& xml) {
    std::vector<NexradFile> files;

    size_t pos = 0;
    while (true) {
        size_t keyStart = xml.find("<Key>", pos);
        if (keyStart == std::string::npos) break;
        keyStart += 5;

        size_t keyEnd = xml.find("</Key>", keyStart);
        if (keyEnd == std::string::npos) break;

        std::string key = xml.substr(keyStart, keyEnd - keyStart);
        pos = keyEnd + 6;

        // Skip MDM (metadata) files
        if (key.find("_MDM") != std::string::npos) continue;

        // Parse size if available
        size_t sizeVal = 0;
        size_t sizeStart = xml.find("<Size>", pos);
        if (sizeStart != std::string::npos && sizeStart < xml.find("<Key>", pos)) {
            sizeStart += 6;
            size_t sizeEnd = xml.find("</Size>", sizeStart);
            if (sizeEnd != std::string::npos) {
                sizeVal = std::stoull(xml.substr(sizeStart, sizeEnd - sizeStart));
            }
        }

        NexradFile f;
        f.key = key;
        f.url = "/" + key;
        f.size = sizeVal;
        files.push_back(std::move(f));
    }

    // Sort by key (which includes timestamp) so latest is last
    std::sort(files.begin(), files.end(),
              [](const NexradFile& a, const NexradFile& b) {
                  return a.key < b.key;
              });

    return files;
}
