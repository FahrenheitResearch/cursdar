#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

// ── Byte-swap utilities (NEXRAD is big-endian) ──────────────
#ifdef _MSC_VER
#include <intrin.h>
inline uint16_t bswap16(uint16_t v) { return _byteswap_ushort(v); }
inline uint32_t bswap32(uint32_t v) { return _byteswap_ulong(v); }
#else
inline uint16_t bswap16(uint16_t v) { return __builtin_bswap16(v); }
inline uint32_t bswap32(uint32_t v) { return __builtin_bswap32(v); }
#endif

inline float bswapf(float v) {
    uint32_t i;
    memcpy(&i, &v, 4);
    i = bswap32(i);
    float r;
    memcpy(&r, &i, 4);
    return r;
}

inline int16_t bswap16s(int16_t v) { return (int16_t)bswap16((uint16_t)v); }

// ── Raw binary structures (packed, big-endian on disk) ──────

#pragma pack(push, 1)

struct VolumeHeader {
    char     tape[9];       // e.g. "AR2V0006."
    char     extension[3];  // extension number
    uint32_t date_be;       // Modified Julian date (big-endian)
    uint32_t time_be;       // Milliseconds since midnight (big-endian)
    char     icao[4];       // Station ICAO (e.g. "KTLX")

    uint32_t date() const { return bswap32(date_be); }
    uint32_t time() const { return bswap32(time_be); }
    std::string station() const { return std::string(icao, 4); }
};
static_assert(sizeof(VolumeHeader) == 24, "VolumeHeader must be 24 bytes");

// CTM header - legacy, always zeros in modern files
struct CtmHeader {
    uint8_t data[12];
};

struct MessageHeader {
    uint16_t message_size_be;    // in halfwords (×2 for bytes)
    uint8_t  channel_id;
    uint8_t  message_type;
    uint16_t id_sequence_be;
    uint16_t julian_date_be;
    uint32_t milliseconds_be;
    uint16_t num_segments_be;
    uint16_t segment_number_be;

    uint16_t messageSize()  const { return bswap16(message_size_be); }
    uint8_t  messageType()  const { return message_type; }
    uint16_t numSegments()  const { return bswap16(num_segments_be); }
    uint16_t segmentNumber() const { return bswap16(segment_number_be); }
    uint16_t sequenceId()   const { return bswap16(id_sequence_be); }
    uint32_t milliseconds() const { return bswap32(milliseconds_be); }
};
static_assert(sizeof(MessageHeader) == 16, "MessageHeader must be 16 bytes");

// Message Type 31 data header
struct Msg31Header {
    char     radar_id[4];
    uint32_t collection_time_be;   // ms since midnight
    uint16_t collection_date_be;   // Modified Julian date
    uint16_t azimuth_number_be;
    float    azimuth_angle_be;     // degrees
    uint8_t  compression;
    uint8_t  spare;
    uint16_t radial_length_be;
    uint8_t  azimuth_resolution;   // 1=0.5deg, 2=1.0deg
    uint8_t  radial_status;        // 0=start_elev, 1=intermed, 2=end_elev, 3=start_vol, 4=end_vol
    uint8_t  elevation_number;
    uint8_t  cut_sector;
    float    elevation_angle_be;
    uint8_t  blanking;
    uint8_t  azimuth_indexing;
    uint16_t data_block_count_be;

    float    azimuth()       const { return bswapf(azimuth_angle_be); }
    float    elevation()     const { return bswapf(elevation_angle_be); }
    uint16_t azimuthNumber() const { return bswap16(azimuth_number_be); }
    uint16_t dataBlockCount()const { return bswap16(data_block_count_be); }
    uint16_t radialLength()  const { return bswap16(radial_length_be); }
};

// Data block pointer (uint32_t big-endian offset from start of Msg31Header)
// Follows immediately after Msg31Header, one per data block

// Generic data block header (first 4 bytes of any data block)
struct DataBlockId {
    char block_type;   // 'R' for metadata, 'D' for moment data
    char name[3];      // "VOL", "ELV", "RAD", "REF", "VEL", etc.
};

// Volume data block
struct VolumeDataBlock {
    DataBlockId id;        // type='R', name="VOL"
    uint16_t    size_be;
    uint8_t     version_major;
    uint8_t     version_minor;
    float       lat_be;
    float       lon_be;
    uint16_t    height_be;     // meters above MSL
    uint8_t     feedhorn_type;
    float       calibration_be;

    float lat() const { return bswapf(lat_be); }
    float lon() const { return bswapf(lon_be); }
};

// Moment data block (REF, VEL, SW, ZDR, PHI, RHO, CFP)
struct MomentDataBlock {
    DataBlockId id;              // type='D'
    uint32_t    reserved_be;
    uint16_t    num_gates_be;
    uint16_t    first_gate_be;   // meters
    uint16_t    gate_spacing_be; // meters
    uint16_t    threshold_be;
    int16_t     snr_threshold_be;
    uint8_t     control_flags;
    uint8_t     data_word_size;  // 8 or 16 bits
    float       scale_be;
    float       offset_be;

    uint16_t numGates()     const { return bswap16(num_gates_be); }
    uint16_t firstGate()    const { return bswap16(first_gate_be); }
    uint16_t gateSpacing()  const { return bswap16(gate_spacing_be); }
    float    scale()        const { return bswapf(scale_be); }
    float    offset()       const { return bswapf(offset_be); }

    // Pointer to raw gate data (immediately follows this struct)
    const uint8_t* gateData() const {
        return reinterpret_cast<const uint8_t*>(this) + 28;
    }
};

#pragma pack(pop)

// ── Parsed data structures (host-side, native endian) ───────

struct ParsedMoment {
    int      product_index;     // Product enum index (-1 if unknown)
    uint16_t num_gates;
    uint16_t first_gate_m;      // meters
    uint16_t gate_spacing_m;    // meters
    float    scale;
    float    offset;
    std::vector<uint16_t> gates; // Raw gate values (8-bit promoted to 16)
};

struct ParsedRadial {
    float azimuth;              // degrees [0, 360)
    float elevation;            // degrees
    uint8_t radial_status;      // sweep boundary indicator
    std::vector<ParsedMoment> moments;
};

struct ParsedSweep {
    float elevation_angle;
    int   sweep_number;          // Message 31 elevation_number (cut index within volume)
    std::vector<ParsedRadial> radials; // sorted by azimuth
};

struct ParsedRadarData {
    std::string station_id;
    float       station_lat = 0;
    float       station_lon = 0;
    uint16_t    station_height_m = 0;
    std::vector<ParsedSweep> sweeps;

    // Get the lowest elevation sweep (most useful for plan view)
    const ParsedSweep* lowestSweep() const {
        if (sweeps.empty()) return nullptr;
        const ParsedSweep* best = &sweeps[0];
        for (auto& s : sweeps) {
            if (s.elevation_angle < best->elevation_angle)
                best = &s;
        }
        return best;
    }
};
