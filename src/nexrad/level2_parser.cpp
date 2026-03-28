#include "level2_parser.h"
#include "products.h"
#include <bzlib.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

// Decompress gzip data using Windows Compression API (Win8+) or zlib on Linux.
#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0602
#endif
#include <windows.h>
#include <compressapi.h>
#pragma comment(lib, "cabinet.lib")
#ifndef COMPRESS_ALGORITHM_GZIP
#define COMPRESS_ALGORITHM_GZIP 0x0A
#endif
#elif defined(__has_include)
#if __has_include(<zlib.h>)
#include <zlib.h>
#define CURSDAR_HAVE_ZLIB 1
#endif
#endif

namespace {

constexpr size_t kArchiveHeaderSize = sizeof(VolumeHeader);
constexpr size_t kRecordStride = 2432;
constexpr size_t kMaxDecompressedBlockBytes = 256u * 1024u * 1024u;
constexpr size_t kMaxCombinedBytes = 512u * 1024u * 1024u;
constexpr int kMaxMsg31Blocks = 32;

struct BlockDesc {
    size_t offset = 0;
    size_t size = 0;
    bool compressed = false;
};

template <typename T>
bool readStruct(const uint8_t* data, size_t size, size_t offset, T& out) {
    if (offset > size || sizeof(T) > size - offset) return false;
    std::memcpy(&out, data + offset, sizeof(T));
    return true;
}

bool readBe32(const uint8_t* data, size_t size, size_t offset, uint32_t& out) {
    uint32_t raw = 0;
    if (!readStruct(data, size, offset, raw)) return false;
    out = bswap32(raw);
    return true;
}

bool readBe16Value(const uint8_t* data, size_t size, size_t offset, uint16_t& out) {
    uint16_t raw = 0;
    if (!readStruct(data, size, offset, raw)) return false;
    out = bswap16(raw);
    return true;
}

bool isBZ2Magic(const uint8_t* data, size_t size) {
    return size >= 4 &&
           data[0] == 'B' &&
           data[1] == 'Z' &&
           data[2] == 'h' &&
           data[3] >= '1' &&
           data[3] <= '9';
}

bool appendBytes(std::vector<uint8_t>& out, const uint8_t* data, size_t size) {
    if (size > kMaxCombinedBytes - out.size()) return false;
    out.insert(out.end(), data, data + size);
    return true;
}

static std::vector<BlockDesc> findBlocks(const uint8_t* data, size_t size) {
    std::vector<BlockDesc> blocks;
    size_t pos = 0;

    while (pos + 4 <= size) {
        if (isBZ2Magic(data + pos, size - pos)) {
            blocks.clear();
            return blocks;
        }

        uint32_t rawSize = 0;
        if (!readStruct(data, size, pos, rawSize)) {
            blocks.clear();
            return blocks;
        }

        const int32_t sizeVal = (int32_t)bswap32(rawSize);
        pos += 4;

        if (sizeVal == 0) continue;

        const uint64_t blockSize64 =
            sizeVal < 0 ? uint64_t(-(int64_t)sizeVal) : uint64_t(sizeVal);
        if (blockSize64 == 0 || blockSize64 > size - pos) {
            blocks.clear();
            return blocks;
        }

        const size_t blockSize = (size_t)blockSize64;
        const bool hasBZ2 = isBZ2Magic(data + pos, size - pos);
        const bool compressed = hasBZ2;
        if (sizeVal < 0 && !hasBZ2) {
            blocks.clear();
            return blocks;
        }

        blocks.push_back({pos, blockSize, compressed});
        pos += blockSize;
    }

    return blocks;
}

bool decompressBZ2Stream(const uint8_t* data, size_t size, std::vector<uint8_t>& output,
                         size_t* consumedOut = nullptr) {
    if (!isBZ2Magic(data, size)) return false;

    size_t outCap = std::max<size_t>(65536, std::min(size * 10, kMaxDecompressedBlockBytes));
    if (outCap == 0) outCap = 65536;
    output.assign(outCap, 0);

    bz_stream strm = {};
    if (BZ2_bzDecompressInit(&strm, 0, 0) != BZ_OK) return false;

    strm.next_in = (char*)data;
    strm.avail_in = (unsigned int)std::min<size_t>(size, std::numeric_limits<unsigned int>::max());
    strm.next_out = (char*)output.data();
    strm.avail_out = (unsigned int)std::min<size_t>(outCap, std::numeric_limits<unsigned int>::max());

    size_t totalOut = 0;
    bool success = false;
    while (true) {
        const int ret = BZ2_bzDecompress(&strm);
        totalOut = output.size() - strm.avail_out;

        if (ret == BZ_STREAM_END) {
            success = true;
            break;
        }
        if (ret != BZ_OK) {
            success = false;
            break;
        }

        if (strm.avail_out == 0) {
            if (output.size() >= kMaxDecompressedBlockBytes) {
                success = false;
                break;
            }
            const size_t newCap = std::min(output.size() * 2, kMaxDecompressedBlockBytes);
            output.resize(newCap);
            strm.next_out = (char*)output.data() + totalOut;
            strm.avail_out =
                (unsigned int)std::min<size_t>(newCap - totalOut, std::numeric_limits<unsigned int>::max());
        }
    }

    if (consumedOut) {
        *consumedOut = size - strm.avail_in;
    }

    BZ2_bzDecompressEnd(&strm);
    if (!success) {
        output.clear();
        return false;
    }

    output.resize(totalOut);
    return true;
}

std::vector<std::vector<uint8_t>> decompressLegacyStreams(const uint8_t* data, size_t size) {
    std::vector<std::vector<uint8_t>> blocks;
    size_t pos = 0;

    while (pos + 4 <= size) {
        while (pos + 4 <= size && !isBZ2Magic(data + pos, size - pos)) pos++;
        if (pos + 4 > size) break;

        std::vector<uint8_t> block;
        size_t consumed = 0;
        if (decompressBZ2Stream(data + pos, size - pos, block, &consumed) &&
            !block.empty() && consumed > 0) {
            blocks.push_back(std::move(block));
            pos += consumed;
        } else {
            pos++;
        }
    }

    return blocks;
}

std::string trimRadarId(const char radarId[4]) {
    std::string id(radarId, 4);
    while (!id.empty() && id.back() == ' ')
        id.pop_back();
    return id;
}

static std::vector<uint8_t> decompressGzip(const std::vector<uint8_t>& data) {
#ifdef _WIN32
    DECOMPRESSOR_HANDLE h = NULL;
    if (!CreateDecompressor(COMPRESS_ALGORITHM_GZIP, NULL, &h)) return {};

    SIZE_T outSize = 0;
    Decompress(h, data.data(), data.size(), NULL, 0, &outSize);
    if (outSize == 0 || outSize > 200u * 1024u * 1024u) {
        CloseDecompressor(h);
        return {};
    }

    std::vector<uint8_t> output(outSize);
    SIZE_T actualSize = 0;
    const BOOL ok = Decompress(h, data.data(), data.size(),
                               output.data(), outSize, &actualSize);
    CloseDecompressor(h);

    if (!ok) return {};
    output.resize(actualSize);
    return output;
#elif defined(CURSDAR_HAVE_ZLIB)
    z_stream strm = {};
    strm.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(data.data()));
    strm.avail_in = (uInt)std::min<size_t>(data.size(), std::numeric_limits<uInt>::max());

    if (inflateInit2(&strm, 16 + MAX_WBITS) != Z_OK) return {};

    std::vector<uint8_t> output(std::max<size_t>(65536, data.size() * 3));
    int ret = Z_OK;
    while (ret == Z_OK) {
        if (strm.total_out >= output.size()) {
            const size_t newSize = std::min(output.size() * 2, 200u * 1024u * 1024u);
            if (newSize <= output.size()) {
                inflateEnd(&strm);
                return {};
            }
            output.resize(newSize);
        }

        strm.next_out = reinterpret_cast<Bytef*>(output.data() + strm.total_out);
        strm.avail_out =
            (uInt)std::min<size_t>(output.size() - strm.total_out, std::numeric_limits<uInt>::max());
        ret = inflate(&strm, Z_NO_FLUSH);
    }

    inflateEnd(&strm);
    if (ret != Z_STREAM_END) return {};
    output.resize(strm.total_out);
    return output;
#else
    return {};
#endif
}

} // namespace

std::vector<size_t> Level2Parser::findBZ2Blocks(const uint8_t* data, size_t size) {
    std::vector<size_t> offsets;
    for (size_t i = 0; i + 4 <= size; i++) {
        if (isBZ2Magic(data + i, size - i))
            offsets.push_back(i);
    }
    return offsets;
}

std::vector<uint8_t> Level2Parser::decompressBZ2Block(const uint8_t* data, size_t maxSize) {
    std::vector<uint8_t> output;
    if (!decompressBZ2Stream(data, maxSize, output, nullptr))
        output.clear();
    return output;
}

void Level2Parser::parseMessages(const uint8_t* data, size_t size, ParsedRadarData& out) {
    size_t pos = 0;

    while (pos + sizeof(CtmHeader) + sizeof(MessageHeader) <= size) {
        MessageHeader mh = {};
        if (!readStruct(data, size, pos + sizeof(CtmHeader), mh)) break;

        const uint8_t mtype = mh.messageType();
        const uint16_t msize = mh.messageSize();

        if (mtype == 31 && msize > 0 && msize < 30000) {
            // MSG31 messages can span multiple 2432-byte records. The message_size
            // field already includes the 16-byte message header, so the next CTM
            // header may begin mid-record rather than on the next 2432-byte boundary.
            const size_t msgSize = (size_t)msize * 2;
            const size_t msgDataOffset = pos + sizeof(CtmHeader) + sizeof(MessageHeader);
            if (msgDataOffset < size) {
                parseMsg31(data + msgDataOffset, std::min(msgSize, size - msgDataOffset), out);
            }
            pos += std::max(kRecordStride, msgSize + sizeof(CtmHeader));
        } else {
            pos += kRecordStride;
        }
    }
}

void Level2Parser::parseMsg31(const uint8_t* data, size_t size, ParsedRadarData& out) {
    Msg31Header hdr = {};
    if (!readStruct(data, size, 0, hdr)) return;

    size_t radialSize = size;
    const uint16_t radialLength = hdr.radialLength();
    if (radialLength >= sizeof(Msg31Header) && radialLength <= size)
        radialSize = radialLength;

    ParsedRadial radial;
    radial.azimuth = hdr.azimuth();
    radial.elevation = hdr.elevation();
    radial.radial_status = hdr.radial_status;

    if (!std::isfinite(radial.azimuth) || !std::isfinite(radial.elevation)) return;
    if (radial.azimuth < 0.0f || radial.azimuth >= 360.0f) return;
    if (radial.elevation < -2.0f || radial.elevation > 90.0f) return;

    const int numBlocks = hdr.dataBlockCount();
    if (numBlocks < 1 || numBlocks > kMaxMsg31Blocks) return;

    const size_t ptrTableOffset = sizeof(Msg31Header);
    if (ptrTableOffset + (size_t)numBlocks * sizeof(uint32_t) > radialSize) return;

    const std::string radarId = trimRadarId(hdr.radar_id);

    for (int i = 0; i < numBlocks; i++) {
        uint32_t ptr = 0;
        if (!readBe32(data, radialSize, ptrTableOffset + (size_t)i * sizeof(uint32_t), ptr)) continue;
        if (ptr == 0 || ptr >= radialSize) continue;

        DataBlockId blockId = {};
        if (!readStruct(data, radialSize, ptr, blockId)) continue;

        if (blockId.block_type == 'R' &&
            blockId.name[0] == 'V' &&
            blockId.name[1] == 'O' &&
            blockId.name[2] == 'L') {
            VolumeDataBlock vol = {};
            if (!readStruct(data, radialSize, ptr, vol)) continue;

            const float lat = vol.lat();
            const float lon = vol.lon();
            if (std::isfinite(lat) && std::isfinite(lon) &&
                lat >= -90.0f && lat <= 90.0f &&
                lon >= -180.0f && lon <= 180.0f) {
                if (out.station_lat == 0.0f && out.station_lon == 0.0f) {
                    out.station_lat = lat;
                    out.station_lon = lon;
                    if (!radarId.empty()) out.station_id = radarId;
                    out.station_height_m = bswap16(vol.height_be);
                }
            }
            continue;
        }

        if (blockId.block_type != 'D') continue;

        MomentDataBlock mom = {};
        if (!readStruct(data, radialSize, ptr, mom)) continue;

        if (mom.data_word_size != 8 && mom.data_word_size != 16) continue;

        const int prodIdx = productFromCode(blockId.name);
        if (prodIdx < 0) continue;

        ParsedMoment moment;
        moment.product_index = prodIdx;
        moment.num_gates = mom.numGates();
        moment.first_gate_m = mom.firstGate();
        moment.gate_spacing_m = mom.gateSpacing();
        moment.scale = mom.scale();
        moment.offset = mom.offset();

        if (moment.num_gates == 0 || moment.num_gates > 2000) continue;
        if (moment.gate_spacing_m == 0) continue;
        if (!std::isfinite(moment.scale) || moment.scale == 0.0f) continue;
        if (!std::isfinite(moment.offset)) continue;

        const size_t gateDataOffset = ptr + sizeof(MomentDataBlock);
        const size_t bytesPerGate = mom.data_word_size == 16 ? 2u : 1u;
        const size_t gateBytes = (size_t)moment.num_gates * bytesPerGate;
        if (gateDataOffset > radialSize || gateBytes > radialSize - gateDataOffset) continue;

        moment.gates.resize(moment.num_gates);
        if (mom.data_word_size == 16) {
            for (uint16_t g = 0; g < moment.num_gates; g++) {
                uint16_t value = 0;
                if (!readBe16Value(data, radialSize, gateDataOffset + (size_t)g * 2, value)) {
                    moment.gates.clear();
                    break;
                }
                moment.gates[g] = value;
            }
            if (moment.gates.empty()) continue;
        } else {
            std::transform(data + gateDataOffset, data + gateDataOffset + gateBytes,
                           moment.gates.begin(),
                           [](uint8_t v) { return (uint16_t)v; });
        }

        radial.moments.push_back(std::move(moment));
    }

    if (radial.moments.empty()) return;

    if (out.sweeps.empty()) {
        out.sweeps.push_back({});
    }

    const float elev = roundf(radial.elevation * 10.0f) / 10.0f;
    const int sweepId = (int)hdr.elevation_number;

    ParsedSweep* targetSweep = nullptr;
    for (auto& s : out.sweeps) {
        if (s.sweep_number == sweepId) {
            targetSweep = &s;
            break;
        }
    }

    if (!targetSweep) {
        out.sweeps.push_back({});
        targetSweep = &out.sweeps.back();
        targetSweep->elevation_angle = elev;
        targetSweep->sweep_number = sweepId;
    }

    targetSweep->radials.push_back(std::move(radial));
}

void Level2Parser::organizeSweeps(ParsedRadarData& out) {
    out.sweeps.erase(
        std::remove_if(out.sweeps.begin(), out.sweeps.end(),
                       [](const ParsedSweep& s) { return s.radials.size() < 10; }),
        out.sweeps.end());

    for (auto& sweep : out.sweeps) {
        std::sort(sweep.radials.begin(), sweep.radials.end(),
                  [](const ParsedRadial& a, const ParsedRadial& b) {
                      return a.azimuth < b.azimuth;
                  });

        if (sweep.radials.size() > 1) {
            auto it = std::unique(sweep.radials.begin(), sweep.radials.end(),
                                  [](const ParsedRadial& a, const ParsedRadial& b) {
                                      return fabsf(a.azimuth - b.azimuth) < 0.01f;
                                  });
            sweep.radials.erase(it, sweep.radials.end());
        }
    }

    std::sort(out.sweeps.begin(), out.sweeps.end(),
              [](const ParsedSweep& a, const ParsedSweep& b) {
                  if (fabsf(a.elevation_angle - b.elevation_angle) > 0.05f)
                      return a.elevation_angle < b.elevation_angle;
                  return a.sweep_number < b.sweep_number;
              });
}

ParsedRadarData Level2Parser::parse(const std::vector<uint8_t>& fileData) {
    return parse(fileData, nullptr);
}

ParsedRadarData Level2Parser::parse(const std::vector<uint8_t>& fileData,
                                    ProgressCallback cb) {
    ParsedRadarData result;
    if (fileData.size() < kArchiveHeaderSize) return result;

    if (fileData.size() >= 2 && fileData[0] == 0x1F && fileData[1] == 0x8B) {
        auto decompressed = decompressGzip(fileData);
        if (decompressed.empty()) {
            std::printf("  gzip decompression failed\n");
            return result;
        }
        return parse(decompressed, cb);
    }

    VolumeHeader vh = {};
    if (readStruct(fileData.data(), fileData.size(), 0, vh) &&
        fileData[0] == 'A' && fileData[1] == 'R') {
        result.station_id = vh.station();
    }

    const uint8_t* archiveData = fileData.data() + kArchiveHeaderSize;
    const size_t archiveSize = fileData.size() - kArchiveHeaderSize;

    std::vector<uint8_t> combined;
    int totalBlocks = 0;

    const auto blocks = findBlocks(archiveData, archiveSize);
    if (!blocks.empty()) {
        totalBlocks = (int)blocks.size();
        for (int i = 0; i < totalBlocks; i++) {
            const auto& block = blocks[i];
            std::vector<uint8_t> decoded;
            if (block.compressed) {
                decoded = decompressBZ2Block(archiveData + block.offset, block.size);
            } else {
                decoded.assign(archiveData + block.offset, archiveData + block.offset + block.size);
            }

            if (!decoded.empty() && !appendBytes(combined, decoded.data(), decoded.size())) {
                combined.clear();
                break;
            }

            if (cb) cb(i + 1, totalBlocks);
        }
    }

    if (combined.empty()) {
        const auto fallbackBlocks = decompressLegacyStreams(archiveData, archiveSize);
        totalBlocks = (int)fallbackBlocks.size();
        for (int i = 0; i < totalBlocks; i++) {
            if (!appendBytes(combined, fallbackBlocks[i].data(), fallbackBlocks[i].size())) {
                combined.clear();
                break;
            }
            if (cb) cb(i + 1, totalBlocks);
        }
    }

    if (!combined.empty()) {
        parseMessages(combined.data(), combined.size(), result);
    }

    organizeSweeps(result);

    int totalRadials = 0;
    for (const auto& sweep : result.sweeps)
        totalRadials += (int)sweep.radials.size();

    std::printf("Parsed %s: %d sweeps, %d radials, %d blocks, %zu bytes, station=(%f, %f)\n",
                result.station_id.c_str(), (int)result.sweeps.size(), totalRadials,
                totalBlocks, fileData.size(), result.station_lat, result.station_lon);

    return result;
}
