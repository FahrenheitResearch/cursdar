#include "level2_parser.h"
#include "products.h"
#include <bzlib.h>
#include <cstring>
#include <algorithm>
#include <thread>
#include <future>
#include <cstdio>
#include <cmath>

// ── BZ2 Block Finding ───────────────────────────────────────

// Block descriptor: offset and size of a compressed or uncompressed block
struct BlockDesc {
    size_t offset;
    size_t size;
    bool   compressed; // true = BZ2, false = uncompressed
};

static bool isBZ2Magic(const uint8_t* p) {
    return p[0] == 'B' && p[1] == 'Z' && p[2] == 'h' && p[3] >= '1' && p[3] <= '9';
}

// Find blocks using the proper size-prefix format:
// After volume header, each block starts with a 4-byte big-endian int32:
//   negative: |value| bytes of BZ2-compressed data follows
//   positive: value bytes of uncompressed data follows
// If the 4 bytes are BZ2 magic, there's no size prefix (handle gracefully)
static std::vector<BlockDesc> findBlocks(const uint8_t* data, size_t size) {
    std::vector<BlockDesc> blocks;
    size_t pos = 0;

    while (pos + 4 < size) {
        const uint8_t* p = data + pos;

        // Check if this is BZ2 magic (no size prefix)
        if (isBZ2Magic(p)) {
            // Scan for the next BZ2 block or EOF
            size_t end = pos + 4;
            while (end + 4 < size) {
                // Check for next size prefix (negative int32 where |val| < remaining)
                int32_t peek = (int32_t)bswap32(*(const uint32_t*)(data + end));
                if (peek < 0 && -peek < (int32_t)(size - end) && -peek > 100) {
                    // Looks like a size prefix for the next block
                    break;
                }
                // Check for BZ2 magic
                if (isBZ2Magic(data + end)) break;
                end++;
            }
            blocks.push_back({pos, end - pos, true});
            pos = end;
            continue;
        }

        // Read 4-byte size prefix
        int32_t sizeVal = (int32_t)bswap32(*(const uint32_t*)p);
        pos += 4;

        if (sizeVal == 0) continue; // skip empty

        if (sizeVal < 0) {
            // BZ2 compressed block
            size_t blockSize = (size_t)(-sizeVal);
            if (pos + blockSize > size) blockSize = size - pos;
            if (blockSize > 0) {
                blocks.push_back({pos, blockSize, true});
            }
            pos += blockSize;
        } else {
            // Uncompressed block
            size_t blockSize = (size_t)sizeVal;
            if (pos + blockSize > size) blockSize = size - pos;
            if (blockSize > 0) {
                blocks.push_back({pos, blockSize, false});
            }
            pos += blockSize;
        }
    }
    return blocks;
}

// Legacy fallback: scan for BZ2 magic bytes
std::vector<size_t> Level2Parser::findBZ2Blocks(const uint8_t* data, size_t size) {
    std::vector<size_t> offsets;
    for (size_t i = 0; i + 4 < size; i++) {
        if (isBZ2Magic(data + i)) {
            offsets.push_back(i);
        }
    }
    return offsets;
}

// ── BZ2 Decompression ───────────────────────────────────────

std::vector<uint8_t> Level2Parser::decompressBZ2Block(const uint8_t* data, size_t maxSize) {
    // Initial output buffer: 10x compressed size, grow if needed
    size_t outCap = maxSize * 10;
    if (outCap < 65536) outCap = 65536;
    if (outCap > 50 * 1024 * 1024) outCap = 50 * 1024 * 1024;

    std::vector<uint8_t> output(outCap);

    bz_stream strm = {};
    int ret = BZ2_bzDecompressInit(&strm, 0, 0);
    if (ret != BZ_OK) return {};

    strm.next_in = (char*)data;
    strm.avail_in = (unsigned int)maxSize;
    strm.next_out = (char*)output.data();
    strm.avail_out = (unsigned int)outCap;

    size_t totalOut = 0;
    while (true) {
        ret = BZ2_bzDecompress(&strm);
        totalOut = outCap - strm.avail_out;

        if (ret == BZ_STREAM_END) break;
        if (ret != BZ_OK) {
            BZ2_bzDecompressEnd(&strm);
            output.resize(totalOut);
            return output; // Return what we got
        }

        if (strm.avail_out == 0) {
            // Grow buffer
            size_t newCap = outCap * 2;
            output.resize(newCap);
            strm.next_out = (char*)output.data() + totalOut;
            strm.avail_out = (unsigned int)(newCap - totalOut);
            outCap = newCap;
        }
    }

    BZ2_bzDecompressEnd(&strm);
    output.resize(totalOut);
    return output;
}

// ── Message Parsing ─────────────────────────────────────────

void Level2Parser::parseMessages(const uint8_t* data, size_t size, ParsedRadarData& out) {
    // Match rustdar's approach: sequential read with CTM(12) + MsgHeader(16)
    // Variable-length stepping: (message_size * 2 + 12) for MSG31, 2432 for others

    size_t pos = 0;

    while (pos + 28 < size) {
        // Each record: CTM header (12 bytes) + Message Header (16 bytes) + data
        size_t ctm_start = pos;

        if (pos + 12 + 16 > size) break;

        // Skip CTM header (12 bytes)
        const MessageHeader* mh = reinterpret_cast<const MessageHeader*>(data + pos + 12);
        uint8_t mtype = mh->messageType();
        uint16_t msize = mh->messageSize();

        if (mtype == 31 && msize > 0 && msize < 30000) {
            size_t msgDataOffset = pos + 12 + 16; // after CTM + MessageHeader
            size_t msgSize = (size_t)msize * 2;

            if (msgDataOffset + 30 < size) {
                parseMsg31(data + msgDataOffset,
                           std::min(msgSize, size - msgDataOffset), out);
            }

            // Advance: (message_size_halfwords * 2) + 12 for CTM, minimum 2432
            size_t recordLen = (size_t)msize * 2 + 12;
            if (recordLen < 2432) recordLen = 2432;
            pos += recordLen;
        } else {
            // Non-MSG31 or invalid: fixed 2432-byte step
            pos += 2432;
        }
    }
}

void Level2Parser::parseMsg31(const uint8_t* data, size_t size, ParsedRadarData& out) {
    if (size < sizeof(Msg31Header)) return;

    const Msg31Header* hdr = reinterpret_cast<const Msg31Header*>(data);

    ParsedRadial radial;
    radial.azimuth = hdr->azimuth();
    radial.elevation = hdr->elevation();
    radial.radial_status = hdr->radial_status;

    // Validate
    if (radial.azimuth < 0 || radial.azimuth >= 360.0f) return;
    if (radial.elevation < -2.0f || radial.elevation > 90.0f) return;

    int numBlocks = hdr->dataBlockCount();
    if (numBlocks < 1 || numBlocks > 20) return;

    // Data block pointers follow the header at offset 32
    const uint32_t* blockPtrs = reinterpret_cast<const uint32_t*>(data + 32);

    for (int i = 0; i < numBlocks && i < 10; i++) {
        uint32_t ptr = bswap32(blockPtrs[i]);
        if (ptr == 0 || ptr >= size) continue;

        const DataBlockId* blockId = reinterpret_cast<const DataBlockId*>(data + ptr);

        // Check for volume data block (station lat/lon)
        if (blockId->block_type == 'R' && blockId->name[0] == 'V' &&
            blockId->name[1] == 'O' && blockId->name[2] == 'L') {
            if (ptr + sizeof(VolumeDataBlock) <= size) {
                const VolumeDataBlock* vol = reinterpret_cast<const VolumeDataBlock*>(data + ptr);
                if (out.station_lat == 0 && out.station_lon == 0) {
                    out.station_lat = vol->lat();
                    out.station_lon = vol->lon();
                    out.station_id = std::string(hdr->radar_id, 4);
                    // Trim trailing spaces
                    while (!out.station_id.empty() && out.station_id.back() == ' ')
                        out.station_id.pop_back();
                }
            }
        }

        // Check for moment data block
        if (blockId->block_type == 'D') {
            if (ptr + 28 > size) continue;
            const MomentDataBlock* mom = reinterpret_cast<const MomentDataBlock*>(data + ptr);

            int prodIdx = productFromCode(blockId->name);
            if (prodIdx < 0) continue;

            ParsedMoment moment;
            moment.product_index = prodIdx;
            moment.num_gates = mom->numGates();
            moment.first_gate_m = mom->firstGate();
            moment.gate_spacing_m = mom->gateSpacing();
            moment.scale = mom->scale();
            moment.offset = mom->offset();

            if (moment.num_gates == 0 || moment.num_gates > 2000) continue;
            if (moment.gate_spacing_m == 0) continue;
            if (moment.scale == 0.0f) continue;

            size_t gateDataOffset = ptr + 28;
            size_t gateBytes = moment.num_gates * (mom->data_word_size == 16 ? 2 : 1);
            if (gateDataOffset + gateBytes > size) continue;

            moment.gates.resize(moment.num_gates);
            const uint8_t* gd = data + gateDataOffset;

            if (mom->data_word_size == 16) {
                for (int g = 0; g < moment.num_gates; g++) {
                    moment.gates[g] = bswap16(*(const uint16_t*)(gd + g * 2));
                }
            } else {
                for (int g = 0; g < moment.num_gates; g++) {
                    moment.gates[g] = gd[g];
                }
            }

            radial.moments.push_back(std::move(moment));
        }
    }

    if (!radial.moments.empty()) {
        // Store radial - we'll organize into sweeps later
        // Use a temporary "sweep 0" to collect all radials
        if (out.sweeps.empty()) {
            out.sweeps.push_back({});
        }

        // Find or create sweep for this elevation AND gate configuration
        // Split cuts produce two sweeps at the same elevation with different gate params
        // (surveillance mode has more REF gates, Doppler mode has VEL/SW gates)
        float elev = roundf(radial.elevation * 10.0f) / 10.0f;

        // Get the REF gate count as a "scan mode" discriminator
        int refGates = 0;
        for (auto& m : radial.moments) {
            if (m.product_index == 0) { refGates = m.num_gates; break; }
        }

        ParsedSweep* targetSweep = nullptr;
        for (auto& s : out.sweeps) {
            if (fabsf(s.elevation_angle - elev) < 0.15f &&
                s.sweep_number == refGates) { // reuse sweep_number to store gate key
                targetSweep = &s;
                break;
            }
        }
        if (!targetSweep) {
            out.sweeps.push_back({});
            targetSweep = &out.sweeps.back();
            targetSweep->elevation_angle = elev;
            targetSweep->sweep_number = refGates; // gate key for matching
        }
        targetSweep->radials.push_back(std::move(radial));
    }
}

// ── Sweep Organization ──────────────────────────────────────

void Level2Parser::organizeSweeps(ParsedRadarData& out) {
    // Remove sweeps with too few radials
    out.sweeps.erase(
        std::remove_if(out.sweeps.begin(), out.sweeps.end(),
                       [](const ParsedSweep& s) { return s.radials.size() < 10; }),
        out.sweeps.end());

    for (auto& sweep : out.sweeps) {
        // Sort radials by azimuth
        std::sort(sweep.radials.begin(), sweep.radials.end(),
                  [](const ParsedRadial& a, const ParsedRadial& b) {
                      return a.azimuth < b.azimuth;
                  });

        // Remove exact azimuth duplicates
        if (sweep.radials.size() > 1) {
            auto it = std::unique(sweep.radials.begin(), sweep.radials.end(),
                                  [](const ParsedRadial& a, const ParsedRadial& b) {
                                      return fabsf(a.azimuth - b.azimuth) < 0.01f;
                                  });
            sweep.radials.erase(it, sweep.radials.end());
        }
    }

    // Sort sweeps by elevation angle, then by gate count (more gates = surveillance = first)
    std::sort(out.sweeps.begin(), out.sweeps.end(),
              [](const ParsedSweep& a, const ParsedSweep& b) {
                  if (fabsf(a.elevation_angle - b.elevation_angle) > 0.05f)
                      return a.elevation_angle < b.elevation_angle;
                  return a.sweep_number > b.sweep_number; // more gates first
              });

    // Re-number sweeps
    for (int i = 0; i < (int)out.sweeps.size(); i++)
        out.sweeps[i].sweep_number = i;
}

// ── Main Parse Entry Point ──────────────────────────────────

ParsedRadarData Level2Parser::parse(const std::vector<uint8_t>& fileData) {
    return parse(fileData, nullptr);
}

ParsedRadarData Level2Parser::parse(const std::vector<uint8_t>& fileData,
                                     ProgressCallback cb) {
    ParsedRadarData result;

    if (fileData.size() < 24) return result;

    // Read volume header
    const VolumeHeader* vh = reinterpret_cast<const VolumeHeader*>(fileData.data());

    // Verify it looks like a Level 2 file
    if (fileData[0] == 'A' && fileData[1] == 'R') {
        result.station_id = vh->station();
    }

    // Sequential BZ2 stream decompression - properly finds stream boundaries
    // by letting the BZ2 decompressor tell us where each stream ends
    std::vector<std::vector<uint8_t>> decompressedBlocks;
    {
        const uint8_t* ptr = fileData.data() + 24;
        size_t remaining = fileData.size() - 24;

        while (remaining > 10) {
            // Skip any non-BZ2 data (size prefixes, metadata)
            bool foundBZ2 = false;
            size_t scanLimit = std::min(remaining, (size_t)500);
            for (size_t skip = 0; skip < scanLimit; skip++) {
                if (isBZ2Magic(ptr + skip)) {
                    ptr += skip;
                    remaining -= skip;
                    foundBZ2 = true;
                    break;
                }
            }
            if (!foundBZ2) {
                // No BZ2 within next 100 bytes - skip ahead
                ptr += scanLimit;
                remaining -= scanLimit;
                continue;
            }

            // Decompress this BZ2 stream - the decompressor consumes
            // exactly the bytes it needs and BZ2_bzDecompress returns BZ_STREAM_END
            bz_stream strm = {};
            if (BZ2_bzDecompressInit(&strm, 0, 0) != BZ_OK) {
                ptr++;
                remaining--;
                continue;
            }

            strm.next_in = (char*)ptr;
            strm.avail_in = (unsigned int)remaining;

            std::vector<uint8_t> outBuf(256 * 1024);
            strm.next_out = (char*)outBuf.data();
            strm.avail_out = (unsigned int)outBuf.size();
            size_t totalOut = 0;

            bool success = false;
            while (true) {
                int ret = BZ2_bzDecompress(&strm);
                totalOut = outBuf.size() - strm.avail_out;

                if (ret == BZ_STREAM_END) {
                    success = true;
                    break;
                }
                if (ret != BZ_OK) break;

                if (strm.avail_out == 0) {
                    size_t newSize = outBuf.size() * 2;
                    outBuf.resize(newSize);
                    strm.next_out = (char*)outBuf.data() + totalOut;
                    strm.avail_out = (unsigned int)(newSize - totalOut);
                }
            }

            // How many compressed bytes were consumed
            size_t consumed = remaining - strm.avail_in;
            BZ2_bzDecompressEnd(&strm);

            if (success && totalOut > 0) {
                outBuf.resize(totalOut);
                decompressedBlocks.push_back(std::move(outBuf));
            }

            // Advance past the consumed compressed data
            if (consumed > 0) {
                ptr += consumed;
                remaining -= consumed;
            } else {
                ptr++;
                remaining--;
            }
        }
    }

    int totalBlocks = (int)decompressedBlocks.size();

    if (cb) cb(totalBlocks, totalBlocks);

    // Concatenate all decompressed blocks into one stream (like rustdar)
    size_t totalSize = 0;
    for (auto& b : decompressedBlocks) totalSize += b.size();

    std::vector<uint8_t> combined;
    combined.reserve(totalSize);
    for (auto& b : decompressedBlocks) {
        combined.insert(combined.end(), b.begin(), b.end());
    }
    decompressedBlocks.clear(); // free memory

    // Parse the combined stream
    if (!combined.empty()) {
        parseMessages(combined.data(), combined.size(), result);
    }

    organizeSweeps(result);

    int totalRadials = 0;
    for (auto& s : result.sweeps) totalRadials += (int)s.radials.size();
    printf("Parsed %s: %d sweeps, %d radials, %d BZ2 blocks, %zu bytes, station=(%f, %f)\n",
           result.station_id.c_str(), (int)result.sweeps.size(), totalRadials,
           totalBlocks, fileData.size(),
           result.station_lat, result.station_lon);

    return result;
}
