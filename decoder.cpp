#include "decoder.h"

#include <algorithm>
#include <array>

namespace {
bool consumeTrailingPadding(BitReader &reader, std::string &error) {
    const std::size_t remaining = reader.bitsRemaining();
    if (remaining >= 8) {
        error = "Trailing non-padding bytes found";
        return false;
    }

    if (remaining == 0) {
        return true;
    }

    uint32_t tail = 0;
    if (!reader.readBits(static_cast<int>(remaining), tail)) {
        error = "Failed to consume trailing padding";
        return false;
    }

    if (tail != 0) {
        error = "Trailing padding bits must be zero";
        return false;
    }

    return true;
}
} // namespace

void Decoder::reset() {
    previousFrame_ = Gray4Frame();
    hasPreviousFrame_ = false;
}

bool Decoder::decodeChangeMap(BitReader &reader,
                              int totalBlocks,
                              std::vector<uint8_t> &changed,
                              std::string &error) {
    changed.assign(static_cast<std::size_t>(totalBlocks), 0);

    uint32_t numRuns = 0;
    if (!reader.readBits(16, numRuns)) {
        error = "Failed to read change-map run count";
        return false;
    }

    bool state = false;
    int cursor = 0;

    for (uint32_t runIndex = 0; runIndex < numRuns; ++runIndex) {
        uint32_t runLen = 0;
        if (!reader.readBits(16, runLen)) {
            error = "Failed to read change-map run length";
            return false;
        }

        if (cursor + static_cast<int>(runLen) > totalBlocks) {
            error = "Change-map overflows block count";
            return false;
        }

        if (runLen > 0) {
            std::fill(changed.begin() + cursor,
                      changed.begin() + cursor + static_cast<int>(runLen),
                      state ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0));
            cursor += static_cast<int>(runLen);
        }

        state = !state;
    }

    if (cursor != totalBlocks) {
        error = "Change-map did not cover all blocks";
        return false;
    }

    return true;
}

DecodeResult Decoder::decode(const std::vector<uint8_t> &packetBytes) {
    DecodeResult result;

    BitReader reader(packetBytes);
    BitstreamHeader header;
    if (!readHeader(reader, header, result.error)) {
        return result;
    }

    const int expectedBlocks = totalBlockCount(header.width, header.height, header.blockSize);
    if (expectedBlocks != header.totalBlocks) {
        result.error = "Header block count mismatch";
        return result;
    }

    Gray4Frame frame(header.width, header.height);
    std::vector<uint8_t> rawSamples;

    if (header.frameType == FrameType::Keyframe) {
        for (int block = 0; block < expectedBlocks; ++block) {
            const BlockGeometry geom = blockGeometry(block, header.width, header.height, header.blockSize);
            bool useRaw = false;
            if (!reader.readBit(useRaw)) {
                result.error = "Failed to read keyframe block mode";
                return result;
            }

            if (useRaw) {
                rawSamples.assign(static_cast<std::size_t>(geom.w * geom.h), 0);
                for (std::size_t i = 0; i < rawSamples.size(); ++i) {
                    uint32_t sample = 0;
                    if (!reader.readBits(4, sample)) {
                        result.error = "Failed to read keyframe raw block";
                        return result;
                    }
                    rawSamples[i] = static_cast<uint8_t>(sample);
                }
                writeRawBlock(frame, geom, rawSamples);
            } else {
                std::array<uint8_t, 16> cells{};
                for (int i = 0; i < 16; ++i) {
                    uint32_t value = 0;
                    if (!reader.readBits(4, value)) {
                        result.error = "Failed to read keyframe 4x4 block";
                        return result;
                    }
                    cells[static_cast<std::size_t>(i)] = static_cast<uint8_t>(value);
                }
                paintCells4x4(frame, geom, cells);
            }
        }
    } else {
        if (!hasPreviousFrame_) {
            result.error = "Interframe received without reference frame";
            return result;
        }

        if (previousFrame_.width != header.width || previousFrame_.height != header.height) {
            result.error = "Interframe dimensions do not match reference frame";
            return result;
        }

        frame = previousFrame_;

        std::vector<uint8_t> changed;
        if (!decodeChangeMap(reader, expectedBlocks, changed, result.error)) {
            return result;
        }

        uint16_t changedCount = 0;
        for (int block = 0; block < expectedBlocks; ++block) {
            if (changed[static_cast<std::size_t>(block)] == 0) {
                continue;
            }

            ++changedCount;
            const BlockGeometry geom = blockGeometry(block, header.width, header.height, header.blockSize);

            uint32_t modeBits = 0;
            if (!reader.readBits(2, modeBits)) {
                result.error = "Failed to read inter block mode";
                return result;
            }

            if (modeBits == static_cast<uint32_t>(BlockPayloadMode::Residual4x4)) {
                std::array<int, 16> qResidual{};
                for (int i = 0; i < 16; ++i) {
                    uint32_t packed = 0;
                    if (!reader.readBits(4, packed)) {
                        result.error = "Failed to read residual block";
                        return result;
                    }
                    qResidual[static_cast<std::size_t>(i)] = unpackSigned4(static_cast<uint8_t>(packed));
                }

                const std::array<uint8_t, 16> prevCells = sampleBlockToCells4x4(previousFrame_, geom);
                std::array<uint8_t, 16> reconstructedCells{};
                for (int i = 0; i < 16; ++i) {
                    const int residualStep = std::max(1, static_cast<int>(header.residualStep));
                    const int value = static_cast<int>(prevCells[static_cast<std::size_t>(i)]) +
                                      qResidual[static_cast<std::size_t>(i)] * residualStep;
                    reconstructedCells[static_cast<std::size_t>(i)] = clamp4(value);
                }
                paintCells4x4(frame, geom, reconstructedCells);
            } else if (modeBits == static_cast<uint32_t>(BlockPayloadMode::Absolute4x4)) {
                std::array<uint8_t, 16> cells{};
                for (int i = 0; i < 16; ++i) {
                    uint32_t value = 0;
                    if (!reader.readBits(4, value)) {
                        result.error = "Failed to read absolute block";
                        return result;
                    }
                    cells[static_cast<std::size_t>(i)] = static_cast<uint8_t>(value);
                }
                paintCells4x4(frame, geom, cells);
            } else if (modeBits == static_cast<uint32_t>(BlockPayloadMode::Raw8x8)) {
                rawSamples.assign(static_cast<std::size_t>(geom.w * geom.h), 0);
                for (std::size_t i = 0; i < rawSamples.size(); ++i) {
                    uint32_t sample = 0;
                    if (!reader.readBits(4, sample)) {
                        result.error = "Failed to read inter raw block";
                        return result;
                    }
                    rawSamples[i] = static_cast<uint8_t>(sample);
                }
                writeRawBlock(frame, geom, rawSamples);
            } else {
                result.error = "Reserved block mode encountered";
                return result;
            }
        }

        if (header.changedBlocks != changedCount) {
            // Not fatal: decoder trusts the explicit change map.
        }
    }

    if (!consumeTrailingPadding(reader, result.error)) {
        return result;
    }

    previousFrame_ = frame;
    hasPreviousFrame_ = true;

    result.ok = true;
    result.frame = std::move(frame);
    result.meta.frameType = header.frameType;
    result.meta.mode = header.mode;
    result.meta.frameIndex = header.frameIndex;
    result.meta.totalBlocks = header.totalBlocks;
    result.meta.changedBlocks = header.changedBlocks;
    result.meta.width = header.width;
    result.meta.height = header.height;
    result.meta.keyframeInterval = header.keyframeInterval;
    return result;
}
