#include "codec.h"

#include <algorithm>
#include <cmath>

CodecParams makeCodecParams(CodecMode mode) {
    CodecParams params;
    params.mode = mode;

    if (mode == CodecMode::Safer) {
        params.width = 128;
        params.height = 96;
        params.blockSize = 8;
        params.targetFps = 2.5;
        params.keyframeInterval = 12;
        params.skipAvgThreshold = 1.2;
        params.skipMaxThreshold = 4;
        params.maxChangedFraction = 0.45;
        params.residualStep = 1;
        params.interLowModeMaxMae = 2.3;
        params.keyLowModeMaxMae = 1.8;
    } else {
        params.width = 96;
        params.height = 72;
        params.blockSize = 8;
        params.targetFps = 2.0;
        params.keyframeInterval = 18;
        params.skipAvgThreshold = 1.8;
        params.skipMaxThreshold = 5;
        params.maxChangedFraction = 0.30;
        params.residualStep = 2;
        params.interLowModeMaxMae = 2.8;
        params.keyLowModeMaxMae = 2.4;
    }

    return params;
}

const char *codecModeName(CodecMode mode) {
    return mode == CodecMode::Safer ? "safer" : "aggressive";
}

int blockCountX(int width, int blockSize) {
    return (width + blockSize - 1) / blockSize;
}

int blockCountY(int height, int blockSize) {
    return (height + blockSize - 1) / blockSize;
}

int totalBlockCount(int width, int height, int blockSize) {
    return blockCountX(width, blockSize) * blockCountY(height, blockSize);
}

BlockGeometry blockGeometry(int blockIndex, int width, int height, int blockSize) {
    const int bxCount = blockCountX(width, blockSize);
    const int bx = blockIndex % bxCount;
    const int by = blockIndex / bxCount;

    const int x = bx * blockSize;
    const int y = by * blockSize;
    const int w = std::min(blockSize, width - x);
    const int h = std::min(blockSize, height - y);

    return {x, y, w, h};
}

std::array<uint8_t, 16> sampleBlockToCells4x4(const Gray4Frame &src, const BlockGeometry &geom) {
    std::array<int, 16> sums{};
    std::array<int, 16> counts{};
    std::array<uint8_t, 16> cells{};

    for (int y = 0; y < geom.h; ++y) {
        for (int x = 0; x < geom.w; ++x) {
            const int cellX = (x * 4) / std::max(1, geom.w);
            const int cellY = (y * 4) / std::max(1, geom.h);
            const int cellIndex = cellY * 4 + cellX;
            sums[static_cast<std::size_t>(cellIndex)] += src.at(geom.x + x, geom.y + y);
            counts[static_cast<std::size_t>(cellIndex)] += 1;
        }
    }

    for (int i = 0; i < 16; ++i) {
        if (counts[static_cast<std::size_t>(i)] == 0) {
            cells[static_cast<std::size_t>(i)] = 0;
        } else {
            const int avg = (sums[static_cast<std::size_t>(i)] + counts[static_cast<std::size_t>(i)] / 2) /
                            counts[static_cast<std::size_t>(i)];
            cells[static_cast<std::size_t>(i)] = clamp4(avg);
        }
    }

    return cells;
}

void paintCells4x4(Gray4Frame &dst, const BlockGeometry &geom, const std::array<uint8_t, 16> &cells) {
    for (int y = 0; y < geom.h; ++y) {
        for (int x = 0; x < geom.w; ++x) {
            const int cellX = (x * 4) / std::max(1, geom.w);
            const int cellY = (y * 4) / std::max(1, geom.h);
            const int cellIndex = cellY * 4 + cellX;
            dst.at(geom.x + x, geom.y + y) = cells[static_cast<std::size_t>(cellIndex)];
        }
    }
}

void copyBlock(const Gray4Frame &src, Gray4Frame &dst, const BlockGeometry &geom) {
    for (int y = 0; y < geom.h; ++y) {
        for (int x = 0; x < geom.w; ++x) {
            dst.at(geom.x + x, geom.y + y) = src.at(geom.x + x, geom.y + y);
        }
    }
}

void extractRawBlock(const Gray4Frame &src, const BlockGeometry &geom, std::vector<uint8_t> &samples) {
    samples.clear();
    samples.reserve(static_cast<std::size_t>(geom.w * geom.h));
    for (int y = 0; y < geom.h; ++y) {
        for (int x = 0; x < geom.w; ++x) {
            samples.push_back(src.at(geom.x + x, geom.y + y));
        }
    }
}

void writeRawBlock(Gray4Frame &dst, const BlockGeometry &geom, const std::vector<uint8_t> &samples) {
    const std::size_t expected = static_cast<std::size_t>(geom.w * geom.h);
    if (samples.size() < expected) {
        return;
    }

    std::size_t i = 0;
    for (int y = 0; y < geom.h; ++y) {
        for (int x = 0; x < geom.w; ++x) {
            dst.at(geom.x + x, geom.y + y) = clamp4(samples[i++]);
        }
    }
}

double blockMae(const Gray4Frame &a, const Gray4Frame &b, const BlockGeometry &geom, int *maxDiff) {
    double sum = 0.0;
    int localMax = 0;
    int samples = 0;

    for (int y = 0; y < geom.h; ++y) {
        for (int x = 0; x < geom.w; ++x) {
            const int diff = std::abs(static_cast<int>(a.at(geom.x + x, geom.y + y)) -
                                      static_cast<int>(b.at(geom.x + x, geom.y + y)));
            sum += static_cast<double>(diff);
            localMax = std::max(localMax, diff);
            ++samples;
        }
    }

    if (maxDiff != nullptr) {
        *maxDiff = localMax;
    }

    if (samples == 0) {
        return 0.0;
    }

    return sum / static_cast<double>(samples);
}

double blockMaeToCells(const Gray4Frame &src, const BlockGeometry &geom, const std::array<uint8_t, 16> &cells) {
    double sum = 0.0;
    int samples = 0;

    for (int y = 0; y < geom.h; ++y) {
        for (int x = 0; x < geom.w; ++x) {
            const int cellX = (x * 4) / std::max(1, geom.w);
            const int cellY = (y * 4) / std::max(1, geom.h);
            const int cellIndex = cellY * 4 + cellX;
            const int diff = std::abs(static_cast<int>(src.at(geom.x + x, geom.y + y)) -
                                      static_cast<int>(cells[static_cast<std::size_t>(cellIndex)]));
            sum += static_cast<double>(diff);
            ++samples;
        }
    }

    if (samples == 0) {
        return 0.0;
    }

    return sum / static_cast<double>(samples);
}

uint8_t packSigned4(int value) {
    const int clamped = std::clamp(value, -8, 7);
    return static_cast<uint8_t>(clamped + 8);
}

int unpackSigned4(uint8_t packed) {
    return static_cast<int>(packed) - 8;
}

void writeHeader(BitWriter &writer, const BitstreamHeader &header) {
    writer.writeBits(kFrameSync, 16);
    writer.writeBits(header.version, 4);
    writer.writeBits(static_cast<uint8_t>(header.frameType), 1);
    writer.writeBits(static_cast<uint8_t>(header.mode), 1);
    writer.writeBits(header.width, 10);
    writer.writeBits(header.height, 10);
    writer.writeBits(header.blockSize, 4);
    writer.writeBits(static_cast<uint32_t>(std::max(1, static_cast<int>(header.residualStep)) - 1), 3);
    writer.writeBits(header.frameIndex & 0xFFFFFFU, 24);
    writer.writeBits(header.keyframeInterval, 8);
    writer.writeBits(header.totalBlocks, 16);
    writer.writeBits(header.changedBlocks, 16);
}

bool readHeader(BitReader &reader, BitstreamHeader &header, std::string &error) {
    uint32_t value = 0;

    if (!reader.readBits(16, value)) {
        error = "Failed to read sync";
        return false;
    }
    if (value != kFrameSync) {
        error = "Invalid sync marker";
        return false;
    }

    if (!reader.readBits(4, value)) {
        error = "Failed to read version";
        return false;
    }
    header.version = static_cast<uint8_t>(value);
    if (header.version != kBitstreamVersion) {
        error = "Unsupported bitstream version";
        return false;
    }

    if (!reader.readBits(1, value)) {
        error = "Failed to read frame type";
        return false;
    }
    header.frameType = value == 0 ? FrameType::Keyframe : FrameType::Interframe;

    if (!reader.readBits(1, value)) {
        error = "Failed to read mode";
        return false;
    }
    header.mode = value == 0 ? CodecMode::Safer : CodecMode::Aggressive;

    if (!reader.readBits(10, value)) {
        error = "Failed to read width";
        return false;
    }
    header.width = static_cast<uint16_t>(value);

    if (!reader.readBits(10, value)) {
        error = "Failed to read height";
        return false;
    }
    header.height = static_cast<uint16_t>(value);

    if (!reader.readBits(4, value)) {
        error = "Failed to read block size";
        return false;
    }
    header.blockSize = static_cast<uint8_t>(value);

    if (!reader.readBits(3, value)) {
        error = "Failed to read residual step";
        return false;
    }
    header.residualStep = static_cast<uint8_t>(value + 1);

    if (!reader.readBits(24, value)) {
        error = "Failed to read frame index";
        return false;
    }
    header.frameIndex = value;

    if (!reader.readBits(8, value)) {
        error = "Failed to read keyframe interval";
        return false;
    }
    header.keyframeInterval = static_cast<uint8_t>(value);

    if (!reader.readBits(16, value)) {
        error = "Failed to read total blocks";
        return false;
    }
    header.totalBlocks = static_cast<uint16_t>(value);

    if (!reader.readBits(16, value)) {
        error = "Failed to read changed blocks";
        return false;
    }
    header.changedBlocks = static_cast<uint16_t>(value);

    if (header.width == 0 || header.height == 0 || header.blockSize == 0) {
        error = "Invalid dimensions in header";
        return false;
    }

    return true;
}
