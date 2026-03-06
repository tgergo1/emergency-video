#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "bitstream.h"
#include "frame.h"

enum class CodecMode : uint8_t {
    Safer = 0,
    Aggressive = 1,
};

enum class FrameType : uint8_t {
    Keyframe = 0,
    Interframe = 1,
};

enum class BlockPayloadMode : uint8_t {
    Residual4x4 = 0,
    Absolute4x4 = 1,
    Raw8x8 = 2,
};

struct CodecParams {
    CodecMode mode = CodecMode::Safer;
    int width = 128;
    int height = 96;
    int blockSize = 8;
    double targetFps = 2.5;
    int keyframeInterval = 12;

    double skipAvgThreshold = 1.2;
    int skipMaxThreshold = 4;
    double maxChangedFraction = 0.45;

    int residualStep = 1;
    double interLowModeMaxMae = 2.3;
    double keyLowModeMaxMae = 1.8;
};

struct EncodeMetadata {
    FrameType frameType = FrameType::Keyframe;
    CodecMode mode = CodecMode::Safer;
    uint32_t frameIndex = 0;
    uint16_t totalBlocks = 0;
    uint16_t changedBlocks = 0;
    uint16_t width = 0;
    uint16_t height = 0;
    uint8_t keyframeInterval = 0;
};

struct EncodedPacket {
    std::vector<uint8_t> bytes;
    EncodeMetadata meta;
};

struct DecodeResult {
    bool ok = false;
    Gray4Frame frame;
    EncodeMetadata meta;
    std::string error;
};

struct BitstreamHeader {
    uint8_t version = 1;
    FrameType frameType = FrameType::Keyframe;
    CodecMode mode = CodecMode::Safer;
    uint16_t width = 0;
    uint16_t height = 0;
    uint8_t blockSize = 8;
    uint8_t residualStep = 1;
    uint32_t frameIndex = 0;
    uint8_t keyframeInterval = 0;
    uint16_t totalBlocks = 0;
    uint16_t changedBlocks = 0;
};

struct BlockGeometry {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
};

constexpr uint16_t kFrameSync = 0xEC0D;
constexpr uint8_t kBitstreamVersion = 1;

CodecParams makeCodecParams(CodecMode mode);
const char *codecModeName(CodecMode mode);

int blockCountX(int width, int blockSize);
int blockCountY(int height, int blockSize);
int totalBlockCount(int width, int height, int blockSize);
BlockGeometry blockGeometry(int blockIndex, int width, int height, int blockSize);

double blockMae(const Gray4Frame &a, const Gray4Frame &b, const BlockGeometry &geom, int *maxDiff = nullptr);
double blockMaeToCells(const Gray4Frame &src, const BlockGeometry &geom, const std::array<uint8_t, 16> &cells);

std::array<uint8_t, 16> sampleBlockToCells4x4(const Gray4Frame &src, const BlockGeometry &geom);
void paintCells4x4(Gray4Frame &dst, const BlockGeometry &geom, const std::array<uint8_t, 16> &cells);
void copyBlock(const Gray4Frame &src, Gray4Frame &dst, const BlockGeometry &geom);

void extractRawBlock(const Gray4Frame &src, const BlockGeometry &geom, std::vector<uint8_t> &samples);
void writeRawBlock(Gray4Frame &dst, const BlockGeometry &geom, const std::vector<uint8_t> &samples);

uint8_t packSigned4(int value);
int unpackSigned4(uint8_t packed);

void writeHeader(BitWriter &writer, const BitstreamHeader &header);
bool readHeader(BitReader &reader, BitstreamHeader &header, std::string &error);
