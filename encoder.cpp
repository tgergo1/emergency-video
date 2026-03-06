#include "encoder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace {
int quantizeResidual(int diff, int step) {
    const int safeStep = std::max(1, step);
    if (diff >= 0) {
        return std::clamp((diff + safeStep / 2) / safeStep, -8, 7);
    }
    return std::clamp(-(((-diff) + safeStep / 2) / safeStep), -8, 7);
}

std::array<uint8_t, 16> reconstructResidualCells(const std::array<uint8_t, 16> &base,
                                                  const std::array<int, 16> &qResidual,
                                                  int step) {
    std::array<uint8_t, 16> out{};
    const int safeStep = std::max(1, step);

    for (int i = 0; i < 16; ++i) {
        const int value = static_cast<int>(base[static_cast<std::size_t>(i)]) +
                          qResidual[static_cast<std::size_t>(i)] * safeStep;
        out[static_cast<std::size_t>(i)] = clamp4(value);
    }

    return out;
}

uint16_t countChangedBlocks(const std::vector<uint8_t> &changed) {
    return static_cast<uint16_t>(std::count(changed.begin(), changed.end(), static_cast<uint8_t>(1)));
}
} // namespace

Encoder::Encoder(const CodecParams &params) : params_(params), keyframeInterval_(params.keyframeInterval) {}

void Encoder::setParams(const CodecParams &params) {
    params_ = params;
    keyframeInterval_ = std::clamp(params.keyframeInterval, 1, 255);
    reset();
}

void Encoder::setKeyframeInterval(int interval) {
    keyframeInterval_ = std::clamp(interval, 1, 255);
    forceKeyframe_ = true;
}

void Encoder::forceNextKeyframe() {
    forceKeyframe_ = true;
}

void Encoder::reset() {
    hasPreviousReconstructed_ = false;
    forceKeyframe_ = true;
    frameIndex_ = 0;
    framesSinceKeyframe_ = 0;
    previousReconstructed_ = Gray4Frame();
}

int Encoder::keyframeInterval() const {
    return keyframeInterval_;
}

bool Encoder::shouldEmitKeyframe() const {
    return forceKeyframe_ || !hasPreviousReconstructed_ || framesSinceKeyframe_ >= keyframeInterval_;
}

void Encoder::selectChangedBlocks(const Gray4Frame &frame,
                                  std::vector<uint8_t> &changed,
                                  const std::vector<uint8_t> *roiBlocks) const {
    const int totalBlocks = totalBlockCount(params_.width, params_.height, params_.blockSize);
    changed.assign(static_cast<std::size_t>(totalBlocks), 0);

    std::vector<CandidateBlock> candidates;
    candidates.reserve(static_cast<std::size_t>(totalBlocks));
    std::vector<int> forcedRoiBlocks;
    forcedRoiBlocks.reserve(static_cast<std::size_t>(totalBlocks / 4 + 1));

    for (int block = 0; block < totalBlocks; ++block) {
        const BlockGeometry geom = blockGeometry(block, params_.width, params_.height, params_.blockSize);
        int maxDiff = 0;
        const double mae = blockMae(frame, previousReconstructed_, geom, &maxDiff);
        const bool isRoi = (roiBlocks != nullptr && static_cast<std::size_t>(block) < roiBlocks->size() &&
                            (*roiBlocks)[static_cast<std::size_t>(block)] != 0);

        const double localAvgThreshold = isRoi ? params_.skipAvgThreshold * 0.60 : params_.skipAvgThreshold;
        const int localMaxThreshold = isRoi ? std::max(1, static_cast<int>(std::round(params_.skipMaxThreshold * 0.60)))
                                            : params_.skipMaxThreshold;

        if (mae >= localAvgThreshold || maxDiff >= localMaxThreshold) {
            const double score = mae + static_cast<double>(maxDiff) * 0.25 + (isRoi ? 3.0 : 0.0);
            candidates.push_back({block, score});
        }

        if (isRoi && (mae >= localAvgThreshold * 0.65 || maxDiff >= std::max(1, localMaxThreshold - 1))) {
            forcedRoiBlocks.push_back(block);
        }
    }

    const int maxChanged = std::max(1, static_cast<int>(std::floor(params_.maxChangedFraction * totalBlocks)));
    if (static_cast<int>(candidates.size()) > maxChanged) {
        std::nth_element(candidates.begin(),
                         candidates.begin() + maxChanged,
                         candidates.end(),
                         [](const CandidateBlock &a, const CandidateBlock &b) { return a.score > b.score; });
        candidates.resize(static_cast<std::size_t>(maxChanged));
    }

    for (const CandidateBlock &candidate : candidates) {
        changed[static_cast<std::size_t>(candidate.index)] = 1;
    }

    // Face blocks are never dropped by global update cap.
    for (int block : forcedRoiBlocks) {
        changed[static_cast<std::size_t>(block)] = 1;
    }
}

void Encoder::encodeChangeMap(BitWriter &writer, const std::vector<uint8_t> &changed) {
    std::vector<uint16_t> runs;
    runs.reserve(changed.size() + 1);

    bool state = false; // Start with "unchanged" run.
    std::size_t index = 0;
    while (index < changed.size()) {
        uint16_t run = 0;
        while (index < changed.size() && (changed[index] != 0) == state) {
            ++run;
            ++index;
        }
        runs.push_back(run);
        state = !state;
    }

    if (runs.empty()) {
        runs.push_back(0);
    }

    writer.writeBits(static_cast<uint32_t>(runs.size()), 16);
    for (uint16_t run : runs) {
        writer.writeBits(run, 16);
    }
}

void Encoder::encodeKeyframePayload(const Gray4Frame &input, Gray4Frame &reconstructed, BitWriter &writer) const {
    const int totalBlocks = totalBlockCount(params_.width, params_.height, params_.blockSize);
    std::vector<uint8_t> rawSamples;

    for (int block = 0; block < totalBlocks; ++block) {
        const BlockGeometry geom = blockGeometry(block, params_.width, params_.height, params_.blockSize);
        const std::array<uint8_t, 16> cells = sampleBlockToCells4x4(input, geom);
        const double absoluteMae = blockMaeToCells(input, geom, cells);
        const bool useRaw = absoluteMae > params_.keyLowModeMaxMae;

        writer.writeBit(useRaw);

        if (useRaw) {
            extractRawBlock(input, geom, rawSamples);
            for (uint8_t sample : rawSamples) {
                writer.writeBits(sample, 4);
            }
            writeRawBlock(reconstructed, geom, rawSamples);
        } else {
            for (uint8_t cell : cells) {
                writer.writeBits(cell, 4);
            }
            paintCells4x4(reconstructed, geom, cells);
        }
    }
}

void Encoder::encodeInterPayload(const Gray4Frame &input,
                                 const std::vector<uint8_t> &changed,
                                 Gray4Frame &reconstructed,
                                 BitWriter &writer) const {
    reconstructed = previousReconstructed_;
    std::vector<uint8_t> rawSamples;

    const int totalBlocks = totalBlockCount(params_.width, params_.height, params_.blockSize);
    for (int block = 0; block < totalBlocks; ++block) {
        if (changed[static_cast<std::size_t>(block)] == 0) {
            continue;
        }

        const BlockGeometry geom = blockGeometry(block, params_.width, params_.height, params_.blockSize);
        const std::array<uint8_t, 16> currentCells = sampleBlockToCells4x4(input, geom);
        const std::array<uint8_t, 16> previousCells = sampleBlockToCells4x4(previousReconstructed_, geom);

        std::array<int, 16> quantizedResidual{};
        for (int i = 0; i < 16; ++i) {
            const int diff = static_cast<int>(currentCells[static_cast<std::size_t>(i)]) -
                             static_cast<int>(previousCells[static_cast<std::size_t>(i)]);
            quantizedResidual[static_cast<std::size_t>(i)] = quantizeResidual(diff, params_.residualStep);
        }

        const std::array<uint8_t, 16> residualCells =
            reconstructResidualCells(previousCells, quantizedResidual, params_.residualStep);

        const double residualMae = blockMaeToCells(input, geom, residualCells);
        const double absoluteMae = blockMaeToCells(input, geom, currentCells);

        BlockPayloadMode mode = BlockPayloadMode::Raw8x8;
        if (std::min(residualMae, absoluteMae) <= params_.interLowModeMaxMae) {
            mode = (residualMae <= absoluteMae) ? BlockPayloadMode::Residual4x4 : BlockPayloadMode::Absolute4x4;
        }

        writer.writeBits(static_cast<uint8_t>(mode), 2);

        if (mode == BlockPayloadMode::Residual4x4) {
            for (int q : quantizedResidual) {
                writer.writeBits(packSigned4(q), 4);
            }
            paintCells4x4(reconstructed, geom, residualCells);
        } else if (mode == BlockPayloadMode::Absolute4x4) {
            for (uint8_t cell : currentCells) {
                writer.writeBits(cell, 4);
            }
            paintCells4x4(reconstructed, geom, currentCells);
        } else {
            extractRawBlock(input, geom, rawSamples);
            for (uint8_t sample : rawSamples) {
                writer.writeBits(sample, 4);
            }
            writeRawBlock(reconstructed, geom, rawSamples);
        }
    }
}

EncodedPacket Encoder::encode(const Gray4Frame &frame, const std::vector<uint8_t> *roiBlocks) {
    if (frame.width != params_.width || frame.height != params_.height) {
        throw std::invalid_argument("Input frame dimensions do not match codec profile");
    }

    const bool keyframe = shouldEmitKeyframe();
    const int totalBlocks = totalBlockCount(params_.width, params_.height, params_.blockSize);

    std::vector<uint8_t> changed(static_cast<std::size_t>(totalBlocks), keyframe ? 1 : 0);
    if (!keyframe) {
        selectChangedBlocks(frame, changed, roiBlocks);
    }
    const uint16_t changedBlocks = keyframe ? static_cast<uint16_t>(totalBlocks) : countChangedBlocks(changed);

    BitstreamHeader header;
    header.version = kBitstreamVersion;
    header.frameType = keyframe ? FrameType::Keyframe : FrameType::Interframe;
    header.mode = params_.mode;
    header.width = static_cast<uint16_t>(params_.width);
    header.height = static_cast<uint16_t>(params_.height);
    header.blockSize = static_cast<uint8_t>(params_.blockSize);
    header.residualStep = static_cast<uint8_t>(std::max(1, params_.residualStep));
    header.frameIndex = frameIndex_;
    header.keyframeInterval = static_cast<uint8_t>(keyframeInterval_);
    header.totalBlocks = static_cast<uint16_t>(totalBlocks);
    header.changedBlocks = changedBlocks;

    BitWriter writer;
    writeHeader(writer, header);

    Gray4Frame reconstructed(params_.width, params_.height);
    if (keyframe) {
        encodeKeyframePayload(frame, reconstructed, writer);
    } else {
        encodeChangeMap(writer, changed);
        encodeInterPayload(frame, changed, reconstructed, writer);
    }

    writer.alignToByte();

    previousReconstructed_ = reconstructed;
    hasPreviousReconstructed_ = true;
    forceKeyframe_ = false;

    if (keyframe) {
        framesSinceKeyframe_ = 0;
    } else {
        ++framesSinceKeyframe_;
    }

    EncodedPacket packet;
    packet.bytes = writer.takeBytes();
    packet.meta.frameType = header.frameType;
    packet.meta.mode = header.mode;
    packet.meta.frameIndex = header.frameIndex;
    packet.meta.totalBlocks = header.totalBlocks;
    packet.meta.changedBlocks = header.changedBlocks;
    packet.meta.width = header.width;
    packet.meta.height = header.height;
    packet.meta.keyframeInterval = header.keyframeInterval;

    ++frameIndex_;
    return packet;
}
