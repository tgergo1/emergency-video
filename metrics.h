#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>

#include "codec.h"
#include "frame.h"

struct MetricsSnapshot {
    double elapsedSeconds = 0.0;
    double liveBitrateKbps = 0.0;
    double smoothedBitrateKbps = 0.0;
    double averageBitrateKbps = 0.0;
    double effectiveEncodedFps = 0.0;
    double averageBytesPerFrame = 0.0;
    double lastBytesPerFrame = 0.0;
    double ratioVsRaw4 = 0.0;
    double ratioVsRaw8 = 0.0;
    double keyframePercent = 0.0;
    double changedBlocksPercent = 0.0;
    double psnrDb = 0.0;
    uint64_t totalFrames = 0;
    uint64_t keyframes = 0;
};

double computePsnr4Bit(const Gray4Frame &original, const Gray4Frame &reconstructed);

class MetricsTracker {
public:
    MetricsTracker() = default;
    MetricsTracker(int width, int height);

    void reset(int width, int height);
    void update(const EncodeMetadata &meta,
                std::size_t packetBytes,
                const Gray4Frame &original,
                const Gray4Frame &reconstructed);

    [[nodiscard]] MetricsSnapshot snapshot() const;

private:
    int width_ = 0;
    int height_ = 0;
    std::size_t raw4BytesPerFrame_ = 0;
    std::size_t raw8BytesPerFrame_ = 0;

    std::chrono::steady_clock::time_point startTime_{};

    uint64_t totalFrames_ = 0;
    uint64_t keyframes_ = 0;
    uint64_t totalBytes_ = 0;

    uint64_t totalChangedBlocks_ = 0;
    uint64_t totalInterBlockSlots_ = 0;

    double totalSquaredError_ = 0.0;
    uint64_t totalComparedPixels_ = 0;

    std::size_t lastPacketBytes_ = 0;
    std::chrono::steady_clock::time_point lastUpdateTime_{};
    double liveBitrateKbps_ = 0.0;
    double smoothedBitrateKbps_ = 0.0;
};
