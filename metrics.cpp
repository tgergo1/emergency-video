#include "metrics.h"

#include <algorithm>
#include <cmath>
#include <limits>

double computePsnr4Bit(const Gray4Frame &original, const Gray4Frame &reconstructed) {
    if (original.width != reconstructed.width || original.height != reconstructed.height || original.empty() ||
        reconstructed.empty()) {
        return 0.0;
    }

    double squaredError = 0.0;
    const std::size_t count = original.size();
    for (std::size_t i = 0; i < count; ++i) {
        const double diff = static_cast<double>(original.pixels[i]) - static_cast<double>(reconstructed.pixels[i]);
        squaredError += diff * diff;
    }

    if (count == 0) {
        return 0.0;
    }

    const double mse = squaredError / static_cast<double>(count);
    if (mse <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }

    const double peak = 15.0;
    return 20.0 * std::log10(peak) - 10.0 * std::log10(mse);
}

MetricsTracker::MetricsTracker(int width, int height) {
    reset(width, height);
}

void MetricsTracker::reset(int width, int height) {
    width_ = width;
    height_ = height;
    raw4BytesPerFrame_ = static_cast<std::size_t>((width * height + 1) / 2);
    raw8BytesPerFrame_ = static_cast<std::size_t>(width * height);

    startTime_ = std::chrono::steady_clock::now();

    totalFrames_ = 0;
    keyframes_ = 0;
    totalBytes_ = 0;
    totalChangedBlocks_ = 0;
    totalInterBlockSlots_ = 0;

    totalSquaredError_ = 0.0;
    totalComparedPixels_ = 0;

    lastPacketBytes_ = 0;
    lastUpdateTime_ = std::chrono::steady_clock::time_point{};
    liveBitrateKbps_ = 0.0;
    smoothedBitrateKbps_ = 0.0;
}

void MetricsTracker::update(const EncodeMetadata &meta,
                            std::size_t packetBytes,
                            const Gray4Frame &original,
                            const Gray4Frame &reconstructed) {
    const auto now = std::chrono::steady_clock::now();
    if (lastUpdateTime_.time_since_epoch().count() > 0) {
        const double dtSeconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(now - lastUpdateTime_).count();
        if (dtSeconds > 0.0001) {
            liveBitrateKbps_ = (static_cast<double>(packetBytes) * 8.0) / (dtSeconds * 1000.0);
            if (smoothedBitrateKbps_ <= 0.0) {
                smoothedBitrateKbps_ = liveBitrateKbps_;
            } else {
                constexpr double alpha = 0.20;
                smoothedBitrateKbps_ = smoothedBitrateKbps_ * (1.0 - alpha) + liveBitrateKbps_ * alpha;
            }
        }
    } else {
        liveBitrateKbps_ = 0.0;
        smoothedBitrateKbps_ = 0.0;
    }
    lastUpdateTime_ = now;

    ++totalFrames_;
    if (meta.frameType == FrameType::Keyframe) {
        ++keyframes_;
    }

    totalBytes_ += packetBytes;
    lastPacketBytes_ = packetBytes;

    if (meta.frameType == FrameType::Interframe) {
        totalChangedBlocks_ += meta.changedBlocks;
        totalInterBlockSlots_ += meta.totalBlocks;
    }

    if (original.width == reconstructed.width && original.height == reconstructed.height && !original.empty() &&
        !reconstructed.empty()) {
        const std::size_t count = std::min(original.size(), reconstructed.size());
        for (std::size_t i = 0; i < count; ++i) {
            const double diff = static_cast<double>(original.pixels[i]) - static_cast<double>(reconstructed.pixels[i]);
            totalSquaredError_ += diff * diff;
        }
        totalComparedPixels_ += count;
    }
}

MetricsSnapshot MetricsTracker::snapshot() const {
    MetricsSnapshot out;
    out.totalFrames = totalFrames_;
    out.keyframes = keyframes_;
    out.lastBytesPerFrame = static_cast<double>(lastPacketBytes_);
    out.liveBitrateKbps = liveBitrateKbps_;
    out.smoothedBitrateKbps = smoothedBitrateKbps_;

    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::max(0.0001,
                                    std::chrono::duration_cast<std::chrono::duration<double>>(now - startTime_)
                                        .count());
    out.elapsedSeconds = elapsed;

    if (totalFrames_ > 0) {
        out.averageBytesPerFrame = static_cast<double>(totalBytes_) / static_cast<double>(totalFrames_);
        out.keyframePercent = 100.0 * static_cast<double>(keyframes_) / static_cast<double>(totalFrames_);
    }

    out.averageBitrateKbps = (static_cast<double>(totalBytes_) * 8.0) / elapsed / 1000.0;
    out.effectiveEncodedFps = static_cast<double>(totalFrames_) / elapsed;

    if (out.averageBytesPerFrame > 0.0) {
        out.ratioVsRaw4 = static_cast<double>(raw4BytesPerFrame_) / out.averageBytesPerFrame;
        out.ratioVsRaw8 = static_cast<double>(raw8BytesPerFrame_) / out.averageBytesPerFrame;
    }

    if (totalInterBlockSlots_ > 0) {
        out.changedBlocksPercent =
            100.0 * static_cast<double>(totalChangedBlocks_) / static_cast<double>(totalInterBlockSlots_);
    }

    if (totalComparedPixels_ > 0) {
        const double mse = totalSquaredError_ / static_cast<double>(totalComparedPixels_);
        if (mse <= 0.0) {
            out.psnrDb = std::numeric_limits<double>::infinity();
        } else {
            out.psnrDb = 20.0 * std::log10(15.0) - 10.0 * std::log10(mse);
        }
    }

    return out;
}
