#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>

#include "communicator_protocol.h"

struct FallbackInputWindow {
    std::size_t queueVideo = 0;
    std::size_t queueInflight = 0;
    uint64_t droppedDelta = 0;
    uint64_t retransmitDelta = 0;
    double transportLossPercent = 0.0;
    bool syncLocked = false;
};

class FallbackController {
public:
    explicit FallbackController(FallbackStage maxStage = FallbackStage::TextOnly);

    void reset(FallbackStage stage = FallbackStage::Normal);
    void setMaxStage(FallbackStage maxStage);

    [[nodiscard]] FallbackStage stage() const;

    bool update(const FallbackInputWindow &window, std::chrono::steady_clock::time_point now);

private:
    FallbackStage stage_ = FallbackStage::Normal;
    FallbackStage maxStage_ = FallbackStage::TextOnly;

    int degradedWindows_ = 0;
    int healthyWindows_ = 0;

    bool hasEval_ = false;
    std::chrono::steady_clock::time_point lastEval_{};

    bool hasChange_ = false;
    std::chrono::steady_clock::time_point lastChange_{};

    static FallbackStage clampStage(FallbackStage stage);
    bool isDegraded(const FallbackInputWindow &window) const;
    bool isHealthy(const FallbackInputWindow &window) const;
};
