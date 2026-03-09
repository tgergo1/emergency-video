#include "fallback_controller.h"

#include <algorithm>

namespace {

constexpr auto kEvalInterval = std::chrono::seconds(1);
constexpr auto kMinStageChangeInterval = std::chrono::seconds(5);

constexpr int kEscalateAfterDegradedWindows = 3;
constexpr int kRecoverAfterHealthyWindows = 15;

int stageToInt(FallbackStage stage) {
    return static_cast<int>(stage);
}

FallbackStage intToStage(int value) {
    value = std::clamp(value, stageToInt(FallbackStage::Normal), stageToInt(FallbackStage::TextOnly));
    return static_cast<FallbackStage>(value);
}

} // namespace

FallbackController::FallbackController(FallbackStage maxStage) {
    setMaxStage(maxStage);
}

void FallbackController::reset(FallbackStage stage) {
    stage_ = clampStage(stage);
    if (stageToInt(stage_) > stageToInt(maxStage_)) {
        stage_ = maxStage_;
    }
    degradedWindows_ = 0;
    healthyWindows_ = 0;
    hasEval_ = false;
    hasChange_ = false;
}

void FallbackController::setMaxStage(FallbackStage maxStage) {
    maxStage_ = clampStage(maxStage);
    if (stageToInt(stage_) > stageToInt(maxStage_)) {
        stage_ = maxStage_;
    }
}

FallbackStage FallbackController::stage() const {
    return stage_;
}

FallbackStage FallbackController::clampStage(FallbackStage stage) {
    return intToStage(stageToInt(stage));
}

bool FallbackController::isDegraded(const FallbackInputWindow &window) const {
    if (!window.syncLocked) {
        return true;
    }
    if (window.transportLossPercent >= 18.0) {
        return true;
    }
    if (window.droppedDelta >= 2) {
        return true;
    }
    if (window.retransmitDelta >= 3) {
        return true;
    }
    if (window.queueVideo >= 32) {
        return true;
    }
    if (window.queueInflight >= 24) {
        return true;
    }
    return false;
}

bool FallbackController::isHealthy(const FallbackInputWindow &window) const {
    if (!window.syncLocked) {
        return false;
    }
    if (window.transportLossPercent > 5.0) {
        return false;
    }
    if (window.droppedDelta != 0) {
        return false;
    }
    if (window.retransmitDelta > 1) {
        return false;
    }
    if (window.queueVideo > 8) {
        return false;
    }
    if (window.queueInflight > 6) {
        return false;
    }
    return true;
}

bool FallbackController::update(const FallbackInputWindow &window, std::chrono::steady_clock::time_point now) {
    if (!hasEval_) {
        hasEval_ = true;
        lastEval_ = now;
    } else if (now - lastEval_ < kEvalInterval) {
        return false;
    } else {
        lastEval_ = now;
    }

    if (isDegraded(window)) {
        ++degradedWindows_;
        healthyWindows_ = 0;
    } else if (isHealthy(window)) {
        ++healthyWindows_;
        degradedWindows_ = 0;
    } else {
        degradedWindows_ = 0;
        healthyWindows_ = 0;
    }

    if (hasChange_ && now - lastChange_ < kMinStageChangeInterval) {
        return false;
    }

    if (degradedWindows_ >= kEscalateAfterDegradedWindows && stageToInt(stage_) < stageToInt(maxStage_)) {
        stage_ = intToStage(stageToInt(stage_) + 1);
        degradedWindows_ = 0;
        healthyWindows_ = 0;
        hasChange_ = true;
        lastChange_ = now;
        return true;
    }

    if (healthyWindows_ >= kRecoverAfterHealthyWindows && stageToInt(stage_) > stageToInt(FallbackStage::Normal)) {
        stage_ = intToStage(stageToInt(stage_) - 1);
        degradedWindows_ = 0;
        healthyWindows_ = 0;
        hasChange_ = true;
        lastChange_ = now;
        return true;
    }

    return false;
}
