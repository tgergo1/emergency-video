#pragma once

#include <cstdint>
#include <vector>

#include "acoustic_link.h"

struct ModemParams {
    uint32_t sampleRate = 48000;
    uint16_t symbolSamples = 120;
    uint8_t bins = 16;
    BandMode bandMode = BandMode::Audible;
    float amplitude = 0.55F;
};

ModemParams modemParamsFromSession(const SessionConfig &config);

class MfskModem {
public:
    explicit MfskModem(const ModemParams &params);

    [[nodiscard]] const ModemParams &params() const;

    std::vector<float> modulateFrame(const std::vector<uint8_t> &rawFrame,
                                     uint8_t fecRepetition,
                                     uint8_t interleaveDepth) const;

    bool demodulateBurst(const std::vector<float> &segment,
                         std::vector<uint8_t> &rawFrame,
                         std::size_t *recoveredSymbols = nullptr) const;

private:
    ModemParams params_;
    std::vector<double> frequencies_;
    std::vector<uint8_t> preambleSymbols_;

    void buildFrequencies();
};

class AcousticBurstReceiver {
public:
    explicit AcousticBurstReceiver(const MfskModem &modem);

    void setEnergyThreshold(float thresholdStart, float thresholdEnd);
    void feedSamples(const float *samples, std::size_t count);

    bool popFrame(std::vector<uint8_t> &rawFrame, std::size_t *recoveredSymbols = nullptr);
    void clear();

private:
    const MfskModem &modem_;
    std::vector<float> buffer_;

    float thresholdStart_ = 0.02F;
    float thresholdEnd_ = 0.012F;
    bool inBurst_ = false;
    std::size_t burstStart_ = 0;
    std::size_t quietCount_ = 0;

    std::vector<std::vector<float>> pendingSegments_;

    void extractSegments();
};

std::vector<std::vector<uint8_t>> demodulatePcmBuffer(const std::vector<float> &pcm,
                                                       const MfskModem &modem,
                                                       LinkStats *stats = nullptr);

