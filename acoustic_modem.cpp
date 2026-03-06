#include "acoustic_modem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {
constexpr uint8_t kWireMagicA = 0x6A;
constexpr uint8_t kWireMagicB = 0xC3;
constexpr double kPi = 3.14159265358979323846;

void appendU16(std::vector<uint8_t> &out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xFFU));
    out.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
}

void appendU32(std::vector<uint8_t> &out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xFFU));
    out.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
    out.push_back(static_cast<uint8_t>((value >> 16U) & 0xFFU));
    out.push_back(static_cast<uint8_t>((value >> 24U) & 0xFFU));
}

bool readU16(const std::vector<uint8_t> &in, std::size_t &off, uint16_t &value) {
    if (off + 2 > in.size()) {
        return false;
    }
    value = static_cast<uint16_t>(in[off]) | (static_cast<uint16_t>(in[off + 1]) << 8U);
    off += 2;
    return true;
}

bool readU32(const std::vector<uint8_t> &in, std::size_t &off, uint32_t &value) {
    if (off + 4 > in.size()) {
        return false;
    }
    value = static_cast<uint32_t>(in[off]) | (static_cast<uint32_t>(in[off + 1]) << 8U) |
            (static_cast<uint32_t>(in[off + 2]) << 16U) | (static_cast<uint32_t>(in[off + 3]) << 24U);
    off += 4;
    return true;
}

std::vector<uint8_t> bytesToSymbols(const std::vector<uint8_t> &bytes) {
    std::vector<uint8_t> symbols;
    symbols.reserve(bytes.size() * 2);
    for (uint8_t byte : bytes) {
        symbols.push_back(static_cast<uint8_t>((byte >> 4U) & 0x0FU));
        symbols.push_back(static_cast<uint8_t>(byte & 0x0FU));
    }
    return symbols;
}

bool symbolsToBytes(const std::vector<uint8_t> &symbols, std::vector<uint8_t> &bytes) {
    if (symbols.size() % 2 != 0) {
        return false;
    }

    bytes.clear();
    bytes.reserve(symbols.size() / 2);
    for (std::size_t i = 0; i < symbols.size(); i += 2) {
        const uint8_t hi = symbols[i] & 0x0FU;
        const uint8_t lo = symbols[i + 1] & 0x0FU;
        bytes.push_back(static_cast<uint8_t>((hi << 4U) | lo));
    }
    return true;
}

double goertzelPower(const float *samples, std::size_t count, double freq, uint32_t sampleRate) {
    if (count == 0 || sampleRate == 0 || freq <= 0.0) {
        return 0.0;
    }

    const double k = std::round((static_cast<double>(count) * freq) / static_cast<double>(sampleRate));
    const double omega = (2.0 * kPi * k) / static_cast<double>(count);
    const double coeff = 2.0 * std::cos(omega);

    double q0 = 0.0;
    double q1 = 0.0;
    double q2 = 0.0;

    for (std::size_t i = 0; i < count; ++i) {
        q0 = coeff * q1 - q2 + samples[i];
        q2 = q1;
        q1 = q0;
    }

    return q1 * q1 + q2 * q2 - q1 * q2 * coeff;
}

std::vector<uint8_t> decodeSymbolsAtOffset(const std::vector<float> &segment,
                                           std::size_t offset,
                                           std::size_t symbolSamples,
                                           const std::vector<double> &frequencies,
                                           uint32_t sampleRate,
                                           std::size_t maxSymbols) {
    std::vector<uint8_t> symbols;
    if (symbolSamples == 0 || frequencies.empty() || offset >= segment.size()) {
        return symbols;
    }

    const std::size_t symbolsAvailable = std::min(maxSymbols, (segment.size() - offset) / symbolSamples);
    symbols.reserve(symbolsAvailable);

    for (std::size_t s = 0; s < symbolsAvailable; ++s) {
        const float *ptr = segment.data() + static_cast<std::ptrdiff_t>(offset + s * symbolSamples);

        double best = -1.0;
        uint8_t bestIndex = 0;
        for (std::size_t i = 0; i < frequencies.size(); ++i) {
            const double p = goertzelPower(ptr, symbolSamples, frequencies[i], sampleRate);
            if (p > best) {
                best = p;
                bestIndex = static_cast<uint8_t>(i);
            }
        }
        symbols.push_back(bestIndex);
    }

    return symbols;
}

bool parseWirePayload(const std::vector<uint8_t> &wire,
                      std::vector<uint8_t> &rawFrame,
                      std::size_t *recoveredSymbols) {
    if (wire.size() < 12 || wire[0] != kWireMagicA || wire[1] != kWireMagicB) {
        return false;
    }

    std::size_t off = 2;
    uint16_t rawSize = 0;
    uint16_t encodedSize = 0;
    if (!readU16(wire, off, rawSize) || off + 2 > wire.size()) {
        return false;
    }

    const uint8_t repetition = wire[off++];
    const uint8_t interleave = wire[off++];

    if (!readU16(wire, off, encodedSize)) {
        return false;
    }

    if (off + encodedSize + 4 > wire.size()) {
        return false;
    }

    std::vector<uint8_t> encoded(wire.begin() + static_cast<std::ptrdiff_t>(off),
                                 wire.begin() + static_cast<std::ptrdiff_t>(off + encodedSize));
    off += encodedSize;

    uint32_t packetCrc = 0;
    if (!readU32(wire, off, packetCrc) || off != wire.size()) {
        return false;
    }

    std::vector<uint8_t> withoutCrc(wire.begin(), wire.end() - 4);
    if (crc32(withoutCrc) != packetCrc) {
        return false;
    }

    std::vector<uint8_t> decoded;
    std::size_t recovered = 0;
    if (!fecRecover(encoded, rawSize, repetition, interleave, decoded, &recovered)) {
        return false;
    }

    rawFrame = std::move(decoded);
    if (recoveredSymbols != nullptr) {
        *recoveredSymbols = recovered;
    }

    return true;
}

} // namespace

ModemParams modemParamsFromSession(const SessionConfig &config) {
    ModemParams params;
    params.sampleRate = std::max<uint16_t>(8000, config.sampleRate);
    params.symbolSamples = std::max<uint16_t>(32, config.symbolSamples);
    params.bins = std::clamp<uint8_t>(config.mfskBins, 4, 16);
    params.bandMode = config.bandMode;
    params.amplitude = 0.55F;
    return params;
}

MfskModem::MfskModem(const ModemParams &params) : params_(params) {
    buildFrequencies();

    // Strong repeating preamble for noncoherent burst sync.
    preambleSymbols_.reserve(28);
    for (int i = 0; i < 10; ++i) {
        preambleSymbols_.push_back(static_cast<uint8_t>(i % std::max<int>(2, params_.bins)));
    }
    preambleSymbols_.push_back(3);
    preambleSymbols_.push_back(12);
    preambleSymbols_.push_back(5);
    preambleSymbols_.push_back(10);
    preambleSymbols_.push_back(3);
    preambleSymbols_.push_back(12);
    preambleSymbols_.push_back(5);
    preambleSymbols_.push_back(10);
    for (int i = 0; i < 10; ++i) {
        preambleSymbols_.push_back(static_cast<uint8_t>((params_.bins - 1 - i) % std::max<int>(2, params_.bins)));
    }
}

const ModemParams &MfskModem::params() const {
    return params_;
}

void MfskModem::buildFrequencies() {
    frequencies_.clear();
    frequencies_.reserve(params_.bins);

    const double base = (params_.bandMode == BandMode::Ultrasonic) ? 17100.0 : 1250.0;
    const double spacing = (params_.bandMode == BandMode::Ultrasonic) ? 110.0 : 180.0;

    for (uint8_t i = 0; i < params_.bins; ++i) {
        frequencies_.push_back(base + spacing * static_cast<double>(i));
    }
}

std::vector<float> MfskModem::modulateFrame(const std::vector<uint8_t> &rawFrame,
                                            uint8_t fecRepetition,
                                            uint8_t interleaveDepth) const {
    std::vector<uint8_t> encoded = fecProtect(rawFrame, fecRepetition, interleaveDepth);

    std::vector<uint8_t> wire;
    wire.reserve(10 + encoded.size() + 4);
    wire.push_back(kWireMagicA);
    wire.push_back(kWireMagicB);
    appendU16(wire, static_cast<uint16_t>(std::min<std::size_t>(rawFrame.size(), 0xFFFFU)));
    wire.push_back(std::clamp<uint8_t>(fecRepetition, 1, 5));
    wire.push_back(std::max<uint8_t>(1, interleaveDepth));
    appendU16(wire, static_cast<uint16_t>(std::min<std::size_t>(encoded.size(), 0xFFFFU)));
    wire.insert(wire.end(), encoded.begin(), encoded.end());
    appendU32(wire, crc32(wire));

    std::vector<uint8_t> symbols = preambleSymbols_;
    std::vector<uint8_t> wireSymbols = bytesToSymbols(wire);
    symbols.insert(symbols.end(), wireSymbols.begin(), wireSymbols.end());

    const std::size_t silencePrefix = params_.sampleRate / 40;
    const std::size_t silenceSuffix = params_.sampleRate / 25;
    std::vector<float> pcm;
    pcm.assign(silencePrefix, 0.0F);
    pcm.reserve(silencePrefix + symbols.size() * params_.symbolSamples + silenceSuffix);

    const std::size_t fade = std::max<std::size_t>(2, params_.symbolSamples / 12);

    for (uint8_t symbol : symbols) {
        const double freq = frequencies_[symbol % frequencies_.size()];
        const double step = (2.0 * kPi * freq) / static_cast<double>(params_.sampleRate);
        double phase = 0.0;
        for (uint16_t n = 0; n < params_.symbolSamples; ++n) {
            float s = static_cast<float>(params_.amplitude * std::sin(phase));
            phase += step;
            if (n < fade) {
                s *= static_cast<float>(n) / static_cast<float>(fade);
            } else if (n + fade >= params_.symbolSamples) {
                s *= static_cast<float>(params_.symbolSamples - n) / static_cast<float>(fade);
            }
            pcm.push_back(s);
        }
    }

    pcm.insert(pcm.end(), silenceSuffix, 0.0F);
    return pcm;
}

bool MfskModem::demodulateBurst(const std::vector<float> &segment,
                                std::vector<uint8_t> &rawFrame,
                                std::size_t *recoveredSymbols) const {
    rawFrame.clear();
    if (segment.size() < static_cast<std::size_t>(params_.symbolSamples) * (preambleSymbols_.size() + 10U)) {
        return false;
    }

    const std::size_t step = std::max<std::size_t>(1, params_.symbolSamples / 6);
    const std::size_t maxOffset = std::min<std::size_t>(params_.symbolSamples, segment.size() / 2);

    std::size_t bestOffset = 0;
    int bestScore = -1;
    for (std::size_t offset = 0; offset < maxOffset; offset += step) {
        const std::vector<uint8_t> symbols = decodeSymbolsAtOffset(segment,
                                                                    offset,
                                                                    params_.symbolSamples,
                                                                    frequencies_,
                                                                    params_.sampleRate,
                                                                    preambleSymbols_.size());
        if (symbols.size() < preambleSymbols_.size()) {
            continue;
        }
        int score = 0;
        for (std::size_t i = 0; i < preambleSymbols_.size(); ++i) {
            if (symbols[i] == (preambleSymbols_[i] % params_.bins)) {
                ++score;
            }
        }
        if (score > bestScore) {
            bestScore = score;
            bestOffset = offset;
        }
    }

    if (bestScore < static_cast<int>(preambleSymbols_.size() * 0.55)) {
        return false;
    }

    const std::vector<uint8_t> allSymbols = decodeSymbolsAtOffset(segment,
                                                                   bestOffset,
                                                                   params_.symbolSamples,
                                                                   frequencies_,
                                                                   params_.sampleRate,
                                                                   segment.size() / params_.symbolSamples);
    if (allSymbols.size() <= preambleSymbols_.size()) {
        return false;
    }

    std::vector<uint8_t> payloadSymbols(allSymbols.begin() + static_cast<std::ptrdiff_t>(preambleSymbols_.size()),
                                        allSymbols.end());

    std::vector<uint8_t> wire;
    if (!symbolsToBytes(payloadSymbols, wire)) {
        return false;
    }

    return parseWirePayload(wire, rawFrame, recoveredSymbols);
}

AcousticBurstReceiver::AcousticBurstReceiver(const MfskModem &modem) : modem_(modem) {
    buffer_.reserve(1U << 18U);
}

void AcousticBurstReceiver::setEnergyThreshold(float thresholdStart, float thresholdEnd) {
    thresholdStart_ = std::max(0.001F, thresholdStart);
    thresholdEnd_ = std::max(0.0001F, thresholdEnd);
}

void AcousticBurstReceiver::feedSamples(const float *samples, std::size_t count) {
    if (samples == nullptr || count == 0) {
        return;
    }

    buffer_.insert(buffer_.end(), samples, samples + static_cast<std::ptrdiff_t>(count));
    extractSegments();

    const std::size_t maxBuffer = static_cast<std::size_t>(modem_.params().sampleRate) * 8U;
    if (buffer_.size() > maxBuffer) {
        buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_.size() - maxBuffer));
    }
}

void AcousticBurstReceiver::extractSegments() {
    const std::size_t gapSamples = std::max<std::size_t>(modem_.params().sampleRate / 30U, modem_.params().symbolSamples);

    for (std::size_t i = 0; i < buffer_.size(); ++i) {
        const float a = std::abs(buffer_[i]);

        if (!inBurst_) {
            if (a >= thresholdStart_) {
                inBurst_ = true;
                burstStart_ = i;
                quietCount_ = 0;
            }
            continue;
        }

        if (a < thresholdEnd_) {
            ++quietCount_;
        } else {
            quietCount_ = 0;
        }

        if (quietCount_ >= gapSamples && i > burstStart_ + modem_.params().symbolSamples * 6U) {
            const std::size_t end = i - quietCount_;
            if (end > burstStart_) {
                pendingSegments_.emplace_back(buffer_.begin() + static_cast<std::ptrdiff_t>(burstStart_),
                                              buffer_.begin() + static_cast<std::ptrdiff_t>(end));
            }

            buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(i));
            inBurst_ = false;
            burstStart_ = 0;
            quietCount_ = 0;
            return;
        }
    }
}

bool AcousticBurstReceiver::popFrame(std::vector<uint8_t> &rawFrame, std::size_t *recoveredSymbols) {
    while (!pendingSegments_.empty()) {
        std::vector<float> segment = std::move(pendingSegments_.front());
        pendingSegments_.erase(pendingSegments_.begin());

        if (modem_.demodulateBurst(segment, rawFrame, recoveredSymbols)) {
            return true;
        }
    }

    return false;
}

void AcousticBurstReceiver::clear() {
    buffer_.clear();
    pendingSegments_.clear();
    inBurst_ = false;
    burstStart_ = 0;
    quietCount_ = 0;
}

std::vector<std::vector<uint8_t>> demodulatePcmBuffer(const std::vector<float> &pcm,
                                                       const MfskModem &modem,
                                                       LinkStats *stats) {
    std::vector<std::vector<uint8_t>> out;
    AcousticBurstReceiver rx(modem);

    const std::size_t chunk = std::max<std::size_t>(256, modem.params().sampleRate / 20U);
    for (std::size_t off = 0; off < pcm.size(); off += chunk) {
        const std::size_t n = std::min(chunk, pcm.size() - off);
        rx.feedSamples(pcm.data() + static_cast<std::ptrdiff_t>(off), n);

        while (true) {
            std::vector<uint8_t> frame;
            std::size_t recovered = 0;
            if (!rx.popFrame(frame, &recovered)) {
                break;
            }
            out.push_back(std::move(frame));
            if (stats != nullptr) {
                stats->syncLocked = true;
                stats->framesReceived += 1;
                stats->fecRecoveredCount += recovered;
            }
        }
    }

    return out;
}
