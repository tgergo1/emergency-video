#include "acoustic_link.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace {
constexpr uint16_t kConfigSync = 0xC0DE;
constexpr uint16_t kAcousticFrameSync = 0xA55A;

void appendU8(std::vector<uint8_t> &out, uint8_t value) {
    out.push_back(value);
}

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

bool readU8(const std::vector<uint8_t> &in, std::size_t &off, uint8_t &value) {
    if (off + 1 > in.size()) {
        return false;
    }
    value = in[off++];
    return true;
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

uint32_t fnv1a32(const uint8_t *bytes, std::size_t size) {
    uint32_t hash = 2166136261U;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint32_t>(bytes[i]);
        hash *= 16777619U;
    }
    return hash;
}

uint32_t fnv1a32(const std::vector<uint8_t> &bytes) {
    return fnv1a32(bytes.data(), bytes.size());
}

std::vector<uint8_t> serializeSessionConfigForHash(const SessionConfig &config, bool includeHash) {
    std::vector<uint8_t> out;
    out.reserve(64);

    appendU16(out, kConfigSync);
    appendU8(out, config.version);
    appendU32(out, config.streamId);
    appendU32(out, config.sessionEpochMs);
    appendU16(out, config.configVersion);

    appendU8(out, static_cast<uint8_t>(config.codecMode));
    appendU16(out, config.width);
    appendU16(out, config.height);
    appendU8(out, config.blockSize);
    appendU8(out, config.residualStep);
    appendU8(out, config.keyframeInterval);

    const long long fpsMilli64 =
        std::clamp<long long>(static_cast<long long>(std::llround(config.targetFps * 1000.0)), 1LL, 240000LL);
    uint32_t fpsMilli = static_cast<uint32_t>(fpsMilli64);
    appendU32(out, fpsMilli);

    appendU8(out, static_cast<uint8_t>(config.sessionMode));
    appendU8(out, static_cast<uint8_t>(config.bandMode));
    appendU8(out, config.fecRepetition);
    appendU8(out, config.interleaveDepth);

    appendU8(out, config.arqWindow);
    appendU16(out, config.arqTimeoutMs);
    appendU8(out, config.arqMaxRetransmit);

    appendU16(out, config.sampleRate);
    appendU16(out, config.symbolSamples);
    appendU8(out, config.mfskBins);

    appendU16(out, config.cycleMs);
    appendU16(out, config.txSlotMs);

    if (includeHash) {
        appendU32(out, config.configHash);
    }

    return out;
}

} // namespace

const char *linkModeName(LinkMode mode) {
    switch (mode) {
    case LinkMode::LocalLoopback:
        return "local_loopback";
    case LinkMode::AcousticTx:
        return "acoustic_tx";
    case LinkMode::AcousticRxLive:
        return "acoustic_rx_live";
    case LinkMode::AcousticRxMedia:
        return "acoustic_rx_media";
    case LinkMode::AcousticDuplexArq:
        return "acoustic_duplex_arq";
    }
    return "local_loopback";
}

LinkMode parseLinkMode(const std::string &text, LinkMode fallback) {
    if (text == "local_loopback") {
        return LinkMode::LocalLoopback;
    }
    if (text == "acoustic_tx") {
        return LinkMode::AcousticTx;
    }
    if (text == "acoustic_rx_live") {
        return LinkMode::AcousticRxLive;
    }
    if (text == "acoustic_rx_media") {
        return LinkMode::AcousticRxMedia;
    }
    if (text == "acoustic_duplex_arq") {
        return LinkMode::AcousticDuplexArq;
    }
    return fallback;
}

const char *rxSourceName(RxSource source) {
    return source == RxSource::MediaFile ? "media_file" : "live_mic";
}

RxSource parseRxSource(const std::string &text, RxSource fallback) {
    if (text == "media_file") {
        return RxSource::MediaFile;
    }
    if (text == "live_mic") {
        return RxSource::LiveMic;
    }
    return fallback;
}

const char *sessionModeName(SessionMode mode) {
    return mode == SessionMode::DuplexArq ? "duplex_arq" : "broadcast";
}

SessionMode parseSessionMode(const std::string &text, SessionMode fallback) {
    if (text == "broadcast") {
        return SessionMode::Broadcast;
    }
    if (text == "duplex_arq") {
        return SessionMode::DuplexArq;
    }
    return fallback;
}

const char *bandModeName(BandMode mode) {
    return mode == BandMode::Ultrasonic ? "ultrasonic" : "audible";
}

BandMode parseBandMode(const std::string &text, BandMode fallback) {
    if (text == "audible") {
        return BandMode::Audible;
    }
    if (text == "ultrasonic") {
        return BandMode::Ultrasonic;
    }
    return fallback;
}

uint32_t crc32(const uint8_t *data, std::size_t size) {
    uint32_t crc = 0xFFFFFFFFU;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= static_cast<uint32_t>(data[i]);
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = -(crc & 1U);
            crc = (crc >> 1U) ^ (0xEDB88320U & mask);
        }
    }
    return ~crc;
}

uint32_t crc32(const std::vector<uint8_t> &data) {
    return crc32(data.data(), data.size());
}

uint32_t computeSessionConfigHash(const SessionConfig &config) {
    const std::vector<uint8_t> bytes = serializeSessionConfigForHash(config, false);
    return fnv1a32(bytes);
}

std::vector<uint8_t> serializeSessionConfig(SessionConfig config) {
    config.configHash = computeSessionConfigHash(config);
    std::vector<uint8_t> out = serializeSessionConfigForHash(config, true);
    appendU32(out, crc32(out));
    return out;
}

bool deserializeSessionConfig(const std::vector<uint8_t> &bytes, SessionConfig &config, std::string &error) {
    error.clear();
    if (bytes.size() < 8) {
        error = "config packet too short";
        return false;
    }

    std::size_t off = 0;
    uint16_t sync = 0;
    if (!readU16(bytes, off, sync) || sync != kConfigSync) {
        error = "invalid config sync";
        return false;
    }

    if (!readU8(bytes, off, config.version)) {
        error = "invalid config version";
        return false;
    }

    if (!readU32(bytes, off, config.streamId) || !readU32(bytes, off, config.sessionEpochMs) ||
        !readU16(bytes, off, config.configVersion)) {
        error = "invalid config header";
        return false;
    }

    uint8_t v8 = 0;
    uint32_t v32 = 0;

    if (!readU8(bytes, off, v8)) {
        error = "invalid codec mode";
        return false;
    }
    config.codecMode = (v8 == 0) ? CodecMode::Safer : CodecMode::Aggressive;

    if (!readU16(bytes, off, config.width) || !readU16(bytes, off, config.height) || !readU8(bytes, off, config.blockSize) ||
        !readU8(bytes, off, config.residualStep) || !readU8(bytes, off, config.keyframeInterval) ||
        !readU32(bytes, off, v32)) {
        error = "invalid codec dimensions";
        return false;
    }
    config.targetFps = static_cast<float>(v32) / 1000.0F;

    if (!readU8(bytes, off, v8)) {
        error = "invalid session mode";
        return false;
    }
    config.sessionMode = (v8 == 0) ? SessionMode::Broadcast : SessionMode::DuplexArq;

    if (!readU8(bytes, off, v8)) {
        error = "invalid band mode";
        return false;
    }
    config.bandMode = (v8 == 0) ? BandMode::Audible : BandMode::Ultrasonic;

    if (!readU8(bytes, off, config.fecRepetition) || !readU8(bytes, off, config.interleaveDepth) ||
        !readU8(bytes, off, config.arqWindow) || !readU16(bytes, off, config.arqTimeoutMs) ||
        !readU8(bytes, off, config.arqMaxRetransmit) || !readU16(bytes, off, config.sampleRate) ||
        !readU16(bytes, off, config.symbolSamples) || !readU8(bytes, off, config.mfskBins) ||
        !readU16(bytes, off, config.cycleMs) || !readU16(bytes, off, config.txSlotMs) ||
        !readU32(bytes, off, config.configHash) || !readU32(bytes, off, v32)) {
        error = "invalid config body";
        return false;
    }

    if (off != bytes.size()) {
        error = "config size mismatch";
        return false;
    }

    std::vector<uint8_t> withoutCrc(bytes.begin(), bytes.end() - 4);
    const uint32_t expectedPacketCrc = crc32(withoutCrc);
    if (expectedPacketCrc != v32) {
        error = "config CRC mismatch";
        return false;
    }

    const uint32_t expectedHash = computeSessionConfigHash(config);
    if (expectedHash != config.configHash) {
        error = "config hash mismatch";
        return false;
    }

    return true;
}

std::vector<uint8_t> serializeAckPacket(const AckPacket &ack) {
    std::vector<uint8_t> out;
    out.reserve(16 + ack.selectiveAcks.size() * 4);
    appendU32(out, ack.ackSeq);
    appendU16(out, static_cast<uint16_t>(std::min<std::size_t>(ack.selectiveAcks.size(), 64)));
    appendU16(out, ack.rttHintMs);
    for (std::size_t i = 0; i < ack.selectiveAcks.size() && i < 64; ++i) {
        appendU32(out, ack.selectiveAcks[i]);
    }
    appendU32(out, crc32(out));
    return out;
}

bool deserializeAckPacket(const std::vector<uint8_t> &bytes, AckPacket &ack, std::string &error) {
    error.clear();
    if (bytes.size() < 12) {
        error = "ack packet too short";
        return false;
    }

    std::size_t off = 0;
    uint16_t count = 0;
    uint32_t packetCrc = 0;
    if (!readU32(bytes, off, ack.ackSeq) || !readU16(bytes, off, count) || !readU16(bytes, off, ack.rttHintMs)) {
        error = "invalid ack header";
        return false;
    }

    ack.selectiveAcks.clear();
    ack.selectiveAcks.reserve(count);
    for (uint16_t i = 0; i < count; ++i) {
        uint32_t v = 0;
        if (!readU32(bytes, off, v)) {
            error = "invalid ack selective set";
            return false;
        }
        ack.selectiveAcks.push_back(v);
    }

    if (!readU32(bytes, off, packetCrc) || off != bytes.size()) {
        error = "invalid ack CRC field";
        return false;
    }

    std::vector<uint8_t> withoutCrc(bytes.begin(), bytes.end() - 4);
    if (crc32(withoutCrc) != packetCrc) {
        error = "ack CRC mismatch";
        return false;
    }

    return true;
}

std::vector<uint8_t> serializeAcousticFrame(AcousticFrameHeader header, const std::vector<uint8_t> &payload) {
    header.payloadSize = static_cast<uint16_t>(std::min<std::size_t>(payload.size(), 0xFFFFU));

    std::vector<uint8_t> out;
    out.reserve(40 + payload.size());

    appendU16(out, kAcousticFrameSync);
    appendU8(out, header.version);
    appendU8(out, static_cast<uint8_t>(header.payloadType));
    appendU8(out, header.flags);
    appendU8(out, 0U);

    appendU32(out, header.streamId);
    appendU32(out, header.sessionEpochMs);
    appendU16(out, header.configVersion);
    appendU32(out, header.configHash);
    appendU32(out, header.seq);
    appendU16(out, header.fragIndex);
    appendU16(out, header.fragCount);
    appendU16(out, header.payloadSize);

    const uint32_t headerCrc = crc32(out);
    appendU32(out, headerCrc);

    std::vector<uint8_t> clippedPayload(payload.begin(), payload.begin() + header.payloadSize);
    const uint32_t payloadCrc = crc32(clippedPayload);

    out.insert(out.end(), clippedPayload.begin(), clippedPayload.end());
    appendU32(out, payloadCrc);

    return out;
}

bool deserializeAcousticFrame(const std::vector<uint8_t> &bytes,
                              AcousticFrameHeader &header,
                              std::vector<uint8_t> &payload,
                              std::string &error) {
    error.clear();
    payload.clear();

    if (bytes.size() < 30) {
        error = "frame too short";
        return false;
    }

    std::size_t off = 0;
    uint16_t sync = 0;
    uint8_t payloadType = 0;
    uint8_t reserved = 0;

    if (!readU16(bytes, off, sync) || sync != kAcousticFrameSync) {
        error = "invalid frame sync";
        return false;
    }

    if (!readU8(bytes, off, header.version) || !readU8(bytes, off, payloadType) || !readU8(bytes, off, header.flags) ||
        !readU8(bytes, off, reserved) || !readU32(bytes, off, header.streamId) ||
        !readU32(bytes, off, header.sessionEpochMs) || !readU16(bytes, off, header.configVersion) ||
        !readU32(bytes, off, header.configHash) || !readU32(bytes, off, header.seq) || !readU16(bytes, off, header.fragIndex) ||
        !readU16(bytes, off, header.fragCount) || !readU16(bytes, off, header.payloadSize) ||
        !readU32(bytes, off, header.headerCrc32)) {
        error = "invalid frame header";
        return false;
    }

    if (payloadType > static_cast<uint8_t>(AcousticPayloadType::Ack)) {
        error = "invalid frame payload type";
        return false;
    }
    header.payloadType = static_cast<AcousticPayloadType>(payloadType);

    if (reserved != 0) {
        error = "invalid reserved bits";
        return false;
    }

    const std::size_t payloadStart = off;
    if (payloadStart + header.payloadSize + 4 != bytes.size()) {
        error = "frame size mismatch";
        return false;
    }

    std::vector<uint8_t> headerBytes(bytes.begin(), bytes.begin() + payloadStart - 4);
    const uint32_t expectedHeaderCrc = crc32(headerBytes);
    if (expectedHeaderCrc != header.headerCrc32) {
        error = "header CRC mismatch";
        return false;
    }

    payload.assign(bytes.begin() + static_cast<std::ptrdiff_t>(payloadStart),
                   bytes.begin() + static_cast<std::ptrdiff_t>(payloadStart + header.payloadSize));

    uint32_t payloadCrc = 0;
    off = payloadStart + header.payloadSize;
    if (!readU32(bytes, off, payloadCrc)) {
        error = "payload CRC missing";
        return false;
    }
    header.payloadCrc32 = payloadCrc;

    if (crc32(payload) != payloadCrc) {
        error = "payload CRC mismatch";
        return false;
    }

    return true;
}

std::vector<std::vector<uint8_t>> fragmentPayload(const std::vector<uint8_t> &payload, std::size_t maxPayloadPerFrame) {
    std::vector<std::vector<uint8_t>> fragments;
    const std::size_t chunk = std::max<std::size_t>(1, maxPayloadPerFrame);
    if (payload.empty()) {
        fragments.push_back({});
        return fragments;
    }

    for (std::size_t off = 0; off < payload.size(); off += chunk) {
        const std::size_t end = std::min(payload.size(), off + chunk);
        fragments.emplace_back(payload.begin() + static_cast<std::ptrdiff_t>(off),
                               payload.begin() + static_cast<std::ptrdiff_t>(end));
    }

    return fragments;
}

std::vector<uint8_t> fecProtect(const std::vector<uint8_t> &bytes, uint8_t repetition, uint8_t interleaveDepth) {
    const uint8_t rep = std::clamp<uint8_t>(repetition, 1, 5);
    const uint8_t depth = std::max<uint8_t>(1, interleaveDepth);

    std::vector<uint8_t> repeated;
    repeated.reserve(bytes.size() * static_cast<std::size_t>(rep));

    for (uint8_t r = 0; r < rep; ++r) {
        repeated.insert(repeated.end(), bytes.begin(), bytes.end());
    }

    if (depth <= 1 || repeated.empty()) {
        return repeated;
    }

    std::vector<uint8_t> interleaved(repeated.size(), 0);
    for (std::size_t i = 0; i < repeated.size(); ++i) {
        const std::size_t dst = (i % depth) * ((repeated.size() + depth - 1) / depth) + (i / depth);
        if (dst < interleaved.size()) {
            interleaved[dst] = repeated[i];
        }
    }

    return interleaved;
}

bool fecRecover(const std::vector<uint8_t> &encoded,
                std::size_t originalSize,
                uint8_t repetition,
                uint8_t interleaveDepth,
                std::vector<uint8_t> &decoded,
                std::size_t *recoveredSymbols) {
    decoded.clear();
    if (recoveredSymbols != nullptr) {
        *recoveredSymbols = 0;
    }

    const uint8_t rep = std::clamp<uint8_t>(repetition, 1, 5);
    const uint8_t depth = std::max<uint8_t>(1, interleaveDepth);
    const std::size_t expected = originalSize * static_cast<std::size_t>(rep);
    if (encoded.size() < expected || originalSize == 0) {
        return false;
    }

    std::vector<uint8_t> deinterleaved(expected, 0);
    if (depth <= 1) {
        std::copy(encoded.begin(), encoded.begin() + static_cast<std::ptrdiff_t>(expected), deinterleaved.begin());
    } else {
        for (std::size_t i = 0; i < expected; ++i) {
            const std::size_t src = (i % depth) * ((expected + depth - 1) / depth) + (i / depth);
            if (src < encoded.size()) {
                deinterleaved[i] = encoded[src];
            }
        }
    }

    decoded.resize(originalSize, 0);
    for (std::size_t i = 0; i < originalSize; ++i) {
        std::array<int, 256> hist{};
        for (uint8_t r = 0; r < rep; ++r) {
            const uint8_t v = deinterleaved[static_cast<std::size_t>(r) * originalSize + i];
            hist[v] += 1;
        }

        int bestCount = -1;
        int bestValue = 0;
        for (int value = 0; value < 256; ++value) {
            if (hist[static_cast<std::size_t>(value)] > bestCount) {
                bestCount = hist[static_cast<std::size_t>(value)];
                bestValue = value;
            }
        }
        decoded[i] = static_cast<uint8_t>(bestValue);

        if (recoveredSymbols != nullptr) {
            int disagreements = 0;
            for (uint8_t r = 0; r < rep; ++r) {
                const uint8_t v = deinterleaved[static_cast<std::size_t>(r) * originalSize + i];
                if (v != decoded[i]) {
                    ++disagreements;
                }
            }
            *recoveredSymbols += static_cast<std::size_t>(disagreements);
        }
    }

    return true;
}

FragmentReassembler::FragmentReassembler(std::chrono::milliseconds timeout) : timeout_(timeout) {}

void FragmentReassembler::push(const AcousticFrameHeader &header, const std::vector<uint8_t> &payload) {
    cleanupExpired();

    if (header.fragCount == 0 || header.fragIndex >= header.fragCount) {
        return;
    }

    Entry &entry = entries_[header.seq];
    if (entry.fragments.size() != header.fragCount) {
        entry.fragments.assign(header.fragCount, {});
        entry.present.assign(header.fragCount, 0);
    }

    entry.fragments[header.fragIndex] = payload;
    entry.present[header.fragIndex] = 1;
    entry.lastSeen = std::chrono::steady_clock::now();

    const bool complete = std::all_of(entry.present.begin(), entry.present.end(), [](uint8_t v) { return v != 0; });
    if (!complete) {
        return;
    }

    std::vector<uint8_t> out;
    std::size_t total = 0;
    for (const auto &frag : entry.fragments) {
        total += frag.size();
    }
    out.reserve(total);

    for (const auto &frag : entry.fragments) {
        out.insert(out.end(), frag.begin(), frag.end());
    }

    completed_.push_back({header.seq, std::move(out)});
    entries_.erase(header.seq);
}

bool FragmentReassembler::popComplete(uint32_t &seq, std::vector<uint8_t> &payloadOut) {
    cleanupExpired();
    if (completed_.empty()) {
        return false;
    }

    seq = completed_.front().first;
    payloadOut = std::move(completed_.front().second);
    completed_.pop_front();
    return true;
}

void FragmentReassembler::clear() {
    entries_.clear();
    completed_.clear();
}

void FragmentReassembler::cleanupExpired() {
    const auto now = std::chrono::steady_clock::now();
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (now - it->second.lastSeen > timeout_) {
            it = entries_.erase(it);
        } else {
            ++it;
        }
    }
}

bool isTxSlotNow(const SessionConfig &config,
                 std::chrono::steady_clock::time_point now,
                 bool senderRole) {
    const uint16_t cycle = std::max<uint16_t>(200, config.cycleMs);
    const uint16_t txSlot = std::min<uint16_t>(cycle, std::max<uint16_t>(50, config.txSlotMs));

    const uint64_t ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count());
    const uint64_t phase = ms % cycle;

    if (senderRole) {
        return phase < txSlot;
    }
    return phase >= txSlot;
}
