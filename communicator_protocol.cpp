#include "communicator_protocol.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace {
constexpr uint16_t kEnvelopeSync = 0xEA12;
constexpr uint16_t kSessionSync = 0x52C1;

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

void appendU64(std::vector<uint8_t> &out, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xFFU));
    }
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

bool readU64(const std::vector<uint8_t> &in, std::size_t &off, uint64_t &value) {
    if (off + 8 > in.size()) {
        return false;
    }
    value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(in[off + i]) << (8 * i);
    }
    off += 8;
    return true;
}

uint32_t fnv1a32(const std::vector<uint8_t> &bytes) {
    uint32_t hash = 2166136261U;
    for (uint8_t b : bytes) {
        hash ^= static_cast<uint32_t>(b);
        hash *= 16777619U;
    }
    return hash;
}

} // namespace

const char *transportKindName(TransportKind kind) {
    switch (kind) {
    case TransportKind::Acoustic:
        return "acoustic";
    case TransportKind::Serial:
        return "serial";
    case TransportKind::Optical:
        return "optical";
    case TransportKind::FileRelay:
        return "file_relay";
    }
    return "acoustic";
}

TransportKind parseTransportKind(const std::string &text, TransportKind fallback) {
    if (text == "acoustic") {
        return TransportKind::Acoustic;
    }
    if (text == "serial") {
        return TransportKind::Serial;
    }
    if (text == "optical") {
        return TransportKind::Optical;
    }
    if (text == "file_relay") {
        return TransportKind::FileRelay;
    }
    return fallback;
}

const char *payloadTypeName(CommPayloadType type) {
    switch (type) {
    case CommPayloadType::Config:
        return "config";
    case CommPayloadType::Text:
        return "text";
    case CommPayloadType::Snapshot:
        return "snapshot";
    case CommPayloadType::VideoFrame:
        return "video_frame";
    case CommPayloadType::Ack:
        return "ack";
    case CommPayloadType::Presence:
        return "presence";
    case CommPayloadType::TransportProbe:
        return "transport_probe";
    }
    return "unknown";
}

uint64_t nowUnixMs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch())
                                    .count());
}

uint32_t crc32Comm(const uint8_t *data, std::size_t size) {
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

uint32_t crc32Comm(const std::vector<uint8_t> &data) {
    return crc32Comm(data.data(), data.size());
}

uint32_t computeSessionConfigV2Hash(const SessionConfigV2 &config) {
    std::vector<uint8_t> bytes;
    bytes.reserve(64);
    appendU8(bytes, config.version);
    appendU32(bytes, config.streamId);
    appendU16(bytes, config.configVersion);
    appendU8(bytes, static_cast<uint8_t>(config.codecMode));
    appendU16(bytes, config.width);
    appendU16(bytes, config.height);
    appendU8(bytes, config.blockSize);
    appendU8(bytes, config.residualStep);
    appendU8(bytes, config.keyframeInterval);
    appendU32(bytes, static_cast<uint32_t>(std::clamp(std::llround(config.targetFps * 1000.0), 1LL, 240000LL)));
    appendU8(bytes, config.enableAcoustic ? 1U : 0U);
    appendU8(bytes, config.enableSerial ? 1U : 0U);
    appendU8(bytes, config.enableOptical ? 1U : 0U);
    appendU8(bytes, config.enableFileRelay ? 1U : 0U);
    appendU32(bytes, config.reliableRetryMs);
    appendU8(bytes, config.reliableMaxRetries);
    appendU8(bytes, static_cast<uint8_t>(config.maxFallbackStage));
    return fnv1a32(bytes);
}

std::vector<uint8_t> serializeSessionConfigV2(SessionConfigV2 config) {
    config.configHash = computeSessionConfigV2Hash(config);

    std::vector<uint8_t> out;
    out.reserve(96);
    appendU16(out, kSessionSync);
    appendU8(out, config.version);
    appendU32(out, config.streamId);
    appendU16(out, config.configVersion);
    appendU8(out, static_cast<uint8_t>(config.codecMode));
    appendU16(out, config.width);
    appendU16(out, config.height);
    appendU8(out, config.blockSize);
    appendU8(out, config.residualStep);
    appendU8(out, config.keyframeInterval);
    appendU32(out, static_cast<uint32_t>(std::clamp(std::llround(config.targetFps * 1000.0), 1LL, 240000LL)));
    appendU8(out, config.enableAcoustic ? 1U : 0U);
    appendU8(out, config.enableSerial ? 1U : 0U);
    appendU8(out, config.enableOptical ? 1U : 0U);
    appendU8(out, config.enableFileRelay ? 1U : 0U);
    appendU32(out, config.reliableRetryMs);
    appendU8(out, config.reliableMaxRetries);
    appendU8(out, static_cast<uint8_t>(config.maxFallbackStage));
    appendU32(out, config.configHash);
    appendU32(out, crc32Comm(out));
    return out;
}

bool deserializeSessionConfigV2(const std::vector<uint8_t> &bytes, SessionConfigV2 &config, std::string &error) {
    error.clear();
    if (bytes.size() < 16) {
        error = "session config too short";
        return false;
    }

    std::size_t off = 0;
    uint16_t sync = 0;
    uint8_t b = 0;
    uint32_t u32 = 0;

    if (!readU16(bytes, off, sync) || sync != kSessionSync) {
        error = "invalid session sync";
        return false;
    }
    if (!readU8(bytes, off, config.version) || !readU32(bytes, off, config.streamId) || !readU16(bytes, off, config.configVersion) ||
        !readU8(bytes, off, b)) {
        error = "invalid session header";
        return false;
    }
    config.codecMode = (b == 0) ? CodecMode::Safer : CodecMode::Aggressive;

    if (!readU16(bytes, off, config.width) || !readU16(bytes, off, config.height) || !readU8(bytes, off, config.blockSize) ||
        !readU8(bytes, off, config.residualStep) || !readU8(bytes, off, config.keyframeInterval) || !readU32(bytes, off, u32)) {
        error = "invalid session codec section";
        return false;
    }
    config.targetFps = static_cast<float>(u32) / 1000.0F;

    if (!readU8(bytes, off, b)) {
        error = "invalid acoustic flag";
        return false;
    }
    config.enableAcoustic = b != 0;
    if (!readU8(bytes, off, b)) {
        error = "invalid serial flag";
        return false;
    }
    config.enableSerial = b != 0;
    if (!readU8(bytes, off, b)) {
        error = "invalid optical flag";
        return false;
    }
    config.enableOptical = b != 0;
    if (!readU8(bytes, off, b)) {
        error = "invalid file relay flag";
        return false;
    }
    config.enableFileRelay = b != 0;

    if (!readU32(bytes, off, config.reliableRetryMs) || !readU8(bytes, off, config.reliableMaxRetries) || !readU8(bytes, off, b) ||
        !readU32(bytes, off, config.configHash)) {
        error = "invalid session reliability section";
        return false;
    }
    config.maxFallbackStage = static_cast<FallbackStage>(b);

    uint32_t packetCrc = 0;
    if (!readU32(bytes, off, packetCrc) || off != bytes.size()) {
        error = "invalid session crc section";
        return false;
    }

    const std::vector<uint8_t> withoutCrc(bytes.begin(), bytes.end() - 4);
    if (crc32Comm(withoutCrc) != packetCrc) {
        error = "session crc mismatch";
        return false;
    }

    const uint32_t expectedHash = computeSessionConfigV2Hash(config);
    if (expectedHash != config.configHash) {
        error = "session hash mismatch";
        return false;
    }

    return true;
}

std::vector<uint8_t> serializeAckPayload(const AckPayload &ack) {
    std::vector<uint8_t> out;
    out.reserve(8);
    appendU64(out, ack.ackMsgId);
    return out;
}

bool deserializeAckPayload(const std::vector<uint8_t> &bytes, AckPayload &ack, std::string &error) {
    error.clear();
    std::size_t off = 0;
    if (!readU64(bytes, off, ack.ackMsgId) || off != bytes.size()) {
        error = "invalid ack payload";
        return false;
    }
    return true;
}

std::vector<uint8_t> serializeTextPayload(uint64_t targetNodeId, const std::string &body) {
    std::vector<uint8_t> out;
    out.reserve(16 + body.size());
    appendU64(out, targetNodeId);
    appendU32(out, static_cast<uint32_t>(std::min<std::size_t>(body.size(), 1U << 20U)));
    out.insert(out.end(), body.begin(), body.end());
    return out;
}

bool deserializeTextPayload(const std::vector<uint8_t> &bytes,
                            uint64_t &targetNodeId,
                            std::string &body,
                            std::string &error) {
    error.clear();
    std::size_t off = 0;
    uint32_t bodyLen = 0;
    if (!readU64(bytes, off, targetNodeId) || !readU32(bytes, off, bodyLen)) {
        error = "invalid text payload header";
        return false;
    }
    if (off + bodyLen > bytes.size()) {
        error = "invalid text payload length";
        return false;
    }
    body.assign(reinterpret_cast<const char *>(bytes.data() + off), bodyLen);
    off += bodyLen;
    if (off != bytes.size()) {
        error = "invalid text payload tail";
        return false;
    }
    return true;
}

std::vector<uint8_t> serializeSnapshotPayload(const SnapshotMessage &snapshot) {
    std::vector<uint8_t> out;
    out.reserve(16 + snapshot.jpeg.size());
    appendU64(out, snapshot.msgId);
    appendU16(out, snapshot.width);
    appendU16(out, snapshot.height);
    appendU32(out, static_cast<uint32_t>(std::min<std::size_t>(snapshot.jpeg.size(), 1U << 22U)));
    out.insert(out.end(), snapshot.jpeg.begin(), snapshot.jpeg.end());
    return out;
}

bool deserializeSnapshotPayload(const std::vector<uint8_t> &bytes,
                                SnapshotMessage &snapshot,
                                std::string &error) {
    error.clear();
    std::size_t off = 0;
    uint32_t len = 0;
    if (!readU64(bytes, off, snapshot.msgId) || !readU16(bytes, off, snapshot.width) || !readU16(bytes, off, snapshot.height) ||
        !readU32(bytes, off, len)) {
        error = "invalid snapshot header";
        return false;
    }
    if (off + len > bytes.size()) {
        error = "invalid snapshot payload length";
        return false;
    }
    snapshot.jpeg.assign(bytes.begin() + static_cast<std::ptrdiff_t>(off),
                         bytes.begin() + static_cast<std::ptrdiff_t>(off + len));
    off += len;
    if (off != bytes.size()) {
        error = "invalid snapshot tail";
        return false;
    }
    return true;
}

std::vector<uint8_t> serializeCommEnvelope(const CommEnvelopeHeader &headerIn, const std::vector<uint8_t> &payload) {
    CommEnvelopeHeader header = headerIn;
    header.payloadLen = static_cast<uint32_t>(payload.size());

    std::vector<uint8_t> out;
    out.reserve(96 + payload.size());
    appendU16(out, kEnvelopeSync);
    appendU8(out, header.protoVersion);
    appendU8(out, static_cast<uint8_t>(header.payloadType));
    appendU64(out, header.msgId);
    appendU32(out, header.streamId);
    appendU64(out, header.senderNodeId);
    appendU8(out, static_cast<uint8_t>(header.targetScope));
    appendU64(out, header.targetNodeId);
    appendU32(out, header.seq);
    appendU16(out, header.fragIndex);
    appendU16(out, header.fragCount);
    appendU64(out, header.timestampMs);
    appendU32(out, header.ttlMs);
    appendU32(out, header.payloadLen);
    const std::size_t crcOffset = out.size();
    appendU32(out, 0U); // placeholder crc
    out.insert(out.end(), payload.begin(), payload.end());

    const uint32_t crc = crc32Comm(out.data(), out.size());
    out[crcOffset + 0] = static_cast<uint8_t>(crc & 0xFFU);
    out[crcOffset + 1] = static_cast<uint8_t>((crc >> 8U) & 0xFFU);
    out[crcOffset + 2] = static_cast<uint8_t>((crc >> 16U) & 0xFFU);
    out[crcOffset + 3] = static_cast<uint8_t>((crc >> 24U) & 0xFFU);
    return out;
}

bool deserializeCommEnvelope(const std::vector<uint8_t> &bytes,
                             CommEnvelopeHeader &header,
                             std::vector<uint8_t> &payload,
                             std::string &error) {
    error.clear();
    payload.clear();

    if (bytes.size() < 45) {
        error = "envelope too short";
        return false;
    }

    const std::vector<uint8_t> copy = bytes;
    std::size_t off = 0;
    uint16_t sync = 0;
    uint8_t payloadType = 0;

    if (!readU16(copy, off, sync) || sync != kEnvelopeSync) {
        error = "invalid envelope sync";
        return false;
    }
    if (!readU8(copy, off, header.protoVersion) || !readU8(copy, off, payloadType) || !readU64(copy, off, header.msgId) ||
        !readU32(copy, off, header.streamId) || !readU64(copy, off, header.senderNodeId)) {
        error = "invalid envelope base header";
        return false;
    }
    header.payloadType = static_cast<CommPayloadType>(payloadType);

    uint8_t scope = 0;
    if (!readU8(copy, off, scope) || !readU64(copy, off, header.targetNodeId) || !readU32(copy, off, header.seq) ||
        !readU16(copy, off, header.fragIndex) || !readU16(copy, off, header.fragCount) || !readU64(copy, off, header.timestampMs) ||
        !readU32(copy, off, header.ttlMs) || !readU32(copy, off, header.payloadLen) || !readU32(copy, off, header.crc32)) {
        error = "invalid envelope extended header";
        return false;
    }
    header.targetScope = (scope == 0U) ? TargetScope::Broadcast : TargetScope::Direct;

    if (off + header.payloadLen != copy.size()) {
        error = "envelope payload length mismatch";
        return false;
    }

    const std::size_t crcOffset = off - 4;
    std::vector<uint8_t> crcBuf = copy;
    crcBuf[crcOffset + 0] = 0;
    crcBuf[crcOffset + 1] = 0;
    crcBuf[crcOffset + 2] = 0;
    crcBuf[crcOffset + 3] = 0;
    const uint32_t crc = crc32Comm(crcBuf.data(), crcBuf.size());
    if (crc != header.crc32) {
        error = "envelope crc mismatch";
        return false;
    }

    payload.assign(copy.begin() + static_cast<std::ptrdiff_t>(off), copy.end());
    return true;
}

std::vector<std::vector<uint8_t>> fragmentCommPayload(const std::vector<uint8_t> &payload, std::size_t maxFragmentBytes) {
    std::vector<std::vector<uint8_t>> out;
    const std::size_t safeMax = std::max<std::size_t>(1, maxFragmentBytes);

    if (payload.empty()) {
        out.push_back({});
        return out;
    }

    for (std::size_t off = 0; off < payload.size(); off += safeMax) {
        const std::size_t len = std::min(safeMax, payload.size() - off);
        out.emplace_back(payload.begin() + static_cast<std::ptrdiff_t>(off),
                         payload.begin() + static_cast<std::ptrdiff_t>(off + len));
    }
    return out;
}
