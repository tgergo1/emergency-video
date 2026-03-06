#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "codec.h"

enum class TransportKind : uint8_t {
    Acoustic = 0,
    Serial = 1,
    Optical = 2,
    FileRelay = 3,
};

enum class CommPayloadType : uint8_t {
    Config = 0,
    Text = 1,
    Snapshot = 2,
    VideoFrame = 3,
    Ack = 4,
    Presence = 5,
    TransportProbe = 6,
};

enum class TargetScope : uint8_t {
    Broadcast = 0,
    Direct = 1,
};

enum class DeliveryState : uint8_t {
    Queued = 0,
    Sent = 1,
    Acked = 2,
    Relayed = 3,
    Failed = 4,
};

enum class FallbackStage : uint8_t {
    Normal = 0,
    LowerFps = 1,
    LowerResolution = 2,
    SnapshotOnly = 3,
    TextOnly = 4,
};

struct NodeIdentity {
    uint64_t nodeId = 0;
    std::string alias;
};

struct CommEnvelopeHeader {
    uint8_t protoVersion = 1;
    CommPayloadType payloadType = CommPayloadType::VideoFrame;
    uint64_t msgId = 0;
    uint32_t streamId = 0;
    uint64_t senderNodeId = 0;
    TargetScope targetScope = TargetScope::Broadcast;
    uint64_t targetNodeId = 0;
    uint32_t seq = 0;
    uint16_t fragIndex = 0;
    uint16_t fragCount = 1;
    uint64_t timestampMs = 0;
    uint32_t ttlMs = 0;
    uint32_t payloadLen = 0;
    uint32_t crc32 = 0;
};

struct TextMessage {
    uint64_t msgId = 0;
    uint64_t senderNodeId = 0;
    uint64_t targetNodeId = 0;
    TargetScope targetScope = TargetScope::Broadcast;
    uint64_t timestampMs = 0;
    DeliveryState state = DeliveryState::Queued;
    std::string body;
};

struct SnapshotMessage {
    uint64_t msgId = 0;
    uint64_t senderNodeId = 0;
    uint64_t timestampMs = 0;
    uint16_t width = 0;
    uint16_t height = 0;
    std::vector<uint8_t> jpeg;
};

struct QueueStats {
    std::size_t queuedConfig = 0;
    std::size_t queuedAck = 0;
    std::size_t queuedText = 0;
    std::size_t queuedSnapshot = 0;
    std::size_t queuedVideo = 0;
    std::size_t inFlightReliable = 0;
    std::size_t dropped = 0;
};

struct RelayRecord {
    uint64_t msgId = 0;
    CommPayloadType payloadType = CommPayloadType::Text;
    uint64_t senderNodeId = 0;
    uint64_t targetNodeId = 0;
    TargetScope targetScope = TargetScope::Broadcast;
    uint64_t timestampMs = 0;
    uint32_t ttlMs = 0;
    std::string blobPath;
};

struct SessionConfigV2 {
    uint8_t version = 2;
    uint32_t streamId = 0;
    uint16_t configVersion = 1;

    CodecMode codecMode = CodecMode::Safer;
    uint16_t width = 128;
    uint16_t height = 96;
    uint8_t blockSize = 8;
    uint8_t residualStep = 1;
    uint8_t keyframeInterval = 12;
    float targetFps = 2.5F;

    bool enableAcoustic = true;
    bool enableSerial = true;
    bool enableOptical = true;
    bool enableFileRelay = true;

    uint32_t reliableRetryMs = 1500;
    uint8_t reliableMaxRetries = 8;

    FallbackStage maxFallbackStage = FallbackStage::TextOnly;

    uint32_t configHash = 0;
};

struct AckPayload {
    uint64_t ackMsgId = 0;
};

const char *transportKindName(TransportKind kind);
TransportKind parseTransportKind(const std::string &text, TransportKind fallback = TransportKind::Acoustic);

const char *payloadTypeName(CommPayloadType type);

uint64_t nowUnixMs();
uint32_t crc32Comm(const uint8_t *data, std::size_t size);
uint32_t crc32Comm(const std::vector<uint8_t> &data);

uint32_t computeSessionConfigV2Hash(const SessionConfigV2 &config);
std::vector<uint8_t> serializeSessionConfigV2(SessionConfigV2 config);
bool deserializeSessionConfigV2(const std::vector<uint8_t> &bytes, SessionConfigV2 &config, std::string &error);

std::vector<uint8_t> serializeAckPayload(const AckPayload &ack);
bool deserializeAckPayload(const std::vector<uint8_t> &bytes, AckPayload &ack, std::string &error);

std::vector<uint8_t> serializeTextPayload(uint64_t targetNodeId, const std::string &body);
bool deserializeTextPayload(const std::vector<uint8_t> &bytes,
                            uint64_t &targetNodeId,
                            std::string &body,
                            std::string &error);

std::vector<uint8_t> serializeSnapshotPayload(const SnapshotMessage &snapshot);
bool deserializeSnapshotPayload(const std::vector<uint8_t> &bytes,
                                SnapshotMessage &snapshot,
                                std::string &error);

std::vector<uint8_t> serializeCommEnvelope(const CommEnvelopeHeader &header, const std::vector<uint8_t> &payload);
bool deserializeCommEnvelope(const std::vector<uint8_t> &bytes,
                             CommEnvelopeHeader &header,
                             std::vector<uint8_t> &payload,
                             std::string &error);

std::vector<std::vector<uint8_t>> fragmentCommPayload(const std::vector<uint8_t> &payload, std::size_t maxFragmentBytes);
