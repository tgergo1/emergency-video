#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <vector>

#include "communicator_protocol.h"
#include "queue_manager.h"

struct RouterOutgoing {
    CommEnvelopeHeader header;
    std::vector<uint8_t> payload;
    bool reliable = false;
    bool isRetry = false;
};

struct RouterIncomingVideo {
    CommEnvelopeHeader header;
    std::vector<uint8_t> codecPayload;
};

struct RouterIncomingSnapshot {
    CommEnvelopeHeader header;
    SnapshotMessage snapshot;
};

struct RouterIncomingText {
    CommEnvelopeHeader header;
    TextMessage text;
};

struct RouterIncomingProbe {
    CommEnvelopeHeader header;
    TransportProbePayload probe;
};

struct RouterEvents {
    std::vector<RouterIncomingVideo> videoFrames;
    std::vector<RouterIncomingSnapshot> snapshots;
    std::vector<RouterIncomingText> texts;
    std::vector<RouterIncomingProbe> probes;
};

class Router {
public:
    explicit Router(const NodeIdentity &identity);

    void reset(const NodeIdentity &identity);

    uint64_t enqueueConfig(const std::vector<uint8_t> &configBytes, uint32_t ttlMs = 120000);
    uint64_t enqueuePresence(uint32_t ttlMs = 45000);
    uint64_t enqueueText(const std::string &body,
                         TargetScope scope,
                         uint64_t targetNodeId,
                         uint32_t ttlMs = 600000);
    uint64_t enqueueSnapshot(const SnapshotMessage &snapshot, uint32_t ttlMs = 600000);
    uint64_t enqueueVideoFrame(const std::vector<uint8_t> &codecPayload,
                               bool keyframe,
                               uint32_t ttlMs = 12000);
    uint64_t enqueueTransportProbe(const TransportProbePayload &probe,
                                   TargetScope scope,
                                   uint64_t targetNodeId,
                                   uint32_t ttlMs = 12000);

    std::vector<RouterOutgoing> collectOutgoing(std::size_t budget,
                                                std::chrono::steady_clock::time_point now,
                                                uint32_t retryMs,
                                                uint8_t maxRetries);

    RouterEvents processIncomingEnvelope(const std::vector<uint8_t> &envelopeBytes,
                                         std::chrono::steady_clock::time_point now,
                                         const EnvelopeAuthConfig *auth = nullptr);

    [[nodiscard]] QueueStats queueStats() const;
    [[nodiscard]] std::vector<TextMessage> timelineAfter(uint64_t msgIdCursor, std::size_t limit) const;

private:
    NodeIdentity identity_;
    QueueManager queue_;
    uint64_t nextMsgId_ = 1;
    uint32_t nextSeq_ = 1;

    struct InFlight {
        RouterOutgoing packet;
        uint8_t retries = 0;
        std::chrono::steady_clock::time_point lastSend{};
    };
    std::map<uint64_t, InFlight> inFlightReliable_;

    std::deque<TextMessage> timeline_;

    uint64_t nextMsgId();
    RouterOutgoing makeOutgoing(CommPayloadType payloadType,
                                std::vector<uint8_t> payload,
                                TargetScope scope,
                                uint64_t targetNodeId,
                                uint32_t ttlMs,
                                bool reliable,
                                uint64_t forceMsgId = 0);

    void pushTimeline(TextMessage item);
};
