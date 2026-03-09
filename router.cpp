#include "router.h"

#include <algorithm>
#include <cstring>

namespace {

bool shouldBeReliable(CommPayloadType type) {
    return type == CommPayloadType::Config || type == CommPayloadType::Ack || type == CommPayloadType::Text ||
           type == CommPayloadType::Snapshot || type == CommPayloadType::Presence;
}

} // namespace

Router::Router(const NodeIdentity &identity) : identity_(identity) {}

void Router::reset(const NodeIdentity &identity) {
    identity_ = identity;
    queue_.reset();
    nextMsgId_ = 1;
    nextSeq_ = 1;
    inFlightReliable_.clear();
    timeline_.clear();
}

uint64_t Router::nextMsgId() {
    if (nextMsgId_ == 0) {
        nextMsgId_ = 1;
    }
    return nextMsgId_++;
}

RouterOutgoing Router::makeOutgoing(CommPayloadType payloadType,
                                    std::vector<uint8_t> payload,
                                    TargetScope scope,
                                    uint64_t targetNodeId,
                                    uint32_t ttlMs,
                                    bool reliable,
                                    uint64_t forceMsgId) {
    RouterOutgoing out;
    out.reliable = reliable;
    out.isRetry = false;
    out.payload = std::move(payload);
    out.header.protoVersion = kCommProtocolVersion;
    out.header.payloadType = payloadType;
    out.header.msgId = (forceMsgId == 0) ? nextMsgId() : forceMsgId;
    out.header.senderNodeId = identity_.nodeId;
    out.header.targetScope = scope;
    out.header.targetNodeId = targetNodeId;
    out.header.seq = nextSeq_++;
    out.header.fragIndex = 0;
    out.header.fragCount = 1;
    out.header.timestampMs = nowUnixMs();
    out.header.ttlMs = ttlMs;
    return out;
}

void Router::pushTimeline(TextMessage item) {
    timeline_.push_back(std::move(item));
    while (timeline_.size() > 1024) {
        timeline_.pop_front();
    }
}

uint64_t Router::enqueueConfig(const std::vector<uint8_t> &configBytes, uint32_t ttlMs) {
    RouterOutgoing out = makeOutgoing(CommPayloadType::Config,
                                      configBytes,
                                      TargetScope::Broadcast,
                                      0,
                                      ttlMs,
                                      true);
    queue_.enqueue(out.header, out.payload, out.reliable);
    return out.header.msgId;
}

uint64_t Router::enqueuePresence(uint32_t ttlMs) {
    std::vector<uint8_t> payload(identity_.alias.begin(), identity_.alias.end());
    RouterOutgoing out =
        makeOutgoing(CommPayloadType::Presence, std::move(payload), TargetScope::Broadcast, 0, ttlMs, true);
    queue_.enqueue(out.header, out.payload, out.reliable);
    return out.header.msgId;
}

uint64_t Router::enqueueText(const std::string &body,
                             TargetScope scope,
                             uint64_t targetNodeId,
                             uint32_t ttlMs) {
    RouterOutgoing out = makeOutgoing(CommPayloadType::Text,
                                      serializeTextPayload(targetNodeId, body),
                                      scope,
                                      targetNodeId,
                                      ttlMs,
                                      true);
    queue_.enqueue(out.header, out.payload, out.reliable);

    TextMessage local;
    local.msgId = out.header.msgId;
    local.senderNodeId = identity_.nodeId;
    local.targetNodeId = targetNodeId;
    local.targetScope = scope;
    local.timestampMs = out.header.timestampMs;
    local.state = DeliveryState::Queued;
    local.body = body;
    pushTimeline(std::move(local));

    return out.header.msgId;
}

uint64_t Router::enqueueSnapshot(const SnapshotMessage &snapshot, uint32_t ttlMs) {
    SnapshotMessage send = snapshot;
    if (send.msgId == 0) {
        send.msgId = nextMsgId();
    }

    RouterOutgoing out = makeOutgoing(CommPayloadType::Snapshot,
                                      serializeSnapshotPayload(send),
                                      TargetScope::Broadcast,
                                      0,
                                      ttlMs,
                                      true,
                                      send.msgId);
    queue_.enqueue(out.header, out.payload, out.reliable);

    TextMessage local;
    local.msgId = out.header.msgId;
    local.senderNodeId = identity_.nodeId;
    local.targetNodeId = 0;
    local.targetScope = TargetScope::Broadcast;
    local.timestampMs = out.header.timestampMs;
    local.state = DeliveryState::Queued;
    local.body = "[snapshot " + std::to_string(send.width) + "x" + std::to_string(send.height) + "]";
    pushTimeline(std::move(local));

    return out.header.msgId;
}

uint64_t Router::enqueueVideoFrame(const std::vector<uint8_t> &codecPayload,
                                   bool keyframe,
                                   uint32_t ttlMs) {
    RouterOutgoing out = makeOutgoing(CommPayloadType::VideoFrame,
                                      codecPayload,
                                      TargetScope::Broadcast,
                                      0,
                                      ttlMs,
                                      false);
    if (keyframe) {
        out.header.ttlMs = std::max<uint32_t>(ttlMs, 22000);
    }
    queue_.enqueue(out.header, out.payload, out.reliable);
    return out.header.msgId;
}

uint64_t Router::enqueueTransportProbe(const TransportProbePayload &probe,
                                       TargetScope scope,
                                       uint64_t targetNodeId,
                                       uint32_t ttlMs) {
    RouterOutgoing out = makeOutgoing(CommPayloadType::TransportProbe,
                                      serializeTransportProbePayload(probe),
                                      scope,
                                      targetNodeId,
                                      ttlMs,
                                      false);
    queue_.enqueue(out.header, out.payload, out.reliable);
    return out.header.msgId;
}

std::vector<RouterOutgoing> Router::collectOutgoing(std::size_t budget,
                                                    std::chrono::steady_clock::time_point now,
                                                    uint32_t retryMs,
                                                    uint8_t maxRetries) {
    std::vector<RouterOutgoing> out;
    out.reserve(budget);

    // Retries first: keep reliable control/data alive.
    for (auto it = inFlightReliable_.begin(); it != inFlightReliable_.end();) {
        if (out.size() >= budget) {
            break;
        }

        InFlight &entry = it->second;
        if (entry.retries >= maxRetries) {
            it = inFlightReliable_.erase(it);
            continue;
        }

        const auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - entry.lastSend).count();
        if (age >= static_cast<long long>(retryMs)) {
            entry.lastSend = now;
            entry.retries += 1;
            RouterOutgoing retryPacket = entry.packet;
            retryPacket.isRetry = true;
            out.push_back(std::move(retryPacket));
        }
        ++it;
    }

    QueuedEnvelope item;
    while (out.size() < budget && queue_.popNext(item)) {
        RouterOutgoing send;
        send.header = item.header;
        send.payload = std::move(item.payload);
        send.reliable = item.reliable || shouldBeReliable(send.header.payloadType);
        send.isRetry = false;
        send.header.seq = nextSeq_++;
        send.header.timestampMs = nowUnixMs();
        send.header.fragIndex = 0;
        send.header.fragCount = 1;

        out.push_back(send);

        if (send.reliable) {
            InFlight flight;
            flight.packet = send;
            flight.retries = 0;
            flight.lastSend = now;
            inFlightReliable_[send.header.msgId] = std::move(flight);
        }

        for (TextMessage &msg : timeline_) {
            if (msg.msgId == send.header.msgId && msg.state == DeliveryState::Queued) {
                msg.state = DeliveryState::Sent;
            }
        }
    }

    return out;
}

RouterEvents Router::processIncomingEnvelope(const std::vector<uint8_t> &envelopeBytes,
                                             std::chrono::steady_clock::time_point /*now*/,
                                             const EnvelopeAuthConfig *auth) {
    RouterEvents events;

    CommEnvelopeHeader header;
    std::vector<uint8_t> payload;
    std::string error;
    bool authFailed = false;
    if (!deserializeCommEnvelope(envelopeBytes, header, payload, error, auth, &authFailed)) {
        return events;
    }

    if (queue_.isDuplicate(header.senderNodeId, header.msgId)) {
        return events;
    }

    if (header.payloadType == CommPayloadType::Ack) {
        AckPayload ack;
        if (deserializeAckPayload(payload, ack, error)) {
            inFlightReliable_.erase(ack.ackMsgId);
            for (TextMessage &msg : timeline_) {
                if (msg.msgId == ack.ackMsgId) {
                    msg.state = DeliveryState::Acked;
                }
            }
        }
        return events;
    }

    // Send ACK for reliable incoming payloads.
    if (shouldBeReliable(header.payloadType) && header.senderNodeId != identity_.nodeId) {
        AckPayload ack;
        ack.ackMsgId = header.msgId;
        RouterOutgoing ackOut = makeOutgoing(CommPayloadType::Ack,
                                             serializeAckPayload(ack),
                                             TargetScope::Direct,
                                             header.senderNodeId,
                                             120000,
                                             true);
        queue_.enqueue(ackOut.header, ackOut.payload, true);
    }

    if (header.payloadType == CommPayloadType::VideoFrame) {
        RouterIncomingVideo video;
        video.header = header;
        video.codecPayload = std::move(payload);
        events.videoFrames.push_back(std::move(video));
        return events;
    }

    if (header.payloadType == CommPayloadType::Text) {
        RouterIncomingText in;
        in.header = header;
        uint64_t target = 0;
        std::string body;
        if (deserializeTextPayload(payload, target, body, error)) {
            in.text.msgId = header.msgId;
            in.text.senderNodeId = header.senderNodeId;
            in.text.targetNodeId = target;
            in.text.targetScope = header.targetScope;
            in.text.timestampMs = header.timestampMs;
            in.text.state = DeliveryState::Relayed;
            in.text.body = body;
            events.texts.push_back(in);
            pushTimeline(in.text);
        }
        return events;
    }

    if (header.payloadType == CommPayloadType::Snapshot) {
        RouterIncomingSnapshot in;
        in.header = header;
        if (deserializeSnapshotPayload(payload, in.snapshot, error)) {
            in.snapshot.senderNodeId = header.senderNodeId;
            in.snapshot.timestampMs = header.timestampMs;
            events.snapshots.push_back(std::move(in));

            TextMessage meta;
            meta.msgId = header.msgId;
            meta.senderNodeId = header.senderNodeId;
            meta.targetNodeId = header.targetNodeId;
            meta.targetScope = header.targetScope;
            meta.timestampMs = header.timestampMs;
            meta.state = DeliveryState::Relayed;
            meta.body = "[snapshot " + std::to_string(header.payloadLen) + " bytes]";
            pushTimeline(std::move(meta));
        }
        return events;
    }

    if (header.payloadType == CommPayloadType::Presence) {
        return events;
    }

    if (header.payloadType == CommPayloadType::Config) {
        return events;
    }

    if (header.payloadType == CommPayloadType::TransportProbe) {
        TransportProbePayload probe;
        if (!deserializeTransportProbePayload(payload, probe, error)) {
            return events;
        }

        RouterIncomingProbe in;
        in.header = header;
        in.probe = probe;
        events.probes.push_back(in);

        if (probe.kind == ProbeKind::Ping && header.senderNodeId != identity_.nodeId) {
            TransportProbePayload pong;
            pong.probeId = probe.probeId;
            pong.kind = ProbeKind::Pong;
            pong.sentTsMs = probe.sentTsMs;
            RouterOutgoing pongOut = makeOutgoing(CommPayloadType::TransportProbe,
                                                  serializeTransportProbePayload(pong),
                                                  TargetScope::Direct,
                                                  header.senderNodeId,
                                                  10000,
                                                  false);
            queue_.enqueue(pongOut.header, pongOut.payload, pongOut.reliable);
        }
        return events;
    }

    return events;
}

QueueStats Router::queueStats() const {
    QueueStats stats = queue_.stats();
    stats.inFlightReliable = inFlightReliable_.size();
    return stats;
}

std::vector<TextMessage> Router::timelineAfter(uint64_t msgIdCursor, std::size_t limit) const {
    std::vector<TextMessage> out;
    out.reserve(limit);
    for (const TextMessage &msg : timeline_) {
        if (msg.msgId <= msgIdCursor) {
            continue;
        }
        out.push_back(msg);
        if (out.size() >= limit) {
            break;
        }
    }
    return out;
}
