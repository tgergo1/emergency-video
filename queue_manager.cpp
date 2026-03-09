#include "queue_manager.h"

#include <algorithm>

namespace {

std::deque<QueuedEnvelope> *queueForType(CommPayloadType type,
                                         std::deque<QueuedEnvelope> &configQ,
                                         std::deque<QueuedEnvelope> &ackQ,
                                         std::deque<QueuedEnvelope> &textQ,
                                         std::deque<QueuedEnvelope> &snapshotQ,
                                         std::deque<QueuedEnvelope> &videoQ) {
    switch (type) {
    case CommPayloadType::Config:
        return &configQ;
    case CommPayloadType::Ack:
        return &ackQ;
    case CommPayloadType::Text:
        return &textQ;
    case CommPayloadType::Snapshot:
        return &snapshotQ;
    case CommPayloadType::VideoFrame:
        return &videoQ;
    case CommPayloadType::Presence:
        return &textQ;
    case CommPayloadType::TransportProbe:
        return &ackQ;
    }
    return &videoQ;
}

} // namespace

void QueueManager::reset() {
    configQ_.clear();
    ackQ_.clear();
    textQ_.clear();
    snapshotQ_.clear();
    videoQ_.clear();
    dedupOrder_.clear();
    dedupSet_.clear();
    dropped_ = 0;
}

uint64_t QueueManager::enqueue(const CommEnvelopeHeader &header, std::vector<uint8_t> payload, bool reliable) {
    QueuedEnvelope item;
    item.header = header;
    item.payload = std::move(payload);
    item.reliable = reliable;

    std::deque<QueuedEnvelope> *queue =
        queueForType(item.header.payloadType, configQ_, ackQ_, textQ_, snapshotQ_, videoQ_);
    queue->push_back(std::move(item));
    trimQueues();
    return header.msgId;
}

bool QueueManager::popNext(QueuedEnvelope &item) {
    auto popFrom = [&](std::deque<QueuedEnvelope> &q) -> bool {
        if (q.empty()) {
            return false;
        }
        item = std::move(q.front());
        q.pop_front();
        return true;
    };

    if (popFrom(configQ_)) {
        return true;
    }
    if (popFrom(ackQ_)) {
        return true;
    }
    if (popFrom(textQ_)) {
        return true;
    }
    if (popFrom(snapshotQ_)) {
        return true;
    }
    return popFrom(videoQ_);
}

bool QueueManager::isDuplicate(uint64_t senderNodeId, uint64_t msgId) {
    const std::pair<uint64_t, uint64_t> key{senderNodeId, msgId};
    if (dedupSet_.find(key) != dedupSet_.end()) {
        return true;
    }

    dedupSet_.insert(key);
    dedupOrder_.push_back(key);
    while (dedupOrder_.size() > kMaxDedup) {
        dedupSet_.erase(dedupOrder_.front());
        dedupOrder_.pop_front();
    }
    return false;
}

QueueStats QueueManager::stats() const {
    QueueStats out;
    out.queuedConfig = configQ_.size();
    out.queuedAck = ackQ_.size();
    out.queuedText = textQ_.size();
    out.queuedSnapshot = snapshotQ_.size();
    out.queuedVideo = videoQ_.size();
    out.dropped = dropped_;
    return out;
}

void QueueManager::trimQueues() {
    auto trim = [&](std::deque<QueuedEnvelope> &q) {
        while (q.size() > kMaxQueuedPerClass) {
            q.pop_front();
            ++dropped_;
        }
    };

    trim(configQ_);
    trim(ackQ_);
    trim(textQ_);
    trim(snapshotQ_);
    trim(videoQ_);
}
