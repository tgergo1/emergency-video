#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "communicator_protocol.h"

struct QueuedEnvelope {
    CommEnvelopeHeader header;
    std::vector<uint8_t> payload;
    bool reliable = false;
    uint8_t retries = 0;
    std::chrono::steady_clock::time_point lastSend{};
};

class QueueManager {
public:
    QueueManager() = default;

    void reset();

    uint64_t enqueue(const CommEnvelopeHeader &header, std::vector<uint8_t> payload, bool reliable);

    bool popNext(QueuedEnvelope &item);

    void markSent(uint64_t msgId, std::chrono::steady_clock::time_point ts);
    void markAcked(uint64_t msgId);

    std::vector<QueuedEnvelope> collectRetries(std::chrono::steady_clock::time_point now,
                                               uint32_t retryMs,
                                               uint8_t maxRetries,
                                               std::size_t maxCount);

    bool isDuplicate(uint64_t senderNodeId, uint64_t msgId);

    [[nodiscard]] QueueStats stats() const;

private:
    std::deque<QueuedEnvelope> configQ_;
    std::deque<QueuedEnvelope> ackQ_;
    std::deque<QueuedEnvelope> textQ_;
    std::deque<QueuedEnvelope> snapshotQ_;
    std::deque<QueuedEnvelope> videoQ_;

    std::map<uint64_t, QueuedEnvelope> inFlightReliable_;
    std::deque<std::pair<uint64_t, uint64_t>> dedupOrder_;
    std::set<std::pair<uint64_t, uint64_t>> dedupSet_;

    std::size_t dropped_ = 0;

    void trimQueues();
    static constexpr std::size_t kMaxQueuedPerClass = 256;
    static constexpr std::size_t kMaxDedup = 4096;
};
