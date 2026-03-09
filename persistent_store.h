#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "communicator_protocol.h"

class PersistentStore {
public:
    PersistentStore();

    bool init(const std::string &rootDir, std::string &error);

    bool persistOutbound(const CommEnvelopeHeader &header,
                         const std::vector<uint8_t> &payload,
                         bool relayable,
                         std::string &error,
                         const EnvelopeAuthConfig *auth = nullptr);
    bool persistInbound(const CommEnvelopeHeader &header,
                        const std::vector<uint8_t> &payload,
                        bool relayable,
                        std::string &error,
                        const EnvelopeAuthConfig *auth = nullptr);

    bool exportRelayBundle(const std::string &outputPath,
                           std::size_t maxRecords,
                           std::size_t *exportedCount,
                           std::string &error);
    bool importRelayBundle(const std::string &inputPath,
                           std::vector<std::vector<uint8_t>> &envelopes,
                           std::string &error);

    bool pruneIfDiskLow(std::string &error);

    [[nodiscard]] const std::string &rootDir() const;

private:
    std::string rootDir_;
    std::string journalPath_;
    std::string blobsDir_;

    bool appendJournalLine(const std::string &line, std::string &error);
    bool persistInternal(const CommEnvelopeHeader &header,
                         const std::vector<uint8_t> &payload,
                         bool relayable,
                         bool outbound,
                         std::string &error,
                         const EnvelopeAuthConfig *auth);
    std::string makeBlobPath(uint64_t msgId, uint64_t tsMs, bool outbound) const;
};
