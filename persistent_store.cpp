#include "persistent_store.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace {

void writeU32(std::ofstream &out, uint32_t value) {
    const uint8_t bytes[4] = {
        static_cast<uint8_t>(value & 0xFFU),
        static_cast<uint8_t>((value >> 8U) & 0xFFU),
        static_cast<uint8_t>((value >> 16U) & 0xFFU),
        static_cast<uint8_t>((value >> 24U) & 0xFFU),
    };
    out.write(reinterpret_cast<const char *>(bytes), 4);
}

bool readU32(std::ifstream &in, uint32_t &value) {
    uint8_t bytes[4] = {};
    in.read(reinterpret_cast<char *>(bytes), 4);
    if (!in) {
        return false;
    }
    value = static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8U) |
            (static_cast<uint32_t>(bytes[2]) << 16U) | (static_cast<uint32_t>(bytes[3]) << 24U);
    return true;
}

} // namespace

PersistentStore::PersistentStore() = default;

bool PersistentStore::init(const std::string &rootDir, std::string &error) {
    error.clear();
    rootDir_ = rootDir;
    blobsDir_ = (fs::path(rootDir_) / "blobs").string();
    journalPath_ = (fs::path(rootDir_) / "journal.log").string();

    std::error_code ec;
    fs::create_directories(blobsDir_, ec);
    if (ec) {
        error = "failed to create store directories: " + ec.message();
        return false;
    }

    if (!fs::exists(journalPath_)) {
        std::ofstream create(journalPath_, std::ios::binary);
        if (!create.is_open()) {
            error = "failed to create journal";
            return false;
        }
    }

    return true;
}

bool PersistentStore::appendJournalLine(const std::string &line, std::string &error) {
    std::ofstream out(journalPath_, std::ios::binary | std::ios::app);
    if (!out.is_open()) {
        error = "failed to open journal";
        return false;
    }
    out << line << '\n';
    if (!out.good()) {
        error = "failed to append journal";
        return false;
    }
    return true;
}

std::string PersistentStore::makeBlobPath(uint64_t msgId, uint64_t tsMs, bool outbound) const {
    std::ostringstream oss;
    oss << (outbound ? "out_" : "in_") << tsMs << "_" << msgId << ".bin";
    return (fs::path(blobsDir_) / oss.str()).string();
}

bool PersistentStore::persistOutbound(const CommEnvelopeHeader &header,
                                      const std::vector<uint8_t> &payload,
                                      bool relayable,
                                      std::string &error) {
    return persistInternal(header, payload, relayable, true, error);
}

bool PersistentStore::persistInbound(const CommEnvelopeHeader &header,
                                     const std::vector<uint8_t> &payload,
                                     bool relayable,
                                     std::string &error) {
    return persistInternal(header, payload, relayable, false, error);
}

bool PersistentStore::persistInternal(const CommEnvelopeHeader &header,
                                      const std::vector<uint8_t> &payload,
                                      bool relayable,
                                      bool outbound,
                                      std::string &error) {
    error.clear();
    if (!relayable) {
        return true;
    }

    const std::string blobPath = makeBlobPath(header.msgId, header.timestampMs, outbound);
    std::ofstream out(blobPath, std::ios::binary);
    if (!out.is_open()) {
        error = "failed to create payload blob";
        pruneIfDiskLow(error);
        return false;
    }

    const std::vector<uint8_t> envelope = serializeCommEnvelope(header, payload);
    out.write(reinterpret_cast<const char *>(envelope.data()), static_cast<std::streamsize>(envelope.size()));
    if (!out.good()) {
        error = "failed to write payload blob";
        pruneIfDiskLow(error);
        return false;
    }

    std::ostringstream line;
    line << header.timestampMs << "|" << header.msgId << "|" << static_cast<int>(header.payloadType) << "|"
         << header.senderNodeId << "|" << header.targetNodeId << "|" << static_cast<int>(header.targetScope) << "|"
         << blobPath;
    if (!appendJournalLine(line.str(), error)) {
        return false;
    }

    pruneIfDiskLow(error);
    return true;
}

bool PersistentStore::exportRelayBundle(const std::string &outputPath,
                                        std::size_t maxRecords,
                                        std::size_t *exportedCount,
                                        std::string &error) {
    error.clear();
    if (exportedCount != nullptr) {
        *exportedCount = 0;
    }

    std::ifstream in(journalPath_);
    if (!in.is_open()) {
        error = "failed to open journal for export";
        return false;
    }

    std::vector<std::string> paths;
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t lastSep = line.rfind('|');
        if (lastSep == std::string::npos || lastSep + 1 >= line.size()) {
            continue;
        }
        paths.push_back(line.substr(lastSep + 1));
        if (paths.size() >= maxRecords) {
            break;
        }
    }

    std::ofstream out(outputPath, std::ios::binary);
    if (!out.is_open()) {
        error = "failed to open output relay bundle";
        return false;
    }

    static constexpr char kMagic[5] = {'E', 'V', 'R', 'L', '1'};
    out.write(kMagic, 5);
    writeU32(out, static_cast<uint32_t>(paths.size()));

    std::size_t exported = 0;
    for (const std::string &path : paths) {
        std::ifstream blob(path, std::ios::binary);
        if (!blob.is_open()) {
            continue;
        }
        std::vector<uint8_t> data((std::istreambuf_iterator<char>(blob)), std::istreambuf_iterator<char>());
        if (data.empty()) {
            continue;
        }
        writeU32(out, static_cast<uint32_t>(data.size()));
        out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!out.good()) {
            error = "failed to write relay bundle";
            return false;
        }
        ++exported;
    }

    if (exportedCount != nullptr) {
        *exportedCount = exported;
    }
    return true;
}

bool PersistentStore::importRelayBundle(const std::string &inputPath,
                                        std::vector<std::vector<uint8_t>> &envelopes,
                                        std::string &error) {
    error.clear();
    envelopes.clear();

    std::ifstream in(inputPath, std::ios::binary);
    if (!in.is_open()) {
        error = "failed to open relay bundle";
        return false;
    }

    char magic[5] = {};
    in.read(magic, 5);
    if (!in || std::string(magic, 5) != "EVRL1") {
        error = "invalid relay bundle magic";
        return false;
    }

    uint32_t count = 0;
    if (!readU32(in, count)) {
        error = "invalid relay bundle count";
        return false;
    }

    for (uint32_t i = 0; i < count; ++i) {
        uint32_t len = 0;
        if (!readU32(in, len)) {
            error = "invalid relay entry length";
            return false;
        }
        if (len == 0 || len > (1U << 24U)) {
            error = "relay entry length out of range";
            return false;
        }
        std::vector<uint8_t> bytes(len);
        in.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(len));
        if (!in) {
            error = "relay entry read failed";
            return false;
        }
        envelopes.push_back(std::move(bytes));
    }

    return true;
}

bool PersistentStore::pruneIfDiskLow(std::string &error) {
    error.clear();
    if (rootDir_.empty()) {
        return true;
    }

    std::error_code ec;
    const fs::space_info info = fs::space(rootDir_, ec);
    if (ec) {
        return true;
    }

    // Unlimited-by-default policy, prune only when very low free space.
    constexpr uint64_t kMinFreeBytes = 64ULL * 1024ULL * 1024ULL;
    if (info.available > kMinFreeBytes) {
        return true;
    }

    std::vector<fs::directory_entry> entries;
    for (const auto &entry : fs::directory_iterator(blobsDir_, ec)) {
        if (ec) {
            break;
        }
        if (entry.is_regular_file()) {
            entries.push_back(entry);
        }
    }

    std::sort(entries.begin(), entries.end(), [](const fs::directory_entry &a, const fs::directory_entry &b) {
        return a.last_write_time() < b.last_write_time();
    });

    for (const auto &entry : entries) {
        fs::remove(entry.path(), ec);
        if (ec) {
            continue;
        }
        const fs::space_info updated = fs::space(rootDir_, ec);
        if (!ec && updated.available > kMinFreeBytes) {
            break;
        }
    }

    return true;
}

const std::string &PersistentStore::rootDir() const {
    return rootDir_;
}
