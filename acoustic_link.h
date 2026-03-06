#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "codec.h"

enum class LinkMode : uint8_t {
    LocalLoopback = 0,
    AcousticTx = 1,
    AcousticRxLive = 2,
    AcousticRxMedia = 3,
    AcousticDuplexArq = 4,
};

enum class RxSource : uint8_t {
    LiveMic = 0,
    MediaFile = 1,
};

enum class SessionMode : uint8_t {
    Broadcast = 0,
    DuplexArq = 1,
};

enum class BandMode : uint8_t {
    Audible = 0,
    Ultrasonic = 1,
};

enum class AcousticPayloadType : uint8_t {
    Config = 0,
    Data = 1,
    Ack = 2,
};

struct SessionConfig {
    uint8_t version = 1;
    uint32_t streamId = 0;
    uint32_t sessionEpochMs = 0;
    uint16_t configVersion = 1;

    CodecMode codecMode = CodecMode::Safer;
    uint16_t width = 128;
    uint16_t height = 96;
    uint8_t blockSize = 8;
    uint8_t residualStep = 1;
    uint8_t keyframeInterval = 12;
    float targetFps = 2.5F;

    SessionMode sessionMode = SessionMode::Broadcast;
    BandMode bandMode = BandMode::Audible;
    uint8_t fecRepetition = 3;
    uint8_t interleaveDepth = 8;

    uint8_t arqWindow = 12;
    uint16_t arqTimeoutMs = 1200;
    uint8_t arqMaxRetransmit = 5;

    uint16_t sampleRate = 48000;
    uint16_t symbolSamples = 120;
    uint8_t mfskBins = 16;

    uint16_t cycleMs = 1500;
    uint16_t txSlotMs = 1100;

    uint32_t configHash = 0;
};

struct AcousticFrameHeader {
    uint8_t version = 1;
    AcousticPayloadType payloadType = AcousticPayloadType::Data;
    uint8_t flags = 0;

    uint32_t streamId = 0;
    uint32_t sessionEpochMs = 0;
    uint16_t configVersion = 0;
    uint32_t configHash = 0;

    uint32_t seq = 0;
    uint16_t fragIndex = 0;
    uint16_t fragCount = 1;
    uint16_t payloadSize = 0;

    uint32_t headerCrc32 = 0;
    uint32_t payloadCrc32 = 0;
};

struct LinkStats {
    bool syncLocked = false;
    double berEstimate = 0.0;
    uint64_t fecRecoveredCount = 0;
    uint64_t framesReceived = 0;
    uint64_t framesDropped = 0;
    uint64_t retransmitCount = 0;
    double rttMs = 0.0;
    double effectivePayloadKbps = 0.0;
};

struct AckPacket {
    uint32_t ackSeq = 0;
    std::vector<uint32_t> selectiveAcks;
    uint16_t rttHintMs = 0;
};

const char *linkModeName(LinkMode mode);
LinkMode parseLinkMode(const std::string &text, LinkMode fallback = LinkMode::LocalLoopback);

const char *rxSourceName(RxSource source);
RxSource parseRxSource(const std::string &text, RxSource fallback = RxSource::LiveMic);

const char *sessionModeName(SessionMode mode);
SessionMode parseSessionMode(const std::string &text, SessionMode fallback = SessionMode::Broadcast);

const char *bandModeName(BandMode mode);
BandMode parseBandMode(const std::string &text, BandMode fallback = BandMode::Audible);

uint32_t crc32(const uint8_t *data, std::size_t size);
uint32_t crc32(const std::vector<uint8_t> &data);

uint32_t computeSessionConfigHash(const SessionConfig &config);
std::vector<uint8_t> serializeSessionConfig(SessionConfig config);
bool deserializeSessionConfig(const std::vector<uint8_t> &bytes, SessionConfig &config, std::string &error);

std::vector<uint8_t> serializeAckPacket(const AckPacket &ack);
bool deserializeAckPacket(const std::vector<uint8_t> &bytes, AckPacket &ack, std::string &error);

std::vector<uint8_t> serializeAcousticFrame(AcousticFrameHeader header, const std::vector<uint8_t> &payload);
bool deserializeAcousticFrame(const std::vector<uint8_t> &bytes,
                              AcousticFrameHeader &header,
                              std::vector<uint8_t> &payload,
                              std::string &error);

std::vector<std::vector<uint8_t>> fragmentPayload(const std::vector<uint8_t> &payload, std::size_t maxPayloadPerFrame);

std::vector<uint8_t> fecProtect(const std::vector<uint8_t> &bytes, uint8_t repetition, uint8_t interleaveDepth);
bool fecRecover(const std::vector<uint8_t> &encoded,
                std::size_t originalSize,
                uint8_t repetition,
                uint8_t interleaveDepth,
                std::vector<uint8_t> &decoded,
                std::size_t *recoveredSymbols = nullptr);

class FragmentReassembler {
public:
    explicit FragmentReassembler(std::chrono::milliseconds timeout = std::chrono::milliseconds(4000));

    void push(const AcousticFrameHeader &header, const std::vector<uint8_t> &payload);
    bool popComplete(uint32_t &seq, std::vector<uint8_t> &payloadOut);
    void clear();

private:
    struct Entry {
        std::vector<std::vector<uint8_t>> fragments;
        std::vector<uint8_t> present;
        std::chrono::steady_clock::time_point lastSeen{};
    };

    std::chrono::milliseconds timeout_;
    std::map<uint32_t, Entry> entries_;
    std::deque<std::pair<uint32_t, std::vector<uint8_t>>> completed_;

    void cleanupExpired();
};

bool isTxSlotNow(const SessionConfig &config,
                 std::chrono::steady_clock::time_point now,
                 bool senderRole);

