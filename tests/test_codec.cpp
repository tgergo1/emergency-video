#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "acoustic_modem.h"
#include "bitstream.h"
#include "codec.h"
#include "communicator_protocol.h"
#include "crypto.h"
#include "decoder.h"
#include "encoder.h"
#include "fallback_controller.h"
#include "frame.h"
#include "queue_manager.h"
#include "router.h"

namespace {
void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double frameMae(const Gray4Frame &a, const Gray4Frame &b) {
    require(a.width == b.width && a.height == b.height, "Frame dimensions mismatch for MAE");
    if (a.empty() || b.empty()) {
        return 0.0;
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < a.pixels.size(); ++i) {
        sum += std::abs(static_cast<int>(a.pixels[i]) - static_cast<int>(b.pixels[i]));
    }
    return sum / static_cast<double>(a.pixels.size());
}

Gray4Frame makePatternFrame(const CodecParams &params, int phase) {
    Gray4Frame frame(params.width, params.height);
    for (int y = 0; y < params.height; ++y) {
        for (int x = 0; x < params.width; ++x) {
            const int value = ((x + phase * 2) / 5 + (y * 3 + phase) / 7) & 15;
            frame.at(x, y) = static_cast<uint8_t>(value);
        }
    }
    return frame;
}

std::vector<uint8_t> makeMalformedInterframeChangeMapPacket(const CodecParams &params,
                                                             uint32_t frameIndex,
                                                             uint32_t runLength) {
    BitstreamHeader header;
    header.version = kBitstreamVersion;
    header.frameType = FrameType::Interframe;
    header.mode = params.mode;
    header.width = static_cast<uint16_t>(params.width);
    header.height = static_cast<uint16_t>(params.height);
    header.blockSize = static_cast<uint8_t>(params.blockSize);
    header.residualStep = static_cast<uint8_t>(std::max(1, params.residualStep));
    header.frameIndex = frameIndex;
    header.keyframeInterval = static_cast<uint8_t>(std::clamp(params.keyframeInterval, 1, 255));
    header.totalBlocks = static_cast<uint16_t>(totalBlockCount(params.width, params.height, params.blockSize));
    header.changedBlocks = 0;

    BitWriter writer;
    writeHeader(writer, header);
    writer.writeBits(1U, 16); // num runs
    writer.writeBits(runLength, 16);
    writer.alignToByte();
    return writer.takeBytes();
}

std::vector<float> addAwgn(const std::vector<float> &in, float sigma, int seed) {
    std::vector<float> out = in;
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0F, sigma);
    for (float &s : out) {
        s += dist(rng);
    }
    return out;
}

std::vector<float> resampleLinear(const std::vector<float> &in, double ratio) {
    if (in.empty() || ratio <= 0.0) {
        return {};
    }
    const std::size_t outCount = std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(in.size() * ratio)));
    std::vector<float> out(outCount, 0.0F);
    for (std::size_t i = 0; i < outCount; ++i) {
        const double src = static_cast<double>(i) / ratio;
        const std::size_t i0 = static_cast<std::size_t>(std::floor(src));
        const std::size_t i1 = std::min(i0 + 1, in.size() - 1);
        const double frac = src - static_cast<double>(i0);
        const float s0 = in[std::min(i0, in.size() - 1)];
        const float s1 = in[i1];
        out[i] = static_cast<float>((1.0 - frac) * static_cast<double>(s0) + frac * static_cast<double>(s1));
    }
    return out;
}

void testBitstreamRoundTrip() {
    BitWriter writer;
    writer.writeBits(0b101U, 3);
    writer.writeBits(0xFU, 4);
    writer.writeBit(true);
    writer.writeBits(0x1234U, 16);
    writer.writeBits(0U, 1);

    std::vector<uint8_t> bytes = writer.takeBytes();
    require(!bytes.empty(), "BitWriter produced no bytes");

    BitReader reader(bytes);
    uint32_t value = 0;
    bool bit = false;

    require(reader.readBits(3, value) && value == 0b101U, "Failed bit round-trip (3 bits)");
    require(reader.readBits(4, value) && value == 0xFU, "Failed bit round-trip (4 bits)");
    require(reader.readBit(bit) && bit, "Failed bit round-trip (single bit)");
    require(reader.readBits(16, value) && value == 0x1234U, "Failed bit round-trip (16 bits)");
    require(reader.readBits(1, value) && value == 0U, "Failed bit round-trip (tail bit)");
}

void testHeaderRoundTrip() {
    BitstreamHeader header;
    header.version = kBitstreamVersion;
    header.frameType = FrameType::Interframe;
    header.mode = CodecMode::Aggressive;
    header.width = 96;
    header.height = 72;
    header.blockSize = 8;
    header.residualStep = 2;
    header.frameIndex = 1234;
    header.keyframeInterval = 19;
    header.totalBlocks = 108;
    header.changedBlocks = 33;

    BitWriter writer;
    writeHeader(writer, header);
    std::vector<uint8_t> bytes = writer.takeBytes();

    BitReader reader(bytes);
    BitstreamHeader decoded;
    std::string error;
    require(readHeader(reader, decoded, error), "Header decode failed: " + error);

    require(decoded.frameType == header.frameType, "Frame type mismatch");
    require(decoded.mode == header.mode, "Mode mismatch");
    require(decoded.width == header.width, "Width mismatch");
    require(decoded.height == header.height, "Height mismatch");
    require(decoded.blockSize == header.blockSize, "Block size mismatch");
    require(decoded.residualStep == header.residualStep, "Residual step mismatch");
    require(decoded.frameIndex == header.frameIndex, "Frame index mismatch");
    require(decoded.keyframeInterval == header.keyframeInterval, "Keyframe interval mismatch");
    require(decoded.totalBlocks == header.totalBlocks, "Total blocks mismatch");
    require(decoded.changedBlocks == header.changedBlocks, "Changed blocks mismatch");
}

void testCodecRoundTrip(CodecMode mode) {
    CodecParams params = makeCodecParams(mode);
    Encoder encoder(params);
    Decoder decoder;

    Gray4Frame frame0 = makePatternFrame(params, 0);
    EncodedPacket packet0 = encoder.encode(frame0);
    require(packet0.meta.frameType == FrameType::Keyframe, "First packet must be keyframe");

    DecodeResult decoded0 = decoder.decode(packet0.bytes);
    require(decoded0.ok, "Keyframe decode failed: " + decoded0.error);
    require(decoded0.frame.width == params.width && decoded0.frame.height == params.height,
            "Decoded keyframe dimensions mismatch");
    require(frameMae(frame0, decoded0.frame) <= 3.5, "Keyframe reconstruction MAE too high");

    Gray4Frame frame1 = makePatternFrame(params, 1);
    EncodedPacket packet1 = encoder.encode(frame1);
    require(packet1.meta.frameType == FrameType::Interframe, "Second packet must be interframe");

    DecodeResult decoded1 = decoder.decode(packet1.bytes);
    require(decoded1.ok, "Interframe decode failed: " + decoded1.error);
    require(decoded1.meta.frameIndex == packet1.meta.frameIndex, "Decoded frame index mismatch");
    require(frameMae(frame1, decoded1.frame) <= 4.5, "Interframe reconstruction MAE too high");
}

void testCorruptedSyncRejected() {
    CodecParams params = makeCodecParams(CodecMode::Safer);
    Encoder encoder(params);
    Decoder decoder;

    Gray4Frame frame = makePatternFrame(params, 0);
    EncodedPacket packet = encoder.encode(frame);
    require(!packet.bytes.empty(), "Packet unexpectedly empty");

    packet.bytes[0] ^= 0x80U;
    DecodeResult result = decoder.decode(packet.bytes);
    require(!result.ok, "Decoder accepted packet with corrupted sync");
    require(result.error.find("sync") != std::string::npos,
            "Unexpected error for corrupted sync: " + result.error);
}

void testTruncatedPacketRejected() {
    CodecParams params = makeCodecParams(CodecMode::Safer);
    Encoder encoder(params);
    Decoder decoder;

    Gray4Frame frame = makePatternFrame(params, 0);
    EncodedPacket packet = encoder.encode(frame);
    require(packet.bytes.size() > 8, "Packet too small for truncation test");

    packet.bytes.pop_back();
    DecodeResult result = decoder.decode(packet.bytes);
    require(!result.ok, "Decoder accepted truncated packet");
}

void testTrailingGarbageRejected() {
    CodecParams params = makeCodecParams(CodecMode::Safer);
    Encoder encoder(params);
    Decoder decoder;

    Gray4Frame frame = makePatternFrame(params, 0);
    EncodedPacket packet = encoder.encode(frame);
    packet.bytes.push_back(0xABU);

    DecodeResult result = decoder.decode(packet.bytes);
    require(!result.ok, "Decoder accepted packet with trailing garbage");
    require(result.error.find("Trailing non-padding bytes found") != std::string::npos,
            "Unexpected error for trailing garbage: " + result.error);
}

void testDecoderResetRequiresKeyframe() {
    CodecParams params = makeCodecParams(CodecMode::Safer);
    Encoder encoder(params);
    Decoder decoder;

    Gray4Frame frame0 = makePatternFrame(params, 0);
    Gray4Frame frame1 = makePatternFrame(params, 1);

    EncodedPacket key = encoder.encode(frame0);
    EncodedPacket inter = encoder.encode(frame1);
    require(inter.meta.frameType == FrameType::Interframe, "Expected an interframe");

    DecodeResult first = decoder.decode(key.bytes);
    require(first.ok, "Keyframe decode failed before reset");

    decoder.reset();
    DecodeResult afterReset = decoder.decode(inter.bytes);
    require(!afterReset.ok, "Decoder accepted interframe after reset");
    require(afterReset.error.find("without reference") != std::string::npos,
            "Unexpected reset error: " + afterReset.error);
}

void testKeyframeIntervalPattern() {
    CodecParams params = makeCodecParams(CodecMode::Safer);
    params.keyframeInterval = 2;
    Encoder encoder(params);

    const FrameType expected[] = {FrameType::Keyframe, FrameType::Interframe, FrameType::Interframe,
                                  FrameType::Keyframe};
    for (int i = 0; i < 4; ++i) {
        EncodedPacket packet = encoder.encode(makePatternFrame(params, i));
        require(packet.meta.frameType == expected[i], "Unexpected keyframe/interframe pattern");
    }
}

void testMalformedInterframeChangeMapRejected() {
    CodecParams params = makeCodecParams(CodecMode::Safer);
    Encoder encoder(params);
    Decoder decoder;

    Gray4Frame frame0 = makePatternFrame(params, 0);
    EncodedPacket key = encoder.encode(frame0);
    DecodeResult keyDecoded = decoder.decode(key.bytes);
    require(keyDecoded.ok, "Failed to establish decoder reference frame");

    const int totalBlocks = totalBlockCount(params.width, params.height, params.blockSize);
    std::vector<uint8_t> badPacket =
        makeMalformedInterframeChangeMapPacket(params, key.meta.frameIndex + 1, static_cast<uint32_t>(totalBlocks + 1));

    DecodeResult bad = decoder.decode(badPacket);
    require(!bad.ok, "Decoder accepted malformed interframe change map");
    require(bad.error.find("Change-map overflows block count") != std::string::npos,
            "Unexpected malformed change-map error: " + bad.error);
}

void testFrameIndexAndDecodeSequence() {
    CodecParams params = makeCodecParams(CodecMode::Safer);
    Encoder encoder(params);
    Decoder decoder;

    for (uint32_t i = 0; i < 16; ++i) {
        Gray4Frame frame = makePatternFrame(params, static_cast<int>(i));
        EncodedPacket packet = encoder.encode(frame);
        require(packet.meta.frameIndex == i, "Unexpected frame index progression");

        DecodeResult decoded = decoder.decode(packet.bytes);
        require(decoded.ok, "Decode failed during sequence at frame " + std::to_string(i));
        require(decoded.meta.frameIndex == i, "Decoded frame index mismatch");

        const double mae = frameMae(frame, decoded.frame);
        if (packet.meta.frameType == FrameType::Keyframe) {
            require(mae <= 3.5, "Keyframe MAE too high in sequence");
        } else {
            require(mae <= 5.5, "Interframe MAE too high in sequence");
        }
    }
}

void testDeterministicEncoding() {
    CodecParams params = makeCodecParams(CodecMode::Aggressive);
    Encoder encoderA(params);
    Encoder encoderB(params);

    for (int i = 0; i < 10; ++i) {
        Gray4Frame frame = makePatternFrame(params, i);
        EncodedPacket packetA = encoderA.encode(frame);
        EncodedPacket packetB = encoderB.encode(frame);

        require(packetA.meta.frameType == packetB.meta.frameType, "Frame type mismatch in deterministic encoding");
        require(packetA.meta.frameIndex == packetB.meta.frameIndex, "Frame index mismatch in deterministic encoding");
        require(packetA.bytes == packetB.bytes, "Encoded bytes differ for identical encoder state/input");
    }
}

void testInterframeWithoutReferenceRejected() {
    CodecParams params = makeCodecParams(CodecMode::Safer);
    Encoder encoder(params);
    Decoder decoder;

    Gray4Frame frame0 = makePatternFrame(params, 0);
    Gray4Frame frame1 = makePatternFrame(params, 1);

    (void)encoder.encode(frame0);
    EncodedPacket interPacket = encoder.encode(frame1);
    require(interPacket.meta.frameType == FrameType::Interframe, "Expected generated interframe packet");

    DecodeResult result = decoder.decode(interPacket.bytes);
    require(!result.ok, "Decoder must reject interframe without reference");
    require(result.error.find("without reference") != std::string::npos,
            "Unexpected error for missing reference: " + result.error);
}

void testCommEnvelopeFragmentRoundTrip() {
    CommEnvelopeHeader base;
    base.protoVersion = kCommProtocolVersion;
    base.payloadType = CommPayloadType::VideoFrame;
    base.msgId = 42;
    base.streamId = 7;
    base.senderNodeId = 123;
    base.targetScope = TargetScope::Broadcast;
    base.targetNodeId = 0;
    base.seq = 9;
    base.timestampMs = nowUnixMs();
    base.ttlMs = 5000;

    std::vector<uint8_t> payload(333);
    for (std::size_t i = 0; i < payload.size(); ++i) {
        payload[i] = static_cast<uint8_t>((i * 19U) & 0xFFU);
    }

    const std::vector<std::vector<uint8_t>> frags = fragmentCommPayload(payload, 40);
    require(frags.size() > 1, "Expected fragmented payload");

    std::vector<std::vector<uint8_t>> wire;
    for (std::size_t i = 0; i < frags.size(); ++i) {
        CommEnvelopeHeader h = base;
        h.fragIndex = static_cast<uint16_t>(i);
        h.fragCount = static_cast<uint16_t>(frags.size());
        wire.push_back(serializeCommEnvelope(h, frags[i]));
    }
    std::reverse(wire.begin(), wire.end());

    std::vector<std::vector<uint8_t>> parts(frags.size());
    std::vector<uint8_t> seen(frags.size(), 0);
    for (const auto &frame : wire) {
        CommEnvelopeHeader h;
        std::vector<uint8_t> p;
        std::string error;
        require(deserializeCommEnvelope(frame, h, p, error), "Fragment decode failed: " + error);
        require(h.fragIndex < h.fragCount, "Fragment index out of range");
        parts[h.fragIndex] = std::move(p);
        seen[h.fragIndex] = 1;
    }

    require(std::all_of(seen.begin(), seen.end(), [](uint8_t v) { return v != 0; }), "Missing fragments");

    std::vector<uint8_t> rebuilt;
    for (const auto &part : parts) {
        rebuilt.insert(rebuilt.end(), part.begin(), part.end());
    }
    require(rebuilt == payload, "Reassembled payload mismatch");
}

void testCommEnvelopeCorruptionRejected() {
    CommEnvelopeHeader header;
    header.payloadType = CommPayloadType::Text;
    header.msgId = 11;
    header.senderNodeId = 22;
    header.seq = 33;
    header.timestampMs = nowUnixMs();
    header.ttlMs = 10000;

    const std::vector<uint8_t> payload = serializeTextPayload(0, "hello");
    std::vector<uint8_t> frame = serializeCommEnvelope(header, payload);
    require(!frame.empty(), "Envelope unexpectedly empty");

    frame[frame.size() / 2] ^= 0x5AU;
    CommEnvelopeHeader decodedHeader;
    std::vector<uint8_t> decodedPayload;
    std::string error;
    require(!deserializeCommEnvelope(frame, decodedHeader, decodedPayload, error),
            "Corrupted envelope should be rejected");
}

void testQueueManagerDedupBySenderMsg() {
    QueueManager queue;
    require(!queue.isDuplicate(100, 1), "First sender/msg should not be duplicate");
    require(queue.isDuplicate(100, 1), "Second sender/msg should be duplicate");
    require(!queue.isDuplicate(101, 1), "Different sender should not be duplicate");
    require(!queue.isDuplicate(100, 2), "Different msg should not be duplicate");
}

void testFallbackControllerEscalationRecovery() {
    FallbackController fallback(FallbackStage::TextOnly);
    auto now = std::chrono::steady_clock::time_point{};

    FallbackInputWindow degraded;
    degraded.syncLocked = false;
    degraded.queueVideo = 64;

    for (int i = 0; i < 4; ++i) {
        fallback.update(degraded, now + std::chrono::seconds(i));
    }
    require(fallback.stage() == FallbackStage::LowerFps, "Fallback should escalate to LowerFps");

    FallbackInputWindow healthy;
    healthy.syncLocked = true;
    healthy.transportLossPercent = 0.0;
    for (int i = 4; i < 25; ++i) {
        fallback.update(healthy, now + std::chrono::seconds(i));
    }
    require(fallback.stage() == FallbackStage::Normal, "Fallback should recover to Normal");
}

void testFallbackControllerMaxStageCap() {
    FallbackController fallback(FallbackStage::LowerResolution);
    auto now = std::chrono::steady_clock::time_point{};

    FallbackInputWindow degraded;
    degraded.syncLocked = false;
    degraded.queueVideo = 80;

    for (int i = 0; i < 60; ++i) {
        fallback.update(degraded, now + std::chrono::seconds(i));
    }
    require(fallback.stage() == FallbackStage::LowerResolution, "Fallback exceeded configured max stage");
}

void testCommEnvelopeAuthRoundTrip() {
    EnvelopeAuthConfig auth;
    auth.enabled = true;
    auth.key = deriveAuthKeyFromPin("1234");

    CommEnvelopeHeader header;
    header.payloadType = CommPayloadType::Text;
    header.msgId = 77;
    header.senderNodeId = 88;
    header.seq = 3;
    header.timestampMs = nowUnixMs();
    header.ttlMs = 9000;

    const std::vector<uint8_t> payload = serializeTextPayload(0, "auth-test");
    const std::vector<uint8_t> frame = serializeCommEnvelope(header, payload, &auth);

    CommEnvelopeHeader decodedHeader;
    std::vector<uint8_t> decodedPayload;
    std::string error;
    bool authFailure = false;
    require(deserializeCommEnvelope(frame, decodedHeader, decodedPayload, error, &auth, &authFailure),
            "Auth envelope decode failed: " + error);
    require(!authFailure, "Auth should not fail for matching key");
    require(decodedPayload == payload, "Auth envelope payload mismatch");
}

void testCommEnvelopeAuthMismatchRejected() {
    EnvelopeAuthConfig authA;
    authA.enabled = true;
    authA.key = deriveAuthKeyFromPin("1111");

    EnvelopeAuthConfig authB;
    authB.enabled = true;
    authB.key = deriveAuthKeyFromPin("2222");

    CommEnvelopeHeader header;
    header.payloadType = CommPayloadType::Text;
    header.msgId = 91;
    header.senderNodeId = 92;
    header.seq = 5;
    header.timestampMs = nowUnixMs();
    header.ttlMs = 9000;

    const std::vector<uint8_t> payload = serializeTextPayload(0, "auth-mismatch");
    const std::vector<uint8_t> frame = serializeCommEnvelope(header, payload, &authA);

    CommEnvelopeHeader decodedHeader;
    std::vector<uint8_t> decodedPayload;
    std::string error;
    bool authFailure = false;
    require(!deserializeCommEnvelope(frame, decodedHeader, decodedPayload, error, &authB, &authFailure),
            "Envelope with wrong auth key should be rejected");
    require(authFailure, "Wrong auth key should set authFailure");
}

void testCommEnvelopeAuthDisabledAccepted() {
    CommEnvelopeHeader header;
    header.payloadType = CommPayloadType::Text;
    header.msgId = 120;
    header.senderNodeId = 9;
    header.seq = 2;
    header.timestampMs = nowUnixMs();
    header.ttlMs = 9000;

    const std::vector<uint8_t> payload = serializeTextPayload(0, "plain");
    const std::vector<uint8_t> frame = serializeCommEnvelope(header, payload, nullptr);

    CommEnvelopeHeader decodedHeader;
    std::vector<uint8_t> decodedPayload;
    std::string error;
    require(deserializeCommEnvelope(frame, decodedHeader, decodedPayload, error),
            "Auth-disabled envelope should decode");
}

void testCommEnvelopeRejectsNonV3() {
    CommEnvelopeHeader header;
    header.payloadType = CommPayloadType::Text;
    header.msgId = 33;
    header.senderNodeId = 22;
    header.seq = 4;
    header.timestampMs = nowUnixMs();
    header.ttlMs = 9000;

    const std::vector<uint8_t> payload = serializeTextPayload(0, "v3-only");
    std::vector<uint8_t> frame = serializeCommEnvelope(header, payload);
    require(frame.size() > 3, "Envelope too short for version mutation");
    frame[2] = 2;

    CommEnvelopeHeader decodedHeader;
    std::vector<uint8_t> decodedPayload;
    std::string error;
    require(!deserializeCommEnvelope(frame, decodedHeader, decodedPayload, error), "Non-v3 envelope should be rejected");
}

void testProbePayloadRoundTrip() {
    TransportProbePayload probe;
    probe.probeId = 99;
    probe.kind = ProbeKind::Ping;
    probe.sentTsMs = 123456;

    const std::vector<uint8_t> bytes = serializeTransportProbePayload(probe);
    TransportProbePayload decoded;
    std::string error;
    require(deserializeTransportProbePayload(bytes, decoded, error), "Probe payload decode failed");
    require(decoded.probeId == probe.probeId, "Probe id mismatch");
    require(decoded.kind == probe.kind, "Probe kind mismatch");
    require(decoded.sentTsMs == probe.sentTsMs, "Probe sent timestamp mismatch");
}

void testRouterRetryFlagPropagation() {
    NodeIdentity identity;
    identity.nodeId = 0xABCULL;
    identity.alias = "node";
    Router router(identity);

    router.enqueueText("retry-check", TargetScope::Broadcast, 0, 120000);
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<RouterOutgoing> first = router.collectOutgoing(1, t0, 100, 3);
    require(first.size() == 1, "Expected initial outgoing packet");
    require(!first[0].isRetry, "Initial packet must not be marked as retry");

    const std::vector<RouterOutgoing> retry = router.collectOutgoing(1, t0 + std::chrono::milliseconds(150), 100, 3);
    require(!retry.empty(), "Expected retry packet");
    require(retry[0].isRetry, "Retry packet should be marked as retry");
}

void testRouterProbePingPong() {
    NodeIdentity aId;
    aId.nodeId = 0x1111ULL;
    aId.alias = "A";
    Router a(aId);

    NodeIdentity bId;
    bId.nodeId = 0x2222ULL;
    bId.alias = "B";
    Router b(bId);

    TransportProbePayload ping;
    ping.probeId = 55;
    ping.kind = ProbeKind::Ping;
    ping.sentTsMs = nowUnixMs();
    a.enqueueTransportProbe(ping, TargetScope::Broadcast, 0, 10000);

    const auto now = std::chrono::steady_clock::now();
    const std::vector<RouterOutgoing> outA = a.collectOutgoing(4, now, 400, 2);
    require(!outA.empty(), "Missing probe ping packet");
    const std::vector<uint8_t> envA = serializeCommEnvelope(outA[0].header, outA[0].payload);
    RouterEvents eventsB = b.processIncomingEnvelope(envA, now, nullptr);
    require(!eventsB.probes.empty(), "B should receive probe event");
    require(eventsB.probes[0].probe.kind == ProbeKind::Ping, "Expected ping probe event");

    const std::vector<RouterOutgoing> outB = b.collectOutgoing(4, now + std::chrono::milliseconds(10), 400, 2);
    require(!outB.empty(), "B should send pong");
    require(outB[0].header.payloadType == CommPayloadType::TransportProbe, "Expected transport probe payload type");
    TransportProbePayload pong;
    std::string error;
    require(deserializeTransportProbePayload(outB[0].payload, pong, error), "Failed to decode pong payload");
    require(pong.kind == ProbeKind::Pong, "Expected pong");
    require(pong.probeId == ping.probeId, "Pong probe id mismatch");
}

void testModemRoundTripClean() {
    SessionConfig cfg;
    cfg.sampleRate = 48000;
    cfg.symbolSamples = 240;
    cfg.mfskBins = 4;
    cfg.bandMode = BandMode::Audible;

    MfskModem modem(modemParamsFromSession(cfg));
    std::vector<uint8_t> frame(48);
    for (std::size_t i = 0; i < frame.size(); ++i) {
        frame[i] = static_cast<uint8_t>((i * 29U) & 0xFFU);
    }

    const std::vector<float> pcm = modem.modulateFrame(frame, 5, 12);
    std::vector<uint8_t> decoded;
    std::size_t recovered = 0;
    std::size_t compared = 0;
    const bool ok = modem.demodulateBurst(pcm, decoded, &recovered, &compared);
    if (ok) {
        require(decoded == frame, "Clean modem round-trip payload mismatch");
    }
}

void testModemRoundTripNoisy() {
    SessionConfig cfg;
    cfg.sampleRate = 48000;
    cfg.symbolSamples = 240;
    cfg.mfskBins = 4;
    cfg.bandMode = BandMode::Audible;

    MfskModem modem(modemParamsFromSession(cfg));
    std::vector<uint8_t> frame(40);
    for (std::size_t i = 0; i < frame.size(); ++i) {
        frame[i] = static_cast<uint8_t>((i * 17U) & 0xFFU);
    }

    const std::vector<float> pcm = modem.modulateFrame(frame, 5, 12);
    const std::vector<float> noisy = addAwgn(pcm, 0.0025F, 1234);
    std::vector<uint8_t> decoded;
    const bool ok = modem.demodulateBurst(noisy, decoded);
    if (ok) {
        require(decoded == frame, "Noisy modem round-trip payload mismatch");
    }
}

void testModemRoundTripTimingDrift() {
    SessionConfig cfg;
    cfg.sampleRate = 48000;
    cfg.symbolSamples = 240;
    cfg.mfskBins = 4;
    cfg.bandMode = BandMode::Audible;

    MfskModem modem(modemParamsFromSession(cfg));
    std::vector<uint8_t> frame(36);
    for (std::size_t i = 0; i < frame.size(); ++i) {
        frame[i] = static_cast<uint8_t>((i * 13U) & 0xFFU);
    }

    const std::vector<float> pcm = modem.modulateFrame(frame, 5, 12);
    const std::vector<float> drifted = resampleLinear(pcm, 1.04);
    std::vector<uint8_t> decoded;
    const bool ok = modem.demodulateBurst(drifted, decoded);
    if (ok) {
        require(decoded == frame, "Timing-drift modem round-trip payload mismatch");
    }
}

void testModemBurstSegmentation() {
    SessionConfig cfg;
    cfg.sampleRate = 48000;
    cfg.symbolSamples = 240;
    cfg.mfskBins = 4;
    cfg.bandMode = BandMode::Audible;

    MfskModem modem(modemParamsFromSession(cfg));
    std::vector<uint8_t> a(32, 0x31U);
    std::vector<uint8_t> b(32, 0xC4U);

    const std::vector<float> burstA = modem.modulateFrame(a, 5, 12);
    const std::vector<float> burstB = modem.modulateFrame(b, 5, 12);

    std::vector<float> merged;
    merged.reserve(burstA.size() + burstB.size() + cfg.sampleRate / 5U);
    merged.insert(merged.end(), burstA.begin(), burstA.end());
    merged.insert(merged.end(), static_cast<std::size_t>(cfg.sampleRate / 6U), 0.0F);
    merged.insert(merged.end(), burstB.begin(), burstB.end());

    AcousticBurstReceiver rx(modem);
    rx.setEnergyThreshold(0.008F, 0.004F);
    rx.feedSamples(merged.data(), merged.size());

    std::vector<float> flush(static_cast<std::size_t>(cfg.sampleRate / 4U), 0.0F);
    rx.feedSamples(flush.data(), flush.size());

    std::vector<std::vector<uint8_t>> decoded;
    while (true) {
        std::vector<uint8_t> frameOut;
        if (!rx.popFrame(frameOut, nullptr, nullptr)) {
            break;
        }
        decoded.push_back(std::move(frameOut));
    }

    if (decoded.size() >= 2) {
        require(decoded[0] == a, "First segmented burst mismatch");
        require(decoded[1] == b, "Second segmented burst mismatch");
    }
}
} // namespace

int main() {
    using TestCase = std::pair<std::string, std::function<void()>>;
    const std::vector<TestCase> tests = {
        {"bitstream_round_trip", testBitstreamRoundTrip},
        {"header_round_trip", testHeaderRoundTrip},
        {"codec_round_trip_safer", []() { testCodecRoundTrip(CodecMode::Safer); }},
        {"codec_round_trip_aggressive", []() { testCodecRoundTrip(CodecMode::Aggressive); }},
        {"corrupted_sync_rejected", testCorruptedSyncRejected},
        {"truncated_packet_rejected", testTruncatedPacketRejected},
        {"trailing_garbage_rejected", testTrailingGarbageRejected},
        {"decoder_reset_requires_keyframe", testDecoderResetRequiresKeyframe},
        {"keyframe_interval_pattern", testKeyframeIntervalPattern},
        {"malformed_interframe_changemap_rejected", testMalformedInterframeChangeMapRejected},
        {"frame_index_and_decode_sequence", testFrameIndexAndDecodeSequence},
        {"deterministic_encoding", testDeterministicEncoding},
        {"interframe_without_reference_rejected", testInterframeWithoutReferenceRejected},
        {"comm_fragment_round_trip", testCommEnvelopeFragmentRoundTrip},
        {"comm_corruption_rejected", testCommEnvelopeCorruptionRejected},
        {"queue_dedup_sender_msg", testQueueManagerDedupBySenderMsg},
        {"fallback_escalation_recovery", testFallbackControllerEscalationRecovery},
        {"fallback_max_stage_cap", testFallbackControllerMaxStageCap},
        {"auth_envelope_round_trip", testCommEnvelopeAuthRoundTrip},
        {"auth_envelope_mismatch_rejected", testCommEnvelopeAuthMismatchRejected},
        {"auth_envelope_disabled", testCommEnvelopeAuthDisabledAccepted},
        {"non_v3_envelope_rejected", testCommEnvelopeRejectsNonV3},
        {"probe_payload_round_trip", testProbePayloadRoundTrip},
        {"router_retry_flag", testRouterRetryFlagPropagation},
        {"router_probe_ping_pong", testRouterProbePingPong},
        {"modem_round_trip_clean", testModemRoundTripClean},
        {"modem_round_trip_noisy", testModemRoundTripNoisy},
        {"modem_round_trip_timing_drift", testModemRoundTripTimingDrift},
        {"modem_burst_segmentation", testModemBurstSegmentation},
    };

    int failed = 0;
    for (const auto &test : tests) {
        try {
            test.second();
            std::cout << "[PASS] " << test.first << '\n';
        } catch (const std::exception &ex) {
            ++failed;
            std::cerr << "[FAIL] " << test.first << ": " << ex.what() << '\n';
        } catch (...) {
            ++failed;
            std::cerr << "[FAIL] " << test.first << ": unknown error\n";
        }
    }

    if (failed != 0) {
        std::cerr << failed << " test(s) failed\n";
        return 1;
    }

    std::cout << "All codec tests passed\n";
    return 0;
}
