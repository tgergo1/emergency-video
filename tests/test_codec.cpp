#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bitstream.h"
#include "codec.h"
#include "decoder.h"
#include "encoder.h"
#include "frame.h"

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
