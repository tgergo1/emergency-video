#pragma once

#include <cstdint>
#include <vector>

#include "codec.h"

class Encoder {
public:
    explicit Encoder(const CodecParams &params);

    void setParams(const CodecParams &params);
    void setKeyframeInterval(int interval);
    void forceNextKeyframe();
    void reset();

    [[nodiscard]] int keyframeInterval() const;
    EncodedPacket encode(const Gray4Frame &frame, const std::vector<uint8_t> *roiBlocks = nullptr);

private:
    struct CandidateBlock {
        int index = 0;
        double score = 0.0;
    };

    [[nodiscard]] bool shouldEmitKeyframe() const;
    void selectChangedBlocks(const Gray4Frame &frame,
                             std::vector<uint8_t> &changed,
                             const std::vector<uint8_t> *roiBlocks) const;
    static void encodeChangeMap(BitWriter &writer, const std::vector<uint8_t> &changed);

    void encodeKeyframePayload(const Gray4Frame &input, Gray4Frame &reconstructed, BitWriter &writer) const;
    void encodeInterPayload(const Gray4Frame &input,
                            const std::vector<uint8_t> &changed,
                            Gray4Frame &reconstructed,
                            BitWriter &writer) const;

    CodecParams params_;
    Gray4Frame previousReconstructed_;
    bool hasPreviousReconstructed_ = false;
    bool forceKeyframe_ = true;
    uint32_t frameIndex_ = 0;
    int framesSinceKeyframe_ = 0;
    int keyframeInterval_ = 12;
};
