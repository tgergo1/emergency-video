#pragma once

#include <vector>

#include "codec.h"

class Decoder {
public:
    void reset();
    DecodeResult decode(const std::vector<uint8_t> &packetBytes);

private:
    static bool decodeChangeMap(BitReader &reader,
                                int totalBlocks,
                                std::vector<uint8_t> &changed,
                                std::string &error);

    Gray4Frame previousFrame_;
    bool hasPreviousFrame_ = false;
};
