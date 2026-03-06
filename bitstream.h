#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

class BitWriter {
public:
    void writeBits(uint32_t value, int bitCount);
    void writeBit(bool bit);
    void alignToByte();
    [[nodiscard]] std::size_t bitCount() const;
    std::vector<uint8_t> takeBytes();

private:
    std::vector<uint8_t> data_;
    uint8_t currentByte_ = 0;
    int bitsInCurrentByte_ = 0;
    std::size_t writtenBits_ = 0;
};

class BitReader {
public:
    explicit BitReader(const std::vector<uint8_t> &data);

    bool readBits(int bitCount, uint32_t &outValue);
    bool readBit(bool &outBit);

    [[nodiscard]] bool ok() const;
    [[nodiscard]] std::size_t bitsRemaining() const;

private:
    const std::vector<uint8_t> &data_;
    std::size_t bitOffset_ = 0;
    bool ok_ = true;
};
