#include "bitstream.h"

#include <stdexcept>

void BitWriter::writeBits(uint32_t value, int bitCount) {
    if (bitCount <= 0 || bitCount > 32) {
        throw std::invalid_argument("Bit count must be between 1 and 32");
    }

    for (int i = bitCount - 1; i >= 0; --i) {
        const uint8_t bit = static_cast<uint8_t>((value >> i) & 1U);
        currentByte_ = static_cast<uint8_t>((currentByte_ << 1) | bit);
        ++bitsInCurrentByte_;
        ++writtenBits_;

        if (bitsInCurrentByte_ == 8) {
            data_.push_back(currentByte_);
            currentByte_ = 0;
            bitsInCurrentByte_ = 0;
        }
    }
}

void BitWriter::writeBit(bool bit) {
    writeBits(bit ? 1U : 0U, 1);
}

void BitWriter::alignToByte() {
    if (bitsInCurrentByte_ == 0) {
        return;
    }

    const int padBits = 8 - bitsInCurrentByte_;
    currentByte_ = static_cast<uint8_t>(currentByte_ << padBits);
    data_.push_back(currentByte_);
    currentByte_ = 0;
    bitsInCurrentByte_ = 0;
}

std::size_t BitWriter::bitCount() const {
    return writtenBits_;
}

std::vector<uint8_t> BitWriter::takeBytes() {
    alignToByte();
    return std::move(data_);
}

BitReader::BitReader(const std::vector<uint8_t> &data) : data_(data) {}

bool BitReader::readBits(int bitCount, uint32_t &outValue) {
    if (!ok_ || bitCount < 0 || bitCount > 32 || bitsRemaining() < static_cast<std::size_t>(bitCount)) {
        ok_ = false;
        outValue = 0;
        return false;
    }

    outValue = 0;
    for (int i = 0; i < bitCount; ++i) {
        const std::size_t absoluteBit = bitOffset_ + static_cast<std::size_t>(i);
        const std::size_t byteIndex = absoluteBit / 8;
        const int shift = 7 - static_cast<int>(absoluteBit % 8);
        const uint32_t bit = (data_[byteIndex] >> shift) & 1U;
        outValue = (outValue << 1U) | bit;
    }

    bitOffset_ += static_cast<std::size_t>(bitCount);
    return true;
}

bool BitReader::readBit(bool &outBit) {
    uint32_t value = 0;
    if (!readBits(1, value)) {
        return false;
    }
    outBit = (value != 0);
    return true;
}

bool BitReader::ok() const {
    return ok_;
}

std::size_t BitReader::bitsRemaining() const {
    return data_.size() * 8U - bitOffset_;
}
