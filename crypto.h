#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using Sha256Digest = std::array<uint8_t, 32>;

Sha256Digest sha256(const uint8_t *data, std::size_t size);
Sha256Digest sha256(const std::vector<uint8_t> &data);

Sha256Digest hmacSha256(const uint8_t *key,
                        std::size_t keySize,
                        const uint8_t *data,
                        std::size_t dataSize);
Sha256Digest hmacSha256(const std::vector<uint8_t> &key, const std::vector<uint8_t> &data);

Sha256Digest deriveAuthKeyFromPin(const std::string &pin);
