#include "crypto.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace {

constexpr std::array<uint32_t, 64> kK = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
    0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
    0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
    0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
    0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
    0xc67178f2U,
};

inline uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

inline uint32_t bsig0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

inline uint32_t bsig1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

inline uint32_t ssig0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

inline uint32_t ssig1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

void processBlock(const uint8_t *block, std::array<uint32_t, 8> &h) {
    std::array<uint32_t, 64> w{};
    for (std::size_t i = 0; i < 16; ++i) {
        const std::size_t off = i * 4;
        w[i] = (static_cast<uint32_t>(block[off]) << 24U) | (static_cast<uint32_t>(block[off + 1]) << 16U) |
               (static_cast<uint32_t>(block[off + 2]) << 8U) | static_cast<uint32_t>(block[off + 3]);
    }
    for (std::size_t i = 16; i < 64; ++i) {
        w[i] = ssig1(w[i - 2]) + w[i - 7] + ssig0(w[i - 15]) + w[i - 16];
    }

    uint32_t a = h[0];
    uint32_t b = h[1];
    uint32_t c = h[2];
    uint32_t d = h[3];
    uint32_t e = h[4];
    uint32_t f = h[5];
    uint32_t g = h[6];
    uint32_t hh = h[7];

    for (std::size_t i = 0; i < 64; ++i) {
        const uint32_t t1 = hh + bsig1(e) + ch(e, f, g) + kK[i] + w[i];
        const uint32_t t2 = bsig0(a) + maj(a, b, c);
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
}

} // namespace

Sha256Digest sha256(const uint8_t *data, std::size_t size) {
    std::array<uint32_t, 8> h = {
        0x6a09e667U,
        0xbb67ae85U,
        0x3c6ef372U,
        0xa54ff53aU,
        0x510e527fU,
        0x9b05688cU,
        0x1f83d9abU,
        0x5be0cd19U,
    };

    const uint64_t bitLen = static_cast<uint64_t>(size) * 8ULL;

    std::size_t fullBlocks = size / 64;
    for (std::size_t i = 0; i < fullBlocks; ++i) {
        processBlock(data + i * 64, h);
    }

    std::array<uint8_t, 128> tail{};
    const std::size_t rem = size % 64;
    if (rem > 0 && data != nullptr) {
        std::memcpy(tail.data(), data + fullBlocks * 64, rem);
    }
    tail[rem] = 0x80U;

    std::size_t tailLen = (rem + 1 <= 56) ? 64 : 128;
    for (int i = 0; i < 8; ++i) {
        tail[tailLen - 1 - i] = static_cast<uint8_t>((bitLen >> (8 * i)) & 0xFFU);
    }

    processBlock(tail.data(), h);
    if (tailLen == 128) {
        processBlock(tail.data() + 64, h);
    }

    Sha256Digest out{};
    for (std::size_t i = 0; i < h.size(); ++i) {
        out[i * 4 + 0] = static_cast<uint8_t>((h[i] >> 24U) & 0xFFU);
        out[i * 4 + 1] = static_cast<uint8_t>((h[i] >> 16U) & 0xFFU);
        out[i * 4 + 2] = static_cast<uint8_t>((h[i] >> 8U) & 0xFFU);
        out[i * 4 + 3] = static_cast<uint8_t>(h[i] & 0xFFU);
    }

    return out;
}

Sha256Digest sha256(const std::vector<uint8_t> &data) {
    return sha256(data.data(), data.size());
}

Sha256Digest hmacSha256(const uint8_t *key,
                        std::size_t keySize,
                        const uint8_t *data,
                        std::size_t dataSize) {
    constexpr std::size_t kBlockSize = 64;

    std::array<uint8_t, kBlockSize> keyBlock{};
    if (key == nullptr || keySize == 0) {
        keyBlock.fill(0);
    } else if (keySize > kBlockSize) {
        const Sha256Digest keyHash = sha256(key, keySize);
        std::copy(keyHash.begin(), keyHash.end(), keyBlock.begin());
    } else {
        std::memcpy(keyBlock.data(), key, keySize);
    }

    std::array<uint8_t, kBlockSize> oPad{};
    std::array<uint8_t, kBlockSize> iPad{};
    for (std::size_t i = 0; i < kBlockSize; ++i) {
        oPad[i] = static_cast<uint8_t>(keyBlock[i] ^ 0x5cU);
        iPad[i] = static_cast<uint8_t>(keyBlock[i] ^ 0x36U);
    }

    std::vector<uint8_t> inner;
    inner.reserve(kBlockSize + dataSize);
    inner.insert(inner.end(), iPad.begin(), iPad.end());
    if (data != nullptr && dataSize > 0) {
        inner.insert(inner.end(), data, data + static_cast<std::ptrdiff_t>(dataSize));
    }

    const Sha256Digest innerHash = sha256(inner);

    std::vector<uint8_t> outer;
    outer.reserve(kBlockSize + innerHash.size());
    outer.insert(outer.end(), oPad.begin(), oPad.end());
    outer.insert(outer.end(), innerHash.begin(), innerHash.end());

    return sha256(outer);
}

Sha256Digest hmacSha256(const std::vector<uint8_t> &key, const std::vector<uint8_t> &data) {
    return hmacSha256(key.data(), key.size(), data.data(), data.size());
}

Sha256Digest deriveAuthKeyFromPin(const std::string &pin) {
    const std::string seeded = "EV3|" + pin;
    return sha256(reinterpret_cast<const uint8_t *>(seeded.data()), seeded.size());
}
