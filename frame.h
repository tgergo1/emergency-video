#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

struct Gray4Frame {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;

    Gray4Frame() = default;

    Gray4Frame(int w, int h)
        : width(w), height(h), pixels(static_cast<std::size_t>(w * h), 0) {
        if (w <= 0 || h <= 0) {
            throw std::invalid_argument("Frame dimensions must be positive");
        }
    }

    [[nodiscard]] bool empty() const {
        return width <= 0 || height <= 0 || pixels.size() != static_cast<std::size_t>(width * height);
    }

    [[nodiscard]] std::size_t size() const {
        return pixels.size();
    }

    uint8_t &at(int x, int y) {
        return pixels[static_cast<std::size_t>(y * width + x)];
    }

    [[nodiscard]] uint8_t at(int x, int y) const {
        return pixels[static_cast<std::size_t>(y * width + x)];
    }
};

inline uint8_t clamp4(int value) {
    if (value < 0) {
        return 0;
    }
    if (value > 15) {
        return 15;
    }
    return static_cast<uint8_t>(value);
}
