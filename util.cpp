#include "util.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <sstream>

#include <opencv2/imgproc.hpp>

namespace {
uint8_t clamp8(int value) {
    if (value < 0) {
        return 0;
    }
    if (value > 255) {
        return 255;
    }
    return static_cast<uint8_t>(value);
}

std::string truncateToWidth(const std::string &text, int maxWidth, int fontFace, double fontScale, int thickness) {
    if (maxWidth <= 0 || text.empty()) {
        return text;
    }

    int baseline = 0;
    if (cv::getTextSize(text, fontFace, fontScale, thickness, &baseline).width <= maxWidth) {
        return text;
    }

    const std::string ellipsis = "...";
    const int ellipsisWidth = cv::getTextSize(ellipsis, fontFace, fontScale, thickness, &baseline).width;
    if (ellipsisWidth >= maxWidth) {
        return ellipsis;
    }

    std::string out = text;
    while (!out.empty()) {
        out.pop_back();
        const int width = cv::getTextSize(out, fontFace, fontScale, thickness, &baseline).width;
        if (width + ellipsisWidth <= maxWidth) {
            return out + ellipsis;
        }
    }

    return ellipsis;
}

void applyHistogramStretch(cv::Mat &gray) {
    uint8_t minValue = 255;
    uint8_t maxValue = 0;

    for (int y = 0; y < gray.rows; ++y) {
        const auto *row = gray.ptr<uint8_t>(y);
        for (int x = 0; x < gray.cols; ++x) {
            minValue = std::min(minValue, row[x]);
            maxValue = std::max(maxValue, row[x]);
        }
    }

    if (maxValue <= minValue + 4) {
        return;
    }

    const int range = static_cast<int>(maxValue) - static_cast<int>(minValue);
    for (int y = 0; y < gray.rows; ++y) {
        auto *row = gray.ptr<uint8_t>(y);
        for (int x = 0; x < gray.cols; ++x) {
            const int stretched = (static_cast<int>(row[x]) - static_cast<int>(minValue)) * 255 / range;
            row[x] = clamp8(stretched);
        }
    }
}

void applyDeblockAndEdgeAware(cv::Mat &gray, int blockSize) {
    if (gray.rows < 4 || gray.cols < 4) {
        return;
    }

    const int bs = std::max(2, blockSize);
    cv::Mat work = gray.clone();

    // Deblock vertical boundaries with edge awareness.
    for (int x = bs; x < gray.cols; x += bs) {
        if (x <= 1 || x >= gray.cols - 1) {
            continue;
        }
        for (int y = 1; y < gray.rows - 1; ++y) {
            const int left = work.at<uint8_t>(y, x - 1);
            const int right = work.at<uint8_t>(y, x);
            const int step = std::abs(left - right);
            const int localEdge = std::abs(work.at<uint8_t>(y, x - 2) - work.at<uint8_t>(y, x + 1));
            if (step < 42 && localEdge < 56) {
                const int avg = (left + right) / 2;
                gray.at<uint8_t>(y, x - 1) = clamp8((left * 3 + avg) / 4);
                gray.at<uint8_t>(y, x) = clamp8((right * 3 + avg) / 4);
            }
        }
    }

    work = gray.clone();
    // Deblock horizontal boundaries with edge awareness.
    for (int y = bs; y < gray.rows; y += bs) {
        if (y <= 1 || y >= gray.rows - 1) {
            continue;
        }
        for (int x = 1; x < gray.cols - 1; ++x) {
            const int up = work.at<uint8_t>(y - 1, x);
            const int down = work.at<uint8_t>(y, x);
            const int step = std::abs(up - down);
            const int localEdge = std::abs(work.at<uint8_t>(y - 2, x) - work.at<uint8_t>(y + 1, x));
            if (step < 42 && localEdge < 56) {
                const int avg = (up + down) / 2;
                gray.at<uint8_t>(y - 1, x) = clamp8((up * 3 + avg) / 4);
                gray.at<uint8_t>(y, x) = clamp8((down * 3 + avg) / 4);
            }
        }
    }

    // Edge-aware sharpen: stronger on real edges, weak on flat/noisy areas.
    work = gray.clone();
    for (int y = 1; y < gray.rows - 1; ++y) {
        for (int x = 1; x < gray.cols - 1; ++x) {
            const int center = work.at<uint8_t>(y, x);
            const int blur = (work.at<uint8_t>(y, x) * 4 + work.at<uint8_t>(y - 1, x) +
                              work.at<uint8_t>(y + 1, x) + work.at<uint8_t>(y, x - 1) +
                              work.at<uint8_t>(y, x + 1)) /
                             8;

            const int gx = std::abs(work.at<uint8_t>(y, x + 1) - work.at<uint8_t>(y, x - 1));
            const int gy = std::abs(work.at<uint8_t>(y + 1, x) - work.at<uint8_t>(y - 1, x));
            const int edgeStrength = gx + gy;

            const int gain = (edgeStrength > 24) ? 3 : 1;
            const int sharpened = center + ((center - blur) * gain) / 4;
            gray.at<uint8_t>(y, x) = clamp8(sharpened);
        }
    }
}

void writeLittleEndianU32(std::ofstream &out, uint32_t value) {
    const uint8_t bytes[4] = {
        static_cast<uint8_t>(value & 0xFFU),
        static_cast<uint8_t>((value >> 8) & 0xFFU),
        static_cast<uint8_t>((value >> 16) & 0xFFU),
        static_cast<uint8_t>((value >> 24) & 0xFFU),
    };
    out.write(reinterpret_cast<const char *>(bytes), 4);
}

bool readLittleEndianU32(std::ifstream &in, uint32_t &value) {
    uint8_t bytes[4] = {};
    in.read(reinterpret_cast<char *>(bytes), 4);
    if (!in) {
        return false;
    }

    value = static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8U) |
            (static_cast<uint32_t>(bytes[2]) << 16U) | (static_cast<uint32_t>(bytes[3]) << 24U);
    return true;
}
} // namespace

cv::Mat resizeAndGrayscale(const cv::Mat &input, int width, int height) {
    cv::Mat gray;
    if (input.channels() == 1) {
        gray = input;
    } else {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_AREA);
    return resized;
}

Gray4Frame quantizeTo4Bit(const cv::Mat &gray8) {
    Gray4Frame frame(gray8.cols, gray8.rows);

    for (int y = 0; y < gray8.rows; ++y) {
        const auto *row = gray8.ptr<uint8_t>(y);
        for (int x = 0; x < gray8.cols; ++x) {
            const int value4 = (static_cast<int>(row[x]) + 8) / 17;
            frame.at(x, y) = clamp4(value4);
        }
    }

    return frame;
}

cv::Mat renderForDisplay(const Gray4Frame &frame, cv::Size outputSize, bool enhance, bool dither, int blockSize) {
    static constexpr int kBayer4[4][4] = {
        {0, 8, 2, 10},
        {12, 4, 14, 6},
        {3, 11, 1, 9},
        {15, 7, 13, 5},
    };

    cv::Mat small(frame.height, frame.width, CV_8UC1);
    for (int y = 0; y < frame.height; ++y) {
        auto *row = small.ptr<uint8_t>(y);
        for (int x = 0; x < frame.width; ++x) {
            int value = static_cast<int>(frame.at(x, y)) * 17;
            if (dither) {
                const int threshold = kBayer4[y & 3][x & 3] - 8;
                value += threshold * 2;
            }
            row[x] = clamp8(value);
        }
    }

    if (enhance) {
        applyHistogramStretch(small);
        applyDeblockAndEdgeAware(small, blockSize);
    }

    cv::Mat scaled;
    cv::resize(small, scaled, outputSize, 0.0, 0.0, cv::INTER_NEAREST);

    cv::Mat bgr;
    cv::cvtColor(scaled, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

void drawHud(cv::Mat &image, const std::vector<std::string> &lines, cv::Point origin, int maxWidth) {
    constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;
    constexpr double kScale = 0.55;
    constexpr int kThickness = 1;

    int y = origin.y;
    for (const std::string &line : lines) {
        const std::string clipped = truncateToWidth(line, maxWidth, kFont, kScale, kThickness);
        cv::putText(image, clipped, cv::Point(origin.x, y), kFont, kScale, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(image, clipped, cv::Point(origin.x, y), kFont, kScale, cv::Scalar(120, 255, 120), 1, cv::LINE_AA);
        y += 22;
    }
}

std::string makeRecordingFileName() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t timeNow = std::chrono::system_clock::to_time_t(now);

    std::tm localTm{};
#if defined(_WIN32)
    localtime_s(&localTm, &timeNow);
#else
    localtime_r(&timeNow, &localTm);
#endif

    std::ostringstream oss;
    oss << "encoded_stream_" << std::put_time(&localTm, "%Y%m%d_%H%M%S") << ".evs";
    return oss.str();
}

bool openRecordingFile(std::ofstream &output, const std::string &path) {
    output.open(path, std::ios::binary);
    if (!output.is_open()) {
        return false;
    }

    static constexpr char kMagic[4] = {'E', 'V', 'S', '1'};
    output.write(kMagic, 4);
    return output.good();
}

bool appendRecordingPacket(std::ofstream &output, const std::vector<uint8_t> &packet) {
    if (!output.is_open()) {
        return false;
    }

    writeLittleEndianU32(output, static_cast<uint32_t>(packet.size()));
    if (!packet.empty()) {
        output.write(reinterpret_cast<const char *>(packet.data()), static_cast<std::streamsize>(packet.size()));
    }

    return output.good();
}

bool loadRecordingFile(const std::string &path, std::vector<std::vector<uint8_t>> &packets) {
    packets.clear();

    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        return false;
    }

    char magic[4] = {};
    input.read(magic, 4);
    if (!input || std::string(magic, 4) != "EVS1") {
        return false;
    }

    while (true) {
        uint32_t packetSize = 0;
        if (!readLittleEndianU32(input, packetSize)) {
            break;
        }

        if (packetSize == 0 || packetSize > (1U << 22U)) {
            return false;
        }

        std::vector<uint8_t> bytes(packetSize);
        input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(packetSize));
        if (!input) {
            return false;
        }

        packets.push_back(std::move(bytes));
    }

    return !packets.empty();
}
