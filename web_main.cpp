#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "third_party/httplib.h"

#include "codec.h"
#include "decoder.h"
#include "encoder.h"
#include "metrics.h"
#include "util.h"
#include "acoustic_link.h"
#include "acoustic_modem.h"
#include "audio_io.h"
#include "media_ffmpeg.h"
#include "communicator_protocol.h"
#include "crypto.h"
#include "fallback_controller.h"
#include "persistent_store.h"
#include "router.h"
#include "transport_manager.h"

namespace {
constexpr const char *kBindHost = "0.0.0.0";
constexpr int kPort = 8080;
constexpr int kMaxCameraProbe = 8;

const std::vector<cv::Size> kResolutionLevels = {
    {96, 72},
    {128, 96},
    {160, 120},
    {192, 144},
};

constexpr double kMinTargetFps = 0.2;

enum class LinkRole : uint8_t {
    Send = 0,
    Receive = 1,
    Duplex = 2,
};

const char *linkRoleName(LinkRole role) {
    switch (role) {
    case LinkRole::Send:
        return "send";
    case LinkRole::Receive:
        return "receive";
    case LinkRole::Duplex:
        return "duplex";
    }
    return "duplex";
}

LinkRole parseLinkRole(const std::string &text, LinkRole fallback = LinkRole::Duplex) {
    if (text == "send") {
        return LinkRole::Send;
    }
    if (text == "receive") {
        return LinkRole::Receive;
    }
    if (text == "duplex") {
        return LinkRole::Duplex;
    }
    return fallback;
}

struct MotionVector {
    int dx = 0;
    int dy = 0;
};

struct InterpolationState {
    Gray4Frame previous;
    Gray4Frame current;
    std::vector<MotionVector> motion;
    bool hasCurrent = false;
    bool hasPair = false;
    std::chrono::steady_clock::time_point blendStart{};
};

struct ControlState {
    std::mutex mutex;

    CodecMode mode = CodecMode::Safer;
    int resolutionIndex = 1;
    double targetFps = 2.5;
    bool shortKeyframeInterval = false;
    bool useReceivedEnhancement = true;
    bool useReceivedDithering = false;
    bool recordingEnabled = false;
    int requestedCameraIndex = 0;
    LinkRole linkRole = LinkRole::Duplex;
    RxSource rxSource = RxSource::LiveMic;
    SessionMode sessionMode = SessionMode::Broadcast;
    BandMode bandMode = BandMode::Audible;
    int requestedAudioInputDevice = -1;
    int requestedAudioOutputDevice = -1;
    bool linkRunning = false;
    std::string mediaPath;
    TransportKind transportKind = TransportKind::Acoustic;
    std::string serialPort;
    int serialBaud = 115200;
    std::string nodeAlias = "Field-Unit";
    std::string relayExportPath = "./relay_out/export.evrelay";
    std::string relayImportPath;
    bool authEnabled = false;
    std::string authPin;
    std::string outgoingText;
    TargetScope outgoingTextScope = TargetScope::Broadcast;
    uint64_t outgoingTextTarget = 0;
    bool sendSnapshot = false;
    bool runLinkTest = false;

    bool forceNextKeyframe = false;
    bool rescanCameras = false;
    bool rescanAudio = false;
    bool startLink = false;
    bool stopLink = false;
    bool sendText = false;
    bool exportRelay = false;
    bool importRelay = false;
};

struct SharedState {
    std::mutex mutex;

    std::vector<uint8_t> rawJpeg;
    std::vector<uint8_t> sentJpeg;
    std::vector<uint8_t> receivedJpeg;

    MetricsSnapshot metrics;
    EncodeMetadata meta{};
    std::size_t latestPacketBytes = 0;
    bool haveStats = false;

    bool faceDetectorReady = false;
    int facesNow = 0;
    std::size_t facesGathered = 0;

    CodecMode mode = CodecMode::Safer;
    int width = 128;
    int height = 96;
    double targetFps = 2.5;
    bool shortKeyframeInterval = false;
    bool useReceivedEnhancement = true;
    bool useReceivedDithering = false;
    bool recordingEnabled = false;
    int keyframeInterval = 12;

    int cameraIndex = -1;
    std::vector<int> cameras;

    std::string status = "initializing";

    LinkRole linkRole = LinkRole::Duplex;
    RxSource rxSource = RxSource::LiveMic;
    SessionMode sessionMode = SessionMode::Broadcast;
    BandMode bandMode = BandMode::Audible;
    bool linkRunning = false;
    bool linkTxSlot = false;
    int audioInputDevice = -1;
    int audioOutputDevice = -1;
    std::string mediaPath;
    std::vector<AudioDeviceInfo> audioInputs;
    std::vector<AudioDeviceInfo> audioOutputs;
    LinkStats linkStats;
    uint32_t streamId = 0;
    uint16_t configVersion = 0;
    uint32_t configHash = 0;
    TransportKind transportKind = TransportKind::Acoustic;
    std::string transportStatus = "idle";
    std::string serialPort;
    int serialBaud = 115200;
    std::string relayExportPath = "./relay_out/export.evrelay";
    std::string relayImportPath;
    std::string nodeAlias = "Field-Unit";
    FallbackStage fallbackStage = FallbackStage::Normal;
    bool authEnabled = false;
    QueueStats queueStats{};
    std::vector<TextMessage> messages;
    uint64_t latestTextCursor = 0;
};

std::atomic<bool> gStop{false};

std::string formatDouble(double value, int precision = 2) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

std::string jsonNumber(double value, int precision = 3) {
    if (!std::isfinite(value)) {
        return "0";
    }
    return formatDouble(value, precision);
}

int findNearestResolutionIndex(int width, int height) {
    int bestIndex = 0;
    long bestScore = std::numeric_limits<long>::max();

    for (int i = 0; i < static_cast<int>(kResolutionLevels.size()); ++i) {
        const cv::Size candidate = kResolutionLevels[static_cast<std::size_t>(i)];
        const long areaDiff =
            std::labs(static_cast<long>(candidate.width * candidate.height) - static_cast<long>(width * height));
        const long shapeDiff = std::labs(static_cast<long>(candidate.width - width)) +
                               std::labs(static_cast<long>(candidate.height - height));
        const long score = areaDiff * 3 + shapeDiff * 20;
        if (score < bestScore) {
            bestScore = score;
            bestIndex = i;
        }
    }

    return bestIndex;
}

CodecParams makeRuntimeParams(CodecMode mode, int resolutionIndex, double targetFps) {
    CodecParams params = makeCodecParams(mode);

    resolutionIndex = std::clamp(resolutionIndex, 0, static_cast<int>(kResolutionLevels.size()) - 1);

    params.width = kResolutionLevels[static_cast<std::size_t>(resolutionIndex)].width;
    params.height = kResolutionLevels[static_cast<std::size_t>(resolutionIndex)].height;
    params.targetFps = std::max(kMinTargetFps, targetFps);

    return params;
}

int activeKeyframeInterval(const CodecParams &params, bool shortIntervalEnabled) {
    return shortIntervalEnabled ? std::max(4, params.keyframeInterval / 2) : params.keyframeInterval;
}

void signalHandler(int /*sig*/) {
    gStop.store(true);
}

int preferredCameraIndex() {
    if (const char *env = std::getenv("EV_CAMERA_INDEX")) {
        char *end = nullptr;
        const long value = std::strtol(env, &end, 10);
        if (end != env && value >= 0 && value <= 63) {
            return static_cast<int>(value);
        }
    }
    return 1;
}

bool openCaptureByIndex(cv::VideoCapture &capture, int index) {
#if defined(__APPLE__)
    if (capture.open(index, cv::CAP_AVFOUNDATION)) {
        return true;
    }
    capture.release();
#endif
    return capture.open(index, cv::CAP_ANY);
}

bool probeCameraIndex(int index) {
    cv::VideoCapture probe;
    if (!openCaptureByIndex(probe, index)) {
        return false;
    }

    cv::Mat frame;
    probe >> frame;
    if (frame.empty()) {
        probe.release();
        return false;
    }

    probe.release();
    return true;
}

std::vector<int> probeAvailableCameraIndices(int maxIndexExclusive) {
    std::vector<int> out;
    int consecutiveMisses = 0;
    for (int index = 0; index < maxIndexExclusive; ++index) {
        if (probeCameraIndex(index)) {
            out.push_back(index);
            consecutiveMisses = 0;
        } else {
            ++consecutiveMisses;
            // If we already found devices and then hit a miss, stop probing to
            // avoid noisy out-of-range backend logs.
            if (!out.empty() && consecutiveMisses >= 1) {
                break;
            }
        }
    }
    return out;
}

bool switchCamera(cv::VideoCapture &active, int newIndex) {
    cv::VideoCapture candidate;
    if (!openCaptureByIndex(candidate, newIndex)) {
        return false;
    }

    cv::Mat test;
    candidate >> test;
    if (test.empty()) {
        candidate.release();
        return false;
    }

    active.release();
    active = std::move(candidate);
    return true;
}

uint8_t clamp4(int value) {
    if (value < 0) {
        return 0;
    }
    if (value > 15) {
        return 15;
    }
    return static_cast<uint8_t>(value);
}

int blockSadShift(const Gray4Frame &a,
                  const Gray4Frame &b,
                  const BlockGeometry &geom,
                  int dx,
                  int dy,
                  int *samplesOut = nullptr) {
    int sad = 0;
    int samples = 0;

    for (int y = 0; y < geom.h; ++y) {
        for (int x = 0; x < geom.w; ++x) {
            const int ax = geom.x + x;
            const int ay = geom.y + y;
            const int bx = std::clamp(ax + dx, 0, b.width - 1);
            const int by = std::clamp(ay + dy, 0, b.height - 1);
            sad += std::abs(static_cast<int>(a.at(ax, ay)) - static_cast<int>(b.at(bx, by)));
            ++samples;
        }
    }

    if (samplesOut != nullptr) {
        *samplesOut = samples;
    }
    return sad;
}

std::vector<MotionVector> estimateBlockMotion(const Gray4Frame &previous,
                                              const Gray4Frame &current,
                                              int blockSize,
                                              int searchRadius) {
    const int totalBlocks = totalBlockCount(previous.width, previous.height, blockSize);
    std::vector<MotionVector> motion(static_cast<std::size_t>(totalBlocks));

    for (int block = 0; block < totalBlocks; ++block) {
        const BlockGeometry geom = blockGeometry(block, previous.width, previous.height, blockSize);
        int bestDx = 0;
        int bestDy = 0;
        int sampleCount = 0;
        int bestSad = blockSadShift(previous, current, geom, 0, 0, &sampleCount);

        for (int dy = -searchRadius; dy <= searchRadius; ++dy) {
            for (int dx = -searchRadius; dx <= searchRadius; ++dx) {
                const int sad = blockSadShift(previous, current, geom, dx, dy);
                const int motionPenalty = (std::abs(dx) + std::abs(dy)) * std::max(1, sampleCount / 8);
                if (sad + motionPenalty < bestSad) {
                    bestSad = sad + motionPenalty;
                    bestDx = dx;
                    bestDy = dy;
                }
            }
        }

        motion[static_cast<std::size_t>(block)] = {bestDx, bestDy};
    }

    return motion;
}

Gray4Frame interpolateMotionCompensated(const Gray4Frame &previous,
                                        const Gray4Frame &current,
                                        const std::vector<MotionVector> &motion,
                                        int blockSize,
                                        double alpha) {
    if (previous.empty() || current.empty() || previous.width != current.width || previous.height != current.height) {
        return current;
    }

    alpha = std::clamp(alpha, 0.0, 1.0);
    Gray4Frame out(previous.width, previous.height);

    std::vector<int> accum(previous.width * previous.height, 0);
    std::vector<int> counts(previous.width * previous.height, 0);

    const int totalBlocks = totalBlockCount(previous.width, previous.height, blockSize);
    for (int block = 0; block < totalBlocks; ++block) {
        const BlockGeometry geom = blockGeometry(block, previous.width, previous.height, blockSize);
        const MotionVector mv = static_cast<std::size_t>(block) < motion.size() ? motion[static_cast<std::size_t>(block)]
                                                                                 : MotionVector{};

        const int shiftX = static_cast<int>(std::lround(alpha * static_cast<double>(mv.dx)));
        const int shiftY = static_cast<int>(std::lround(alpha * static_cast<double>(mv.dy)));

        for (int y = 0; y < geom.h; ++y) {
            for (int x = 0; x < geom.w; ++x) {
                const int sx = geom.x + x;
                const int sy = geom.y + y;
                const int tx = std::clamp(sx + shiftX, 0, previous.width - 1);
                const int ty = std::clamp(sy + shiftY, 0, previous.height - 1);
                const std::size_t index = static_cast<std::size_t>(ty * previous.width + tx);
                accum[index] += previous.at(sx, sy);
                counts[index] += 1;
            }
        }
    }

    for (int y = 0; y < previous.height; ++y) {
        for (int x = 0; x < previous.width; ++x) {
            const std::size_t index = static_cast<std::size_t>(y * previous.width + x);
            const int warped = counts[index] > 0
                                   ? (accum[index] + counts[index] / 2) / counts[index]
                                   : static_cast<int>(previous.at(x, y));
            const int blended = static_cast<int>(std::lround((1.0 - alpha) * warped +
                                                             alpha * static_cast<double>(current.at(x, y))));
            out.at(x, y) = clamp4(blended);
        }
    }

    return out;
}

std::vector<cv::Rect> scaleRects(const std::vector<cv::Rect> &sourceRects, cv::Size sourceSize, cv::Size targetSize) {
    std::vector<cv::Rect> out;
    out.reserve(sourceRects.size());

    if (sourceSize.width <= 0 || sourceSize.height <= 0 || targetSize.width <= 0 || targetSize.height <= 0) {
        return out;
    }

    const double sx = static_cast<double>(targetSize.width) / static_cast<double>(sourceSize.width);
    const double sy = static_cast<double>(targetSize.height) / static_cast<double>(sourceSize.height);

    for (const cv::Rect &rect : sourceRects) {
        cv::Rect scaled(static_cast<int>(std::lround(rect.x * sx)),
                        static_cast<int>(std::lround(rect.y * sy)),
                        std::max(1, static_cast<int>(std::lround(rect.width * sx))),
                        std::max(1, static_cast<int>(std::lround(rect.height * sy))));

        scaled &= cv::Rect(0, 0, targetSize.width, targetSize.height);
        if (scaled.width > 0 && scaled.height > 0) {
            out.push_back(scaled);
        }
    }

    return out;
}

void drawFaceRects(cv::Mat &image, const std::vector<cv::Rect> &rects, const cv::Scalar &color) {
    for (const cv::Rect &rect : rects) {
        cv::rectangle(image, rect, color, 2, cv::LINE_AA);
    }
}

std::vector<uint8_t> buildFaceRoiBlocks(const std::vector<cv::Rect> &faces,
                                        cv::Size sourceSize,
                                        int codecWidth,
                                        int codecHeight,
                                        int blockSize) {
    const int totalBlocks = totalBlockCount(codecWidth, codecHeight, blockSize);
    std::vector<uint8_t> roi(static_cast<std::size_t>(totalBlocks), 0);

    if (faces.empty() || sourceSize.width <= 0 || sourceSize.height <= 0) {
        return roi;
    }

    const double sx = static_cast<double>(codecWidth) / static_cast<double>(sourceSize.width);
    const double sy = static_cast<double>(codecHeight) / static_cast<double>(sourceSize.height);

    for (const cv::Rect &face : faces) {
        const int marginX = std::max(2, face.width / 6);
        const int marginY = std::max(2, face.height / 6);
        cv::Rect expanded(face.x - marginX,
                          face.y - marginY,
                          face.width + 2 * marginX,
                          face.height + 2 * marginY);
        expanded &= cv::Rect(0, 0, sourceSize.width, sourceSize.height);
        if (expanded.width <= 0 || expanded.height <= 0) {
            continue;
        }

        cv::Rect codecRect(static_cast<int>(std::floor(expanded.x * sx)),
                           static_cast<int>(std::floor(expanded.y * sy)),
                           std::max(1, static_cast<int>(std::ceil(expanded.width * sx))),
                           std::max(1, static_cast<int>(std::ceil(expanded.height * sy))));
        codecRect &= cv::Rect(0, 0, codecWidth, codecHeight);
        if (codecRect.width <= 0 || codecRect.height <= 0) {
            continue;
        }

        for (int block = 0; block < totalBlocks; ++block) {
            const BlockGeometry geom = blockGeometry(block, codecWidth, codecHeight, blockSize);
            const cv::Rect blockRect(geom.x, geom.y, geom.w, geom.h);
            if ((blockRect & codecRect).area() > 0) {
                roi[static_cast<std::size_t>(block)] = 1;
            }
        }
    }

    return roi;
}

std::vector<std::string> faceCascadeCandidates() {
    std::vector<std::string> candidates;

    if (const char *envPath = std::getenv("EV_FACE_CASCADE")) {
        if (*envPath != '\0') {
            candidates.emplace_back(envPath);
        }
    }

    candidates.emplace_back("haarcascade_frontalface_default.xml");
    candidates.emplace_back("data/haarcascades/haarcascade_frontalface_default.xml");
    candidates.emplace_back("build/_deps/opencv-src/data/haarcascades/haarcascade_frontalface_default.xml");
    candidates.emplace_back("/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");
    candidates.emplace_back("/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");

    try {
        const std::string samplePath = cv::samples::findFile("haarcascade_frontalface_default.xml", false, false);
        if (!samplePath.empty()) {
            candidates.push_back(samplePath);
        }
    } catch (const cv::Exception &) {
        // Optional probe only.
    }

    return candidates;
}

bool loadFaceDetector(cv::CascadeClassifier &detector, std::string &loadedPath) {
    for (const std::string &candidate : faceCascadeCandidates()) {
        if (candidate.empty()) {
            continue;
        }
        std::error_code ec;
        if (!std::filesystem::exists(candidate, ec)) {
            continue;
        }
        if (detector.load(candidate)) {
            loadedPath = candidate;
            return true;
        }
    }

    loadedPath.clear();
    return false;
}

std::vector<cv::Rect> detectFaces(const cv::Mat &bgr, cv::CascadeClassifier &detector, cv::Mat &grayOut) {
    std::vector<cv::Rect> faces;
    grayOut.release();

    if (detector.empty() || bgr.empty()) {
        return faces;
    }

    cv::cvtColor(bgr, grayOut, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(grayOut, grayOut);
    detector.detectMultiScale(grayOut, faces, 1.12, 3, 0, cv::Size(26, 26));
    return faces;
}

void gatherFacePatches(const cv::Mat &gray,
                       const std::vector<cv::Rect> &faces,
                       std::vector<cv::Mat> &gallery,
                       std::size_t maxFaces) {
    if (gray.empty() || faces.empty() || gallery.size() >= maxFaces) {
        return;
    }

    for (const cv::Rect &face : faces) {
        if (gallery.size() >= maxFaces) {
            break;
        }
        cv::Rect clipped = face & cv::Rect(0, 0, gray.cols, gray.rows);
        if (clipped.width < 20 || clipped.height < 20) {
            continue;
        }

        cv::Mat patch = gray(clipped).clone();
        cv::resize(patch, patch, cv::Size(48, 48), 0.0, 0.0, cv::INTER_AREA);
        gallery.push_back(std::move(patch));
    }
}

bool boolFromParam(const std::string &value, bool fallback = false) {
    if (value == "1" || value == "true" || value == "on" || value == "yes") {
        return true;
    }
    if (value == "0" || value == "false" || value == "off" || value == "no") {
        return false;
    }
    return fallback;
}

std::string urlDecode(const std::string &value) {
    std::string out;
    out.reserve(value.size());

    for (std::size_t i = 0; i < value.size(); ++i) {
        const char ch = value[i];
        if (ch == '+') {
            out.push_back(' ');
            continue;
        }
        if (ch == '%' && i + 2 < value.size()) {
            const auto hexValue = [](char c) -> int {
                if (c >= '0' && c <= '9') {
                    return c - '0';
                }
                if (c >= 'a' && c <= 'f') {
                    return 10 + (c - 'a');
                }
                if (c >= 'A' && c <= 'F') {
                    return 10 + (c - 'A');
                }
                return -1;
            };

            const int hi = hexValue(value[i + 1]);
            const int lo = hexValue(value[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
                continue;
            }
        }
        out.push_back(ch);
    }

    return out;
}

std::map<std::string, std::string> queryMapFromTarget(const std::string &target) {
    std::map<std::string, std::string> out;
    const std::size_t queryPos = target.find('?');
    if (queryPos == std::string::npos || queryPos + 1 >= target.size()) {
        return out;
    }

    const std::string query = target.substr(queryPos + 1);
    std::size_t start = 0;
    while (start < query.size()) {
        const std::size_t amp = query.find('&', start);
        const std::size_t end = (amp == std::string::npos) ? query.size() : amp;
        const std::string token = query.substr(start, end - start);
        if (!token.empty()) {
            const std::size_t eq = token.find('=');
            if (eq == std::string::npos) {
                out[urlDecode(token)] = "";
            } else {
                const std::string key = urlDecode(token.substr(0, eq));
                const std::string value = urlDecode(token.substr(eq + 1));
                out[key] = value;
            }
        }
        if (amp == std::string::npos) {
            break;
        }
        start = amp + 1;
    }

    return out;
}

std::string trimWhitespace(std::string value) {
    auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [&](char c) { return !isSpace(c); }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [&](char c) { return !isSpace(c); }).base(), value.end());
    return value;
}

bool parseFlatJsonObject(const std::string &json, std::map<std::string, std::string> &out) {
    out.clear();
    std::size_t i = 0;
    auto skipWs = [&]() {
        while (i < json.size() && std::isspace(static_cast<unsigned char>(json[i])) != 0) {
            ++i;
        }
    };

    auto parseString = [&](std::string &value) -> bool {
        value.clear();
        if (i >= json.size() || json[i] != '"') {
            return false;
        }
        ++i;
        while (i < json.size()) {
            const char ch = json[i++];
            if (ch == '"') {
                return true;
            }
            if (ch != '\\') {
                value.push_back(ch);
                continue;
            }
            if (i >= json.size()) {
                return false;
            }
            const char esc = json[i++];
            switch (esc) {
            case '"':
            case '\\':
            case '/':
                value.push_back(esc);
                break;
            case 'b':
                value.push_back('\b');
                break;
            case 'f':
                value.push_back('\f');
                break;
            case 'n':
                value.push_back('\n');
                break;
            case 'r':
                value.push_back('\r');
                break;
            case 't':
                value.push_back('\t');
                break;
            case 'u':
                // Keep parser lightweight: skip unicode code unit and preserve placeholder.
                if (i + 4 > json.size()) {
                    return false;
                }
                i += 4;
                value.push_back('?');
                break;
            default:
                return false;
            }
        }
        return false;
    };

    skipWs();
    if (i >= json.size() || json[i] != '{') {
        return false;
    }
    ++i;

    while (true) {
        skipWs();
        if (i < json.size() && json[i] == '}') {
            ++i;
            break;
        }

        std::string key;
        if (!parseString(key)) {
            return false;
        }
        skipWs();
        if (i >= json.size() || json[i] != ':') {
            return false;
        }
        ++i;
        skipWs();

        std::string value;
        if (i < json.size() && json[i] == '"') {
            if (!parseString(value)) {
                return false;
            }
        } else {
            const std::size_t start = i;
            while (i < json.size() && json[i] != ',' && json[i] != '}') {
                ++i;
            }
            value = trimWhitespace(json.substr(start, i - start));
        }
        out[key] = value;

        skipWs();
        if (i < json.size() && json[i] == ',') {
            ++i;
            continue;
        }
        if (i < json.size() && json[i] == '}') {
            ++i;
            break;
        }
        return false;
    }

    skipWs();
    return i == json.size();
}

std::map<std::string, std::string> requestParamMap(const httplib::Request &req) {
    std::map<std::string, std::string> out;
    const std::map<std::string, std::string> queryMap = queryMapFromTarget(req.target);
    out.insert(queryMap.begin(), queryMap.end());

    for (const auto &entry : req.params) {
        out[entry.first] = entry.second;
    }

    const auto ctypeIt = req.headers.find("Content-Type");
    const bool jsonBody = (ctypeIt != req.headers.end() &&
                           ctypeIt->second.find("application/json") != std::string::npos);
    if (jsonBody && !req.body.empty()) {
        std::map<std::string, std::string> parsed;
        if (parseFlatJsonObject(req.body, parsed)) {
            out.insert(parsed.begin(), parsed.end());
            for (const auto &entry : parsed) {
                out[entry.first] = entry.second;
            }
        }
    }

    return out;
}

std::string jsonEscape(const std::string &value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
        case '"':
            out << "\\\"";
            break;
        case '\\':
            out << "\\\\";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (c < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
            } else {
                out << static_cast<char>(c);
            }
            break;
        }
    }
    return out.str();
}

std::string camerasJsonArray(const std::vector<int> &cameras) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << cameras[i];
    }
    oss << "]";
    return oss.str();
}

std::string audioDevicesJsonArray(const std::vector<AudioDeviceInfo> &devices) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < devices.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << "{";
        oss << "\"index\":" << devices[i].index << ",";
        oss << "\"name\":\"" << jsonEscape(devices[i].name) << "\",";
        oss << "\"default\":" << (devices[i].isDefault ? "true" : "false");
        oss << "}";
    }
    oss << "]";
    return oss.str();
}

std::string textMessagesJsonArray(const std::vector<TextMessage> &messages) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << "{";
        oss << "\"msg_id\":" << messages[i].msgId << ",";
        oss << "\"sender\":" << messages[i].senderNodeId << ",";
        oss << "\"target\":" << messages[i].targetNodeId << ",";
        oss << "\"scope\":\"" << (messages[i].targetScope == TargetScope::Broadcast ? "broadcast" : "direct") << "\",";
        oss << "\"timestamp_ms\":" << messages[i].timestampMs << ",";
        oss << "\"state\":" << static_cast<int>(messages[i].state) << ",";
        oss << "\"body\":\"" << jsonEscape(messages[i].body) << "\"";
        oss << "}";
    }
    oss << "]";
    return oss.str();
}

std::string buildStateJson(const SharedState &state) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"mode\":\"" << codecModeName(state.mode) << "\",";
    oss << "\"resolution\":\"" << state.width << "x" << state.height << "\",";
    oss << "\"width\":" << state.width << ",";
    oss << "\"height\":" << state.height << ",";
    oss << "\"target_fps\":" << jsonNumber(state.targetFps, 2) << ",";
    oss << "\"short_keyframe\":" << (state.shortKeyframeInterval ? "true" : "false") << ",";
    oss << "\"enhance\":" << (state.useReceivedEnhancement ? "true" : "false") << ",";
    oss << "\"dither\":" << (state.useReceivedDithering ? "true" : "false") << ",";
    oss << "\"recording\":" << (state.recordingEnabled ? "true" : "false") << ",";
    oss << "\"camera\":" << state.cameraIndex << ",";
    oss << "\"cameras\":" << camerasJsonArray(state.cameras) << ",";
    oss << "\"link_role\":\"" << linkRoleName(state.linkRole) << "\",";
    oss << "\"transport_kind\":\"" << transportKindName(state.transportKind) << "\",";
    oss << "\"transport_status\":\"" << jsonEscape(state.transportStatus) << "\",";
    oss << "\"node_alias\":\"" << jsonEscape(state.nodeAlias) << "\",";
    oss << "\"auth_enabled\":" << (state.authEnabled ? "true" : "false") << ",";
    oss << "\"rx_source\":\"" << rxSourceName(state.rxSource) << "\",";
    oss << "\"session_mode\":\"" << sessionModeName(state.sessionMode) << "\",";
    oss << "\"band_mode\":\"" << bandModeName(state.bandMode) << "\",";
    oss << "\"link_running\":" << (state.linkRunning ? "true" : "false") << ",";
    oss << "\"link_tx_slot\":" << (state.linkTxSlot ? "true" : "false") << ",";
    oss << "\"audio_in_device\":" << state.audioInputDevice << ",";
    oss << "\"audio_out_device\":" << state.audioOutputDevice << ",";
    oss << "\"audio_inputs\":" << audioDevicesJsonArray(state.audioInputs) << ",";
    oss << "\"audio_outputs\":" << audioDevicesJsonArray(state.audioOutputs) << ",";
    oss << "\"media_path\":\"" << jsonEscape(state.mediaPath) << "\",";
    oss << "\"serial_port\":\"" << jsonEscape(state.serialPort) << "\",";
    oss << "\"serial_baud\":" << state.serialBaud << ",";
    oss << "\"relay_export_path\":\"" << jsonEscape(state.relayExportPath) << "\",";
    oss << "\"relay_import_path\":\"" << jsonEscape(state.relayImportPath) << "\",";
    oss << "\"stream_id\":" << state.streamId << ",";
    oss << "\"config_version\":" << state.configVersion << ",";
    oss << "\"config_hash\":" << state.configHash << ",";
    oss << "\"fallback_stage\":" << static_cast<int>(state.fallbackStage) << ",";
    oss << "\"queue\":{";
    oss << "\"config\":" << state.queueStats.queuedConfig << ",";
    oss << "\"ack\":" << state.queueStats.queuedAck << ",";
    oss << "\"text\":" << state.queueStats.queuedText << ",";
    oss << "\"snapshot\":" << state.queueStats.queuedSnapshot << ",";
    oss << "\"video\":" << state.queueStats.queuedVideo << ",";
    oss << "\"inflight\":" << state.queueStats.inFlightReliable << ",";
    oss << "\"dropped\":" << state.queueStats.dropped;
    oss << "},";
    oss << "\"latest_text_cursor\":" << state.latestTextCursor << ",";
    oss << "\"messages\":" << textMessagesJsonArray(state.messages) << ",";
    oss << "\"status\":\"" << jsonEscape(state.status) << "\",";
    oss << "\"have_stats\":" << (state.haveStats ? "true" : "false") << ",";
    oss << "\"packet_bytes\":" << state.latestPacketBytes << ",";
    oss << "\"keyframe_interval\":" << state.keyframeInterval << ",";
    oss << "\"faces_now\":" << state.facesNow << ",";
    oss << "\"faces_gathered\":" << state.facesGathered << ",";
    oss << "\"face_detector\":" << (state.faceDetectorReady ? "true" : "false") << ",";
    oss << "\"metrics\":{";
    oss << "\"live_kbps\":" << jsonNumber(state.metrics.liveBitrateKbps, 3) << ",";
    oss << "\"smooth_kbps\":" << jsonNumber(state.metrics.smoothedBitrateKbps, 3) << ",";
    oss << "\"avg_kbps\":" << jsonNumber(state.metrics.averageBitrateKbps, 3) << ",";
    oss << "\"fps\":" << jsonNumber(state.metrics.effectiveEncodedFps, 3) << ",";
    oss << "\"ratio_raw4\":" << jsonNumber(state.metrics.ratioVsRaw4, 3) << ",";
    oss << "\"ratio_raw8\":" << jsonNumber(state.metrics.ratioVsRaw8, 3) << ",";
    oss << "\"psnr\":" << jsonNumber(state.metrics.psnrDb, 3) << ",";
    oss << "\"keyframe_percent\":" << jsonNumber(state.metrics.keyframePercent, 2) << ",";
    oss << "\"changed_percent\":" << jsonNumber(state.metrics.changedBlocksPercent, 2);
    oss << "},";
    oss << "\"link_stats\":{";
    oss << "\"sync_locked\":" << (state.linkStats.syncLocked ? "true" : "false") << ",";
    oss << "\"ber\":" << jsonNumber(state.linkStats.berEstimate, 6) << ",";
    oss << "\"fec_recovered\":" << state.linkStats.fecRecoveredCount << ",";
    oss << "\"frames_received\":" << state.linkStats.framesReceived << ",";
    oss << "\"frames_dropped\":" << state.linkStats.framesDropped << ",";
    oss << "\"retransmit_count\":" << state.linkStats.retransmitCount << ",";
    oss << "\"transport_frames_in\":" << state.linkStats.transportFramesIn << ",";
    oss << "\"transport_frames_dropped\":" << state.linkStats.transportFramesDropped << ",";
    oss << "\"transport_loss_percent\":" << jsonNumber(state.linkStats.transportLossPercent, 2) << ",";
    oss << "\"auth_failures\":" << state.linkStats.authFailures << ",";
    oss << "\"probe_sent\":" << state.linkStats.probeSent << ",";
    oss << "\"probe_acked\":" << state.linkStats.probeAcked << ",";
    oss << "\"probe_loss_percent\":" << jsonNumber(state.linkStats.probeLossPercent, 2) << ",";
    oss << "\"rtt_ms\":" << jsonNumber(state.linkStats.rttMs, 2) << ",";
    oss << "\"effective_payload_kbps\":" << jsonNumber(state.linkStats.effectivePayloadKbps, 3);
    oss << "},";
    oss << "\"frame\":{";
    oss << "\"index\":" << state.meta.frameIndex << ",";
    oss << "\"type\":\"" << (state.meta.frameType == FrameType::Keyframe ? "K" : "P") << "\",";
    oss << "\"changed\":" << state.meta.changedBlocks << ",";
    oss << "\"total\":" << state.meta.totalBlocks;
    oss << "}";
    oss << "}";
    return oss.str();
}

std::vector<uint8_t> encodeJpeg(const cv::Mat &image, int quality = 80) {
    std::vector<uint8_t> out;
    if (image.empty()) {
        return out;
    }

    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, std::clamp(quality, 30, 95)};
    cv::imencode(".jpg", image, out, params);
    return out;
}

std::string makeIndexHtml() {
    return R"HTML(
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Emergency Communicator</title>
  <style>
    :root {
      --bg: #0f1513;
      --panel: #192521;
      --panel-soft: #1e2e29;
      --line: #30433c;
      --ink: #ecf5f1;
      --muted: #a7bbb3;
      --accent: #45d37d;
      --accent-soft: #225b3d;
      --danger: #d97575;
      --warn: #ffd58a;
    }
    * { box-sizing: border-box; min-width: 0; }
    body {
      margin: 0;
      background: radial-gradient(circle at 90% -25%, #2d4139 0%, var(--bg) 58%);
      color: var(--ink);
      font-family: "SF Pro Text", "Segoe UI", system-ui, sans-serif;
    }
    .shell {
      width: min(1880px, 100vw);
      margin: 0 auto;
      padding: 14px;
      display: grid;
      gap: 12px;
    }
    /* === TOPBAR === */
    .topbar {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 11px 16px;
      display: flex;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
    }
    .topbar-left { flex: 1; min-width: 180px; }
    .topbar-left h1 { margin: 0; font-size: 18px; line-height: 1.2; }
    .topbar-center {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 14px;
      font-weight: 600;
    }
    .topbar-right {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 13px;
      color: var(--muted);
    }
    .conn-dot {
      width: 11px;
      height: 11px;
      border-radius: 50%;
      background: var(--danger);
      flex-shrink: 0;
      transition: background 0.3s;
    }
    .conn-dot.searching { background: var(--warn); animation: blink 1.2s ease-in-out infinite; }
    .conn-dot.linked { background: var(--accent); }
    @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
    .bw-label { color: var(--accent); font-weight: 600; }
    /* === LAYOUT === */
    .layout {
      display: grid;
      gap: 12px;
      grid-template-columns: minmax(0, 2.3fr) minmax(420px, 1fr);
      align-items: start;
    }
    /* === FEEDS AREA === */
    .feeds-area {
      display: grid;
      gap: 10px;
      grid-template-columns: 1fr;
    }
    [data-role="send"] .feeds-area { grid-template-columns: minmax(0, 2fr) minmax(0, 1fr); }
    [data-role="duplex"] .feeds-area { grid-template-columns: 1fr 1fr; }
    .feed-secondary-card { display: none; }
    [data-role="send"] .feed-secondary-card,
    [data-role="duplex"] .feed-secondary-card { display: block; }
    .feed-wrap { position: relative; }
    .feed-overlay {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(10,16,12,0.82);
      color: var(--muted);
      font-size: 14px;
      font-weight: 600;
      letter-spacing: 0.03em;
      z-index: 2;
    }
    .feed-overlay.hidden { display: none !important; }
    /* === CARD === */
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
    }
    .label {
      padding: 9px 10px;
      background: var(--panel-soft);
      border-bottom: 1px solid var(--line);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .08em;
      text-transform: uppercase;
      color: #bde8cd;
    }
    .feedimg {
      width: 100%;
      aspect-ratio: 4/3;
      object-fit: cover;
      background: #0a100c;
      display: block;
    }
    /* === SIDE PANEL === */
    .side {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px;
      display: grid;
      gap: 10px;
      max-height: calc(100vh - 26px);
      overflow: hidden;
    }
    .tabs {
      display: grid;
      gap: 8px;
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }
    .tabs button, .btn {
      border-radius: 10px;
      border: 1px solid #3a5146;
      color: var(--ink);
      background: #1a2a24;
      padding: 9px 10px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      width: 100%;
      white-space: nowrap;
      text-overflow: ellipsis;
      overflow: hidden;
    }
    .tabs button.active { background: #204232; border-color: #5aa97b; color: #deffe9; }
    .btn.primary { background: var(--accent-soft); border-color: #5aa97b; color: #e7fff0; }
    .btn.danger { background: #3f2323; border-color: #8a4c4c; color: #ffdcdc; }
    .panel-scroll {
      overflow: auto;
      padding-right: 2px;
      display: grid;
      gap: 10px;
      align-content: start;
    }
    .tab { display: none; gap: 10px; align-content: start; }
    .tab.active { display: grid; }
    .section {
      background: var(--panel-soft);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
      display: grid;
      gap: 8px;
    }
    .section h3 { margin: 0; font-size: 14px; }
    .hint { color: var(--muted); font-size: 12px; line-height: 1.3; }
    .grid { display: grid; gap: 8px; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); }
    .field { display: grid; gap: 4px; }
    .field.full { grid-column: 1 / -1; }
    .field label { color: var(--muted); font-size: 12px; font-weight: 600; }
    select, input, textarea {
      border-radius: 10px;
      border: 1px solid #3a4d44;
      background: #101713;
      color: var(--ink);
      padding: 8px 9px;
      font-size: 13px;
      width: 100%;
    }
    textarea { min-height: 92px; resize: vertical; }
    .actions {
      display: grid;
      gap: 8px;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    }
    /* === ROLE PICKER === */
    .role-picker {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }
    .role-btn {
      padding: 12px 8px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #1a2a24;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 5px;
      transition: background 0.15s, border-color 0.15s, color 0.15s;
      width: 100%;
    }
    .role-btn .role-icon { font-size: 18px; line-height: 1; }
    .role-btn.active { background: var(--accent-soft); border-color: var(--accent); color: #deffe9; }
    /* === LINK TOGGLE BUTTON === */
    .link-btn {
      width: 100%;
      padding: 14px;
      border-radius: 12px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      letter-spacing: 0.02em;
    }
    .link-btn.start { background: var(--accent-soft); border: 1px solid var(--accent); color: #deffe9; }
    .link-btn.stop { background: #3f2323; border: 1px solid #8a4c4c; color: #ffdcdc; }
    /* === MESSAGES === */
    .chat {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #101712;
      max-height: 300px;
      overflow: auto;
      padding: 10px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      font-size: 12px;
    }
    .msg-out, .msg-in {
      padding: 8px 10px;
      border-radius: 10px;
      max-width: 88%;
      line-height: 1.45;
      word-break: break-word;
    }
    .msg-out { background: var(--accent-soft); border: 1px solid #3a6b4d; align-self: flex-end; }
    .msg-in { background: #1a2a24; border: 1px solid var(--line); align-self: flex-start; }
    .msg-meta {
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 3px;
      display: flex;
      align-items: center;
      gap: 5px;
    }
    .msg-state { font-size: 10px; padding: 1px 5px; border-radius: 999px; font-weight: 700; }
    .s0 { background: #253530; color: var(--muted); }
    .s1 { background: #1e3d2b; color: #7acca0; }
    .s2 { background: #1b4530; color: var(--accent); }
    .s3 { background: #3d3010; color: var(--warn); }
    .s4 { background: #3f2323; color: var(--danger); }
    /* === STATUS TAB === */
    .stat-grid { display: grid; gap: 4px; }
    .stat-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 6px 8px;
      border-radius: 8px;
      background: #12201a;
      border: 1px solid #273d30;
    }
    .stat-label { color: var(--muted); font-size: 12px; }
    .stat-val { font-size: 12px; font-weight: 600; color: var(--ink); }
    /* === RELAY INFO === */
    .relay-info-box {
      background: #1a2d22;
      border: 1px solid #3a5140;
      border-radius: 10px;
      padding: 10px;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.5;
    }
    /* === TRANSPORT / ROLE CONDITIONAL === */
    .acoustic-section { display: none; }
    [data-transport="acoustic"] .acoustic-section { display: grid; }
    .serial-section { display: none; }
    [data-transport="serial"] .serial-section { display: grid; }
    .relay-section { display: none; }
    [data-transport="file_relay"] .relay-section { display: grid; }
    [data-role="receive"] .camera-field { display: none; }
    /* === MISC === */
    .hidden { display: none !important; }
    /* === MEDIA QUERIES === */
    @media (max-width: 1480px) {
      .layout { grid-template-columns: minmax(0, 1.7fr) minmax(360px, 1fr); }
    }
    @media (max-width: 1260px) {
      .layout { grid-template-columns: 1fr; }
      .side { max-height: none; }
      .tabs { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="shell" data-role="duplex" data-transport="acoustic">
    <header class="topbar">
      <div class="topbar-left">
        <h1>Emergency Communicator</h1>
      </div>
      <div class="topbar-center">
        <span class="conn-dot" id="connDot"></span>
        <span id="connLabel">Stopped</span>
      </div>
      <div class="topbar-right">
        <span id="topbarTransport">Acoustic</span>
        <span id="topbarBw" class="bw-label"></span>
      </div>
    </header>
    <main class="layout">
      <section class="feeds-area">
        <article class="card">
          <div class="label" id="feedMainLabel">Raw Input</div>
          <div class="feed-wrap">
            <img id="feedMain" class="feedimg" alt="main feed" />
            <div class="feed-overlay hidden" id="feedOverlay">Waiting for signal...</div>
          </div>
        </article>
        <article class="card feed-secondary-card">
          <div class="label" id="feedSecondaryLabel">Received Stream</div>
          <img id="feedSecondary" class="feedimg" alt="secondary feed" />
        </article>
      </section>
      <aside class="side">
        <nav class="tabs">
          <button id="tabBtnSetup" class="active">Setup</button>
          <button id="tabBtnMessages">Messages</button>
          <button id="tabBtnStatus">Status</button>
          <button id="tabBtnAdvanced">Advanced</button>
        </nav>
        <div class="panel-scroll">

          <!-- SETUP TAB -->
          <section id="tabSetup" class="tab active">
            <div class="section">
              <h3>Role</h3>
              <div class="role-picker">
                <button class="role-btn" id="roleBtnSend">
                  <span class="role-icon">&#8593;</span>SEND
                </button>
                <button class="role-btn" id="roleBtnReceive">
                  <span class="role-icon">&#8595;</span>RECEIVE
                </button>
                <button class="role-btn active" id="roleBtnDuplex">
                  <span class="role-icon">&#8597;</span>DUPLEX
                </button>
              </div>
              <select id="roleSelect" style="display:none">
                <option value="duplex">Duplex</option>
                <option value="send">Send</option>
                <option value="receive">Receive</option>
              </select>
            </div>
            <div class="section">
              <h3>Connection</h3>
              <div class="grid">
                <div class="field"><label>Node Alias</label><input id="aliasInput" type="text" placeholder="my-node" /></div>
                <div class="field"><label>Transport</label>
                  <select id="transportSelect">
                    <option value="acoustic">Acoustic</option>
                    <option value="serial">Serial</option>
                    <option value="optical">Optical</option>
                    <option value="file_relay">File Relay</option>
                  </select>
                </div>
                <div class="field"><label>Quality</label>
                  <select id="modeSelect">
                    <option value="safer">Safer</option>
                    <option value="aggressive">Aggressive</option>
                  </select>
                </div>
                <div class="field"><label>Resolution</label>
                  <select id="resSelect">
                    <option value="96x72">96x72</option>
                    <option value="128x96">128x96</option>
                    <option value="160x120">160x120</option>
                    <option value="192x144">192x144</option>
                  </select>
                </div>
                <div class="field"><label>Target FPS</label><input id="fpsInput" type="number" min="0.2" step="0.1" /></div>
              </div>
            </div>
            <div class="section camera-field">
              <h3>Camera</h3>
              <div class="grid">
                <div class="field full"><label>Camera Source</label><select id="cameraSelect"></select></div>
              </div>
            </div>
            <div class="section relay-section">
              <h3>Relay Paths</h3>
              <div class="hint">Configure bundle paths before starting.</div>
              <div class="grid">
                <div class="field full">
                  <label>Export Bundle File</label>
                  <input id="relayExportInput" type="text" placeholder="./relay_out/export.evrelay" />
                </div>
                <div class="field full">
                  <label>Import Bundle File</label>
                  <input id="relayImportInput" type="text" placeholder="./relay_in/import.evrelay" />
                </div>
              </div>
              <div class="actions">
                <button id="exportRelayBtn" class="btn">Export</button>
                <button id="importRelayBtn" class="btn">Import</button>
              </div>
            </div>
            <button id="linkToggleBtn" class="link-btn start">Start Link</button>
          </section>

          <!-- MESSAGES TAB -->
          <section id="tabMessages" class="tab">
            <div class="section">
              <h3>Compose</h3>
              <div class="grid">
                <div class="field"><label>Target Scope</label>
                  <select id="textScopeSelect">
                    <option value="broadcast">Broadcast</option>
                    <option value="direct">Direct</option>
                  </select>
                </div>
                <div class="field" id="textTargetField" style="display:none">
                  <label>Target Node ID</label>
                  <input id="textTargetInput" type="number" min="0" step="1" value="0" />
                </div>
                <div class="field full">
                  <label>Message</label>
                  <textarea id="textBodyInput" placeholder="Type emergency message..."></textarea>
                </div>
              </div>
              <div class="actions">
                <button id="sendTextBtn" class="btn primary">Send Message</button>
                <button id="quickNeedBtn" class="btn">Need Help</button>
                <button id="quickSafeBtn" class="btn">Safe Here</button>
                <button id="quickMoveBtn" class="btn">Move North</button>
              </div>
            </div>
            <div class="section">
              <h3>Message History</h3>
              <div id="chatLog" class="chat"></div>
            </div>
          </section>

          <!-- STATUS TAB -->
          <section id="tabStatus" class="tab">
            <div class="section">
              <h3>Connection</h3>
              <div class="stat-grid">
                <div class="stat-row"><span class="stat-label">Signal</span><span class="stat-val" id="statSignal">--</span></div>
                <div class="stat-row"><span class="stat-label">Bandwidth</span><span class="stat-val" id="statBw">--</span></div>
                <div class="stat-row"><span class="stat-label">RTT</span><span class="stat-val" id="statRtt">--</span></div>
                <div class="stat-row"><span class="stat-label">BER</span><span class="stat-val" id="statBer">--</span></div>
                <div class="stat-row"><span class="stat-label">Transport Loss</span><span class="stat-val" id="statTransportLoss">--</span></div>
                <div class="stat-row"><span class="stat-label">FPS</span><span class="stat-val" id="statFps">--</span></div>
                <div class="stat-row"><span class="stat-label">Auth Failures</span><span class="stat-val" id="statAuthFailures">--</span></div>
                <div class="stat-row"><span class="stat-label">Probe</span><span class="stat-val" id="statProbe">--</span></div>
              </div>
            </div>
            <div class="section">
              <h3>Queue</h3>
              <div class="stat-grid">
                <div class="stat-row"><span class="stat-label">Text</span><span class="stat-val" id="statQueueText">--</span></div>
                <div class="stat-row"><span class="stat-label">Video</span><span class="stat-val" id="statQueueVideo">--</span></div>
                <div class="stat-row"><span class="stat-label">In-flight</span><span class="stat-val" id="statQueueInflight">--</span></div>
                <div class="stat-row"><span class="stat-label">Dropped</span><span class="stat-val" id="statDropped">--</span></div>
              </div>
            </div>
            <div class="section">
              <h3>Device</h3>
              <div class="stat-grid">
                <div class="stat-row"><span class="stat-label">Transport</span><span class="stat-val" id="statTransportStatus">--</span></div>
                <div class="stat-row"><span class="stat-label">Backend</span><span class="stat-val" id="statBackend">--</span></div>
                <div class="stat-row"><span class="stat-label">Faces</span><span class="stat-val" id="statFaces">--</span></div>
              </div>
            </div>
          </section>

          <!-- ADVANCED TAB -->
          <section id="tabAdvanced" class="tab">
            <div class="section">
              <h3>Codec &amp; Display</h3>
              <div class="actions">
                <button id="keyBtn" class="btn">Keyframe: Default</button>
                <button id="enhanceBtn" class="btn">Enhancement: Off</button>
                <button id="ditherBtn" class="btn">Dithering: Off</button>
                <button id="recordBtn" class="btn">Recording: Off</button>
                <button id="forceKfBtn" class="btn">Force Keyframe</button>
                <button id="rescanBtn" class="btn">Rescan Devices</button>
                <button id="sendSnapshotBtn" class="btn">Send Snapshot</button>
                <button id="runLinkTestBtn" class="btn">Run Link Test</button>
              </div>
            </div>
            <div class="section">
              <h3>Authentication (V3)</h3>
              <div class="grid">
                <div class="field">
                  <label>Auth Enabled</label>
                  <select id="authEnabledSelect">
                    <option value="false">Disabled</option>
                    <option value="true">Enabled</option>
                  </select>
                </div>
                <div class="field">
                  <label>PIN</label>
                  <input id="authPinInput" type="password" placeholder="PIN (runtime only)" />
                </div>
              </div>
            </div>
            <div class="section acoustic-section">
              <h3>Acoustic Link</h3>
              <div class="grid">
                <div class="field"><label>RX Source</label>
                  <select id="rxSourceSelect">
                    <option value="live_mic">Live Mic</option>
                    <option value="media_file">Media File</option>
                  </select>
                </div>
                <div class="field"><label>Session</label>
                  <select id="sessionModeSelect">
                    <option value="broadcast">Broadcast</option>
                    <option value="duplex_arq">Duplex ARQ</option>
                  </select>
                </div>
                <div class="field"><label>Band</label>
                  <select id="bandModeSelect">
                    <option value="audible">Audible</option>
                    <option value="ultrasonic">Ultrasonic</option>
                  </select>
                </div>
                <div class="field"><label>Audio Input</label><select id="audioInSelect"></select></div>
                <div class="field"><label>Audio Output</label><select id="audioOutSelect"></select></div>
                <div class="field full"><label>Media Path</label><input id="mediaPathInput" type="text" placeholder="/path/to/recording.mp4" /></div>
              </div>
            </div>
            <div class="section serial-section">
              <h3>Serial Link</h3>
              <div class="grid">
                <div class="field"><label>Serial Port</label><input id="serialPortInput" type="text" placeholder="/dev/cu.usbserial-*" /></div>
                <div class="field"><label>Serial Baud</label><input id="serialBaudInput" type="number" min="1200" step="1" value="115200" /></div>
              </div>
            </div>
            <div class="section relay-section">
              <h3>File Relay</h3>
              <div class="relay-info-box">
                Use relay bundles to transfer communications via USB drive when a direct link is unavailable.
                Export a bundle on one device, physically copy it, import on the other.
              </div>
              <div class="grid">
                <div class="field full">
                  <label>Export Bundle File</label>
                  <input id="relayExportAdvInput" type="text" placeholder="./relay_out/export.evrelay" />
                </div>
                <div class="field full">
                  <label>Import Bundle File</label>
                  <input id="relayImportAdvInput" type="text" placeholder="./relay_in/import.evrelay" />
                </div>
              </div>
              <div class="actions">
                <button id="exportRelayAdvBtn" class="btn">Export</button>
                <button id="importRelayAdvBtn" class="btn">Import</button>
              </div>
            </div>
            <div class="actions">
              <button id="applyAdvancedBtn" class="btn primary">Apply Advanced</button>
            </div>
          </section>

        </div>
      </aside>
    </main>
  </div>

  <script>
    const stateStore = {
      latest: null,
      formDirty: false,
      cursor: 0,
      mainFeedUrl: '/api/v2/frame/raw.jpg',
      secondaryFeedUrl: '/api/v2/frame/received.jpg'
    };

    const TRANSPORT_NAMES = {
      acoustic: 'Acoustic', serial: 'Serial', optical: 'Optical', file_relay: 'File Relay'
    };

    function escapeHtml(s) {
      return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    const TAB_IDS = ['Setup', 'Messages', 'Status', 'Advanced'];
    function setTab(name) {
      TAB_IDS.forEach(t => {
        document.getElementById('tabBtn' + t).classList.toggle('active', t === name);
        document.getElementById('tab' + t).classList.toggle('active', t === name);
      });
    }

    function num(v, d) {
      const n = Number(v);
      return Number.isFinite(n) ? n : (d !== undefined ? d : 0);
    }

    function text(v, d) {
      return (typeof v === 'string' && v.length > 0) ? v : (d || '');
    }

    function setRole(role) {
      document.getElementById('roleSelect').value = role;
      stateStore.formDirty = true;
      applyRoleToUI(role);
    }

    function applyRoleToUI(role) {
      document.querySelector('.shell').setAttribute('data-role', role || 'duplex');
      ['Send', 'Receive', 'Duplex'].forEach(function(r) {
        document.getElementById('roleBtn' + r).classList.toggle('active', r.toLowerCase() === role);
      });
      if (role === 'receive') {
        stateStore.mainFeedUrl = '/api/v2/frame/received.jpg';
        stateStore.secondaryFeedUrl = null;
        document.getElementById('feedMainLabel').textContent = 'Received Stream';
      } else if (role === 'send') {
        stateStore.mainFeedUrl = '/api/v2/frame/raw.jpg';
        stateStore.secondaryFeedUrl = '/api/v2/frame/sent.jpg';
        document.getElementById('feedMainLabel').textContent = 'Raw Input';
        document.getElementById('feedSecondaryLabel').textContent = 'Sent Stream';
      } else {
        stateStore.mainFeedUrl = '/api/v2/frame/raw.jpg';
        stateStore.secondaryFeedUrl = '/api/v2/frame/received.jpg';
        document.getElementById('feedMainLabel').textContent = 'Raw Input';
        document.getElementById('feedSecondaryLabel').textContent = 'Received Stream';
      }
    }

    function syncTransportVisibility(transport) {
      document.querySelector('.shell').setAttribute('data-transport', transport || 'acoustic');
      document.getElementById('topbarTransport').textContent = TRANSPORT_NAMES[transport] || transport || '--';
    }

    function updateFeedOverlay(data) {
      var overlay = document.getElementById('feedOverlay');
      var running = !!(data.link_running);
      var locked = !!(data.link_stats && data.link_stats.sync_locked);
      var role = data.link_role || 'duplex';
      if (role === 'receive' && running && !locked) {
        overlay.classList.remove('hidden');
      } else {
        overlay.classList.add('hidden');
      }
    }

    async function postJson(url, body) {
      await fetch(url, {
        method: 'POST',
        cache: 'no-store',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body || {})
      });
    }

    async function refreshState() {
      try {
        var resp = await fetch('/api/v2/state', { cache: 'no-store' });
        if (!resp.ok) return;
        var data = await resp.json();
        stateStore.latest = data;
        renderState(data);
      } catch (_) {}
    }

    function renderTopbar(data) {
      var dot = document.getElementById('connDot');
      var label = document.getElementById('connLabel');
      var bwLabel = document.getElementById('topbarBw');
      var running = !!(data.link_running);
      var locked = !!(data.link_stats && data.link_stats.sync_locked);
      dot.className = 'conn-dot';
      if (!running) {
        label.textContent = 'Stopped';
        bwLabel.textContent = '';
      } else if (locked) {
        dot.classList.add('linked');
        label.textContent = 'Linked';
        bwLabel.textContent = num(data.link_stats && data.link_stats.effective_payload_kbps).toFixed(1) + ' kbps';
      } else {
        dot.classList.add('searching');
        label.textContent = 'Searching...';
        bwLabel.textContent = '';
      }
    }

    function renderStatusTab(data) {
      var running = !!(data.link_running);
      var locked = !!(data.link_stats && data.link_stats.sync_locked);
      function sv(id, val) { var el = document.getElementById(id); if (el) el.textContent = val; }
      sv('statSignal', locked ? 'Good' : (running ? 'Searching...' : 'Not running'));
      sv('statBw', num(data.link_stats && data.link_stats.effective_payload_kbps).toFixed(1) + ' kbps');
      sv('statRtt', num(data.link_stats && data.link_stats.rtt_ms).toFixed(0) + ' ms');
      sv('statBer', num(data.link_stats && data.link_stats.ber).toFixed(5));
      sv('statTransportLoss', num(data.link_stats && data.link_stats.transport_loss_percent).toFixed(1) + '%');
      sv('statAuthFailures', num(data.link_stats && data.link_stats.auth_failures).toFixed(0));
      sv('statProbe',
        num(data.link_stats && data.link_stats.probe_acked).toFixed(0) + '/' +
        num(data.link_stats && data.link_stats.probe_sent).toFixed(0) + ' (' +
        num(data.link_stats && data.link_stats.probe_loss_percent).toFixed(1) + '%)');
      sv('statFps', num(data.metrics && data.metrics.fps).toFixed(1));
      sv('statQueueText', num(data.queue && data.queue.text).toFixed(0));
      sv('statQueueVideo', num(data.queue && data.queue.video).toFixed(0));
      sv('statQueueInflight', num(data.queue && data.queue.inflight).toFixed(0));
      sv('statDropped', num(data.queue && data.queue.dropped).toFixed(0));
      sv('statTransportStatus', (TRANSPORT_NAMES[data.transport_kind] || data.transport_kind || '--') + ' \u2014 ' + (data.transport_status || '--'));
      sv('statBackend', data.status || '--');
      sv('statFaces', 'now ' + num(data.faces_now).toFixed(0) + ' / gathered ' + num(data.faces_gathered).toFixed(0));
    }

    function renderMessages(messages) {
      var chat = document.getElementById('chatLog');
      var msgArr = Array.isArray(messages) ? messages : [];
      chat.innerHTML = '';
      var stateLabels = ['Queued', 'Sent', 'Acked', 'Relayed', 'Failed'];
      for (var i = 0; i < msgArr.length; i++) {
        var m = msgArr[i];
        var state = Number(m.state || 0);
        var isIncoming = state === 3;
        var div = document.createElement('div');
        div.className = isIncoming ? 'msg-in' : 'msg-out';
        var tsMs = Number(m.timestamp_ms);
        var timeStr = (Number.isFinite(tsMs) && tsMs > 0)
          ? new Date(tsMs).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
          : '';
        var senderStr = isIncoming
          ? (' from ' + ('0000' + Number(m.sender).toString(16)).slice(-4).toUpperCase())
          : '';
        var stateLabel = stateLabels[state] || 'Unknown';
        div.innerHTML =
          '<div class="msg-meta">' +
          escapeHtml(timeStr + senderStr) +
          ' <span class="msg-state s' + state + '">' + escapeHtml(stateLabel) + '</span>' +
          '</div>' +
          '<div>' + escapeHtml(m.body || '') + '</div>';
        chat.appendChild(div);
      }
      chat.scrollTop = chat.scrollHeight;
    }

    function syncSelectOptions(selectEl, entries, current, labelFn) {
      var list = Array.isArray(entries) ? entries : [];
      var seen = new Set(Array.from(selectEl.options).map(function(o) { return String(o.value); }));
      for (var i = 0; i < list.length; i++) {
        var entry = list[i];
        var value = String(typeof entry === 'object' ? entry.index : entry);
        if (!seen.has(value)) {
          var opt = document.createElement('option');
          opt.value = value;
          opt.textContent = labelFn(entry);
          selectEl.appendChild(opt);
        }
      }
      Array.from(selectEl.options).forEach(function(o) {
        if (!list.some(function(e) { return String(typeof e === 'object' ? e.index : e) === o.value; })) {
          selectEl.removeChild(o);
        }
      });
      if (current !== undefined && current !== null) {
        selectEl.value = String(current);
      }
    }

    function renderState(data) {
      syncSelectOptions(document.getElementById('cameraSelect'), data.cameras || [], data.camera,
        function(id) { return 'Camera ' + id; });
      syncSelectOptions(document.getElementById('audioInSelect'), data.audio_inputs || [], data.audio_in_device,
        function(d) { return d.default ? d.name + ' (Default)' : d.name; });
      syncSelectOptions(document.getElementById('audioOutSelect'), data.audio_outputs || [], data.audio_out_device,
        function(d) { return d.default ? d.name + ' (Default)' : d.name; });

      if (!stateStore.formDirty) {
        document.getElementById('aliasInput').value = data.node_alias || '';
        document.getElementById('modeSelect').value = data.mode || 'safer';
        document.getElementById('resSelect').value = data.resolution || '128x96';
        document.getElementById('fpsInput').value = num(data.target_fps, 2.5).toFixed(1);
        document.getElementById('transportSelect').value = data.transport_kind || 'acoustic';
        document.getElementById('roleSelect').value = data.link_role || 'duplex';
        document.getElementById('rxSourceSelect').value = data.rx_source || 'live_mic';
        document.getElementById('sessionModeSelect').value = data.session_mode || 'broadcast';
        document.getElementById('bandModeSelect').value = data.band_mode || 'audible';
        document.getElementById('mediaPathInput').value = data.media_path || '';
        document.getElementById('serialPortInput').value = data.serial_port || '';
        document.getElementById('serialBaudInput').value = num(data.serial_baud, 115200).toFixed(0);
        document.getElementById('relayExportInput').value = data.relay_export_path || '';
        document.getElementById('relayImportInput').value = data.relay_import_path || '';
        document.getElementById('relayExportAdvInput').value = data.relay_export_path || '';
        document.getElementById('relayImportAdvInput').value = data.relay_import_path || '';
        document.getElementById('authEnabledSelect').value = data.auth_enabled ? 'true' : 'false';
        applyRoleToUI(data.link_role || 'duplex');
      }

      syncTransportVisibility(document.getElementById('transportSelect').value);

      var lBtn = document.getElementById('linkToggleBtn');
      if (data.link_running) {
        lBtn.textContent = 'Stop Link';
        lBtn.className = 'link-btn stop';
      } else {
        lBtn.textContent = 'Start Link';
        lBtn.className = 'link-btn start';
      }

      document.getElementById('keyBtn').textContent = 'Keyframe: ' + (data.short_keyframe ? 'Short' : 'Default');
      document.getElementById('enhanceBtn').textContent = 'Enhancement: ' + (data.enhance ? 'On' : 'Off');
      document.getElementById('ditherBtn').textContent = 'Dithering: ' + (data.dither ? 'On' : 'Off');
      document.getElementById('recordBtn').textContent = 'Recording: ' + (data.recording ? 'On' : 'Off');

      renderTopbar(data);
      renderStatusTab(data);
      renderMessages(data.messages || []);
      updateFeedOverlay(data);
      stateStore.cursor = Math.max(stateStore.cursor, num(data.latest_text_cursor));
    }

    function refreshFrames() {
      var ts = Date.now();
      document.getElementById('feedMain').src = stateStore.mainFeedUrl + '?t=' + ts;
      if (stateStore.secondaryFeedUrl) {
        document.getElementById('feedSecondary').src = stateStore.secondaryFeedUrl + '?t=' + ts;
      }
    }

    function collectControl() {
      return {
        mode: document.getElementById('modeSelect').value,
        resolution: document.getElementById('resSelect').value,
        target_fps: document.getElementById('fpsInput').value,
        camera: document.getElementById('cameraSelect').value,
        link_role: document.getElementById('roleSelect').value,
        rx_source: document.getElementById('rxSourceSelect').value,
        session_mode: document.getElementById('sessionModeSelect').value,
        band_mode: document.getElementById('bandModeSelect').value,
        audio_in_device: document.getElementById('audioInSelect').value,
        audio_out_device: document.getElementById('audioOutSelect').value,
        media_path: document.getElementById('mediaPathInput').value,
        transport_kind: document.getElementById('transportSelect').value,
        serial_port: document.getElementById('serialPortInput').value,
        serial_baud: document.getElementById('serialBaudInput').value,
        node_alias: document.getElementById('aliasInput').value,
        relay_export_path: document.getElementById('relayExportInput').value || document.getElementById('relayExportAdvInput').value,
        relay_import_path: document.getElementById('relayImportInput').value || document.getElementById('relayImportAdvInput').value,
        auth_enabled: document.getElementById('authEnabledSelect').value,
        auth_pin: document.getElementById('authPinInput').value
      };
    }

    async function applyControl(extra) {
      var body = Object.assign(collectControl(), extra || {});
      await postJson('/api/v2/control', body);
      stateStore.formDirty = false;
      await refreshState();
    }

    function bind() {
      document.getElementById('tabBtnSetup').onclick = function() { setTab('Setup'); };
      document.getElementById('tabBtnMessages').onclick = function() { setTab('Messages'); };
      document.getElementById('tabBtnStatus').onclick = function() { setTab('Status'); };
      document.getElementById('tabBtnAdvanced').onclick = function() { setTab('Advanced'); };

      document.getElementById('roleBtnSend').onclick = function() { setRole('send'); };
      document.getElementById('roleBtnReceive').onclick = function() { setRole('receive'); };
      document.getElementById('roleBtnDuplex').onclick = function() { setRole('duplex'); };

      document.getElementById('transportSelect').addEventListener('change', function() {
        stateStore.formDirty = true;
        syncTransportVisibility(this.value);
      });

      document.getElementById('textScopeSelect').addEventListener('change', function() {
        document.getElementById('textTargetField').style.display = this.value === 'direct' ? 'grid' : 'none';
      });

      ['aliasInput','cameraSelect','modeSelect','resSelect','fpsInput','roleSelect','rxSourceSelect',
       'sessionModeSelect','bandModeSelect','audioInSelect','audioOutSelect','mediaPathInput',
       'serialPortInput','serialBaudInput','relayExportInput','relayImportInput',
       'relayExportAdvInput','relayImportAdvInput','authEnabledSelect','authPinInput']
        .forEach(function(id) {
          document.getElementById(id).addEventListener('input', function() { stateStore.formDirty = true; });
        });

      document.getElementById('linkToggleBtn').onclick = async function() {
        var data = stateStore.latest;
        if (data && data.link_running) {
          await postJson('/api/v2/link/stop', {});
        } else {
          await applyControl({});
          await postJson('/api/v2/link/start', {});
        }
        await refreshState();
      };

      document.getElementById('applyAdvancedBtn').onclick = async function() { await applyControl({}); };
      document.getElementById('forceKfBtn').onclick = async function() { await applyControl({ force_keyframe: true }); };
      document.getElementById('rescanBtn').onclick = async function() { await applyControl({ rescan_cameras: true, rescan_audio: true }); };
      document.getElementById('keyBtn').onclick = async function() { await applyControl({ short_keyframe: !(stateStore.latest && stateStore.latest.short_keyframe) }); };
      document.getElementById('enhanceBtn').onclick = async function() { await applyControl({ enhance: !(stateStore.latest && stateStore.latest.enhance) }); };
      document.getElementById('ditherBtn').onclick = async function() { await applyControl({ dither: !(stateStore.latest && stateStore.latest.dither) }); };
      document.getElementById('recordBtn').onclick = async function() { await applyControl({ recording: !(stateStore.latest && stateStore.latest.recording) }); };
      document.getElementById('exportRelayBtn').onclick = async function() { await applyControl({ export_relay: true }); };
      document.getElementById('importRelayBtn').onclick = async function() { await applyControl({ import_relay: true }); };
      document.getElementById('exportRelayAdvBtn').onclick = async function() { await applyControl({ export_relay: true }); };
      document.getElementById('importRelayAdvBtn').onclick = async function() { await applyControl({ import_relay: true }); };
      document.getElementById('sendSnapshotBtn').onclick = async function() { await applyControl({ send_snapshot: true }); };
      document.getElementById('runLinkTestBtn').onclick = async function() { await applyControl({ run_link_test: true }); };

      async function sendText(txt) {
        await postJson('/api/v2/messages/send', {
          body: txt,
          scope: document.getElementById('textScopeSelect').value,
          target_node_id: Number(document.getElementById('textTargetInput').value || 0)
        });
      }

      document.getElementById('sendTextBtn').onclick = async function() {
        var box = document.getElementById('textBodyInput');
        var txt = (box.value || '').trim();
        if (!txt) return;
        await sendText(txt);
        box.value = '';
      };
      document.getElementById('quickNeedBtn').onclick = async function() { await sendText('Need medical help at current location.'); };
      document.getElementById('quickSafeBtn').onclick = async function() { await sendText('We are safe and holding position.'); };
      document.getElementById('quickMoveBtn').onclick = async function() { await sendText('Moving north to extraction point.'); };
    }

    bind();
    setTab('Setup');
    applyRoleToUI('duplex');
    refreshState();
    refreshFrames();
    setInterval(refreshState, 500);
    setInterval(refreshFrames, 160);
  </script>
</body>
</html>
)HTML";
}

bool parseResolutionString(const std::string &text, int &indexOut) {
    for (int i = 0; i < static_cast<int>(kResolutionLevels.size()); ++i) {
        const cv::Size s = kResolutionLevels[static_cast<std::size_t>(i)];
        if (text == (std::to_string(s.width) + "x" + std::to_string(s.height))) {
            indexOut = i;
            return true;
        }
    }
    return false;
}

void applyControlRequest(const httplib::Request &req, ControlState &control) {
    const std::map<std::string, std::string> params = requestParamMap(req);

    auto itValue = [&](const char *name) -> std::optional<std::string> {
        const auto it = params.find(name);
        if (it == params.end()) {
            return std::nullopt;
        }
        return it->second;
    };

    std::lock_guard<std::mutex> lock(control.mutex);

    if (auto value = itValue("mode")) {
        if (*value == "safer") {
            control.mode = CodecMode::Safer;
        } else if (*value == "aggressive") {
            control.mode = CodecMode::Aggressive;
        }
    }

    if (auto value = itValue("target_fps")) {
        try {
            const double fps = std::stod(*value);
            control.targetFps = std::max(kMinTargetFps, fps);
        } catch (const std::exception &) {
        }
    }

    if (auto value = itValue("resolution")) {
        int idx = control.resolutionIndex;
        if (parseResolutionString(*value, idx)) {
            control.resolutionIndex = idx;
        }
    }

    if (auto value = itValue("resolution_index")) {
        try {
            const int idx = std::stoi(*value);
            control.resolutionIndex = std::clamp(idx, 0, static_cast<int>(kResolutionLevels.size()) - 1);
        } catch (const std::exception &) {
        }
    }

    if (auto value = itValue("short_keyframe")) {
        control.shortKeyframeInterval = boolFromParam(*value, control.shortKeyframeInterval);
    }

    if (auto value = itValue("enhance")) {
        control.useReceivedEnhancement = boolFromParam(*value, control.useReceivedEnhancement);
    }

    if (auto value = itValue("dither")) {
        control.useReceivedDithering = boolFromParam(*value, control.useReceivedDithering);
    }

    if (auto value = itValue("recording")) {
        control.recordingEnabled = boolFromParam(*value, control.recordingEnabled);
    }

    if (auto value = itValue("camera")) {
        try {
            const int index = std::stoi(*value);
            if (index >= 0 && index < 64) {
                control.requestedCameraIndex = index;
            }
        } catch (const std::exception &) {
        }
    }

    if (auto value = itValue("link_role")) {
        control.linkRole = parseLinkRole(*value, control.linkRole);
    }

    if (auto value = itValue("rx_source")) {
        control.rxSource = parseRxSource(*value, control.rxSource);
    }

    if (auto value = itValue("session_mode")) {
        control.sessionMode = parseSessionMode(*value, control.sessionMode);
    }

    if (auto value = itValue("band_mode")) {
        control.bandMode = parseBandMode(*value, control.bandMode);
    }

    if (auto value = itValue("audio_in_device")) {
        try {
            control.requestedAudioInputDevice = std::stoi(*value);
        } catch (const std::exception &) {
        }
    }

    if (auto value = itValue("audio_out_device")) {
        try {
            control.requestedAudioOutputDevice = std::stoi(*value);
        } catch (const std::exception &) {
        }
    }

    if (auto value = itValue("media_path")) {
        control.mediaPath = *value;
    }

    if (auto value = itValue("transport_kind")) {
        control.transportKind = parseTransportKind(*value, control.transportKind);
    }

    if (auto value = itValue("serial_port")) {
        control.serialPort = *value;
    }

    if (auto value = itValue("serial_baud")) {
        try {
            control.serialBaud = std::clamp(std::stoi(*value), 1200, 1000000);
        } catch (const std::exception &) {
        }
    }

    if (auto value = itValue("node_alias")) {
        control.nodeAlias = *value;
    }

    if (auto value = itValue("relay_export_path")) {
        control.relayExportPath = *value;
    }

    if (auto value = itValue("relay_import_path")) {
        control.relayImportPath = *value;
    }

    if (auto value = itValue("auth_enabled")) {
        control.authEnabled = boolFromParam(*value, control.authEnabled);
    }
    if (auto value = itValue("auth_pin")) {
        control.authPin = *value;
    }

    if (auto value = itValue("text_body")) {
        control.outgoingText = *value;
    }
    if (auto value = itValue("text_scope")) {
        control.outgoingTextScope = (*value == "direct") ? TargetScope::Direct : TargetScope::Broadcast;
    }
    if (auto value = itValue("text_target")) {
        try {
            control.outgoingTextTarget = static_cast<uint64_t>(std::stoull(*value));
        } catch (const std::exception &) {
        }
    }
    if (auto value = itValue("send_text")) {
        if (boolFromParam(*value, false)) {
            control.sendText = true;
        }
    }

    if (auto value = itValue("send_snapshot")) {
        if (boolFromParam(*value, false)) {
            control.sendSnapshot = true;
        }
    }

    if (auto value = itValue("run_link_test")) {
        if (boolFromParam(*value, false)) {
            control.runLinkTest = true;
        }
    }

    if (auto value = itValue("force_keyframe")) {
        if (boolFromParam(*value, false)) {
            control.forceNextKeyframe = true;
        }
    }

    if (auto value = itValue("rescan_cameras")) {
        if (boolFromParam(*value, false)) {
            control.rescanCameras = true;
        }
    }

    if (auto value = itValue("rescan_audio")) {
        if (boolFromParam(*value, false)) {
            control.rescanAudio = true;
        }
    }

    if (auto value = itValue("export_relay")) {
        if (boolFromParam(*value, false)) {
            control.exportRelay = true;
        }
    }

    if (auto value = itValue("import_relay")) {
        if (boolFromParam(*value, false)) {
            control.importRelay = true;
        }
    }

    if (auto value = itValue("start_link")) {
        if (boolFromParam(*value, false)) {
            control.startLink = true;
            control.stopLink = false;
            control.linkRunning = true;
        }
    }

    if (auto value = itValue("stop_link")) {
        if (boolFromParam(*value, false)) {
            control.stopLink = true;
            control.startLink = false;
            control.linkRunning = false;
        }
    }
}

struct ControlSnapshot {
    CodecMode mode = CodecMode::Safer;
    int resolutionIndex = 1;
    double targetFps = 2.5;
    bool shortKey = false;
    bool enhance = true;
    bool dither = false;
    bool recording = false;
    int camera = 0;
    LinkRole linkRole = LinkRole::Duplex;
    RxSource rxSource = RxSource::LiveMic;
    SessionMode sessionMode = SessionMode::Broadcast;
    BandMode bandMode = BandMode::Audible;
    int audioInDevice = -1;
    int audioOutDevice = -1;
    std::string mediaPath;
    TransportKind transportKind = TransportKind::Acoustic;
    std::string serialPort;
    int serialBaud = 115200;
    std::string nodeAlias = "Field-Unit";
    std::string relayExportPath = "./relay_out/export.evrelay";
    std::string relayImportPath;
    bool authEnabled = false;
    std::string authPin;
    bool linkRunning = false;
    bool forceKey = false;
    bool rescanCamera = false;
    bool rescanAudio = false;
    bool startLink = false;
    bool stopLink = false;
    bool sendText = false;
    bool sendSnapshot = false;
    bool runLinkTest = false;
    std::string outgoingText;
    TargetScope outgoingTextScope = TargetScope::Broadcast;
    uint64_t outgoingTextTarget = 0;
    bool exportRelay = false;
    bool importRelay = false;
};

ControlSnapshot copyControlSnapshot(ControlState &control) {
    ControlSnapshot out;
    std::lock_guard<std::mutex> lock(control.mutex);
    out.mode = control.mode;
    out.resolutionIndex = control.resolutionIndex;
    out.targetFps = control.targetFps;
    out.shortKey = control.shortKeyframeInterval;
    out.enhance = control.useReceivedEnhancement;
    out.dither = control.useReceivedDithering;
    out.recording = control.recordingEnabled;
    out.camera = control.requestedCameraIndex;
    out.linkRole = control.linkRole;
    out.rxSource = control.rxSource;
    out.sessionMode = control.sessionMode;
    out.bandMode = control.bandMode;
    out.audioInDevice = control.requestedAudioInputDevice;
    out.audioOutDevice = control.requestedAudioOutputDevice;
    out.mediaPath = control.mediaPath;
    out.transportKind = control.transportKind;
    out.serialPort = control.serialPort;
    out.serialBaud = control.serialBaud;
    out.nodeAlias = control.nodeAlias;
    out.relayExportPath = control.relayExportPath;
    out.relayImportPath = control.relayImportPath;
    out.authEnabled = control.authEnabled;
    out.authPin = control.authPin;
    out.linkRunning = control.linkRunning;
    out.forceKey = control.forceNextKeyframe;
    out.rescanCamera = control.rescanCameras;
    out.rescanAudio = control.rescanAudio;
    out.startLink = control.startLink;
    out.stopLink = control.stopLink;
    out.sendText = control.sendText;
    out.sendSnapshot = control.sendSnapshot;
    out.runLinkTest = control.runLinkTest;
    out.outgoingText = control.outgoingText;
    out.outgoingTextScope = control.outgoingTextScope;
    out.outgoingTextTarget = control.outgoingTextTarget;
    out.exportRelay = control.exportRelay;
    out.importRelay = control.importRelay;
    return out;
}

void clearOneShotControlFlags(ControlState &control,
                              bool clearForceKey,
                              bool clearRescanCamera,
                              bool clearRescanAudio,
                              bool clearStartLink,
                              bool clearStopLink,
                              bool clearSendText,
                              bool clearSendSnapshot,
                              bool clearRunLinkTest,
                              bool clearExportRelay,
                              bool clearImportRelay) {
    std::lock_guard<std::mutex> lock(control.mutex);
    if (clearForceKey) {
        control.forceNextKeyframe = false;
    }
    if (clearRescanCamera) {
        control.rescanCameras = false;
    }
    if (clearRescanAudio) {
        control.rescanAudio = false;
    }
    if (clearStartLink) {
        control.startLink = false;
    }
    if (clearStopLink) {
        control.stopLink = false;
    }
    if (clearSendText) {
        control.sendText = false;
        control.outgoingText.clear();
    }
    if (clearSendSnapshot) {
        control.sendSnapshot = false;
    }
    if (clearRunLinkTest) {
        control.runLinkTest = false;
    }
    if (clearExportRelay) {
        control.exportRelay = false;
    }
    if (clearImportRelay) {
        control.importRelay = false;
    }
}

void writeStateStatus(SharedState &shared, const std::string &status) {
    std::lock_guard<std::mutex> lock(shared.mutex);
    shared.status = status;
}

LinkMode acousticLinkModeFromRole(LinkRole role, RxSource rxSource) {
    switch (role) {
    case LinkRole::Send:
        return LinkMode::AcousticTx;
    case LinkRole::Receive:
        return (rxSource == RxSource::MediaFile) ? LinkMode::AcousticRxMedia : LinkMode::AcousticRxLive;
    case LinkRole::Duplex:
        return LinkMode::AcousticDuplexArq;
    }
    return LinkMode::AcousticDuplexArq;
}

std::size_t transportFragmentBytes(TransportKind kind) {
    switch (kind) {
    case TransportKind::Acoustic:
        return 96;
    case TransportKind::Serial:
        return 700;
    case TransportKind::Optical:
        return 40;
    case TransportKind::FileRelay:
        return 1300;
    }
    return 256;
}

bool isRelayablePayloadType(CommPayloadType type) {
    return type == CommPayloadType::Text || type == CommPayloadType::Snapshot;
}

struct EnvelopeReassemblyEntry {
    CommEnvelopeHeader header{};
    std::vector<std::vector<uint8_t>> frags;
    std::vector<uint8_t> present;
    std::chrono::steady_clock::time_point lastSeen{};
};

class EnvelopeFragmentReassembler {
public:
    explicit EnvelopeFragmentReassembler(std::chrono::milliseconds timeout) : timeout_(timeout) {}

    bool push(const CommEnvelopeHeader &header,
              const std::vector<uint8_t> &payload,
              std::vector<uint8_t> &completeEnvelope,
              std::string &error,
              const EnvelopeAuthConfig *auth) {
        completeEnvelope.clear();
        error.clear();
        cleanupExpired();

        if (header.fragCount <= 1) {
            completeEnvelope = serializeCommEnvelope(header, payload, auth);
            return true;
        }

        if (header.fragIndex >= header.fragCount) {
            error = "invalid fragment index";
            return false;
        }

        const Key key{header.senderNodeId, header.msgId, header.seq};
        auto &entry = entries_[key];
        if (entry.frags.empty()) {
            entry.header = header;
            entry.frags.assign(header.fragCount, {});
            entry.present.assign(header.fragCount, 0);
        }
        if (entry.frags.size() != header.fragCount) {
            entries_.erase(key);
            error = "fragment count mismatch";
            return false;
        }

        entry.lastSeen = std::chrono::steady_clock::now();
        const std::size_t idx = static_cast<std::size_t>(header.fragIndex);
        entry.frags[idx] = payload;
        entry.present[idx] = 1;

        if (!std::all_of(entry.present.begin(), entry.present.end(), [](uint8_t v) { return v != 0; })) {
            return false;
        }

        std::vector<uint8_t> assembled;
        for (const auto &frag : entry.frags) {
            assembled.insert(assembled.end(), frag.begin(), frag.end());
        }

        CommEnvelopeHeader merged = entry.header;
        merged.fragIndex = 0;
        merged.fragCount = 1;
        completeEnvelope = serializeCommEnvelope(merged, assembled, auth);
        entries_.erase(key);
        return true;
    }

private:
    struct Key {
        uint64_t sender = 0;
        uint64_t msg = 0;
        uint32_t seq = 0;
        bool operator<(const Key &other) const {
            if (sender != other.sender) {
                return sender < other.sender;
            }
            if (msg != other.msg) {
                return msg < other.msg;
            }
            return seq < other.seq;
        }
    };

    void cleanupExpired() {
        const auto now = std::chrono::steady_clock::now();
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (now - it->second.lastSeen > timeout_) {
                it = entries_.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::chrono::milliseconds timeout_;
    std::map<Key, EnvelopeReassemblyEntry> entries_;
};

} // namespace

int main() {
    std::signal(SIGINT, signalHandler);
#if defined(SIGTERM)
    std::signal(SIGTERM, signalHandler);
#endif

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    const int preferredIndex = preferredCameraIndex();
    CodecMode mode = CodecMode::Safer;
    CodecParams defaults = makeCodecParams(mode);
    int resolutionIndex = findNearestResolutionIndex(defaults.width, defaults.height);
    double targetFps = defaults.targetFps;

    CodecParams params = makeRuntimeParams(mode, resolutionIndex, targetFps);

    Encoder encoder(params);
    Decoder decoder;
    MetricsTracker metrics(params.width, params.height);

    bool shortKeyframeInterval = false;
    bool useReceivedEnhancement = true;
    bool useReceivedDithering = false;

    bool recordingEnabled = false;
    std::ofstream recordingFile;
    std::string lastRecordingPath;

    cv::VideoCapture camera;
    int cameraIndex = preferredIndex;
    std::vector<int> initialCameraList = probeAvailableCameraIndices(kMaxCameraProbe);
    if (!switchCamera(camera, cameraIndex)) {
        bool opened = false;
        for (int idx : initialCameraList) {
            if (switchCamera(camera, idx)) {
                cameraIndex = idx;
                opened = true;
                break;
            }
        }
        if (!opened) {
            std::cerr << "Failed to open any webcam index in discovered list\n";
            return 1;
        }
    }

    std::cout << "Web UI listening on http://" << kBindHost << ":" << kPort << "\n";
    std::cout << "Using webcam index " << cameraIndex << " (override with EV_CAMERA_INDEX)\n";

    cv::CascadeClassifier faceDetector;
    std::string faceCascadePath;
    const bool faceDetectorReady = loadFaceDetector(faceDetector, faceCascadePath);
    if (faceDetectorReady) {
        std::cout << "Face detector loaded: " << faceCascadePath << '\n';
    } else {
        std::cout << "Face detector unavailable. Set EV_FACE_CASCADE to haarcascade_frontalface_default.xml\n";
    }

    std::vector<cv::Rect> lastFacesRaw;
    std::vector<cv::Mat> gatheredFaces;

    bool haveSentFrame = false;
    Gray4Frame latestSentFrame;

    InterpolationState interpolation;

    MetricsSnapshot latestMetrics;
    EncodeMetadata latestMeta{};
    std::size_t latestPacketBytes = 0;
    bool haveStats = false;

    AudioEngine audioEngine;
    if (!audioEngine.init()) {
        std::cerr << "Audio backend init failed; acoustic modes disabled\n";
    }
    std::vector<AudioDeviceInfo> audioInputs = audioEngine.listInputDevices();
    std::vector<AudioDeviceInfo> audioOutputs = audioEngine.listOutputDevices();
    int audioInputDevice = audioInputs.empty() ? -1 : audioInputs.front().index;
    int audioOutputDevice = audioOutputs.empty() ? -1 : audioOutputs.front().index;

    LinkRole linkRole = LinkRole::Duplex;
    RxSource rxSource = RxSource::LiveMic;
    SessionMode sessionMode = SessionMode::Broadcast;
    BandMode bandMode = BandMode::Audible;
    bool linkRunning = false;
    std::string mediaPath;
    TransportKind transportKind = TransportKind::Acoustic;
    std::string serialPort;
    int serialBaud = 115200;
    std::string nodeAlias = "Field-Unit";
    std::string relayExportPath = "./relay_out/export.evrelay";
    std::string relayImportPath;
    bool authEnabled = false;
    std::string authPin;
    EnvelopeAuthConfig envelopeAuth;

    SessionConfig sessionConfig;
    sessionConfig.streamId = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                       std::chrono::steady_clock::now().time_since_epoch())
                                                       .count() &
                                                   0xFFFFFFFFULL);
    sessionConfig.sessionEpochMs = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                              std::chrono::steady_clock::now().time_since_epoch())
                                                              .count() &
                                                          0xFFFFFFFFULL);
    sessionConfig.configVersion = 1;
    sessionConfig.codecMode = params.mode;
    sessionConfig.width = static_cast<uint16_t>(params.width);
    sessionConfig.height = static_cast<uint16_t>(params.height);
    sessionConfig.blockSize = static_cast<uint8_t>(params.blockSize);
    sessionConfig.residualStep = static_cast<uint8_t>(std::max(1, params.residualStep));
    sessionConfig.keyframeInterval = static_cast<uint8_t>(activeKeyframeInterval(params, shortKeyframeInterval));
    sessionConfig.targetFps = static_cast<float>(params.targetFps);
    sessionConfig.sessionMode = sessionMode;
    sessionConfig.bandMode = bandMode;
    sessionConfig.fecRepetition = 3;
    sessionConfig.interleaveDepth = 8;
    sessionConfig.arqWindow = 12;
    sessionConfig.arqTimeoutMs = 1200;
    sessionConfig.arqMaxRetransmit = 5;
    sessionConfig.sampleRate = 48000;
    sessionConfig.symbolSamples = 120;
    sessionConfig.mfskBins = 16;
    sessionConfig.cycleMs = 1500;
    sessionConfig.txSlotMs = 1100;
    sessionConfig.configHash = computeSessionConfigHash(sessionConfig);

    LinkStats linkStats;
    uint64_t linkPayloadBytesReceived = 0;
    int startupConfigBurstRemaining = 0;
    auto lastConfigBeacon = std::chrono::steady_clock::now();
    auto linkStartTime = std::chrono::steady_clock::now();
    uint64_t linkPayloadBytesSent = 0;

    ControlState control;
    {
        std::lock_guard<std::mutex> lock(control.mutex);
        control.mode = mode;
        control.resolutionIndex = resolutionIndex;
        control.targetFps = targetFps;
        control.useReceivedEnhancement = useReceivedEnhancement;
        control.useReceivedDithering = useReceivedDithering;
        control.shortKeyframeInterval = shortKeyframeInterval;
        control.recordingEnabled = recordingEnabled;
        control.requestedCameraIndex = cameraIndex;
        control.linkRole = linkRole;
        control.rxSource = rxSource;
        control.sessionMode = sessionMode;
        control.bandMode = bandMode;
        control.requestedAudioInputDevice = audioInputDevice;
        control.requestedAudioOutputDevice = audioOutputDevice;
        control.linkRunning = linkRunning;
        control.mediaPath = mediaPath;
        control.transportKind = transportKind;
        control.serialPort = serialPort;
        control.serialBaud = serialBaud;
        control.nodeAlias = nodeAlias;
        control.relayExportPath = relayExportPath;
        control.relayImportPath = relayImportPath;
        control.authEnabled = authEnabled;
    }

    SharedState shared;
    {
        std::lock_guard<std::mutex> lock(shared.mutex);
        shared.mode = mode;
        shared.width = params.width;
        shared.height = params.height;
        shared.targetFps = params.targetFps;
        shared.shortKeyframeInterval = shortKeyframeInterval;
        shared.useReceivedEnhancement = useReceivedEnhancement;
        shared.useReceivedDithering = useReceivedDithering;
        shared.recordingEnabled = recordingEnabled;
        shared.keyframeInterval = activeKeyframeInterval(params, shortKeyframeInterval);
        shared.cameraIndex = cameraIndex;
        shared.cameras = initialCameraList;
        if (std::find(shared.cameras.begin(), shared.cameras.end(), cameraIndex) == shared.cameras.end()) {
            shared.cameras.push_back(cameraIndex);
            std::sort(shared.cameras.begin(), shared.cameras.end());
        }
        shared.status = "running";
        shared.faceDetectorReady = faceDetectorReady;
        shared.linkRole = linkRole;
        shared.rxSource = rxSource;
        shared.sessionMode = sessionMode;
        shared.bandMode = bandMode;
        shared.linkRunning = linkRunning;
        shared.audioInputDevice = audioInputDevice;
        shared.audioOutputDevice = audioOutputDevice;
        shared.audioInputs = audioInputs;
        shared.audioOutputs = audioOutputs;
        shared.mediaPath = mediaPath;
        shared.linkStats = linkStats;
        shared.streamId = sessionConfig.streamId;
        shared.configVersion = sessionConfig.configVersion;
        shared.configHash = sessionConfig.configHash;
        shared.transportKind = transportKind;
        shared.serialPort = serialPort;
        shared.serialBaud = serialBaud;
        shared.relayExportPath = relayExportPath;
        shared.relayImportPath = relayImportPath;
        shared.nodeAlias = nodeAlias;
        shared.authEnabled = authEnabled;
    }

    const std::string indexHtml = makeIndexHtml();

    httplib::Server server;
    server.Get("/", [&](const httplib::Request &, httplib::Response &res) {
        res.set_content(indexHtml, "text/html; charset=utf-8");
    });

    server.Get("/api/v2/state", [&](const httplib::Request &, httplib::Response &res) {
        std::string payload;
        {
            std::lock_guard<std::mutex> lock(shared.mutex);
            payload = buildStateJson(shared);
        }
        res.set_content(payload, "application/json");
        res.set_header("Cache-Control", "no-store, no-cache, must-revalidate");
    });

    server.Post("/api/v2/control", [&](const httplib::Request &req, httplib::Response &res) {
        applyControlRequest(req, control);
        res.set_content("{\"ok\":true}", "application/json");
    });

    server.Post("/api/v2/link/start", [&](const httplib::Request &, httplib::Response &res) {
        std::lock_guard<std::mutex> lock(control.mutex);
        control.startLink = true;
        control.stopLink = false;
        control.linkRunning = true;
        res.set_content("{\"ok\":true}", "application/json");
    });

    server.Post("/api/v2/link/stop", [&](const httplib::Request &, httplib::Response &res) {
        std::lock_guard<std::mutex> lock(control.mutex);
        control.stopLink = true;
        control.startLink = false;
        control.linkRunning = false;
        res.set_content("{\"ok\":true}", "application/json");
    });

    server.Post("/api/v2/messages/send", [&](const httplib::Request &req, httplib::Response &res) {
        const std::map<std::string, std::string> params = requestParamMap(req);
        std::lock_guard<std::mutex> lock(control.mutex);
        auto itBody = params.find("body");
        if (itBody != params.end()) {
            control.outgoingText = itBody->second;
            control.sendText = !control.outgoingText.empty();
        }
        auto itScope = params.find("scope");
        if (itScope != params.end()) {
            control.outgoingTextScope = (itScope->second == "direct") ? TargetScope::Direct : TargetScope::Broadcast;
        }
        auto itTarget = params.find("target_node_id");
        if (itTarget != params.end()) {
            try {
                control.outgoingTextTarget = static_cast<uint64_t>(std::stoull(itTarget->second));
            } catch (const std::exception &) {
            }
        }
        res.set_content("{\"ok\":true}", "application/json");
    });

    auto serveFrame = [&](const std::vector<uint8_t> SharedState::*field, httplib::Response &res) {
        std::vector<uint8_t> bytes;
        {
            std::lock_guard<std::mutex> lock(shared.mutex);
            bytes = shared.*field;
        }

        if (bytes.empty()) {
            res.status = 503;
            res.set_content("frame not ready", "text/plain");
            return;
        }

        res.set_content(reinterpret_cast<const char *>(bytes.data()), bytes.size(), "image/jpeg");
        res.set_header("Cache-Control", "no-store, no-cache, must-revalidate");
    };

    server.Get("/api/v2/frame/raw.jpg", [&](const httplib::Request &, httplib::Response &res) {
        serveFrame(&SharedState::rawJpeg, res);
    });
    server.Get("/api/v2/frame/sent.jpg", [&](const httplib::Request &, httplib::Response &res) {
        serveFrame(&SharedState::sentJpeg, res);
    });
    server.Get("/api/v2/frame/received.jpg", [&](const httplib::Request &, httplib::Response &res) {
        serveFrame(&SharedState::receivedJpeg, res);
    });

    std::thread serverThread([&]() {
        if (!server.listen(kBindHost, kPort)) {
            std::cerr << "Web server failed to listen on " << kBindHost << ":" << kPort << '\n';
            gStop.store(true);
        }
    });

    using Clock = std::chrono::steady_clock;
    auto nextEncodeTime = Clock::now();
    auto lastConsolePrint = Clock::now();
    auto resetCodecPipeline = [&]() {
        encoder.setParams(params);
        decoder.reset();
        metrics.reset(params.width, params.height);
        encoder.setKeyframeInterval(activeKeyframeInterval(params, shortKeyframeInterval));
        haveSentFrame = false;
        interpolation = InterpolationState{};
        haveStats = false;
        nextEncodeTime = Clock::now();
    };

    auto updateInterpolationFrame = [&](const Gray4Frame &decodedFrame, const Clock::time_point &ts) {
        if (!interpolation.hasCurrent) {
            interpolation.current = decodedFrame;
            interpolation.previous = decodedFrame;
            interpolation.motion.assign(
                static_cast<std::size_t>(totalBlockCount(decodedFrame.width, decodedFrame.height, params.blockSize)),
                MotionVector{});
            interpolation.hasCurrent = true;
            interpolation.hasPair = false;
            interpolation.blendStart = ts;
            return;
        }

        interpolation.previous = interpolation.current;
        interpolation.current = decodedFrame;
        interpolation.motion = estimateBlockMotion(interpolation.previous, interpolation.current, params.blockSize, 2);
        interpolation.hasPair = true;
        interpolation.blendStart = ts;
    };

    TransportManager transportManager;
    QueueStats queueStats{};
    linkStats = LinkStats{};
    linkPayloadBytesSent = 0;
    linkPayloadBytesReceived = 0;
    linkStartTime = Clock::now();
    lastConfigBeacon = Clock::now() - std::chrono::seconds(5);
    startupConfigBurstRemaining = 8;
    bool forceConfigBurst = true;
    int emptyFrames = 0;

    NodeIdentity nodeIdentity;
    nodeIdentity.nodeId = (nowUnixMs() << 1U) ^ 0x5A17ULL;
    nodeIdentity.alias = nodeAlias;
    Router router(nodeIdentity);

    auto enqueueSnapshotFromFrame = [&](const cv::Mat &frame) {
        if (frame.empty()) {
            return;
        }
        SnapshotMessage snapshot;
        snapshot.width = static_cast<uint16_t>(std::clamp(frame.cols, 0, 65535));
        snapshot.height = static_cast<uint16_t>(std::clamp(frame.rows, 0, 65535));
        snapshot.jpeg = encodeJpeg(frame, 84);
        if (!snapshot.jpeg.empty()) {
            router.enqueueSnapshot(snapshot, 120000);
        }
    };

    PersistentStore store;
    std::string storeError;
    if (!store.init("./relay_store", storeError)) {
        std::cerr << "Persistent store init failed: " << storeError << '\n';
    }

    SessionConfigV2 sessionConfigV2;
    sessionConfigV2.streamId = sessionConfig.streamId;
    sessionConfigV2.configVersion = sessionConfig.configVersion;

    auto rebuildSessionConfig = [&](bool bumpVersion) {
        if (bumpVersion) {
            sessionConfig.configVersion = static_cast<uint16_t>(std::max<uint16_t>(1, sessionConfig.configVersion + 1));
            sessionConfigV2.configVersion = static_cast<uint16_t>(std::max<uint16_t>(1, sessionConfigV2.configVersion + 1));
        }
        sessionConfig.codecMode = params.mode;
        sessionConfig.width = static_cast<uint16_t>(params.width);
        sessionConfig.height = static_cast<uint16_t>(params.height);
        sessionConfig.blockSize = static_cast<uint8_t>(params.blockSize);
        sessionConfig.residualStep = static_cast<uint8_t>(std::max(1, params.residualStep));
        sessionConfig.keyframeInterval = static_cast<uint8_t>(activeKeyframeInterval(params, shortKeyframeInterval));
        sessionConfig.targetFps = static_cast<float>(params.targetFps);
        sessionConfig.sessionMode = sessionMode;
        sessionConfig.bandMode = bandMode;
        sessionConfig.configHash = computeSessionConfigHash(sessionConfig);

        sessionConfigV2.codecMode = params.mode;
        sessionConfigV2.width = static_cast<uint16_t>(params.width);
        sessionConfigV2.height = static_cast<uint16_t>(params.height);
        sessionConfigV2.blockSize = static_cast<uint8_t>(params.blockSize);
        sessionConfigV2.residualStep = static_cast<uint8_t>(std::max(1, params.residualStep));
        sessionConfigV2.keyframeInterval = static_cast<uint8_t>(activeKeyframeInterval(params, shortKeyframeInterval));
        sessionConfigV2.targetFps = static_cast<float>(params.targetFps);
        sessionConfigV2.version = 3;
        sessionConfigV2.configHash = computeSessionConfigV2Hash(sessionConfigV2);
    };
    rebuildSessionConfig(false);

    auto stopAllTransports = [&]() {
        transportManager.acoustic().stop();
        transportManager.serial().stop();
        transportManager.optical().stop();
        transportManager.fileRelay().stop();
    };

    auto configureTransports = [&]() {
        transportManager.setActive(transportKind);
        transportManager.serial().setPreferredPort(serialPort);
        transportManager.serial().setBaud(serialBaud);

        const std::filesystem::path exportPath =
            relayExportPath.empty() ? std::filesystem::path("./relay_out/export.evrelay") : std::filesystem::path(relayExportPath);
        const std::filesystem::path importPath =
            relayImportPath.empty() ? std::filesystem::path("./relay_in/import.evrelay") : std::filesystem::path(relayImportPath);
        std::filesystem::path exportDir = exportPath.has_parent_path() ? exportPath.parent_path() : std::filesystem::path(".");
        std::filesystem::path importDir = importPath.has_parent_path() ? importPath.parent_path() : std::filesystem::path(".");
        if (exportDir.empty()) {
            exportDir = ".";
        }
        if (importDir.empty()) {
            importDir = ".";
        }
        transportManager.fileRelay().setDirectories(exportDir.string(), importDir.string());

        const LinkMode acousticMode = acousticLinkModeFromRole(linkRole, rxSource);
        transportManager.acoustic().configure(
            sessionConfig, acousticMode, rxSource, sessionMode, audioInputDevice, audioOutputDevice, mediaPath);
    };
    configureTransports();

    auto refreshEnvelopeAuth = [&]() {
        envelopeAuth.enabled = authEnabled && !authPin.empty();
        if (envelopeAuth.enabled) {
            envelopeAuth.key = deriveAuthKeyFromPin(authPin);
        } else {
            envelopeAuth.key.fill(0);
        }
    };
    refreshEnvelopeAuth();

    FallbackController fallbackController(sessionConfigV2.maxFallbackStage);
    FallbackStage fallbackStage = FallbackStage::Normal;
    uint64_t prevDroppedForFallback = 0;
    uint64_t prevRetransForFallback = 0;
    auto lastFallbackSnapshot = Clock::now() - std::chrono::seconds(10);

    struct ProbeRunState {
        bool active = false;
        int remaining = 0;
        uint64_t nextProbeId = 1;
        std::chrono::steady_clock::time_point nextSend{};
        std::map<uint64_t, uint64_t> pendingSentMs;
    };
    ProbeRunState probeRun;

    cv::Mat latestReceivedSnapshot;

    auto startActiveTransport = [&]() {
        if (!linkRunning) {
            return;
        }
        std::string error;
        if (TransportAdapter *adapter = transportManager.activeAdapter()) {
            const bool wasRunning = adapter->running();
            if (!wasRunning && !adapter->start(error)) {
                writeStateStatus(shared, "transport start failed: " + error);
                linkRunning = false;
                std::lock_guard<std::mutex> lock(control.mutex);
                control.linkRunning = false;
                return;
            }
            if (!wasRunning) {
                linkStartTime = Clock::now();
                startupConfigBurstRemaining = 8;
                forceConfigBurst = true;
            }
        }
    };

    EnvelopeFragmentReassembler envelopeReassembler(std::chrono::milliseconds(4000));
    std::map<uint64_t, std::unique_ptr<Decoder>> remoteDecoders;

    auto processCompleteEnvelope = [&](const std::vector<uint8_t> &envelope, const Clock::time_point &now) {
        CommEnvelopeHeader header;
        std::vector<uint8_t> payload;
        std::string error;
        bool authFailure = false;
        if (!deserializeCommEnvelope(envelope, header, payload, error, &envelopeAuth, &authFailure)) {
            linkStats.framesDropped += 1;
            if (authFailure) {
                linkStats.authFailures += 1;
            }
            return;
        }

        linkPayloadBytesReceived += payload.size();
        linkStats.framesReceived += 1;
        linkStats.syncLocked = true;

        std::string persistError;
        (void)store.persistInbound(header, payload, isRelayablePayloadType(header.payloadType), persistError, &envelopeAuth);

        RouterEvents events = router.processIncomingEnvelope(envelope, now, &envelopeAuth);
        for (const auto &video : events.videoFrames) {
            auto &decoderPtr = remoteDecoders[video.header.senderNodeId];
            if (!decoderPtr) {
                decoderPtr = std::make_unique<Decoder>();
                decoderPtr->reset();
            }
            DecodeResult decoded = decoderPtr->decode(video.codecPayload);
            if (!decoded.ok) {
                linkStats.framesDropped += 1;
                continue;
            }
            updateInterpolationFrame(decoded.frame, now);
            latestMeta = decoded.meta;
            latestPacketBytes = video.codecPayload.size();
            haveStats = true;
        }
        for (const auto &snapshot : events.snapshots) {
            const cv::Mat jpegMat = cv::imdecode(snapshot.snapshot.jpeg, cv::IMREAD_COLOR);
            if (!jpegMat.empty()) {
                latestReceivedSnapshot = jpegMat;
            }
        }
        for (const auto &text : events.texts) {
            (void)text;
        }
        for (const auto &probeEvent : events.probes) {
            if (probeEvent.probe.kind != ProbeKind::Pong) {
                continue;
            }
            const auto it = probeRun.pendingSentMs.find(probeEvent.probe.probeId);
            if (it == probeRun.pendingSentMs.end()) {
                continue;
            }
            const uint64_t nowMs = nowUnixMs();
            const double rttSample = static_cast<double>(nowMs - probeEvent.probe.sentTsMs);
            linkStats.rttMs = (linkStats.rttMs <= 0.0) ? rttSample : (0.75 * linkStats.rttMs + 0.25 * rttSample);
            linkStats.probeAcked += 1;
            probeRun.pendingSentMs.erase(it);
        }
    };

    if (linkRunning) {
        startActiveTransport();
    }

    while (!gStop.load()) {
        const ControlSnapshot desired = copyControlSnapshot(control);
        bool transportNeedsReconfigure = false;
        bool bumpConfigVersion = false;
        bool oneShotSnapshot = false;

        if (desired.rescanCamera) {
            const std::vector<int> cameras = probeAvailableCameraIndices(kMaxCameraProbe);
            {
                std::lock_guard<std::mutex> lock(shared.mutex);
                shared.cameras = cameras;
                if (std::find(shared.cameras.begin(), shared.cameras.end(), shared.cameraIndex) == shared.cameras.end() &&
                    shared.cameraIndex >= 0) {
                    shared.cameras.push_back(shared.cameraIndex);
                    std::sort(shared.cameras.begin(), shared.cameras.end());
                }
                shared.status = "camera list refreshed";
            }
            clearOneShotControlFlags(control, false, true, false, false, false, false, false, false, false, false);
        }
        if (desired.rescanAudio) {
            audioInputs = audioEngine.listInputDevices();
            audioOutputs = audioEngine.listOutputDevices();
            {
                std::lock_guard<std::mutex> lock(shared.mutex);
                shared.audioInputs = audioInputs;
                shared.audioOutputs = audioOutputs;
                shared.status = "audio list refreshed";
            }
            clearOneShotControlFlags(control, false, false, true, false, false, false, false, false, false, false);
        }
        if (desired.forceKey) {
            encoder.forceNextKeyframe();
            clearOneShotControlFlags(control, true, false, false, false, false, false, false, false, false, false);
        }

        const int desiredResolutionIndex =
            std::clamp(desired.resolutionIndex, 0, static_cast<int>(kResolutionLevels.size()) - 1);
        const double desiredFps = std::max(kMinTargetFps, desired.targetFps);

        if (desired.camera != cameraIndex) {
            if (switchCamera(camera, desired.camera)) {
                cameraIndex = desired.camera;
                emptyFrames = 0;
                writeStateStatus(shared, "camera switched to index " + std::to_string(cameraIndex));
            } else {
                writeStateStatus(shared, "camera switch failed for index " + std::to_string(desired.camera));
            }
        }

        if (desired.mode != mode || desiredResolutionIndex != resolutionIndex || std::abs(desiredFps - targetFps) > 1e-6) {
            mode = desired.mode;
            resolutionIndex = desiredResolutionIndex;
            targetFps = desiredFps;
            bumpConfigVersion = true;
        }

        if (desired.shortKey != shortKeyframeInterval) {
            shortKeyframeInterval = desired.shortKey;
            encoder.setKeyframeInterval(activeKeyframeInterval(params, shortKeyframeInterval));
            bumpConfigVersion = true;
        }

        useReceivedEnhancement = desired.enhance;
        useReceivedDithering = desired.dither;

        if (desired.recording != recordingEnabled) {
            if (desired.recording) {
                lastRecordingPath = makeRecordingFileName();
                if (!openRecordingFile(recordingFile, lastRecordingPath)) {
                    writeStateStatus(shared, "recording start failed");
                } else {
                    recordingEnabled = true;
                    writeStateStatus(shared, "recording started: " + lastRecordingPath);
                }
            } else {
                recordingEnabled = false;
                if (recordingFile.is_open()) {
                    recordingFile.close();
                }
                writeStateStatus(shared, "recording stopped");
            }
        }

        if (desired.linkRole != linkRole) {
            linkRole = desired.linkRole;
            transportNeedsReconfigure = true;
        }
        if (desired.rxSource != rxSource) {
            rxSource = desired.rxSource;
            transportNeedsReconfigure = true;
        }
        if (desired.sessionMode != sessionMode) {
            sessionMode = desired.sessionMode;
            transportNeedsReconfigure = true;
            bumpConfigVersion = true;
        }
        if (desired.bandMode != bandMode) {
            bandMode = desired.bandMode;
            transportNeedsReconfigure = true;
            bumpConfigVersion = true;
        }
        if (desired.audioInDevice != audioInputDevice) {
            audioInputDevice = desired.audioInDevice;
            transportNeedsReconfigure = true;
        }
        if (desired.audioOutDevice != audioOutputDevice) {
            audioOutputDevice = desired.audioOutDevice;
            transportNeedsReconfigure = true;
        }
        if (desired.mediaPath != mediaPath) {
            mediaPath = desired.mediaPath;
            transportNeedsReconfigure = true;
        }
        if (desired.transportKind != transportKind) {
            transportKind = desired.transportKind;
            transportNeedsReconfigure = true;
        }
        if (desired.serialPort != serialPort || desired.serialBaud != serialBaud) {
            serialPort = desired.serialPort;
            serialBaud = desired.serialBaud;
            transportNeedsReconfigure = true;
        }
        if (desired.nodeAlias != nodeAlias) {
            nodeAlias = desired.nodeAlias;
            nodeIdentity.alias = nodeAlias;
            bumpConfigVersion = true;
        }
        if (desired.relayExportPath != relayExportPath || desired.relayImportPath != relayImportPath) {
            relayExportPath = desired.relayExportPath;
            relayImportPath = desired.relayImportPath;
            transportNeedsReconfigure = true;
        }
        if (desired.authEnabled != authEnabled || desired.authPin != authPin) {
            authEnabled = desired.authEnabled;
            authPin = desired.authPin;
            refreshEnvelopeAuth();
        }

        if (desired.startLink) {
            linkRunning = true;
            clearOneShotControlFlags(control, false, false, false, true, false, false, false, false, false, false);
        }
        if (desired.stopLink) {
            linkRunning = false;
            clearOneShotControlFlags(control, false, false, false, false, true, false, false, false, false, false);
        }
        if (desired.linkRunning != linkRunning && !desired.startLink && !desired.stopLink) {
            linkRunning = desired.linkRunning;
        }

        if (desired.sendText) {
            if (!desired.outgoingText.empty()) {
                router.enqueueText(desired.outgoingText, desired.outgoingTextScope, desired.outgoingTextTarget);
            }
            clearOneShotControlFlags(control, false, false, false, false, false, true, false, false, false, false);
        }
        if (desired.sendSnapshot) {
            oneShotSnapshot = true;
            clearOneShotControlFlags(control, false, false, false, false, false, false, true, false, false, false);
        }
        if (desired.runLinkTest) {
            probeRun.active = true;
            probeRun.remaining = 10;
            probeRun.nextSend = Clock::now();
            probeRun.pendingSentMs.clear();
            linkStats.probeSent = 0;
            linkStats.probeAcked = 0;
            clearOneShotControlFlags(control, false, false, false, false, false, false, false, true, false, false);
        }

        if (desired.exportRelay) {
            std::size_t exportedCount = 0;
            std::string error;
            if (!store.exportRelayBundle(relayExportPath, 2048, &exportedCount, error)) {
                writeStateStatus(shared, "relay export failed: " + error);
            } else {
                writeStateStatus(shared, "relay export complete (" + std::to_string(exportedCount) + " records)");
            }
            clearOneShotControlFlags(control, false, false, false, false, false, false, false, false, true, false);
        }
        if (desired.importRelay) {
            std::vector<std::vector<uint8_t>> imported;
            std::string error;
            if (!store.importRelayBundle(relayImportPath, imported, error)) {
                writeStateStatus(shared, "relay import failed: " + error);
            } else {
                const auto now = Clock::now();
                for (const auto &env : imported) {
                    processCompleteEnvelope(env, now);
                }
                writeStateStatus(shared, "relay import complete (" + std::to_string(imported.size()) + " envelopes)");
            }
            clearOneShotControlFlags(control, false, false, false, false, false, false, false, false, false, true);
        }

        fallbackController.setMaxStage(sessionConfigV2.maxFallbackStage);
        if (!linkRunning) {
            fallbackController.reset(FallbackStage::Normal);
        } else {
            const uint64_t droppedNow = linkStats.transportFramesDropped;
            const uint64_t retransNow = linkStats.retransmitCount;
            const uint64_t droppedDelta = (droppedNow >= prevDroppedForFallback) ? (droppedNow - prevDroppedForFallback) : 0;
            const uint64_t retransDelta = (retransNow >= prevRetransForFallback) ? (retransNow - prevRetransForFallback) : 0;
            prevDroppedForFallback = droppedNow;
            prevRetransForFallback = retransNow;

            FallbackInputWindow fallbackInput;
            fallbackInput.queueVideo = queueStats.queuedVideo;
            fallbackInput.queueInflight = queueStats.inFlightReliable;
            fallbackInput.droppedDelta = droppedDelta;
            fallbackInput.retransmitDelta = retransDelta;
            fallbackInput.transportLossPercent = linkStats.transportLossPercent;
            fallbackInput.syncLocked = linkStats.syncLocked;
            if (fallbackController.update(fallbackInput, Clock::now())) {
                bumpConfigVersion = true;
                forceConfigBurst = true;
            }
        }
        fallbackStage = fallbackController.stage();

        int effectiveResolutionIndex = resolutionIndex;
        double effectiveFps = targetFps;
        if (static_cast<int>(fallbackStage) >= static_cast<int>(FallbackStage::LowerFps)) {
            effectiveFps = std::max(0.8, targetFps * 0.65);
        }
        if (static_cast<int>(fallbackStage) >= static_cast<int>(FallbackStage::LowerResolution)) {
            effectiveResolutionIndex = std::max(0, resolutionIndex - 1);
        }

        const CodecParams effectiveParams = makeRuntimeParams(mode, effectiveResolutionIndex, effectiveFps);
        if (effectiveParams.mode != params.mode || effectiveParams.width != params.width || effectiveParams.height != params.height ||
            std::abs(effectiveParams.targetFps - params.targetFps) > 1e-6) {
            params = effectiveParams;
            resetCodecPipeline();
            bumpConfigVersion = true;
        }

        if (bumpConfigVersion) {
            rebuildSessionConfig(true);
            forceConfigBurst = true;
            startupConfigBurstRemaining = 8;
            encoder.forceNextKeyframe();
        }

        if (transportNeedsReconfigure) {
            const bool wasRunning = linkRunning;
            stopAllTransports();
            configureTransports();
            if (wasRunning) {
                startActiveTransport();
            }
            forceConfigBurst = true;
        }

        if (linkRunning) {
            startActiveTransport();
        } else {
            stopAllTransports();
            linkStats.syncLocked = false;
            probeRun.active = false;
            probeRun.pendingSentMs.clear();
        }

        cv::Mat bgr;
        camera >> bgr;
        if (bgr.empty()) {
            ++emptyFrames;
            if (emptyFrames > 16) {
                writeStateStatus(shared, "empty frames from camera; attempting reopen");
                if (!switchCamera(camera, cameraIndex)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(120));
                }
                emptyFrames = 0;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(12));
            }
            continue;
        }
        emptyFrames = 0;

        const auto now = Clock::now();
        const bool roleSend = linkRole != LinkRole::Receive;
        const bool roleReceive = linkRole != LinkRole::Send;

        if (oneShotSnapshot && linkRunning && roleSend) {
            enqueueSnapshotFromFrame(bgr);
        }
        if (linkRunning && roleSend && fallbackStage == FallbackStage::SnapshotOnly &&
            std::chrono::duration_cast<std::chrono::milliseconds>(now - lastFallbackSnapshot).count() >= 1500) {
            enqueueSnapshotFromFrame(bgr);
            lastFallbackSnapshot = now;
        }

        if (now >= nextEncodeTime) {
            const cv::Mat grayCodec = resizeAndGrayscale(bgr, params.width, params.height);
            Gray4Frame inputFrame = quantizeTo4Bit(grayCodec);
            latestSentFrame = inputFrame;
            haveSentFrame = true;

            std::vector<uint8_t> roiBlocks;
            cv::Mat rawGray;
            if (faceDetectorReady) {
                lastFacesRaw = detectFaces(bgr, faceDetector, rawGray);
                gatherFacePatches(rawGray, lastFacesRaw, gatheredFaces, 256);
                roiBlocks = buildFaceRoiBlocks(lastFacesRaw, bgr.size(), params.width, params.height, params.blockSize);
            } else {
                lastFacesRaw.clear();
            }

            EncodedPacket packet;
            try {
                packet = encoder.encode(inputFrame, roiBlocks.empty() ? nullptr : &roiBlocks);
            } catch (const std::exception &ex) {
                writeStateStatus(shared, std::string("encode failure: ") + ex.what());
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }

            if (recordingEnabled && !appendRecordingPacket(recordingFile, packet.bytes)) {
                writeStateStatus(shared, "recording write failed; recording disabled");
                recordingEnabled = false;
                if (recordingFile.is_open()) {
                    recordingFile.close();
                }
                std::lock_guard<std::mutex> lock(control.mutex);
                control.recordingEnabled = false;
            }

            DecodeResult localPreview = decoder.decode(packet.bytes);
            if (localPreview.ok) {
                metrics.update(packet.meta, packet.bytes.size(), inputFrame, localPreview.frame);
                latestMetrics = metrics.snapshot();
                latestMeta = packet.meta;
                latestPacketBytes = packet.bytes.size();
                haveStats = true;
                if (!roleReceive || !linkRunning) {
                    updateInterpolationFrame(localPreview.frame, now);
                }
            }

            if (linkRunning && roleSend && static_cast<int>(fallbackStage) < static_cast<int>(FallbackStage::SnapshotOnly)) {
                router.enqueueVideoFrame(packet.bytes, packet.meta.frameType == FrameType::Keyframe);
            }

            if (std::chrono::duration_cast<std::chrono::seconds>(now - lastConsolePrint).count() >= 1) {
                std::cout << "mode=" << codecModeName(params.mode) << " cam=" << cameraIndex
                          << " bytes=" << packet.bytes.size()
                          << " live=" << formatDouble(latestMetrics.liveBitrateKbps) << "kbps"
                          << " transport=" << transportKindName(transportKind)
                          << " role=" << linkRoleName(linkRole) << '\n';
                lastConsolePrint = now;
            }

            const int encodePeriodMs = std::max(1, static_cast<int>(std::lround(1000.0 / std::max(0.1, params.targetFps))));
            nextEncodeTime = now + std::chrono::milliseconds(encodePeriodMs);
        }

        if (linkRunning) {
            TransportAdapter *adapter = transportManager.activeAdapter();
            if (adapter != nullptr && adapter->running()) {
                if (transportKind == TransportKind::Optical && roleReceive) {
                    transportManager.optical().feedRxFrame(bgr);
                }

                if (roleReceive) {
                    std::vector<std::vector<uint8_t>> incoming;
                    adapter->pollIncoming(incoming);
                    for (const auto &wire : incoming) {
                        linkStats.transportFramesIn += 1;
                        CommEnvelopeHeader header;
                        std::vector<uint8_t> payload;
                        std::string error;
                        bool authFailure = false;
                        if (!deserializeCommEnvelope(wire, header, payload, error, &envelopeAuth, &authFailure)) {
                            linkStats.framesDropped += 1;
                            linkStats.transportFramesDropped += 1;
                            if (authFailure) {
                                linkStats.authFailures += 1;
                            }
                            continue;
                        }
                        std::vector<uint8_t> completeEnvelope;
                        if (!envelopeReassembler.push(header, payload, completeEnvelope, error, &envelopeAuth)) {
                            if (!error.empty()) {
                                linkStats.framesDropped += 1;
                                linkStats.transportFramesDropped += 1;
                            }
                            continue;
                        }
                        processCompleteEnvelope(completeEnvelope, now);
                    }
                }

                if (roleSend) {
                    while (probeRun.active && probeRun.remaining > 0 && now >= probeRun.nextSend) {
                        TransportProbePayload probe;
                        probe.probeId = probeRun.nextProbeId++;
                        probe.kind = ProbeKind::Ping;
                        probe.sentTsMs = nowUnixMs();
                        router.enqueueTransportProbe(probe, TargetScope::Broadcast, 0, 12000);
                        probeRun.pendingSentMs[probe.probeId] = probe.sentTsMs;
                        linkStats.probeSent += 1;
                        probeRun.nextSend += std::chrono::milliseconds(200);
                        probeRun.remaining -= 1;
                    }
                    if (probeRun.active && probeRun.remaining == 0 && probeRun.pendingSentMs.empty()) {
                        probeRun.active = false;
                    }

                    const bool periodicBeacon =
                        std::chrono::duration_cast<std::chrono::milliseconds>(now - lastConfigBeacon).count() >= 2500;
                    if (startupConfigBurstRemaining > 0 || forceConfigBurst || periodicBeacon) {
                        router.enqueueConfig(serializeSessionConfigV2(sessionConfigV2));
                        if (startupConfigBurstRemaining > 0) {
                            --startupConfigBurstRemaining;
                        }
                        if (startupConfigBurstRemaining <= 0) {
                            forceConfigBurst = false;
                        }
                        lastConfigBeacon = now;
                    }

                    const std::size_t sendBudget = (transportKind == TransportKind::Optical) ? 2 : 8;
                    std::vector<RouterOutgoing> outgoing =
                        router.collectOutgoing(sendBudget, now, sessionConfigV2.reliableRetryMs, sessionConfigV2.reliableMaxRetries);
                    for (const auto &packet : outgoing) {
                        std::string persistError;
                        (void)store.persistOutbound(
                            packet.header, packet.payload, isRelayablePayloadType(packet.header.payloadType), persistError, &envelopeAuth);

                        std::vector<std::vector<uint8_t>> frags =
                            fragmentCommPayload(packet.payload, transportFragmentBytes(transportKind));
                        const uint16_t fragCount = static_cast<uint16_t>(std::min<std::size_t>(frags.size(), 0xFFFFU));
                        for (uint16_t frag = 0; frag < fragCount; ++frag) {
                            CommEnvelopeHeader fragmentHeader = packet.header;
                            fragmentHeader.fragIndex = frag;
                            fragmentHeader.fragCount = std::max<uint16_t>(1, fragCount);
                            const std::vector<uint8_t> envelope =
                                serializeCommEnvelope(fragmentHeader, frags[static_cast<std::size_t>(frag)], &envelopeAuth);
                            adapter->sendEnvelope(envelope);
                        }
                        if (packet.isRetry) {
                            linkStats.retransmitCount += 1;
                        }
                        linkPayloadBytesSent += packet.payload.size();
                    }
                }

                if (transportKind == TransportKind::Acoustic) {
                    const LinkStats delta = transportManager.acoustic().takeAndResetLinkStats();
                    linkStats.syncLocked = linkStats.syncLocked || delta.syncLocked;
                    linkStats.fecRecoveredCount += delta.fecRecoveredCount;
                    linkStats.framesReceived += delta.framesReceived;
                    linkStats.framesDropped += delta.framesDropped;
                    if (delta.rttMs > 0.0) {
                        linkStats.rttMs = delta.rttMs;
                    }
                    if (delta.berEstimate > 0.0) {
                        linkStats.berEstimate = (linkStats.berEstimate <= 0.0)
                                                    ? delta.berEstimate
                                                    : (0.8 * linkStats.berEstimate + 0.2 * delta.berEstimate);
                    }
                }
            }
        }

        queueStats = router.queueStats();
        const std::vector<TextMessage> timeline = router.timelineAfter(0, 256);
        const uint64_t latestTextCursor = timeline.empty() ? 0 : timeline.back().msgId;

        const double linkElapsed = std::max(
            0.001, std::chrono::duration_cast<std::chrono::duration<double>>(now - linkStartTime).count());
        linkStats.effectivePayloadKbps =
            (static_cast<double>(linkPayloadBytesSent + linkPayloadBytesReceived) * 8.0) / (linkElapsed * 1000.0);
        linkStats.transportLossPercent =
            (100.0 * static_cast<double>(linkStats.transportFramesDropped)) /
            std::max(1.0, static_cast<double>(linkStats.transportFramesIn));
        linkStats.probeLossPercent =
            (linkStats.probeSent == 0)
                ? 0.0
                : (100.0 * static_cast<double>(linkStats.probeSent - std::min(linkStats.probeAcked, linkStats.probeSent)) /
                   static_cast<double>(linkStats.probeSent));

        const int panelHeight = std::clamp(bgr.rows, 300, 460);
        const int panelWidth = std::max(340, panelHeight * std::max(1, bgr.cols) / std::max(1, bgr.rows));
        const cv::Size panelSize(panelWidth, panelHeight);

        cv::Mat rawPanel;
        cv::resize(bgr, rawPanel, panelSize, 0.0, 0.0, cv::INTER_AREA);
        const std::vector<cv::Rect> rawFacesPanel = scaleRects(lastFacesRaw, bgr.size(), panelSize);
        drawFaceRects(rawPanel, rawFacesPanel, cv::Scalar(0, 255, 255));

        cv::Mat sentPanel(panelHeight, panelWidth, CV_8UC3, cv::Scalar(26, 26, 26));
        if (haveSentFrame) {
            sentPanel = renderForDisplay(latestSentFrame, panelSize, false, true, params.blockSize);
            drawFaceRects(sentPanel, rawFacesPanel, cv::Scalar(0, 220, 220));
        }
        if (transportKind == TransportKind::Optical) {
            cv::Mat optical = transportManager.optical().currentTxPattern();
            if (!optical.empty()) {
                cv::resize(optical, sentPanel, panelSize, 0.0, 0.0, cv::INTER_NEAREST);
            }
        }

        cv::Mat receivedPanel(panelHeight, panelWidth, CV_8UC3, cv::Scalar(20, 20, 20));
        if (interpolation.hasCurrent) {
            Gray4Frame displayFrame = interpolation.current;
            if (interpolation.hasPair) {
                const double interval = 1.0 / std::max(0.1, params.targetFps);
                const double alpha = std::clamp(
                    std::chrono::duration_cast<std::chrono::duration<double>>(now - interpolation.blendStart).count() /
                        std::max(0.001, interval),
                    0.0,
                    1.0);
                displayFrame = interpolateMotionCompensated(
                    interpolation.previous, interpolation.current, interpolation.motion, params.blockSize, alpha);
            }
            receivedPanel =
                renderForDisplay(displayFrame, panelSize, useReceivedEnhancement, useReceivedDithering, params.blockSize);
            drawFaceRects(receivedPanel, rawFacesPanel, cv::Scalar(150, 255, 100));
        } else if (!latestReceivedSnapshot.empty()) {
            cv::resize(latestReceivedSnapshot, receivedPanel, panelSize, 0.0, 0.0, cv::INTER_AREA);
        }

        std::vector<uint8_t> rawJpeg = encodeJpeg(rawPanel, 82);
        std::vector<uint8_t> sentJpeg = encodeJpeg(sentPanel, 82);
        std::vector<uint8_t> receivedJpeg = encodeJpeg(receivedPanel, 82);

        std::string transportStatus = "stopped";
        if (TransportAdapter *adapter = transportManager.activeAdapter()) {
            transportStatus = adapter->status();
        }

        {
            std::lock_guard<std::mutex> lock(shared.mutex);
            shared.rawJpeg = std::move(rawJpeg);
            shared.sentJpeg = std::move(sentJpeg);
            shared.receivedJpeg = std::move(receivedJpeg);
            shared.metrics = latestMetrics;
            shared.meta = latestMeta;
            shared.latestPacketBytes = latestPacketBytes;
            shared.haveStats = haveStats;
            shared.faceDetectorReady = faceDetectorReady;
            shared.facesNow = static_cast<int>(lastFacesRaw.size());
            shared.facesGathered = gatheredFaces.size();
            shared.mode = params.mode;
            shared.width = params.width;
            shared.height = params.height;
            shared.targetFps = params.targetFps;
            shared.shortKeyframeInterval = shortKeyframeInterval;
            shared.useReceivedEnhancement = useReceivedEnhancement;
            shared.useReceivedDithering = useReceivedDithering;
            shared.recordingEnabled = recordingEnabled;
            shared.keyframeInterval = activeKeyframeInterval(params, shortKeyframeInterval);
            shared.cameraIndex = cameraIndex;
            shared.linkRole = linkRole;
            shared.rxSource = rxSource;
            shared.sessionMode = sessionMode;
            shared.bandMode = bandMode;
            shared.linkRunning = linkRunning;
            shared.linkTxSlot = roleSend;
            shared.audioInputDevice = audioInputDevice;
            shared.audioOutputDevice = audioOutputDevice;
            shared.mediaPath = mediaPath;
            shared.linkStats = linkStats;
            shared.streamId = sessionConfig.streamId;
            shared.configVersion = sessionConfig.configVersion;
            shared.configHash = sessionConfig.configHash;
            shared.transportKind = transportKind;
            shared.transportStatus = transportStatus;
            shared.serialPort = serialPort;
            shared.serialBaud = serialBaud;
            shared.nodeAlias = nodeAlias;
            shared.relayExportPath = relayExportPath;
            shared.relayImportPath = relayImportPath;
            shared.authEnabled = authEnabled;
            shared.queueStats = queueStats;
            shared.messages = timeline;
            shared.latestTextCursor = latestTextCursor;
            shared.fallbackStage = fallbackStage;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }

    stopAllTransports();

    if (recordingFile.is_open()) {
        recordingFile.close();
    }

    server.stop();
    if (serverThread.joinable()) {
        serverThread.join();
    }

    std::cout << "Shutdown complete\n";
    return 0;
}
