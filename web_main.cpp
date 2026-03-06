#include <algorithm>
#include <array>
#include <atomic>
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
    LinkMode linkMode = LinkMode::LocalLoopback;
    RxSource rxSource = RxSource::LiveMic;
    SessionMode sessionMode = SessionMode::Broadcast;
    BandMode bandMode = BandMode::Audible;
    int requestedAudioInputDevice = -1;
    int requestedAudioOutputDevice = -1;
    bool linkRunning = false;
    std::string mediaPath;

    bool forceNextKeyframe = false;
    bool rescanCameras = false;
    bool rescanAudio = false;
    bool startLink = false;
    bool stopLink = false;
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

    LinkMode linkMode = LinkMode::LocalLoopback;
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
    oss << "\"link_mode\":\"" << linkModeName(state.linkMode) << "\",";
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
    oss << "\"stream_id\":" << state.streamId << ",";
    oss << "\"config_version\":" << state.configVersion << ",";
    oss << "\"config_hash\":" << state.configHash << ",";
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
  <title>Emergency Video Platform</title>
  <style>
    :root {
      --bg: #0f1512;
      --panel: #16201a;
      --panel-2: #1c2a22;
      --ink: #ecf6ef;
      --muted: #a7bdb0;
      --accent: #5af28c;
      --line: #2c3f34;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "SF Pro Text", "Segoe UI", system-ui, sans-serif;
      background: radial-gradient(1200px 600px at 70% -200px, #22362b 0%, var(--bg) 55%);
      color: var(--ink);
    }
    .shell {
      max-width: 1680px;
      margin: 0 auto;
      padding: 14px;
      display: grid;
      gap: 12px;
    }
    .top {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      display: grid;
      gap: 10px;
    }
    .top h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: .2px;
    }
    .row {
      display: grid;
      gap: 12px;
      grid-template-columns: 1.2fr .9fr;
      align-items: center;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .meter {
      display: grid;
      gap: 6px;
      justify-items: end;
      font-size: 12px;
      color: var(--muted);
    }
    progress {
      width: min(560px, 100%);
      height: 14px;
      appearance: none;
    }
    progress::-webkit-progress-bar { background: #101712; border-radius: 999px; }
    progress::-webkit-progress-value {
      background: linear-gradient(90deg, #39d778, var(--accent));
      border-radius: 999px;
    }
    .main {
      display: grid;
      gap: 12px;
      grid-template-columns: minmax(0, 2.25fr) minmax(320px, 1fr);
      align-items: start;
    }
    .feeds {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
    }
    .card .label {
      padding: 8px 10px;
      background: var(--panel-2);
      border-bottom: 1px solid var(--line);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .08em;
      color: #bce8ca;
    }
    .feedimg {
      width: 100%;
      display: block;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      background: #0b100d;
    }
    .controls {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      display: grid;
      gap: 10px;
      align-content: start;
      position: sticky;
      top: 10px;
      max-height: calc(100vh - 20px);
      overflow-y: auto;
      overflow-x: hidden;
    }
    .controls h2 {
      margin: 0;
      font-size: 16px;
    }
    .help {
      margin: 0;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.35;
    }
    .panel-section {
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
      display: grid;
      gap: 8px;
    }
    .section-title {
      margin: 0;
      font-size: 13px;
      font-weight: 700;
      color: #d8fbe5;
      letter-spacing: .01em;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px;
    }
    .field {
      display: grid;
      gap: 4px;
      min-width: 0;
    }
    .field.full {
      grid-column: 1 / -1;
    }
    .field label {
      font-size: 12px;
      color: var(--muted);
    }
    input, select, button {
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #101712;
      color: var(--ink);
      padding: 9px 10px;
      font-size: 13px;
      width: 100%;
      min-width: 0;
    }
    button {
      cursor: pointer;
      background: #1b2a21;
      border-color: #2f4639;
    }
    button:hover { border-color: #5c896f; }
    button.accent { background: #1d3a2b; border-color: #3e875f; color: #cbffe0; }
    .primary-actions {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .secondary-actions {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 8px;
    }
    .advanced {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #121a15;
      overflow: hidden;
    }
    .advanced summary {
      cursor: pointer;
      list-style: none;
      padding: 10px;
      font-size: 13px;
      font-weight: 700;
      color: #cdebd8;
      background: #17231c;
      border-bottom: 1px solid var(--line);
    }
    .advanced summary::-webkit-details-marker { display: none; }
    .advanced[open] summary { background: #1a2a20; }
    .advanced-inner {
      padding: 10px;
      display: grid;
      gap: 8px;
    }
    .stats {
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
      display: grid;
      gap: 5px;
      font-size: 12px;
      color: var(--muted);
    }
    .stats .strong { color: #dcffe7; font-weight: 600; }
    @media (max-width: 1300px) {
      .main { grid-template-columns: 1fr; }
      .feeds { grid-template-columns: 1fr; }
      .controls {
        position: static;
        max-height: none;
        overflow: visible;
      }
    }
    @media (max-width: 820px) {
      .row { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; }
      .primary-actions { grid-template-columns: 1fr; }
      .secondary-actions { grid-template-columns: 1fr; }
      .shell { padding: 10px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="top">
      <h1>Emergency Communication Platform</h1>
      <div class="row">
        <div class="status" id="statusText">Initializing...</div>
        <div class="meter">
          <progress id="bwBar" max="120" value="0"></progress>
          <div id="bwText">live 0.00 kbps | smooth 0.00 kbps | avg 0.00 kbps</div>
        </div>
      </div>
    </section>

    <section class="main">
      <div class="feeds">
        <article class="card">
          <div class="label">Raw Input</div>
          <img id="feedRaw" class="feedimg" alt="raw" />
        </article>
        <article class="card">
          <div class="label">Sent (Dithered)</div>
          <img id="feedSent" class="feedimg" alt="sent" />
        </article>
        <article class="card">
          <div class="label">Received (Reconstructed)</div>
          <img id="feedReceived" class="feedimg" alt="received" />
        </article>
      </div>

      <aside class="controls">
        <h2>Easy Controls</h2>
        <p class="help">Pick camera and quality, press Apply Settings, then Start Link. Open Advanced only if you need technical tuning.</p>

        <section class="panel-section">
          <h3 class="section-title">Basic Setup</h3>
          <div class="grid">
            <div class="field">
              <label for="cameraSelect">Camera</label>
              <select id="cameraSelect"></select>
            </div>
            <div class="field">
              <label for="modeSelect">Quality Mode</label>
              <select id="modeSelect">
                <option value="safer">Safer (Recommended)</option>
                <option value="aggressive">Low Bandwidth</option>
              </select>
            </div>
            <div class="field">
              <label for="resSelect">Resolution</label>
              <select id="resSelect">
                <option value="96x72">96x72 (Lowest Data)</option>
                <option value="128x96">128x96 (Balanced)</option>
                <option value="160x120">160x120 (Clearer)</option>
                <option value="192x144">192x144 (Highest)</option>
              </select>
            </div>
            <div class="field">
              <label for="fpsInput">Update Speed (FPS)</label>
              <input id="fpsInput" type="number" step="0.1" min="0.2" />
            </div>
          </div>
          <div class="primary-actions">
            <button id="applyBtn" class="accent">Apply Settings</button>
            <button id="startLinkBtn" class="accent">Start Link</button>
            <button id="stopLinkBtn">Stop Link</button>
          </div>
        </section>

        <section class="panel-section">
          <h3 class="section-title">Image Clarity</h3>
          <div class="secondary-actions">
            <button id="enhanceBtn">Enhancement: On</button>
            <button id="ditherBtn">Dithering: Off</button>
          </div>
        </section>

        <details class="advanced" id="advancedSettings">
          <summary>Advanced Technical Settings (Optional)</summary>
          <div class="advanced-inner">
            <div class="grid">
              <div class="field">
                <label for="linkModeSelect">Link Mode</label>
                <select id="linkModeSelect">
                  <option value="local_loopback">Local Test</option>
                  <option value="acoustic_tx">Send by Speaker</option>
                  <option value="acoustic_rx_live">Receive by Microphone</option>
                  <option value="acoustic_rx_media">Receive from Media File</option>
                  <option value="acoustic_duplex_arq">Two-way ARQ (Advanced)</option>
                </select>
              </div>
              <div class="field">
                <label for="rxSourceSelect">Receive Source</label>
                <select id="rxSourceSelect">
                  <option value="live_mic">Live Microphone</option>
                  <option value="media_file">Media File</option>
                </select>
              </div>
              <div class="field">
                <label for="sessionModeSelect">Session Mode</label>
                <select id="sessionModeSelect">
                  <option value="broadcast">Broadcast</option>
                  <option value="duplex_arq">Duplex ARQ</option>
                </select>
              </div>
              <div class="field">
                <label for="bandModeSelect">Audio Band</label>
                <select id="bandModeSelect">
                  <option value="audible">Audible</option>
                  <option value="ultrasonic">Ultrasonic</option>
                </select>
              </div>
              <div class="field">
                <label for="audioInSelect">Audio Input Device</label>
                <select id="audioInSelect"></select>
              </div>
              <div class="field">
                <label for="audioOutSelect">Audio Output Device</label>
                <select id="audioOutSelect"></select>
              </div>
              <div class="field full">
                <label for="mediaPathInput">Media File Path</label>
                <input id="mediaPathInput" type="text" placeholder="/path/to/file.mp4" />
              </div>
            </div>

            <div class="secondary-actions">
              <button id="keyBtn">Keyframe Interval: Default</button>
              <button id="forceKfBtn">Force Keyframe</button>
              <button id="recordBtn">Recording: Off</button>
              <button id="rescanBtn">Rescan Cameras/Audio</button>
            </div>
          </div>
        </details>

        <div class="stats">
          <div id="statFrame" class="strong">frame --</div>
          <div id="statCodec">codec --</div>
          <div id="statLink">link --</div>
          <div id="statComp">compression --</div>
          <div id="statQuality">quality --</div>
          <div id="statFaces">faces --</div>
        </div>
      </aside>
    </section>
  </div>

  <script>
    const stateStore = {
      latest: null,
      refreshFramesEveryMs: 160,
      refreshStateEveryMs: 600,
      formInitialized: false,
      formDirty: false,
      applyInFlight: false
    };

    function num(v, d = 0) {
      const n = Number(v);
      return Number.isFinite(n) ? n : d;
    }

    async function callControl(params) {
      const query = new URLSearchParams(params);
      query.set('_ts', String(Date.now()));
      await fetch('/api/control?' + query.toString(), { cache: 'no-store' });
    }

    function updateCameraOptions(cameras, current, forceCurrentSelection) {
      const sel = document.getElementById('cameraSelect');
      const existing = new Set(Array.from(sel.options).map(o => Number(o.value)));
      const next = Array.isArray(cameras) ? cameras.map(Number) : [];
      const previousValue = sel.value;

      for (const id of next) {
        if (!existing.has(id)) {
          const opt = document.createElement('option');
          opt.value = String(id);
          opt.textContent = `Camera ${id}`;
          sel.appendChild(opt);
        }
      }

      Array.from(sel.options).forEach(opt => {
        if (!next.includes(Number(opt.value))) {
          sel.removeChild(opt);
        }
      });

      if (forceCurrentSelection && current !== undefined && current !== null) {
        sel.value = String(current);
        return;
      }

      if (previousValue && next.includes(Number(previousValue))) {
        sel.value = previousValue;
      } else if (next.length > 0) {
        sel.value = String(next[0]);
      }
    }

    function updateAudioOptions(selectId, devices, current) {
      const sel = document.getElementById(selectId);
      const existing = new Set(Array.from(sel.options).map(o => Number(o.value)));
      const next = Array.isArray(devices) ? devices : [];
      const previousValue = sel.value;

      for (const dev of next) {
        const id = Number(dev.index);
        if (!existing.has(id)) {
          const opt = document.createElement('option');
          opt.value = String(id);
          opt.textContent = dev.default ? `${dev.name} (Default)` : dev.name;
          sel.appendChild(opt);
        }
      }

      Array.from(sel.options).forEach(opt => {
        if (!next.some(d => Number(d.index) === Number(opt.value))) {
          sel.removeChild(opt);
        }
      });

      if (current !== undefined && next.some(d => Number(d.index) === Number(current))) {
        sel.value = String(current);
      } else if (previousValue && next.some(d => Number(d.index) === Number(previousValue))) {
        sel.value = previousValue;
      } else if (next.length > 0) {
        sel.value = String(next[0].index);
      }
    }

    function renderState(data) {
      stateStore.latest = data;

      document.getElementById('statusText').textContent =
        `cam ${data.camera} | mode ${data.mode} | ${data.resolution} | fps target ${num(data.target_fps).toFixed(1)} | link ${data.link_mode} (${data.link_running ? 'on' : 'off'}) | ${data.status}`;

      const live = num(data.metrics?.live_kbps);
      const smooth = num(data.metrics?.smooth_kbps);
      const avg = num(data.metrics?.avg_kbps);
      const bwBar = document.getElementById('bwBar');
      bwBar.value = Math.max(0, Math.min(120, smooth));
      document.getElementById('bwText').textContent =
        `live ${live.toFixed(2)} kbps | smooth ${smooth.toFixed(2)} kbps | avg ${avg.toFixed(2)} kbps`;

      const shouldSyncForm = !stateStore.formDirty && !stateStore.applyInFlight;
      if (!stateStore.formInitialized || shouldSyncForm) {
        document.getElementById('modeSelect').value = data.mode;
        document.getElementById('resSelect').value = data.resolution;
        document.getElementById('fpsInput').value = Number(data.target_fps).toFixed(1);
        document.getElementById('linkModeSelect').value = data.link_mode;
        document.getElementById('rxSourceSelect').value = data.rx_source;
        document.getElementById('sessionModeSelect').value = data.session_mode;
        document.getElementById('bandModeSelect').value = data.band_mode;
        document.getElementById('mediaPathInput').value = data.media_path || '';
        updateCameraOptions(data.cameras || [], data.camera, true);
        updateAudioOptions('audioInSelect', data.audio_inputs || [], data.audio_in_device);
        updateAudioOptions('audioOutSelect', data.audio_outputs || [], data.audio_out_device);
        stateStore.formInitialized = true;
      } else {
        updateCameraOptions(data.cameras || [], data.camera, false);
        updateAudioOptions('audioInSelect', data.audio_inputs || [], data.audio_in_device);
        updateAudioOptions('audioOutSelect', data.audio_outputs || [], data.audio_out_device);
      }

      document.getElementById('keyBtn').textContent = `Keyframe Interval: ${data.short_keyframe ? 'Short' : 'Default'}`;
      document.getElementById('enhanceBtn').textContent = `Enhancement: ${data.enhance ? 'On' : 'Off'}`;
      document.getElementById('ditherBtn').textContent = `Dithering: ${data.dither ? 'On' : 'Off'}`;
      document.getElementById('recordBtn').textContent = `Recording: ${data.recording ? 'On' : 'Off'}`;
      document.getElementById('startLinkBtn').disabled = !!data.link_running;
      document.getElementById('stopLinkBtn').disabled = !data.link_running;

      const f = data.frame || {};
      document.getElementById('statFrame').textContent =
        `frame ${f.index ?? '--'} [${f.type ?? '-'}] bytes ${num(data.packet_bytes).toFixed(0)}`;
      document.getElementById('statCodec').textContent =
        `encoded fps ${num(data.metrics?.fps).toFixed(2)} | key-int ${num(data.keyframe_interval).toFixed(0)} | changed ${num(data.metrics?.changed_percent).toFixed(1)}%`;
      document.getElementById('statLink').textContent =
        `sync ${data.link_stats?.sync_locked ? 'locked' : 'searching'} | BER ${num(data.link_stats?.ber).toFixed(5)} | fec fix ${num(data.link_stats?.fec_recovered).toFixed(0)} | rtx ${num(data.link_stats?.retransmit_count).toFixed(0)} | slot ${data.link_tx_slot ? 'tx' : 'rx'}`;
      document.getElementById('statComp').textContent =
        `ratio raw4 ${num(data.metrics?.ratio_raw4).toFixed(2)}x | raw8 ${num(data.metrics?.ratio_raw8).toFixed(2)}x`;
      document.getElementById('statQuality').textContent =
        `PSNR ${num(data.metrics?.psnr).toFixed(2)} dB | keyframes ${num(data.metrics?.keyframe_percent).toFixed(1)}%`;
      document.getElementById('statFaces').textContent =
        `faces now ${num(data.faces_now).toFixed(0)} | gathered ${num(data.faces_gathered).toFixed(0)} | detector ${data.face_detector ? 'on' : 'missing'}`;
    }

    async function pollState() {
      try {
        const resp = await fetch('/api/state', { cache: 'no-store' });
        if (!resp.ok) return;
        const data = await resp.json();
        renderState(data);
      } catch (_) {
      }
    }

    function refreshFeeds() {
      const stamp = Date.now();
      document.getElementById('feedRaw').src = `/api/frame/raw.jpg?t=${stamp}`;
      document.getElementById('feedSent').src = `/api/frame/sent.jpg?t=${stamp}`;
      document.getElementById('feedReceived').src = `/api/frame/received.jpg?t=${stamp}`;
    }

    function setupEvents() {
      const markDirty = () => { stateStore.formDirty = true; };
      document.getElementById('cameraSelect').addEventListener('change', markDirty);
      document.getElementById('linkModeSelect').addEventListener('change', markDirty);
      document.getElementById('modeSelect').addEventListener('change', markDirty);
      document.getElementById('resSelect').addEventListener('change', markDirty);
      document.getElementById('rxSourceSelect').addEventListener('change', markDirty);
      document.getElementById('sessionModeSelect').addEventListener('change', markDirty);
      document.getElementById('bandModeSelect').addEventListener('change', markDirty);
      document.getElementById('audioInSelect').addEventListener('change', markDirty);
      document.getElementById('audioOutSelect').addEventListener('change', markDirty);
      document.getElementById('fpsInput').addEventListener('input', markDirty);
      document.getElementById('mediaPathInput').addEventListener('input', markDirty);

      document.getElementById('applyBtn').addEventListener('click', async () => {
        if (stateStore.applyInFlight) {
          return;
        }
        stateStore.applyInFlight = true;
        const applyBtn = document.getElementById('applyBtn');
        applyBtn.disabled = true;
        try {
          await callControl({
            mode: document.getElementById('modeSelect').value,
            resolution: document.getElementById('resSelect').value,
            target_fps: document.getElementById('fpsInput').value,
            camera: document.getElementById('cameraSelect').value,
            link_mode: document.getElementById('linkModeSelect').value,
            rx_source: document.getElementById('rxSourceSelect').value,
            session_mode: document.getElementById('sessionModeSelect').value,
            band_mode: document.getElementById('bandModeSelect').value,
            audio_in_device: document.getElementById('audioInSelect').value,
            audio_out_device: document.getElementById('audioOutSelect').value,
            media_path: document.getElementById('mediaPathInput').value
          });
          stateStore.formDirty = false;
          await pollState();
        } finally {
          stateStore.applyInFlight = false;
          applyBtn.disabled = false;
        }
      });

      document.getElementById('rescanBtn').addEventListener('click', async () => {
        await callControl({ rescan_cameras: '1', rescan_audio: '1' });
      });

      document.getElementById('startLinkBtn').addEventListener('click', async () => {
        await callControl({ start_link: '1' });
      });

      document.getElementById('stopLinkBtn').addEventListener('click', async () => {
        await callControl({ stop_link: '1' });
      });

      document.getElementById('forceKfBtn').addEventListener('click', async () => {
        await callControl({ force_keyframe: '1' });
      });

      document.getElementById('keyBtn').addEventListener('click', async () => {
        const next = !(stateStore.latest?.short_keyframe);
        await callControl({ short_keyframe: next ? '1' : '0' });
      });

      document.getElementById('enhanceBtn').addEventListener('click', async () => {
        const next = !(stateStore.latest?.enhance);
        await callControl({ enhance: next ? '1' : '0' });
      });

      document.getElementById('ditherBtn').addEventListener('click', async () => {
        const next = !(stateStore.latest?.dither);
        await callControl({ dither: next ? '1' : '0' });
      });

      document.getElementById('recordBtn').addEventListener('click', async () => {
        const next = !(stateStore.latest?.recording);
        await callControl({ recording: next ? '1' : '0' });
      });
    }

    setupEvents();
    pollState();
    refreshFeeds();
    setInterval(pollState, stateStore.refreshStateEveryMs);
    setInterval(refreshFeeds, stateStore.refreshFramesEveryMs);
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
    std::lock_guard<std::mutex> lock(control.mutex);
    const std::map<std::string, std::string> fallbackParams = queryMapFromTarget(req.target);

    auto getParam = [&](const char *name, std::string &valueOut) -> bool {
        if (req.has_param(name)) {
            valueOut = req.get_param_value(name);
            return true;
        }
        const auto it = fallbackParams.find(name);
        if (it != fallbackParams.end()) {
            valueOut = it->second;
            return true;
        }
        return false;
    };

    std::string value;

    if (getParam("mode", value)) {
        const std::string mode = value;
        if (mode == "safer") {
            control.mode = CodecMode::Safer;
        } else if (mode == "aggressive") {
            control.mode = CodecMode::Aggressive;
        }
    }

    if (getParam("target_fps", value)) {
        try {
            const double fps = std::stod(value);
            control.targetFps = std::max(kMinTargetFps, fps);
        } catch (const std::exception &) {
        }
    }

    if (getParam("resolution", value)) {
        int idx = control.resolutionIndex;
        if (parseResolutionString(value, idx)) {
            control.resolutionIndex = idx;
        }
    }

    if (getParam("resolution_index", value)) {
        try {
            const int idx = std::stoi(value);
            control.resolutionIndex = std::clamp(idx, 0, static_cast<int>(kResolutionLevels.size()) - 1);
        } catch (const std::exception &) {
        }
    }

    if (getParam("short_keyframe", value)) {
        control.shortKeyframeInterval = boolFromParam(value, control.shortKeyframeInterval);
    }

    if (getParam("enhance", value)) {
        control.useReceivedEnhancement = boolFromParam(value, control.useReceivedEnhancement);
    }

    if (getParam("dither", value)) {
        control.useReceivedDithering = boolFromParam(value, control.useReceivedDithering);
    }

    if (getParam("recording", value)) {
        control.recordingEnabled = boolFromParam(value, control.recordingEnabled);
    }

    if (getParam("camera", value)) {
        try {
            const int index = std::stoi(value);
            if (index >= 0 && index < 64) {
                control.requestedCameraIndex = index;
            }
        } catch (const std::exception &) {
        }
    }

    if (getParam("link_mode", value)) {
        control.linkMode = parseLinkMode(value, control.linkMode);
    }

    if (getParam("rx_source", value)) {
        control.rxSource = parseRxSource(value, control.rxSource);
    }

    if (getParam("session_mode", value)) {
        control.sessionMode = parseSessionMode(value, control.sessionMode);
    }

    if (getParam("band_mode", value)) {
        control.bandMode = parseBandMode(value, control.bandMode);
    }

    if (getParam("audio_in_device", value)) {
        try {
            control.requestedAudioInputDevice = std::stoi(value);
        } catch (const std::exception &) {
        }
    }

    if (getParam("audio_out_device", value)) {
        try {
            control.requestedAudioOutputDevice = std::stoi(value);
        } catch (const std::exception &) {
        }
    }

    if (getParam("media_path", value)) {
        control.mediaPath = value;
    }

    if (getParam("force_keyframe", value)) {
        if (boolFromParam(value, false)) {
            control.forceNextKeyframe = true;
        }
    }

    if (getParam("rescan_cameras", value)) {
        if (boolFromParam(value, false)) {
            control.rescanCameras = true;
        }
    }

    if (getParam("rescan_audio", value)) {
        if (boolFromParam(value, false)) {
            control.rescanAudio = true;
        }
    }

    if (getParam("start_link", value)) {
        if (boolFromParam(value, false)) {
            control.startLink = true;
            control.stopLink = false;
            control.linkRunning = true;
        }
    }

    if (getParam("stop_link", value)) {
        if (boolFromParam(value, false)) {
            control.stopLink = true;
            control.startLink = false;
            control.linkRunning = false;
        }
    }
}

void copyControlSnapshot(ControlState &control,
                         CodecMode &mode,
                         int &resolutionIndex,
                         double &targetFps,
                         bool &shortKey,
                         bool &enhance,
                         bool &dither,
                         bool &recording,
                         int &camera,
                         LinkMode &linkMode,
                         RxSource &rxSource,
                         SessionMode &sessionMode,
                         BandMode &bandMode,
                         int &audioInDevice,
                         int &audioOutDevice,
                         std::string &mediaPath,
                         bool &linkRunning,
                         bool &forceKey,
                         bool &rescanCamera,
                         bool &rescanAudio,
                         bool &startLink,
                         bool &stopLink) {
    std::lock_guard<std::mutex> lock(control.mutex);
    mode = control.mode;
    resolutionIndex = control.resolutionIndex;
    targetFps = control.targetFps;
    shortKey = control.shortKeyframeInterval;
    enhance = control.useReceivedEnhancement;
    dither = control.useReceivedDithering;
    recording = control.recordingEnabled;
    camera = control.requestedCameraIndex;
    linkMode = control.linkMode;
    rxSource = control.rxSource;
    sessionMode = control.sessionMode;
    bandMode = control.bandMode;
    audioInDevice = control.requestedAudioInputDevice;
    audioOutDevice = control.requestedAudioOutputDevice;
    mediaPath = control.mediaPath;
    linkRunning = control.linkRunning;
    forceKey = control.forceNextKeyframe;
    rescanCamera = control.rescanCameras;
    rescanAudio = control.rescanAudio;
    startLink = control.startLink;
    stopLink = control.stopLink;
}

void clearOneShotControlFlags(ControlState &control,
                              bool clearForceKey,
                              bool clearRescanCamera,
                              bool clearRescanAudio,
                              bool clearStartLink,
                              bool clearStopLink) {
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
}

void writeStateStatus(SharedState &shared, const std::string &status) {
    std::lock_guard<std::mutex> lock(shared.mutex);
    shared.status = status;
}

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

    LinkMode linkMode = LinkMode::LocalLoopback;
    RxSource rxSource = RxSource::LiveMic;
    SessionMode sessionMode = SessionMode::Broadcast;
    BandMode bandMode = BandMode::Audible;
    bool linkRunning = false;
    std::string mediaPath;

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

    auto modem = std::make_unique<MfskModem>(modemParamsFromSession(sessionConfig));
    auto burstReceiver = std::make_unique<AcousticBurstReceiver>(*modem);
    FragmentReassembler reassembler(std::chrono::milliseconds(4000));
    LinkStats linkStats;

    struct PendingPacket {
        uint32_t seq = 0;
        std::vector<uint8_t> bytes;
        std::chrono::steady_clock::time_point sentAt{};
        int retransmits = 0;
    };
    struct TxCodecPayload {
        std::vector<uint8_t> bytes;
        uint8_t flags = 0;
    };
    std::map<uint32_t, PendingPacket> pendingArq;
    std::deque<std::vector<uint8_t>> txRawFrameQueue;
    std::deque<TxCodecPayload> pendingTxPayloads;
    std::deque<AckPacket> pendingAcks;
    std::deque<std::pair<AcousticFrameHeader, std::vector<uint8_t>>> pendingDataForUnknownConfig;
    std::map<uint16_t, SessionConfig> knownRxConfigs;
    std::map<uint16_t, std::unique_ptr<Decoder>> rxDecoders;
    std::set<uint32_t> rxOutOfOrderSeq;
    uint32_t remoteStreamId = 0;
    uint32_t highestContiguousRxSeq = 0;
    uint64_t linkPayloadBytesReceived = 0;
    uint64_t demodRecoveredSymbols = 0;
    uint64_t demodTotalSymbols = 0;
    int startupConfigBurstRemaining = 0;
    bool configBeaconForced = true;
    uint32_t nextAcousticSeq = 1;
    uint32_t highestAckedSeq = 0;
    auto lastConfigBeacon = std::chrono::steady_clock::now();
    auto linkStartTime = std::chrono::steady_clock::now();
    uint64_t linkPayloadBytesSent = 0;
    std::string mediaLoadedPath;
    std::vector<float> mediaPcm;
    std::size_t mediaCursor = 0;
    auto lastMediaFeedTime = std::chrono::steady_clock::now();

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
        control.linkMode = linkMode;
        control.rxSource = rxSource;
        control.sessionMode = sessionMode;
        control.bandMode = bandMode;
        control.requestedAudioInputDevice = audioInputDevice;
        control.requestedAudioOutputDevice = audioOutputDevice;
        control.linkRunning = linkRunning;
        control.mediaPath = mediaPath;
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
        shared.linkMode = linkMode;
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
    }

    const std::string indexHtml = makeIndexHtml();

    httplib::Server server;
    server.Get("/", [&](const httplib::Request &, httplib::Response &res) {
        res.set_content(indexHtml, "text/html; charset=utf-8");
    });

    server.Get("/api/state", [&](const httplib::Request &, httplib::Response &res) {
        std::string payload;
        {
            std::lock_guard<std::mutex> lock(shared.mutex);
            payload = buildStateJson(shared);
        }
        res.set_content(payload, "application/json");
        res.set_header("Cache-Control", "no-store, no-cache, must-revalidate");
    });

    server.Get("/api/control", [&](const httplib::Request &req, httplib::Response &res) {
        applyControlRequest(req, control);
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

    server.Get("/api/frame/raw.jpg", [&](const httplib::Request &, httplib::Response &res) {
        serveFrame(&SharedState::rawJpeg, res);
    });
    server.Get("/api/frame/sent.jpg", [&](const httplib::Request &, httplib::Response &res) {
        serveFrame(&SharedState::sentJpeg, res);
    });
    server.Get("/api/frame/received.jpg", [&](const httplib::Request &, httplib::Response &res) {
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
        if (shortKeyframeInterval) {
            encoder.setKeyframeInterval(activeKeyframeInterval(params, true));
        }
        haveSentFrame = false;
        interpolation = InterpolationState{};
        haveStats = false;
        nextEncodeTime = Clock::now();
    };

    auto rebuildAcousticSession = [&](bool bumpVersion) {
        if (bumpVersion) {
            sessionConfig.configVersion = static_cast<uint16_t>(std::max<uint16_t>(1, sessionConfig.configVersion + 1));
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

        modem = std::make_unique<MfskModem>(modemParamsFromSession(sessionConfig));
        burstReceiver = std::make_unique<AcousticBurstReceiver>(*modem);
        reassembler.clear();
        pendingArq.clear();
        pendingAcks.clear();
        pendingDataForUnknownConfig.clear();
        rxOutOfOrderSeq.clear();
        txRawFrameQueue.clear();
        pendingTxPayloads.clear();
        remoteStreamId = 0;
        highestContiguousRxSeq = 0;
        nextAcousticSeq = 1;
        highestAckedSeq = 0;
        linkStats = LinkStats{};
        linkPayloadBytesSent = 0;
        linkPayloadBytesReceived = 0;
        demodRecoveredSymbols = 0;
        demodTotalSymbols = 0;
        linkStartTime = Clock::now();
        lastMediaFeedTime = Clock::now();
        lastConfigBeacon = Clock::now() - std::chrono::seconds(5);
        startupConfigBurstRemaining = 8;
        configBeaconForced = true;
        encoder.forceNextKeyframe();
    };

    int emptyFrames = 0;

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

    auto enqueueAcousticPayload = [&](AcousticPayloadType payloadType,
                                      uint32_t seq,
                                      const std::vector<uint8_t> &payload,
                                      uint8_t flags) {
        std::vector<std::vector<uint8_t>> fragments;
        if (payloadType == AcousticPayloadType::Data) {
            fragments = fragmentPayload(payload, 220);
        } else {
            fragments = {payload};
        }

        const uint16_t fragCount = static_cast<uint16_t>(std::min<std::size_t>(fragments.size(), 0xFFFFU));
        for (uint16_t frag = 0; frag < fragCount; ++frag) {
            AcousticFrameHeader header;
            header.payloadType = payloadType;
            header.flags = flags;
            header.streamId = sessionConfig.streamId;
            header.sessionEpochMs = sessionConfig.sessionEpochMs;
            header.configVersion = sessionConfig.configVersion;
            header.configHash = sessionConfig.configHash;
            header.seq = seq;
            header.fragIndex = frag;
            header.fragCount = std::max<uint16_t>(1, fragCount);

            txRawFrameQueue.push_back(
                serializeAcousticFrame(header, fragments[static_cast<std::size_t>(frag)]));
        }

        constexpr std::size_t kMaxQueuedRawFrames = 512;
        while (txRawFrameQueue.size() > kMaxQueuedRawFrames) {
            txRawFrameQueue.pop_front();
        }
    };

    auto handleIncomingAck = [&](const AckPacket &ack) {
        if (pendingArq.empty()) {
            return;
        }

        highestAckedSeq = std::max(highestAckedSeq, ack.ackSeq);
        const auto now = Clock::now();

        for (auto it = pendingArq.begin(); it != pendingArq.end();) {
            if (it->first <= ack.ackSeq) {
                const double rttMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                                         now - it->second.sentAt)
                                         .count();
                if (std::isfinite(rttMs) && rttMs > 0.0) {
                    if (linkStats.rttMs <= 0.0) {
                        linkStats.rttMs = rttMs;
                    } else {
                        linkStats.rttMs = linkStats.rttMs * 0.7 + rttMs * 0.3;
                    }
                }
                it = pendingArq.erase(it);
            } else {
                ++it;
            }
        }

        for (uint32_t selective : ack.selectiveAcks) {
            auto found = pendingArq.find(selective);
            if (found == pendingArq.end()) {
                continue;
            }

            const double rttMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                                     now - found->second.sentAt)
                                     .count();
            if (std::isfinite(rttMs) && rttMs > 0.0) {
                if (linkStats.rttMs <= 0.0) {
                    linkStats.rttMs = rttMs;
                } else {
                    linkStats.rttMs = linkStats.rttMs * 0.7 + rttMs * 0.3;
                }
            }

            pendingArq.erase(found);
        }
    };

    auto processReceivedDataPayload = [&](const AcousticFrameHeader &header,
                                          const std::vector<uint8_t> &codecPayload,
                                          const Clock::time_point &ts) {
        if (sessionMode == SessionMode::DuplexArq && header.seq <= highestContiguousRxSeq) {
            AckPacket ack;
            ack.ackSeq = highestContiguousRxSeq;
            ack.rttHintMs = static_cast<uint16_t>(std::clamp<double>(linkStats.rttMs, 0.0, 65000.0));
            pendingAcks.push_back(std::move(ack));
            return;
        }

        const auto cfgIt = knownRxConfigs.find(header.configVersion);
        if (cfgIt == knownRxConfigs.end() || cfgIt->second.configHash != header.configHash) {
            if (pendingDataForUnknownConfig.size() >= 64) {
                pendingDataForUnknownConfig.pop_front();
            }
            pendingDataForUnknownConfig.push_back({header, codecPayload});
            return;
        }

        auto &decoderPtr = rxDecoders[header.configVersion];
        if (!decoderPtr) {
            decoderPtr = std::make_unique<Decoder>();
            decoderPtr->reset();
        }

        DecodeResult decoded = decoderPtr->decode(codecPayload);
        if (!decoded.ok) {
            linkStats.framesDropped += 1;
            return;
        }

        updateInterpolationFrame(decoded.frame, ts);
        latestMeta = decoded.meta;
        latestPacketBytes = codecPayload.size();
        haveStats = true;
        linkPayloadBytesReceived += codecPayload.size();

        if (header.streamId != sessionConfig.streamId) {
            remoteStreamId = header.streamId;
        }

        if (sessionMode == SessionMode::DuplexArq) {
            if (header.seq == highestContiguousRxSeq + 1) {
                highestContiguousRxSeq = header.seq;
                while (rxOutOfOrderSeq.erase(highestContiguousRxSeq + 1) > 0) {
                    ++highestContiguousRxSeq;
                }
            } else if (header.seq > highestContiguousRxSeq + 1) {
                rxOutOfOrderSeq.insert(header.seq);
            }

            AckPacket ack;
            ack.ackSeq = highestContiguousRxSeq;
            ack.rttHintMs = static_cast<uint16_t>(std::clamp<double>(linkStats.rttMs, 0.0, 65000.0));
            for (uint32_t seq : rxOutOfOrderSeq) {
                ack.selectiveAcks.push_back(seq);
                if (ack.selectiveAcks.size() >= 16) {
                    break;
                }
            }
            pendingAcks.push_back(std::move(ack));
        }
    };

    auto processAcousticWireFrame = [&](const std::vector<uint8_t> &wireFrame, const Clock::time_point &ts) {
        AcousticFrameHeader header;
        std::vector<uint8_t> payload;
        std::string error;
        if (!deserializeAcousticFrame(wireFrame, header, payload, error)) {
            linkStats.framesDropped += 1;
            return;
        }

        linkStats.framesReceived += 1;
        linkStats.syncLocked = true;

        if (header.streamId == sessionConfig.streamId &&
            (linkMode == LinkMode::AcousticTx || linkMode == LinkMode::AcousticDuplexArq)) {
            // Ignore self-looped speaker->mic feedback in local TX roles.
            return;
        }

        if (header.payloadType == AcousticPayloadType::Config) {
            SessionConfig incoming;
            if (!deserializeSessionConfig(payload, incoming, error)) {
                linkStats.framesDropped += 1;
                return;
            }

            knownRxConfigs[incoming.configVersion] = incoming;
            auto decoderPtr = std::make_unique<Decoder>();
            decoderPtr->reset();
            rxDecoders[incoming.configVersion] = std::move(decoderPtr);

            if (knownRxConfigs.size() > 8) {
                knownRxConfigs.erase(knownRxConfigs.begin());
            }
            while (rxDecoders.size() > 8) {
                rxDecoders.erase(rxDecoders.begin());
            }

            remoteStreamId = incoming.streamId;
            linkStats.syncLocked = true;

            for (auto it = pendingDataForUnknownConfig.begin(); it != pendingDataForUnknownConfig.end();) {
                if (it->first.configVersion == incoming.configVersion && it->first.configHash == incoming.configHash) {
                    processReceivedDataPayload(it->first, it->second, ts);
                    it = pendingDataForUnknownConfig.erase(it);
                } else {
                    ++it;
                }
            }

            return;
        }

        if (header.payloadType == AcousticPayloadType::Ack) {
            AckPacket ack;
            if (!deserializeAckPacket(payload, ack, error)) {
                linkStats.framesDropped += 1;
                return;
            }
            handleIncomingAck(ack);
            return;
        }

        reassembler.push(header, payload);
        uint32_t seq = 0;
        std::vector<uint8_t> assembled;
        while (reassembler.popComplete(seq, assembled)) {
            AcousticFrameHeader completeHeader = header;
            completeHeader.seq = seq;
            processReceivedDataPayload(completeHeader, assembled, ts);
        }
    };

    while (!gStop.load()) {
        CodecMode desiredMode = mode;
        int desiredResolutionIndex = resolutionIndex;
        double desiredFps = targetFps;
        bool desiredShortKey = shortKeyframeInterval;
        bool desiredEnhance = useReceivedEnhancement;
        bool desiredDither = useReceivedDithering;
        bool desiredRecording = recordingEnabled;
        int desiredCameraIndex = cameraIndex;
        LinkMode desiredLinkMode = linkMode;
        RxSource desiredRxSource = rxSource;
        SessionMode desiredSessionMode = sessionMode;
        BandMode desiredBandMode = bandMode;
        int desiredAudioInputDevice = audioInputDevice;
        int desiredAudioOutputDevice = audioOutputDevice;
        std::string desiredMediaPath = mediaPath;
        bool desiredLinkRunning = linkRunning;
        bool requestForceKey = false;
        bool requestRescanCamera = false;
        bool requestRescanAudio = false;
        bool requestStartLink = false;
        bool requestStopLink = false;

        copyControlSnapshot(control,
                            desiredMode,
                            desiredResolutionIndex,
                            desiredFps,
                            desiredShortKey,
                            desiredEnhance,
                            desiredDither,
                            desiredRecording,
                            desiredCameraIndex,
                            desiredLinkMode,
                            desiredRxSource,
                            desiredSessionMode,
                            desiredBandMode,
                            desiredAudioInputDevice,
                            desiredAudioOutputDevice,
                            desiredMediaPath,
                            desiredLinkRunning,
                            requestForceKey,
                            requestRescanCamera,
                            requestRescanAudio,
                            requestStartLink,
                            requestStopLink);

        if (requestRescanCamera) {
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
            clearOneShotControlFlags(control, false, true, false, false, false);
        }

        if (requestRescanAudio) {
            audioInputs = audioEngine.listInputDevices();
            audioOutputs = audioEngine.listOutputDevices();
            {
                std::lock_guard<std::mutex> lock(shared.mutex);
                shared.audioInputs = audioInputs;
                shared.audioOutputs = audioOutputs;
                shared.status = "audio list refreshed";
            }
            clearOneShotControlFlags(control, false, false, true, false, false);
        }

        if (requestForceKey) {
            encoder.forceNextKeyframe();
            clearOneShotControlFlags(control, true, false, false, false, false);
        }

        desiredResolutionIndex = std::clamp(desiredResolutionIndex, 0, static_cast<int>(kResolutionLevels.size()) - 1);
        desiredFps = std::max(kMinTargetFps, desiredFps);

        if (desiredCameraIndex != cameraIndex) {
            if (switchCamera(camera, desiredCameraIndex)) {
                cameraIndex = desiredCameraIndex;
                emptyFrames = 0;
                writeStateStatus(shared, "camera switched to index " + std::to_string(cameraIndex));
                std::cout << "Camera switched to index " << cameraIndex << '\n';
            } else {
                writeStateStatus(shared, "camera switch failed for index " + std::to_string(desiredCameraIndex));
            }
        }

        if (desiredMode != mode || desiredResolutionIndex != resolutionIndex) {
            mode = desiredMode;
            resolutionIndex = desiredResolutionIndex;
            params = makeRuntimeParams(mode, resolutionIndex, desiredFps);
            targetFps = desiredFps;
            shortKeyframeInterval = desiredShortKey;
            resetCodecPipeline();
            rebuildAcousticSession(true);
            writeStateStatus(shared, "codec profile updated");
        } else if (std::abs(desiredFps - targetFps) > 1e-6) {
            targetFps = desiredFps;
            params.targetFps = targetFps;
            rebuildAcousticSession(true);
            nextEncodeTime = Clock::now();
        }

        if (desiredShortKey != shortKeyframeInterval) {
            shortKeyframeInterval = desiredShortKey;
            encoder.setKeyframeInterval(activeKeyframeInterval(params, shortKeyframeInterval));
            writeStateStatus(shared,
                             std::string("keyframe interval switched to ") +
                                 std::to_string(activeKeyframeInterval(params, shortKeyframeInterval)));
            rebuildAcousticSession(true);
        }

        useReceivedEnhancement = desiredEnhance;
        useReceivedDithering = desiredDither;

        const bool previousLinkRunning = linkRunning;
        bool acousticConfigChanged = false;
        if (desiredLinkMode != linkMode) {
            linkMode = desiredLinkMode;
            acousticConfigChanged = true;
        }
        if (desiredRxSource != rxSource) {
            rxSource = desiredRxSource;
            acousticConfigChanged = true;
        }
        if (desiredSessionMode != sessionMode) {
            sessionMode = desiredSessionMode;
            acousticConfigChanged = true;
        }
        if (desiredBandMode != bandMode) {
            bandMode = desiredBandMode;
            acousticConfigChanged = true;
        }
        if (desiredAudioInputDevice != audioInputDevice) {
            audioInputDevice = desiredAudioInputDevice;
            acousticConfigChanged = true;
        }
        if (desiredAudioOutputDevice != audioOutputDevice) {
            audioOutputDevice = desiredAudioOutputDevice;
            acousticConfigChanged = true;
        }
        if (desiredMediaPath != mediaPath) {
            mediaPath = desiredMediaPath;
            acousticConfigChanged = true;
            mediaLoadedPath.clear();
            mediaPcm.clear();
            mediaCursor = 0;
        }
        if (desiredLinkRunning != linkRunning) {
            linkRunning = desiredLinkRunning;
        }
        if (requestStartLink) {
            linkRunning = true;
            clearOneShotControlFlags(control, false, false, false, true, false);
        }
        if (requestStopLink) {
            linkRunning = false;
            clearOneShotControlFlags(control, false, false, false, false, true);
        }

        if (acousticConfigChanged) {
            rebuildAcousticSession(true);
        }

        if (!previousLinkRunning && linkRunning) {
            startupConfigBurstRemaining = 8;
            configBeaconForced = true;
            linkStartTime = Clock::now();
            lastConfigBeacon = Clock::now() - std::chrono::seconds(5);
            txRawFrameQueue.clear();
            pendingTxPayloads.clear();
            pendingArq.clear();
            pendingAcks.clear();
            audioEngine.clearCaptureBuffer();
            audioEngine.clearPlaybackBuffer();
        } else if (previousLinkRunning && !linkRunning) {
            txRawFrameQueue.clear();
            pendingTxPayloads.clear();
            pendingArq.clear();
            pendingAcks.clear();
            reassembler.clear();
            audioEngine.clearCaptureBuffer();
            audioEngine.clearPlaybackBuffer();
            linkStats.syncLocked = false;
        }

        const bool captureNeeded = linkRunning &&
                                   ((linkMode == LinkMode::AcousticRxLive && rxSource == RxSource::LiveMic) ||
                                    linkMode == LinkMode::AcousticDuplexArq ||
                                    (linkMode == LinkMode::AcousticTx && sessionMode == SessionMode::DuplexArq));
        const bool playbackNeeded = linkRunning &&
                                    (linkMode == LinkMode::AcousticTx || linkMode == LinkMode::AcousticDuplexArq ||
                                     ((linkMode == LinkMode::AcousticRxLive || linkMode == LinkMode::AcousticRxMedia) &&
                                      sessionMode == SessionMode::DuplexArq));

        if (captureNeeded && !audioEngine.captureRunning()) {
            if (!audioEngine.startCapture(audioInputDevice, sessionConfig.sampleRate)) {
                writeStateStatus(shared, "failed to start audio capture");
            } else {
                audioEngine.clearCaptureBuffer();
            }
        } else if (!captureNeeded && audioEngine.captureRunning()) {
            audioEngine.stopCapture();
            audioEngine.clearCaptureBuffer();
        }

        if (playbackNeeded && !audioEngine.playbackRunning()) {
            if (!audioEngine.startPlayback(audioOutputDevice, sessionConfig.sampleRate)) {
                writeStateStatus(shared, "failed to start audio playback");
            } else {
                audioEngine.clearPlaybackBuffer();
            }
        } else if (!playbackNeeded && audioEngine.playbackRunning()) {
            audioEngine.stopPlayback();
            audioEngine.clearPlaybackBuffer();
        }

        if (desiredRecording != recordingEnabled) {
            if (desiredRecording) {
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
        const bool txVideoEnabled = (linkMode == LinkMode::LocalLoopback || linkMode == LinkMode::AcousticTx ||
                                     linkMode == LinkMode::AcousticDuplexArq);
        const bool rxPathEnabled =
            linkRunning &&
            (linkMode == LinkMode::AcousticRxLive || linkMode == LinkMode::AcousticRxMedia ||
             linkMode == LinkMode::AcousticDuplexArq ||
             (linkMode == LinkMode::AcousticTx && sessionMode == SessionMode::DuplexArq));
        const bool txControlEnabled =
            linkRunning &&
            (linkMode == LinkMode::AcousticTx || linkMode == LinkMode::AcousticDuplexArq ||
             ((linkMode == LinkMode::AcousticRxLive || linkMode == LinkMode::AcousticRxMedia) &&
              sessionMode == SessionMode::DuplexArq));

        const bool senderRole =
            (remoteStreamId == 0) ? ((sessionConfig.streamId & 1U) == 0U) : (sessionConfig.streamId < remoteStreamId);
        const bool txSlot =
            sessionMode == SessionMode::DuplexArq ? isTxSlotNow(sessionConfig, now, senderRole) : true;

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

            if (txVideoEnabled) {
                EncodedPacket packet;
                try {
                    packet = encoder.encode(inputFrame, roiBlocks.empty() ? nullptr : &roiBlocks);
                } catch (const std::exception &ex) {
                    writeStateStatus(shared, std::string("encode failure: ") + ex.what());
                    std::this_thread::sleep_for(std::chrono::milliseconds(40));
                    continue;
                }
                const std::size_t packetBytes = packet.bytes.size();

                if (recordingEnabled) {
                    if (!appendRecordingPacket(recordingFile, packet.bytes)) {
                        writeStateStatus(shared, "recording write failed; recording disabled");
                        recordingEnabled = false;
                        recordingFile.close();
                        {
                            std::lock_guard<std::mutex> lock(control.mutex);
                            control.recordingEnabled = false;
                        }
                    }
                }

                DecodeResult localPreview = decoder.decode(packet.bytes);
                if (!localPreview.ok) {
                    writeStateStatus(shared, std::string("decode failure: ") + localPreview.error);
                    if (linkMode == LinkMode::LocalLoopback) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(40));
                        continue;
                    }
                } else {
                    metrics.update(packet.meta, packetBytes, inputFrame, localPreview.frame);
                    latestMetrics = metrics.snapshot();
                    latestMeta = packet.meta;
                    latestPacketBytes = packetBytes;
                    haveStats = true;

                    if (linkMode == LinkMode::LocalLoopback || linkMode == LinkMode::AcousticTx || remoteStreamId == 0) {
                        updateInterpolationFrame(localPreview.frame, now);
                    }
                }

                if (linkRunning && (linkMode == LinkMode::AcousticTx || linkMode == LinkMode::AcousticDuplexArq)) {
                    TxCodecPayload txPayload;
                    txPayload.bytes = std::move(packet.bytes);
                    txPayload.flags =
                        (packet.meta.frameType == FrameType::Keyframe) ? static_cast<uint8_t>(0x1U) : static_cast<uint8_t>(0U);
                    pendingTxPayloads.push_back(std::move(txPayload));
                    while (pendingTxPayloads.size() > 32) {
                        pendingTxPayloads.pop_front();
                    }
                }

                if (std::chrono::duration_cast<std::chrono::seconds>(now - lastConsolePrint).count() >= 1) {
                    std::cout << "mode=" << codecModeName(params.mode) << " cam=" << cameraIndex
                              << " frame=" << packet.meta.frameIndex << " bytes=" << packetBytes
                              << " live=" << formatDouble(latestMetrics.liveBitrateKbps)
                              << "kbps smooth=" << formatDouble(latestMetrics.smoothedBitrateKbps)
                              << "kbps avg=" << formatDouble(latestMetrics.averageBitrateKbps)
                              << "kbps faces=" << lastFacesRaw.size()
                              << " txq=" << pendingTxPayloads.size() << " arq=" << pendingArq.size() << '\n';
                    lastConsolePrint = now;
                }
            }

            const int encodePeriodMs =
                std::max(1, static_cast<int>(std::lround(1000.0 / std::max(0.1, params.targetFps))));
            nextEncodeTime = now + std::chrono::milliseconds(encodePeriodMs);
        }

        if (linkRunning &&
            (linkMode == LinkMode::AcousticRxMedia ||
             (linkMode == LinkMode::AcousticRxLive && rxSource == RxSource::MediaFile))) {
            if (mediaPath.empty()) {
                writeStateStatus(shared, "set media path for acoustic_rx_media");
            } else if (mediaLoadedPath != mediaPath) {
                std::string ffmpegError;
                std::vector<float> decodedPcm;
                if (!decodeMediaAudioToMonoF32(mediaPath, sessionConfig.sampleRate, decodedPcm, ffmpegError)) {
                    writeStateStatus(shared, std::string("media decode failed: ") + ffmpegError);
                    mediaLoadedPath.clear();
                    mediaPcm.clear();
                    mediaCursor = 0;
                } else {
                    mediaLoadedPath = mediaPath;
                    mediaPcm = std::move(decodedPcm);
                    mediaCursor = 0;
                    burstReceiver->clear();
                    lastMediaFeedTime = now;
                    writeStateStatus(shared, "media loaded for acoustic RX");
                }
            }

            if (!mediaPcm.empty()) {
                const auto feedStep = std::chrono::milliseconds(35);
                while (lastMediaFeedTime + feedStep <= now) {
                    const std::size_t feedCount = std::max<std::size_t>(
                        64, static_cast<std::size_t>((sessionConfig.sampleRate * feedStep.count()) / 1000));
                    std::vector<float> chunk;
                    chunk.reserve(feedCount);
                    for (std::size_t i = 0; i < feedCount; ++i) {
                        if (mediaCursor >= mediaPcm.size()) {
                            mediaCursor = 0;
                        }
                        chunk.push_back(mediaPcm[mediaCursor++]);
                    }
                    burstReceiver->feedSamples(chunk.data(), chunk.size());
                    lastMediaFeedTime += feedStep;
                }
            }
        }

        if (audioEngine.captureRunning() && rxPathEnabled) {
            std::vector<float> captured;
            audioEngine.popCaptured(captured, static_cast<std::size_t>(sessionConfig.sampleRate / 4U));
            if (!captured.empty()) {
                burstReceiver->feedSamples(captured.data(), captured.size());
            }
        }

        while (true) {
            std::vector<uint8_t> rawFrame;
            std::size_t recoveredSymbols = 0;
            if (!burstReceiver->popFrame(rawFrame, &recoveredSymbols)) {
                break;
            }
            linkStats.fecRecoveredCount += recoveredSymbols;
            demodRecoveredSymbols += recoveredSymbols;
            demodTotalSymbols += rawFrame.size() * 8U;
            processAcousticWireFrame(rawFrame, now);
        }

        if (txControlEnabled && txSlot) {
            const bool periodicBeacon =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - lastConfigBeacon).count() >= 2500;
            if (startupConfigBurstRemaining > 0 || configBeaconForced || periodicBeacon) {
                const std::vector<uint8_t> cfgBytes = serializeSessionConfig(sessionConfig);
                enqueueAcousticPayload(AcousticPayloadType::Config, nextAcousticSeq++, cfgBytes, 0x1U);
                lastConfigBeacon = now;
                if (startupConfigBurstRemaining > 0) {
                    --startupConfigBurstRemaining;
                }
                if (startupConfigBurstRemaining <= 0) {
                    configBeaconForced = false;
                }
            }

            if (sessionMode == SessionMode::DuplexArq) {
                int ackBudget = 4;
                while (!pendingAcks.empty() && ackBudget-- > 0) {
                    const std::vector<uint8_t> ackBytes = serializeAckPacket(pendingAcks.front());
                    enqueueAcousticPayload(AcousticPayloadType::Ack, nextAcousticSeq++, ackBytes, 0U);
                    pendingAcks.pop_front();
                }

                std::vector<uint32_t> dropSeq;
                int retransmitBudget = 3;
                for (auto &entry : pendingArq) {
                    PendingPacket &pending = entry.second;
                    const auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - pending.sentAt).count();
                    if (age < sessionConfig.arqTimeoutMs) {
                        continue;
                    }
                    if (pending.retransmits >= sessionConfig.arqMaxRetransmit) {
                        dropSeq.push_back(entry.first);
                        continue;
                    }
                    if (retransmitBudget <= 0) {
                        break;
                    }
                    enqueueAcousticPayload(AcousticPayloadType::Data, pending.seq, pending.bytes, 0x2U);
                    pending.sentAt = now;
                    pending.retransmits += 1;
                    linkStats.retransmitCount += 1;
                    --retransmitBudget;
                }
                for (uint32_t seq : dropSeq) {
                    pendingArq.erase(seq);
                    linkStats.framesDropped += 1;
                }
            }

            while (!pendingTxPayloads.empty() &&
                   (sessionMode != SessionMode::DuplexArq ||
                    pendingArq.size() < std::max<std::size_t>(1, sessionConfig.arqWindow))) {
                TxCodecPayload txPayload = std::move(pendingTxPayloads.front());
                pendingTxPayloads.pop_front();

                const uint32_t seq = nextAcousticSeq++;
                enqueueAcousticPayload(AcousticPayloadType::Data, seq, txPayload.bytes, txPayload.flags);

                if (sessionMode == SessionMode::DuplexArq) {
                    PendingPacket pending;
                    pending.seq = seq;
                    pending.bytes = txPayload.bytes;
                    pending.sentAt = now;
                    pending.retransmits = 0;
                    pendingArq[seq] = std::move(pending);
                }

                linkPayloadBytesSent += txPayload.bytes.size();
            }
        }

        if (txControlEnabled && txSlot && audioEngine.playbackRunning()) {
            int burstBudget = 2;
            while (!txRawFrameQueue.empty() && burstBudget-- > 0) {
                const std::vector<float> pcm = modem->modulateFrame(
                    txRawFrameQueue.front(), sessionConfig.fecRepetition, sessionConfig.interleaveDepth);
                audioEngine.pushPlayback(pcm);
                txRawFrameQueue.pop_front();
            }
        }

        if (demodTotalSymbols > 0) {
            linkStats.berEstimate =
                static_cast<double>(demodRecoveredSymbols) / static_cast<double>(std::max<uint64_t>(1, demodTotalSymbols));
        }
        const double linkElapsed = std::max(
            0.001, std::chrono::duration_cast<std::chrono::duration<double>>(now - linkStartTime).count());
        linkStats.effectivePayloadKbps =
            (static_cast<double>(linkPayloadBytesSent + linkPayloadBytesReceived) * 8.0) / (linkElapsed * 1000.0);

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

        cv::Mat receivedPanel(panelHeight, panelWidth, CV_8UC3, cv::Scalar(20, 20, 20));
        if (interpolation.hasCurrent) {
            Gray4Frame displayFrame = interpolation.current;
            if (interpolation.hasPair) {
                const double interval = 1.0 / std::max(0.1, params.targetFps);
                const double alpha = std::clamp(std::chrono::duration_cast<std::chrono::duration<double>>(now -
                                                                                                           interpolation
                                                                                                               .blendStart)
                                                    .count() /
                                                    std::max(0.001, interval),
                                                0.0,
                                                1.0);
                displayFrame = interpolateMotionCompensated(
                    interpolation.previous, interpolation.current, interpolation.motion, params.blockSize, alpha);
            }

            receivedPanel =
                renderForDisplay(displayFrame, panelSize, useReceivedEnhancement, useReceivedDithering, params.blockSize);
            drawFaceRects(receivedPanel, rawFacesPanel, cv::Scalar(150, 255, 100));
        }

        std::vector<uint8_t> rawJpeg = encodeJpeg(rawPanel, 82);
        std::vector<uint8_t> sentJpeg = encodeJpeg(sentPanel, 82);
        std::vector<uint8_t> receivedJpeg = encodeJpeg(receivedPanel, 82);

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
            shared.linkMode = linkMode;
            shared.rxSource = rxSource;
            shared.sessionMode = sessionMode;
            shared.bandMode = bandMode;
            shared.linkRunning = linkRunning;
            shared.linkTxSlot = txSlot;
            shared.audioInputDevice = audioInputDevice;
            shared.audioOutputDevice = audioOutputDevice;
            shared.mediaPath = mediaPath;
            shared.linkStats = linkStats;
            shared.streamId = sessionConfig.streamId;
            shared.configVersion = sessionConfig.configVersion;
            shared.configHash = sessionConfig.configHash;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }

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
