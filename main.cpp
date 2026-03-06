#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>

#include "codec.h"
#include "decoder.h"
#include "encoder.h"
#include "metrics.h"
#include "util.h"

namespace {
constexpr const char *kDashboardWindow = "Emergency Platform";
constexpr int kDefaultWindowWidth = 1680;
constexpr int kDefaultWindowHeight = 920;
constexpr int kHudHeight = 230;
constexpr int kControlPanelWidth = 360;

const std::vector<cv::Size> kResolutionLevels = {
    {96, 72},
    {128, 96},
    {160, 120},
    {192, 144},
};

constexpr double kMinTargetFps = 0.2;
constexpr double kFpsStep = 0.5;

struct UiButton {
    std::string id;
    std::string label;
    cv::Rect rect;
};

struct UiMouseState {
    bool pendingClick = false;
    cv::Point clickPoint{0, 0};
};

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

std::string formatDouble(double value, int precision = 2) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
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

int preferredCameraIndex() {
    if (const char *env = std::getenv("EV_CAMERA_INDEX")) {
        char *end = nullptr;
        const long value = std::strtol(env, &end, 10);
        if (end != env && value >= 0 && value <= 63) {
            return static_cast<int>(value);
        }
    }
    // Prefer index 1 by default for this deployment.
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

bool openPreferredCamera(cv::VideoCapture &camera, int preferredIndex, int &openedIndex) {
    std::vector<int> candidates;
    candidates.push_back(preferredIndex);
    for (int idx = 0; idx < 8; ++idx) {
        if (idx != preferredIndex) {
            candidates.push_back(idx);
        }
    }

    for (int index : candidates) {
        cv::VideoCapture probe;
        if (!openCaptureByIndex(probe, index)) {
            continue;
        }

        cv::Mat frame;
        probe >> frame;
        if (frame.empty()) {
            probe.release();
            continue;
        }

        camera = std::move(probe);
        openedIndex = index;
        return true;
    }

    openedIndex = -1;
    return false;
}

std::vector<UiButton> buildControlButtons(const cv::Rect &panelRect) {
    if (panelRect.width < 140 || panelRect.height < 110) {
        return {};
    }

    constexpr int margin = 12;
    constexpr int gap = 10;
    constexpr int titleHeight = 24;

    const std::array<std::pair<const char *, const char *>, 4> defs = {
        std::pair{"fps_down", "FPS -"},
        std::pair{"fps_up", "FPS +"},
        std::pair{"res_down", "RES -"},
        std::pair{"res_up", "RES +"},
    };

    const int innerX = panelRect.x + margin;
    const int innerY = panelRect.y + margin + titleHeight;
    const int innerW = panelRect.width - margin * 2;
    const int innerH = panelRect.height - margin * 2 - titleHeight;

    const int buttonWidth = std::max(64, (innerW - gap) / 2);
    const int buttonHeight = std::max(30, (innerH - gap) / 2);

    std::vector<UiButton> buttons;
    buttons.reserve(defs.size());

    for (int i = 0; i < static_cast<int>(defs.size()); ++i) {
        const int col = i % 2;
        const int row = i / 2;
        UiButton button;
        button.id = defs[static_cast<std::size_t>(i)].first;
        button.label = defs[static_cast<std::size_t>(i)].second;
        button.rect = cv::Rect(innerX + col * (buttonWidth + gap),
                               innerY + row * (buttonHeight + gap),
                               buttonWidth,
                               buttonHeight);
        buttons.push_back(std::move(button));
    }

    return buttons;
}

void drawControlButtons(cv::Mat &image, const std::vector<UiButton> &buttons) {
    for (const UiButton &button : buttons) {
        cv::rectangle(image, button.rect, cv::Scalar(22, 22, 22), cv::FILLED);
        cv::rectangle(image, button.rect, cv::Scalar(100, 230, 110), 2);

        int baseline = 0;
        const cv::Size textSize = cv::getTextSize(button.label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
        const cv::Point textPos(button.rect.x + (button.rect.width - textSize.width) / 2,
                                button.rect.y + (button.rect.height + textSize.height) / 2 - 3);

        cv::putText(image, button.label, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 3,
                    cv::LINE_AA);
        cv::putText(image, button.label, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(120, 255, 120), 1,
                    cv::LINE_AA);
    }
}

std::string hitTestButton(const std::vector<UiButton> &buttons, const cv::Point &point) {
    for (const UiButton &button : buttons) {
        if (button.rect.contains(point)) {
            return button.id;
        }
    }
    return {};
}

void onDashboardMouse(int event, int x, int y, int /*flags*/, void *userdata) {
    if (event != cv::EVENT_LBUTTONUP || userdata == nullptr) {
        return;
    }

    auto *state = static_cast<UiMouseState *>(userdata);
    state->pendingClick = true;
    state->clickPoint = cv::Point(x, y);
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
        // Optional path probe only.
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

void drawPanelLabel(cv::Mat &image, const std::string &title) {
    cv::rectangle(image, cv::Rect(0, 0, image.cols, 32), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(image, title, cv::Point(10, 22), cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(image, title, cv::Point(10, 22), cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(110, 255, 110), 1,
                cv::LINE_AA);
}

void drawBandwidthMeter(cv::Mat &canvas, cv::Point topLeft, int width, int height, double liveKbps, double smoothKbps) {
    const cv::Rect bar(topLeft.x, topLeft.y, width, height);
    cv::rectangle(canvas, bar, cv::Scalar(40, 40, 40), cv::FILLED);
    cv::rectangle(canvas, bar, cv::Scalar(95, 220, 95), 1);

    constexpr double kDisplayMaxKbps = 120.0;
    const double ratio = std::clamp(smoothKbps / kDisplayMaxKbps, 0.0, 1.0);
    const int fill = static_cast<int>(std::lround(ratio * static_cast<double>(width)));
    if (fill > 0) {
        cv::rectangle(canvas, cv::Rect(topLeft.x, topLeft.y, fill, height), cv::Scalar(70, 200, 90), cv::FILLED);
    }

    const std::string label = "Bandwidth live " + formatDouble(liveKbps, 2) +
                              " kbps | smooth " + formatDouble(smoothKbps, 2) + " kbps";
    cv::putText(canvas, label, cv::Point(topLeft.x, topLeft.y - 8), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(canvas, label, cv::Point(topLeft.x, topLeft.y - 8), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(120, 255, 120), 1, cv::LINE_AA);
}

std::vector<std::string> buildHudLines(const CodecParams &params,
                                       const MetricsSnapshot &metrics,
                                       const EncodeMetadata &meta,
                                       std::size_t bytes,
                                       bool haveStats,
                                       bool dither,
                                       bool recording,
                                       int facesNow,
                                       std::size_t facesGathered,
                                       bool faceDetectorReady,
                                       int keyInterval,
                                       int cameraIndex) {
    std::vector<std::string> lines;

    lines.push_back("cam " + std::to_string(cameraIndex) + " | mode " + std::string(codecModeName(params.mode)) +
                    " | " + std::to_string(params.width) + "x" + std::to_string(params.height) + " | target fps " +
                    formatDouble(params.targetFps, 1));

    if (haveStats) {
        const std::string frameType = (meta.frameType == FrameType::Keyframe) ? "K" : "P";
        const double changedPercent =
            meta.totalBlocks > 0 ? (100.0 * static_cast<double>(meta.changedBlocks) / static_cast<double>(meta.totalBlocks))
                                 : 0.0;

        lines.push_back("frame " + std::to_string(meta.frameIndex) + " [" + frameType + "] bytes " +
                        std::to_string(bytes) + " | changed " + formatDouble(changedPercent, 1) + "%");
        lines.push_back("avg bitrate " + formatDouble(metrics.averageBitrateKbps, 2) + " kbps | encoded fps " +
                        formatDouble(metrics.effectiveEncodedFps, 2) + " | key-int " + std::to_string(keyInterval));
        lines.push_back("ratio raw4 " + formatDouble(metrics.ratioVsRaw4, 2) + "x | raw8 " +
                        formatDouble(metrics.ratioVsRaw8, 2) + "x | PSNR " + formatDouble(metrics.psnrDb, 2) + " dB");
    } else {
        lines.push_back("Waiting for first encoded frame...");
    }

    lines.push_back("faces now " + std::to_string(facesNow) + " | gathered " + std::to_string(facesGathered) +
                    " | detector " + std::string(faceDetectorReady ? "on" : "missing"));
    lines.push_back("received enhancement: deblock + edge-aware sharpen | dither " +
                    std::string(dither ? "on" : "off") +
                    " | recording " + std::string(recording ? "on" : "off"));
    lines.push_back("keys: q/esc quit | m mode | +/- fps | [] res | k key-int | e enhance | d dither | s rec | f kf");
    lines.push_back("mouse: use right-side panel buttons for FPS and resolution");
    return lines;
}

void printControls() {
    std::cout << "Controls:\n"
              << "  q / ESC : quit\n"
              << "  m       : toggle safer/aggressive profile\n"
              << "  + / -   : increase/decrease target encoded fps (no upper limit)\n"
              << "  ] / [   : increase/decrease internal encoded resolution\n"
              << "  k       : toggle keyframe interval (default/short)\n"
              << "  e       : toggle received enhancement (deblock+edge-aware sharpen)\n"
              << "  d       : toggle dithering for received display\n"
              << "  s       : start/stop stream recording (.evs)\n"
              << "  f       : force next frame to keyframe\n"
              << "  mouse   : click FPS/RES buttons in dashboard view\n"
              << "Camera:\n"
              << "  EV_CAMERA_INDEX=<n> to choose preferred camera index (default 0 for internal webcam)\n";
}
} // namespace

int main() {
    cv::VideoCapture camera;
    const int requestedCameraIndex = preferredCameraIndex();
    int cameraIndex = -1;
    if (!openPreferredCamera(camera, requestedCameraIndex, cameraIndex)) {
        std::cerr << "Failed to open webcam (requested index " << requestedCameraIndex << ")\n";
        return 1;
    }
    std::cout << "Using camera index " << cameraIndex
              << " (set EV_CAMERA_INDEX to override, internal webcam is typically index 0)\n";

    CodecMode mode = CodecMode::Safer;
    const CodecParams modeDefaults = makeCodecParams(mode);
    int resolutionIndex = findNearestResolutionIndex(modeDefaults.width, modeDefaults.height);
    double targetFps = modeDefaults.targetFps;
    CodecParams params = makeRuntimeParams(mode, resolutionIndex, targetFps);

    Encoder encoder(params);
    Decoder decoder;
    MetricsTracker metrics(params.width, params.height);

    bool useReceivedEnhancement = true;
    bool useReceivedDithering = false;
    bool shortKeyframeInterval = false;

    bool recordingEnabled = false;
    std::ofstream recordingFile;
    std::string lastRecordingPath;

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

    cv::namedWindow(kDashboardWindow, cv::WINDOW_NORMAL);
    cv::resizeWindow(kDashboardWindow, kDefaultWindowWidth, kDefaultWindowHeight);

    UiMouseState mouseState;
    cv::setMouseCallback(kDashboardWindow, onDashboardMouse, &mouseState);

    printControls();

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

    auto adjustFps = [&](int delta) {
        targetFps = std::max(kMinTargetFps, targetFps + static_cast<double>(delta) * kFpsStep);
        params.targetFps = targetFps;
        nextEncodeTime = Clock::now();

        std::cout << "Target encoded fps set to " << formatDouble(params.targetFps, 1) << '\n';
    };

    auto adjustResolution = [&](int delta) {
        const int newIndex = std::clamp(resolutionIndex + delta, 0, static_cast<int>(kResolutionLevels.size()) - 1);
        if (newIndex == resolutionIndex) {
            return;
        }

        resolutionIndex = newIndex;
        params = makeRuntimeParams(mode, resolutionIndex, targetFps);
        resetCodecPipeline();

        std::cout << "Resolution set to " << params.width << "x" << params.height << " (target "
                  << formatDouble(params.targetFps, 1) << " fps)\n";
    };

    std::vector<UiButton> controlButtons;

    while (true) {
        cv::Mat bgr;
        camera >> bgr;
        if (bgr.empty()) {
            std::cerr << "Webcam returned empty frame\n";
            break;
        }

        const auto now = Clock::now();

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
                std::cerr << "Encode failure: " << ex.what() << '\n';
                break;
            }

            const DecodeResult decoded = decoder.decode(packet.bytes);
            if (!decoded.ok) {
                std::cerr << "Decode failure: " << decoded.error << '\n';
                break;
            }

            metrics.update(packet.meta, packet.bytes.size(), inputFrame, decoded.frame);
            latestMetrics = metrics.snapshot();
            latestMeta = packet.meta;
            latestPacketBytes = packet.bytes.size();
            haveStats = true;

            if (recordingEnabled) {
                if (!appendRecordingPacket(recordingFile, packet.bytes)) {
                    std::cerr << "Recording write failed. Stopping recording.\n";
                    recordingEnabled = false;
                    recordingFile.close();
                }
            }

            if (!interpolation.hasCurrent) {
                interpolation.current = decoded.frame;
                interpolation.previous = decoded.frame;
                interpolation.motion.assign(
                    static_cast<std::size_t>(totalBlockCount(params.width, params.height, params.blockSize)),
                    MotionVector{});
                interpolation.hasCurrent = true;
                interpolation.hasPair = false;
                interpolation.blendStart = now;
            } else {
                interpolation.previous = interpolation.current;
                interpolation.current = decoded.frame;
                interpolation.motion =
                    estimateBlockMotion(interpolation.previous, interpolation.current, params.blockSize, 2);
                interpolation.hasPair = true;
                interpolation.blendStart = now;
            }

            const int encodePeriodMs =
                std::max(1, static_cast<int>(std::lround(1000.0 / std::max(0.1, params.targetFps))));
            nextEncodeTime = now + std::chrono::milliseconds(encodePeriodMs);

            if (std::chrono::duration_cast<std::chrono::seconds>(now - lastConsolePrint).count() >= 1) {
                std::cout << "mode=" << codecModeName(params.mode) << " frame=" << packet.meta.frameIndex
                          << " bytes=" << packet.bytes.size() << " live="
                          << formatDouble(latestMetrics.liveBitrateKbps) << "kbps smooth="
                          << formatDouble(latestMetrics.smoothedBitrateKbps) << "kbps avg="
                          << formatDouble(latestMetrics.averageBitrateKbps) << "kbps faces=" << lastFacesRaw.size()
                          << " gathered=" << gatheredFaces.size() << '\n';
                lastConsolePrint = now;
            }
        }

        const int panelHeight = std::clamp(bgr.rows, 300, 460);
        const int panelWidth = std::max(340, panelHeight * std::max(1, bgr.cols) / std::max(1, bgr.rows));
        const cv::Size panelSize(panelWidth, panelHeight);

        cv::Mat rawPanel;
        cv::resize(bgr, rawPanel, panelSize, 0.0, 0.0, cv::INTER_AREA);
        const std::vector<cv::Rect> rawFacesPanel = scaleRects(lastFacesRaw, bgr.size(), panelSize);
        drawFaceRects(rawPanel, rawFacesPanel, cv::Scalar(0, 255, 255));
        drawPanelLabel(rawPanel, "RAW INPUT");

        cv::Mat sentPanel(panelHeight, panelWidth, CV_8UC3, cv::Scalar(26, 26, 26));
        if (haveSentFrame) {
            sentPanel = renderForDisplay(latestSentFrame, panelSize, false, true, params.blockSize);
            drawFaceRects(sentPanel, rawFacesPanel, cv::Scalar(0, 220, 220));
        }
        drawPanelLabel(sentPanel, "SENT (DITHERED)");

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
        drawPanelLabel(receivedPanel, "RECEIVED (INTERP+ENH)");

        std::vector<cv::Mat> panels = {rawPanel, sentPanel, receivedPanel};
        cv::Mat feedsRow;
        cv::hconcat(panels, feedsRow);

        cv::Mat canvas(kHudHeight + feedsRow.rows, feedsRow.cols, CV_8UC3, cv::Scalar(15, 15, 15));
        feedsRow.copyTo(canvas(cv::Rect(0, kHudHeight, feedsRow.cols, feedsRow.rows)));

        constexpr int kHudMargin = 12;
        const int desiredControlWidth = std::clamp(canvas.cols / 4, 220, kControlPanelWidth);
        const int hudWidth = std::max(220, canvas.cols - desiredControlWidth - 3 * kHudMargin);
        cv::Rect hudRect(kHudMargin, kHudMargin, hudWidth, kHudHeight - 2 * kHudMargin);
        cv::Rect controlRect(hudRect.x + hudRect.width + kHudMargin,
                             kHudMargin,
                             canvas.cols - (hudRect.x + hudRect.width + 2 * kHudMargin),
                             kHudHeight - 2 * kHudMargin);

        cv::rectangle(canvas, hudRect, cv::Scalar(20, 20, 20), cv::FILLED);
        cv::rectangle(canvas, hudRect, cv::Scalar(70, 70, 70), 1);
        cv::rectangle(canvas, controlRect, cv::Scalar(22, 22, 22), cv::FILLED);
        cv::rectangle(canvas, controlRect, cv::Scalar(90, 90, 90), 1);

        const std::vector<std::string> hudLines = buildHudLines(params,
                                                                 latestMetrics,
                                                                 latestMeta,
                                                                 latestPacketBytes,
                                                                 haveStats,
                                                                 useReceivedDithering,
                                                                 recordingEnabled,
                                                                 static_cast<int>(lastFacesRaw.size()),
                                                                 gatheredFaces.size(),
                                                                 faceDetectorReady,
                                                                 activeKeyframeInterval(params, shortKeyframeInterval),
                                                                 cameraIndex);
        drawHud(canvas, hudLines, cv::Point(hudRect.x + 10, hudRect.y + 24), hudRect.width - 20);
        drawBandwidthMeter(canvas,
                           cv::Point(hudRect.x + 10, hudRect.y + hudRect.height - 18),
                           std::max(180, hudRect.width - 20),
                           14,
                           latestMetrics.liveBitrateKbps,
                           latestMetrics.smoothedBitrateKbps);

        cv::putText(canvas,
                    "QUICK CONTROLS",
                    cv::Point(controlRect.x + 12, controlRect.y + 20),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.56,
                    cv::Scalar(0, 0, 0),
                    3,
                    cv::LINE_AA);
        cv::putText(canvas,
                    "QUICK CONTROLS",
                    cv::Point(controlRect.x + 12, controlRect.y + 20),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.56,
                    cv::Scalar(125, 255, 125),
                    1,
                    cv::LINE_AA);

        controlButtons = buildControlButtons(controlRect);
        drawControlButtons(canvas, controlButtons);

        cv::imshow(kDashboardWindow, canvas);

        const int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {
            break;
        }

        bool handledMouseAction = false;
        if (mouseState.pendingClick) {
            const cv::Point clickPoint = mouseState.clickPoint;
            mouseState.pendingClick = false;

            const std::string action = hitTestButton(controlButtons, clickPoint);
            if (action == "fps_up") {
                adjustFps(+1);
                handledMouseAction = true;
            } else if (action == "fps_down") {
                adjustFps(-1);
                handledMouseAction = true;
            } else if (action == "res_up") {
                adjustResolution(+1);
                handledMouseAction = true;
            } else if (action == "res_down") {
                adjustResolution(-1);
                handledMouseAction = true;
            }
        }

        if (handledMouseAction) {
            continue;
        }

        if (key == 'm') {
            mode = (mode == CodecMode::Safer) ? CodecMode::Aggressive : CodecMode::Safer;
            params = makeRuntimeParams(mode, resolutionIndex, targetFps);
            shortKeyframeInterval = false;
            resetCodecPipeline();

            std::cout << "Switched mode to " << codecModeName(mode) << " (" << params.width << "x" << params.height
                      << ", target " << formatDouble(params.targetFps, 1) << " fps)\n";
            continue;
        }

        if (key == '+' || key == '=') {
            adjustFps(+1);
            continue;
        }

        if (key == '-') {
            adjustFps(-1);
            continue;
        }

        if (key == ']') {
            adjustResolution(+1);
            continue;
        }

        if (key == '[') {
            adjustResolution(-1);
            continue;
        }

        if (key == 'k') {
            shortKeyframeInterval = !shortKeyframeInterval;
            const int interval = activeKeyframeInterval(params, shortKeyframeInterval);
            encoder.setKeyframeInterval(interval);
            std::cout << "Keyframe interval set to " << interval << " frames\n";
            continue;
        }

        if (key == 'e') {
            useReceivedEnhancement = !useReceivedEnhancement;
            std::cout << "Received enhancement " << (useReceivedEnhancement ? "enabled" : "disabled") << '\n';
            continue;
        }

        if (key == 'd') {
            useReceivedDithering = !useReceivedDithering;
            std::cout << "Received dithering " << (useReceivedDithering ? "enabled" : "disabled") << '\n';
            continue;
        }

        if (key == 'f') {
            encoder.forceNextKeyframe();
            std::cout << "Forced keyframe on next encode\n";
            continue;
        }

        if (key == 's') {
            if (!recordingEnabled) {
                lastRecordingPath = makeRecordingFileName();
                if (!openRecordingFile(recordingFile, lastRecordingPath)) {
                    std::cerr << "Failed to open recording file: " << lastRecordingPath << '\n';
                } else {
                    recordingEnabled = true;
                    std::cout << "Recording started: " << lastRecordingPath << '\n';
                }
            } else {
                recordingEnabled = false;
                recordingFile.close();
                std::cout << "Recording stopped: " << lastRecordingPath << '\n';
            }
            continue;
        }
    }

    if (recordingFile.is_open()) {
        recordingFile.close();
    }

    return 0;
}
