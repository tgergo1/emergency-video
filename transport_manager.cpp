#include "transport_manager.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>

#ifndef _WIN32
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#endif

#include <opencv2/imgproc.hpp>

#include "communicator_protocol.h"
#include "media_ffmpeg.h"

namespace fs = std::filesystem;

namespace {

#ifndef _WIN32
speed_t toSpeed(int baud) {
    switch (baud) {
    case 9600:
        return B9600;
    case 19200:
        return B19200;
    case 38400:
        return B38400;
    case 57600:
        return B57600;
    case 115200:
        return B115200;
#ifdef B230400
    case 230400:
        return B230400;
#endif
    default:
        return B115200;
    }
}
#endif

std::vector<uint8_t> addCrc(const std::vector<uint8_t> &payload) {
    std::vector<uint8_t> out = payload;
    const uint32_t crc = crc32Comm(payload);
    out.push_back(static_cast<uint8_t>(crc & 0xFFU));
    out.push_back(static_cast<uint8_t>((crc >> 8U) & 0xFFU));
    out.push_back(static_cast<uint8_t>((crc >> 16U) & 0xFFU));
    out.push_back(static_cast<uint8_t>((crc >> 24U) & 0xFFU));
    return out;
}

bool stripAndVerifyCrc(const std::vector<uint8_t> &framed, std::vector<uint8_t> &payload) {
    payload.clear();
    if (framed.size() < 4) {
        return false;
    }
    const std::size_t bodyLen = framed.size() - 4;
    const uint32_t expected = static_cast<uint32_t>(framed[bodyLen]) |
                              (static_cast<uint32_t>(framed[bodyLen + 1]) << 8U) |
                              (static_cast<uint32_t>(framed[bodyLen + 2]) << 16U) |
                              (static_cast<uint32_t>(framed[bodyLen + 3]) << 24U);
    payload.assign(framed.begin(), framed.begin() + static_cast<std::ptrdiff_t>(bodyLen));
    return crc32Comm(payload) == expected;
}

} // namespace

AcousticAdapter::AcousticAdapter() {
    audioEngine_.init();
}

AcousticAdapter::~AcousticAdapter() {
    stop();
}

void AcousticAdapter::configure(const SessionConfig &session,
                                LinkMode linkMode,
                                RxSource rxSource,
                                SessionMode sessionMode,
                                int audioInDevice,
                                int audioOutDevice,
                                const std::string &mediaPath) {
    session_ = session;
    linkMode_ = linkMode;
    rxSource_ = rxSource;
    sessionMode_ = sessionMode;
    audioInDevice_ = audioInDevice;
    audioOutDevice_ = audioOutDevice;
    mediaPath_ = mediaPath;

    modem_ = std::make_unique<MfskModem>(modemParamsFromSession(session_));
    receiver_ = std::make_unique<AcousticBurstReceiver>(*modem_);
    receiver_->setEnergyThreshold(thresholdStart_, thresholdEnd_);
}

TransportKind AcousticAdapter::kind() const {
    return TransportKind::Acoustic;
}

bool AcousticAdapter::start(std::string &error) {
    error.clear();
    if (!modem_) {
        modem_ = std::make_unique<MfskModem>(modemParamsFromSession(session_));
        receiver_ = std::make_unique<AcousticBurstReceiver>(*modem_);
    }

    running_ = true;
    ensureAudioDevices();
    calibrateAmbientProfile();
    ensureAudioDevices();
    return true;
}

void AcousticAdapter::stop() {
    running_ = false;
    audioEngine_.stopCapture();
    audioEngine_.stopPlayback();
    audioEngine_.clearCaptureBuffer();
    audioEngine_.clearPlaybackBuffer();
    txRawFrames_.clear();
    if (receiver_) {
        receiver_->clear();
    }
}

bool AcousticAdapter::running() const {
    return running_;
}

void AcousticAdapter::ensureAudioDevices() {
    if (!running_) {
        return;
    }

    const bool needCapture = (linkMode_ == LinkMode::AcousticRxLive || linkMode_ == LinkMode::AcousticDuplexArq ||
                              (linkMode_ == LinkMode::AcousticTx && sessionMode_ == SessionMode::DuplexArq));
    const bool needPlayback =
        (linkMode_ == LinkMode::AcousticTx || linkMode_ == LinkMode::AcousticDuplexArq ||
         ((linkMode_ == LinkMode::AcousticRxLive || linkMode_ == LinkMode::AcousticRxMedia) &&
          sessionMode_ == SessionMode::DuplexArq));

    if (needCapture && !audioEngine_.captureRunning()) {
        audioEngine_.startCapture(audioInDevice_, session_.sampleRate);
        audioEngine_.clearCaptureBuffer();
    }
    if (!needCapture && audioEngine_.captureRunning()) {
        audioEngine_.stopCapture();
    }

    if (needPlayback && !audioEngine_.playbackRunning()) {
        audioEngine_.startPlayback(audioOutDevice_, session_.sampleRate);
        audioEngine_.clearPlaybackBuffer();
    }
    if (!needPlayback && audioEngine_.playbackRunning()) {
        audioEngine_.stopPlayback();
    }
}

void AcousticAdapter::calibrateAmbientProfile() {
    if (!running_ || receiver_ == nullptr) {
        return;
    }

    bool startedTempCapture = false;
    if (!audioEngine_.captureRunning()) {
        startedTempCapture = audioEngine_.startCapture(audioInDevice_, session_.sampleRate);
        if (startedTempCapture) {
            audioEngine_.clearCaptureBuffer();
        }
    } else {
        audioEngine_.clearCaptureBuffer();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(620));

    std::vector<float> ambient;
    const std::size_t maxSamples = std::max<std::size_t>(480, static_cast<std::size_t>(session_.sampleRate) * 7U / 10U);
    audioEngine_.popCaptured(ambient, maxSamples);

    double sumSq = 0.0;
    for (float s : ambient) {
        sumSq += static_cast<double>(s) * static_cast<double>(s);
    }
    noiseRms_ = ambient.empty() ? 0.0 : std::sqrt(sumSq / static_cast<double>(ambient.size()));

    thresholdStart_ = std::max(0.008F, static_cast<float>(noiseRms_ * 5.0));
    thresholdEnd_ = std::max(0.004F, static_cast<float>(noiseRms_ * 3.0));
    receiver_->setEnergyThreshold(thresholdStart_, thresholdEnd_);

    if (noiseRms_ < 0.003) {
        session_.fecRepetition = 3;
        session_.interleaveDepth = 8;
    } else if (noiseRms_ < 0.012) {
        session_.fecRepetition = 4;
        session_.interleaveDepth = 10;
    } else {
        session_.fecRepetition = 5;
        session_.interleaveDepth = 12;
    }

    if (startedTempCapture) {
        audioEngine_.stopCapture();
    }
}

void AcousticAdapter::maybeFeedMediaFile() {
    if (!running_ || mediaPath_.empty() || linkMode_ != LinkMode::AcousticRxMedia) {
        return;
    }

    if (mediaPcm_.empty()) {
        std::string error;
        std::vector<float> decoded;
        if (decodeMediaAudioToMonoF32(mediaPath_, session_.sampleRate, decoded, error)) {
            mediaPcm_ = std::move(decoded);
            mediaCursor_ = 0;
        }
    }

    if (mediaPcm_.empty() || !receiver_) {
        return;
    }

    const std::size_t feedCount = std::max<std::size_t>(64, session_.sampleRate / 20U);
    std::vector<float> chunk;
    chunk.reserve(feedCount);
    for (std::size_t i = 0; i < feedCount; ++i) {
        if (mediaCursor_ >= mediaPcm_.size()) {
            mediaCursor_ = 0;
        }
        chunk.push_back(mediaPcm_[mediaCursor_++]);
    }
    receiver_->feedSamples(chunk.data(), chunk.size());
}

void AcousticAdapter::sendEnvelope(const std::vector<uint8_t> &envelope) {
    if (!running_) {
        return;
    }
    txRawFrames_.push_back(envelope);
    while (txRawFrames_.size() > 256) {
        txRawFrames_.pop_front();
    }

    ensureAudioDevices();
    if (audioEngine_.playbackRunning() && modem_) {
        int budget = 2;
        while (!txRawFrames_.empty() && budget-- > 0) {
            const std::vector<float> pcm =
                modem_->modulateFrame(txRawFrames_.front(), session_.fecRepetition, session_.interleaveDepth);
            audioEngine_.pushPlayback(pcm);
            txRawFrames_.pop_front();
        }
    }
}

void AcousticAdapter::pollIncoming(std::vector<std::vector<uint8_t>> &outEnvelopes) {
    outEnvelopes.clear();
    if (!running_) {
        return;
    }

    ensureAudioDevices();
    maybeFeedMediaFile();

    if (audioEngine_.captureRunning() && receiver_) {
        std::vector<float> captured;
        audioEngine_.popCaptured(captured, std::max<std::size_t>(128, session_.sampleRate / 3U));
        if (!captured.empty()) {
            receiver_->feedSamples(captured.data(), captured.size());
        }
    }

    if (!receiver_) {
        return;
    }

    while (true) {
        std::vector<uint8_t> wire;
        std::size_t recoveredSymbols = 0;
        std::size_t comparedSymbols = 0;
        if (!receiver_->popFrame(wire, &recoveredSymbols, &comparedSymbols)) {
            break;
        }
        stats_.framesReceived += 1;
        stats_.fecRecoveredCount += recoveredSymbols;
        stats_.syncLocked = true;
        if (comparedSymbols > 0) {
            const double sample = static_cast<double>(recoveredSymbols) / static_cast<double>(comparedSymbols);
            stats_.berEstimate = (stats_.berEstimate <= 0.0) ? sample : (0.8 * stats_.berEstimate + 0.2 * sample);
        }
        outEnvelopes.push_back(std::move(wire));
    }
}

std::string AcousticAdapter::status() const {
    std::ostringstream oss;
    oss << "acoustic " << (running_ ? "running" : "stopped") << " txq=" << txRawFrames_.size()
        << " rms=" << std::fixed << std::setprecision(4) << noiseRms_
        << " th=" << thresholdStart_ << "/" << thresholdEnd_
        << " fec=" << static_cast<int>(session_.fecRepetition) << "/" << static_cast<int>(session_.interleaveDepth);
    return oss.str();
}

LinkStats AcousticAdapter::takeAndResetLinkStats() {
    LinkStats out = stats_;
    stats_ = LinkStats{};
    return out;
}

SerialAdapter::SerialAdapter() = default;

SerialAdapter::~SerialAdapter() {
    stop();
}

void SerialAdapter::setPreferredPort(const std::string &portPath) {
    preferredPort_ = portPath;
}

void SerialAdapter::setBaud(int baud) {
    baud_ = baud;
}

TransportKind SerialAdapter::kind() const {
    return TransportKind::Serial;
}

std::vector<std::string> SerialAdapter::listCandidatePorts() {
#ifdef _WIN32
    std::vector<std::string> out;
    out.reserve(16);
    for (int i = 1; i <= 16; ++i) {
        out.push_back("COM" + std::to_string(i));
    }
    return out;
#else
    std::vector<std::string> out;
    for (const auto &entry : fs::directory_iterator("/dev")) {
        if (!entry.is_character_file()) {
            continue;
        }
        const std::string p = entry.path().string();
        if (p.find("/dev/cu.") == 0 || p.find("/dev/tty.") == 0 || p.find("/dev/ttyUSB") == 0 ||
            p.find("/dev/ttyACM") == 0) {
            out.push_back(p);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
#endif
}

bool SerialAdapter::openPort(const std::string &path, std::string &error) {
#ifdef _WIN32
    (void)path;
    error = "serial adapter is not supported in this Windows build";
    return false;
#else
    error.clear();
    fd_ = ::open(path.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
        error = "open failed: " + std::string(std::strerror(errno));
        return false;
    }

    termios tty{};
    if (tcgetattr(fd_, &tty) != 0) {
        error = "tcgetattr failed";
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    cfmakeraw(&tty);
    const speed_t speed = toSpeed(baud_);
    cfsetispeed(&tty, speed);
    cfsetospeed(&tty, speed);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
        error = "tcsetattr failed";
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    activePort_ = path;
    return true;
#endif
}

bool SerialAdapter::start(std::string &error) {
#ifdef _WIN32
    error = "serial adapter is not supported in this Windows build";
    return false;
#else
    error.clear();
    if (fd_ >= 0) {
        return true;
    }

    if (!preferredPort_.empty() && openPort(preferredPort_, error)) {
        return true;
    }

    for (const std::string &candidate : listCandidatePorts()) {
        if (openPort(candidate, error)) {
            return true;
        }
    }

    if (error.empty()) {
        error = "no serial port available";
    }
    return false;
#endif
}

void SerialAdapter::stop() {
    if (fd_ >= 0) {
#ifndef _WIN32
        ::close(fd_);
#endif
        fd_ = -1;
    }
    activePort_.clear();
    rxBuffer_.clear();
    txQueue_.clear();
}

bool SerialAdapter::running() const {
#ifdef _WIN32
    return false;
#else
    return fd_ >= 0;
#endif
}

std::vector<uint8_t> SerialAdapter::slipEncode(const std::vector<uint8_t> &bytes) {
    constexpr uint8_t END = 0xC0;
    constexpr uint8_t ESC = 0xDB;
    constexpr uint8_t ESC_END = 0xDC;
    constexpr uint8_t ESC_ESC = 0xDD;

    std::vector<uint8_t> out;
    out.reserve(bytes.size() + 8);
    out.push_back(END);
    for (uint8_t b : bytes) {
        if (b == END) {
            out.push_back(ESC);
            out.push_back(ESC_END);
        } else if (b == ESC) {
            out.push_back(ESC);
            out.push_back(ESC_ESC);
        } else {
            out.push_back(b);
        }
    }
    out.push_back(END);
    return out;
}

bool SerialAdapter::slipTryDecodeOne(std::vector<uint8_t> &buffer, std::vector<uint8_t> &frame) {
    constexpr uint8_t END = 0xC0;
    constexpr uint8_t ESC = 0xDB;
    constexpr uint8_t ESC_END = 0xDC;
    constexpr uint8_t ESC_ESC = 0xDD;

    auto startIt = std::find(buffer.begin(), buffer.end(), END);
    if (startIt == buffer.end()) {
        buffer.clear();
        return false;
    }
    auto endIt = std::find(startIt + 1, buffer.end(), END);
    if (endIt == buffer.end()) {
        buffer.erase(buffer.begin(), startIt);
        return false;
    }

    frame.clear();
    for (auto it = startIt + 1; it != endIt; ++it) {
        uint8_t b = *it;
        if (b == ESC) {
            ++it;
            if (it == endIt) {
                frame.clear();
                break;
            }
            if (*it == ESC_END) {
                frame.push_back(END);
            } else if (*it == ESC_ESC) {
                frame.push_back(ESC);
            }
        } else {
            frame.push_back(b);
        }
    }

    buffer.erase(buffer.begin(), endIt + 1);
    return !frame.empty();
}

void SerialAdapter::sendEnvelope(const std::vector<uint8_t> &envelope) {
#ifdef _WIN32
    (void)envelope;
    return;
#else
    if (fd_ < 0) {
        return;
    }
    txQueue_.push_back(addCrc(envelope));
    while (txQueue_.size() > 256) {
        txQueue_.pop_front();
    }

    int budget = 8;
    while (!txQueue_.empty() && budget-- > 0) {
        const std::vector<uint8_t> framed = slipEncode(txQueue_.front());
        ssize_t written = ::write(fd_, framed.data(), framed.size());
        if (written < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break;
            }
            break;
        }
        txQueue_.pop_front();
    }
#endif
}

void SerialAdapter::pollIncoming(std::vector<std::vector<uint8_t>> &outEnvelopes) {
    outEnvelopes.clear();
#ifdef _WIN32
    return;
#else
    if (fd_ < 0) {
        return;
    }

    std::array<uint8_t, 2048> buf{};
    while (true) {
        const ssize_t n = ::read(fd_, buf.data(), buf.size());
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break;
            }
            break;
        }
        if (n == 0) {
            break;
        }
        rxBuffer_.insert(rxBuffer_.end(), buf.begin(), buf.begin() + n);
    }

    std::vector<uint8_t> framed;
    while (slipTryDecodeOne(rxBuffer_, framed)) {
        std::vector<uint8_t> payload;
        if (stripAndVerifyCrc(framed, payload)) {
            outEnvelopes.push_back(std::move(payload));
        }
    }
#endif
}

std::string SerialAdapter::status() const {
#ifdef _WIN32
    return "serial unsupported on Windows in this build";
#else
    if (fd_ < 0) {
        return "serial stopped";
    }
    return "serial running on " + activePort_;
#endif
}

OpticalAdapter::OpticalAdapter() = default;

TransportKind OpticalAdapter::kind() const {
    return TransportKind::Optical;
}

bool OpticalAdapter::start(std::string &error) {
    error.clear();
    running_ = true;
    return true;
}

void OpticalAdapter::stop() {
    running_ = false;
    txQueue_.clear();
    rxQueue_.clear();
    lastTxPattern_.release();
}

bool OpticalAdapter::running() const {
    return running_;
}

cv::Mat OpticalAdapter::encodePattern(const std::vector<uint8_t> &bytes) {
    // 32x32 monochrome symbol map packed as bytes for screen-camera transfer.
    constexpr int grid = 32;
    constexpr int cell = 10;
    cv::Mat img(grid * cell, grid * cell, CV_8UC1, cv::Scalar(20));

    std::vector<uint8_t> framed = addCrc(bytes);
    std::size_t bitIndex = 0;
    for (int y = 0; y < grid; ++y) {
        for (int x = 0; x < grid; ++x) {
            const std::size_t byteIndex = bitIndex / 8;
            const int bitOff = static_cast<int>(bitIndex % 8);
            uint8_t bit = 0;
            if (byteIndex < framed.size()) {
                bit = (framed[byteIndex] >> bitOff) & 1U;
            }
            const uint8_t v = bit ? 235 : 25;
            cv::rectangle(img, cv::Rect(x * cell, y * cell, cell, cell), cv::Scalar(v), cv::FILLED);
            ++bitIndex;
        }
    }

    cv::Mat bgr;
    cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

bool OpticalAdapter::decodePattern(const cv::Mat &frame, std::vector<uint8_t> &bytes) {
    bytes.clear();
    if (frame.empty()) {
        return false;
    }

    constexpr int grid = 32;
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame;
    }

    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(grid, grid), 0.0, 0.0, cv::INTER_AREA);

    std::vector<uint8_t> raw((grid * grid + 7) / 8, 0);
    std::size_t bitIndex = 0;
    for (int y = 0; y < grid; ++y) {
        for (int x = 0; x < grid; ++x) {
            const uint8_t bit = resized.at<uint8_t>(y, x) > 127 ? 1U : 0U;
            const std::size_t byteIndex = bitIndex / 8;
            const int bitOff = static_cast<int>(bitIndex % 8);
            raw[byteIndex] |= static_cast<uint8_t>(bit << bitOff);
            ++bitIndex;
        }
    }

    if (!stripAndVerifyCrc(raw, bytes)) {
        return false;
    }

    return true;
}

void OpticalAdapter::sendEnvelope(const std::vector<uint8_t> &envelope) {
    if (!running_) {
        return;
    }
    txQueue_.push_back(envelope);
    while (txQueue_.size() > 64) {
        txQueue_.pop_front();
    }

    if (!txQueue_.empty()) {
        lastTxPattern_ = encodePattern(txQueue_.front());
        // In the same process demo, make it available for local loop/decode fallback.
        rxQueue_.push_back(txQueue_.front());
        txQueue_.pop_front();
    }
}

void OpticalAdapter::feedRxFrame(const cv::Mat &frame) {
    if (!running_) {
        return;
    }
    std::vector<uint8_t> bytes;
    if (decodePattern(frame, bytes)) {
        rxQueue_.push_back(std::move(bytes));
    }
}

void OpticalAdapter::pollIncoming(std::vector<std::vector<uint8_t>> &outEnvelopes) {
    outEnvelopes.clear();
    while (!rxQueue_.empty()) {
        outEnvelopes.push_back(std::move(rxQueue_.front()));
        rxQueue_.pop_front();
    }
}

std::string OpticalAdapter::status() const {
    return running_ ? "optical running" : "optical stopped";
}

cv::Mat OpticalAdapter::currentTxPattern() const {
    return lastTxPattern_.clone();
}

FileRelayAdapter::FileRelayAdapter() = default;

void FileRelayAdapter::setDirectories(const std::string &exportDir, const std::string &importDir) {
    exportDir_ = exportDir;
    importDir_ = importDir;
}

TransportKind FileRelayAdapter::kind() const {
    return TransportKind::FileRelay;
}

bool FileRelayAdapter::start(std::string &error) {
    error.clear();
    std::error_code ec;
    if (!exportDir_.empty()) {
        fs::create_directories(exportDir_, ec);
        if (ec) {
            error = "failed to create export dir: " + ec.message();
            return false;
        }
    }
    if (!importDir_.empty()) {
        fs::create_directories(importDir_, ec);
        if (ec) {
            error = "failed to create import dir: " + ec.message();
            return false;
        }
    }
    running_ = true;
    return true;
}

void FileRelayAdapter::stop() {
    running_ = false;
}

bool FileRelayAdapter::running() const {
    return running_;
}

void FileRelayAdapter::sendEnvelope(const std::vector<uint8_t> &envelope) {
    if (!running_ || exportDir_.empty()) {
        return;
    }
    const uint64_t ts = nowUnixMs();
    const fs::path path = fs::path(exportDir_) / ("env_" + std::to_string(ts) + ".evrelay");
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        return;
    }
    static constexpr char kMagic[5] = {'E', 'V', 'R', 'L', '1'};
    out.write(kMagic, 5);
    const uint32_t count = 1;
    out.write(reinterpret_cast<const char *>(&count), sizeof(count));
    const uint32_t len = static_cast<uint32_t>(envelope.size());
    out.write(reinterpret_cast<const char *>(&len), sizeof(len));
    out.write(reinterpret_cast<const char *>(envelope.data()), static_cast<std::streamsize>(envelope.size()));
}

void FileRelayAdapter::pollIncoming(std::vector<std::vector<uint8_t>> &outEnvelopes) {
    outEnvelopes.clear();
    if (!running_ || importDir_.empty()) {
        return;
    }

    std::error_code ec;
    for (const auto &entry : fs::directory_iterator(importDir_, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file()) {
            continue;
        }
        const fs::path path = entry.path();
        if (path.extension() != ".evrelay") {
            continue;
        }

        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            continue;
        }

        char magic[5] = {};
        in.read(magic, 5);
        if (!in || std::string(magic, 5) != "EVRL1") {
            continue;
        }

        uint32_t count = 0;
        in.read(reinterpret_cast<char *>(&count), sizeof(count));
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t len = 0;
            in.read(reinterpret_cast<char *>(&len), sizeof(len));
            if (!in || len == 0 || len > (1U << 24U)) {
                break;
            }
            std::vector<uint8_t> bytes(len);
            in.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(len));
            if (!in) {
                break;
            }
            outEnvelopes.push_back(std::move(bytes));
        }

        fs::remove(path, ec);
    }
}

std::string FileRelayAdapter::status() const {
    return running_ ? "file relay running" : "file relay stopped";
}

void TransportManager::setActive(TransportKind kind) {
    active_ = kind;
}

TransportKind TransportManager::active() const {
    return active_;
}

TransportAdapter *TransportManager::activeAdapter() {
    switch (active_) {
    case TransportKind::Acoustic:
        return &acoustic_;
    case TransportKind::Serial:
        return &serial_;
    case TransportKind::Optical:
        return &optical_;
    case TransportKind::FileRelay:
        return &fileRelay_;
    }
    return &acoustic_;
}

AcousticAdapter &TransportManager::acoustic() {
    return acoustic_;
}

SerialAdapter &TransportManager::serial() {
    return serial_;
}

OpticalAdapter &TransportManager::optical() {
    return optical_;
}

FileRelayAdapter &TransportManager::fileRelay() {
    return fileRelay_;
}
