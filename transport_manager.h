#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "acoustic_modem.h"
#include "audio_io.h"
#include "communicator_protocol.h"

class TransportAdapter {
public:
    virtual ~TransportAdapter() = default;

    virtual TransportKind kind() const = 0;
    virtual bool start(std::string &error) = 0;
    virtual void stop() = 0;
    virtual bool running() const = 0;

    virtual void sendEnvelope(const std::vector<uint8_t> &envelope) = 0;
    virtual void pollIncoming(std::vector<std::vector<uint8_t>> &outEnvelopes) = 0;
    virtual std::string status() const = 0;
};

class AcousticAdapter final : public TransportAdapter {
public:
    AcousticAdapter();
    ~AcousticAdapter() override;

    void configure(const SessionConfig &session,
                   LinkMode linkMode,
                   RxSource rxSource,
                   SessionMode sessionMode,
                   int audioInDevice,
                   int audioOutDevice,
                   const std::string &mediaPath);

    TransportKind kind() const override;
    bool start(std::string &error) override;
    void stop() override;
    bool running() const override;

    void sendEnvelope(const std::vector<uint8_t> &envelope) override;
    void pollIncoming(std::vector<std::vector<uint8_t>> &outEnvelopes) override;
    std::string status() const override;

    LinkStats takeAndResetLinkStats();

private:
    AudioEngine audioEngine_;

    SessionConfig session_{};
    LinkMode linkMode_ = LinkMode::AcousticTx;
    RxSource rxSource_ = RxSource::LiveMic;
    SessionMode sessionMode_ = SessionMode::Broadcast;
    int audioInDevice_ = -1;
    int audioOutDevice_ = -1;
    std::string mediaPath_;

    std::unique_ptr<MfskModem> modem_;
    std::unique_ptr<AcousticBurstReceiver> receiver_;

    std::deque<std::vector<uint8_t>> txRawFrames_;
    std::vector<float> mediaPcm_;
    std::size_t mediaCursor_ = 0;

    bool running_ = false;
    LinkStats stats_;
    float thresholdStart_ = 0.02F;
    float thresholdEnd_ = 0.011F;
    double noiseRms_ = 0.0;

    void ensureAudioDevices();
    void calibrateAmbientProfile();
    void maybeFeedMediaFile();
};

class SerialAdapter final : public TransportAdapter {
public:
    SerialAdapter();
    ~SerialAdapter() override;

    void setPreferredPort(const std::string &portPath);
    void setBaud(int baud);

    TransportKind kind() const override;
    bool start(std::string &error) override;
    void stop() override;
    bool running() const override;

    void sendEnvelope(const std::vector<uint8_t> &envelope) override;
    void pollIncoming(std::vector<std::vector<uint8_t>> &outEnvelopes) override;
    std::string status() const override;

    static std::vector<std::string> listCandidatePorts();

private:
    std::string preferredPort_;
    std::string activePort_;
    int baud_ = 115200;
    int fd_ = -1;

    std::vector<uint8_t> rxBuffer_;
    std::deque<std::vector<uint8_t>> txQueue_;

    bool openPort(const std::string &path, std::string &error);
    static std::vector<uint8_t> slipEncode(const std::vector<uint8_t> &bytes);
    static bool slipTryDecodeOne(std::vector<uint8_t> &buffer, std::vector<uint8_t> &frame);
};

class OpticalAdapter final : public TransportAdapter {
public:
    OpticalAdapter();

    TransportKind kind() const override;
    bool start(std::string &error) override;
    void stop() override;
    bool running() const override;

    void sendEnvelope(const std::vector<uint8_t> &envelope) override;
    void pollIncoming(std::vector<std::vector<uint8_t>> &outEnvelopes) override;
    std::string status() const override;

    cv::Mat currentTxPattern() const;
    void feedRxFrame(const cv::Mat &frame);

private:
    bool running_ = false;
    std::deque<std::vector<uint8_t>> txQueue_;
    std::deque<std::vector<uint8_t>> rxQueue_;
    cv::Mat lastTxPattern_;

    static cv::Mat encodePattern(const std::vector<uint8_t> &bytes);
    static bool decodePattern(const cv::Mat &frame, std::vector<uint8_t> &bytes);
};

class FileRelayAdapter final : public TransportAdapter {
public:
    FileRelayAdapter();

    void setDirectories(const std::string &exportDir, const std::string &importDir);

    TransportKind kind() const override;
    bool start(std::string &error) override;
    void stop() override;
    bool running() const override;

    void sendEnvelope(const std::vector<uint8_t> &envelope) override;
    void pollIncoming(std::vector<std::vector<uint8_t>> &outEnvelopes) override;
    std::string status() const override;

private:
    std::string exportDir_;
    std::string importDir_;
    bool running_ = false;
};

class TransportManager {
public:
    void setActive(TransportKind kind);
    TransportKind active() const;

    TransportAdapter *activeAdapter();

    AcousticAdapter &acoustic();
    SerialAdapter &serial();
    OpticalAdapter &optical();
    FileRelayAdapter &fileRelay();

private:
    TransportKind active_ = TransportKind::Acoustic;
    AcousticAdapter acoustic_;
    SerialAdapter serial_;
    OpticalAdapter optical_;
    FileRelayAdapter fileRelay_;
};
