#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

struct AudioDeviceInfo {
    int index = -1;
    std::string name;
    bool isDefault = false;
};

class AudioEngine {
public:
    AudioEngine();
    ~AudioEngine();

    bool init();
    void shutdown();

    std::vector<AudioDeviceInfo> listInputDevices();
    std::vector<AudioDeviceInfo> listOutputDevices();

    bool startCapture(int deviceIndex, uint32_t sampleRate);
    bool startPlayback(int deviceIndex, uint32_t sampleRate);

    void stopCapture();
    void stopPlayback();

    void clearCaptureBuffer();
    void clearPlaybackBuffer();

    std::size_t popCaptured(std::vector<float> &out, std::size_t maxSamples);
    void pushPlayback(const std::vector<float> &samples);

    [[nodiscard]] bool captureRunning() const;
    [[nodiscard]] bool playbackRunning() const;

private:
    struct Impl;
    Impl *impl_ = nullptr;
};

