#include "audio_io.h"

#include <algorithm>
#include <cstring>
#include <memory>

#define MA_NO_ENCODING
#include "third_party/miniaudio.h"

namespace {
constexpr std::size_t kMaxCaptureQueueSamples = 48000U * 15U;
constexpr std::size_t kMaxPlaybackQueueSamples = 48000U * 15U;

void trimDeque(std::deque<float> &queue, std::size_t maxSize) {
    while (queue.size() > maxSize) {
        queue.pop_front();
    }
}

} // namespace

struct AudioEngine::Impl {
    ma_context context{};
    bool contextReady = false;

    ma_device captureDevice{};
    bool captureActive = false;

    ma_device playbackDevice{};
    bool playbackActive = false;

    std::mutex captureMutex;
    std::deque<float> captureQueue;

    std::mutex playbackMutex;
    std::deque<float> playbackQueue;

    static void captureCallback(ma_device *device, void *output, const void *input, ma_uint32 frameCount) {
        (void)output;
        if (device == nullptr || input == nullptr) {
            return;
        }

        auto *self = static_cast<Impl *>(device->pUserData);
        if (self == nullptr) {
            return;
        }

        const auto *samples = static_cast<const float *>(input);
        std::lock_guard<std::mutex> lock(self->captureMutex);
        for (ma_uint32 i = 0; i < frameCount; ++i) {
            self->captureQueue.push_back(samples[i]);
        }
        trimDeque(self->captureQueue, kMaxCaptureQueueSamples);
    }

    static void playbackCallback(ma_device *device, void *output, const void *input, ma_uint32 frameCount) {
        (void)input;
        if (device == nullptr || output == nullptr) {
            return;
        }

        auto *self = static_cast<Impl *>(device->pUserData);
        auto *samples = static_cast<float *>(output);

        if (self == nullptr) {
            std::memset(samples, 0, sizeof(float) * frameCount);
            return;
        }

        std::lock_guard<std::mutex> lock(self->playbackMutex);
        for (ma_uint32 i = 0; i < frameCount; ++i) {
            if (!self->playbackQueue.empty()) {
                samples[i] = self->playbackQueue.front();
                self->playbackQueue.pop_front();
            } else {
                samples[i] = 0.0F;
            }
        }
    }

    std::vector<AudioDeviceInfo> listDevices(ma_device_type type) {
        std::vector<AudioDeviceInfo> out;
        if (!contextReady) {
            return out;
        }

        ma_device_info *playbacks = nullptr;
        ma_uint32 playbackCount = 0;
        ma_device_info *captures = nullptr;
        ma_uint32 captureCount = 0;

        if (ma_context_get_devices(&context, &playbacks, &playbackCount, &captures, &captureCount) != MA_SUCCESS) {
            return out;
        }

        if (type == ma_device_type_capture) {
            out.reserve(captureCount);
            for (ma_uint32 i = 0; i < captureCount; ++i) {
                AudioDeviceInfo info;
                info.index = static_cast<int>(i);
                info.name = captures[i].name;
                info.isDefault = captures[i].isDefault != 0;
                out.push_back(std::move(info));
            }
        } else {
            out.reserve(playbackCount);
            for (ma_uint32 i = 0; i < playbackCount; ++i) {
                AudioDeviceInfo info;
                info.index = static_cast<int>(i);
                info.name = playbacks[i].name;
                info.isDefault = playbacks[i].isDefault != 0;
                out.push_back(std::move(info));
            }
        }

        return out;
    }

    bool startCaptureDevice(int deviceIndex, uint32_t sampleRate) {
        stopCaptureDevice();

        ma_device_config config = ma_device_config_init(ma_device_type_capture);
        config.capture.format = ma_format_f32;
        config.capture.channels = 1;
        config.sampleRate = sampleRate;
        config.dataCallback = captureCallback;
        config.pUserData = this;

        ma_device_info *playbacks = nullptr;
        ma_uint32 playbackCount = 0;
        ma_device_info *captures = nullptr;
        ma_uint32 captureCount = 0;

        if (ma_context_get_devices(&context, &playbacks, &playbackCount, &captures, &captureCount) != MA_SUCCESS) {
            return false;
        }

        if (deviceIndex >= 0 && static_cast<ma_uint32>(deviceIndex) < captureCount) {
            config.capture.pDeviceID = &captures[static_cast<ma_uint32>(deviceIndex)].id;
        }

        if (ma_device_init(&context, &config, &captureDevice) != MA_SUCCESS) {
            return false;
        }

        if (ma_device_start(&captureDevice) != MA_SUCCESS) {
            ma_device_uninit(&captureDevice);
            return false;
        }

        captureActive = true;
        return true;
    }

    bool startPlaybackDevice(int deviceIndex, uint32_t sampleRate) {
        stopPlaybackDevice();

        ma_device_config config = ma_device_config_init(ma_device_type_playback);
        config.playback.format = ma_format_f32;
        config.playback.channels = 1;
        config.sampleRate = sampleRate;
        config.dataCallback = playbackCallback;
        config.pUserData = this;

        ma_device_info *playbacks = nullptr;
        ma_uint32 playbackCount = 0;
        ma_device_info *captures = nullptr;
        ma_uint32 captureCount = 0;

        if (ma_context_get_devices(&context, &playbacks, &playbackCount, &captures, &captureCount) != MA_SUCCESS) {
            return false;
        }

        if (deviceIndex >= 0 && static_cast<ma_uint32>(deviceIndex) < playbackCount) {
            config.playback.pDeviceID = &playbacks[static_cast<ma_uint32>(deviceIndex)].id;
        }

        if (ma_device_init(&context, &config, &playbackDevice) != MA_SUCCESS) {
            return false;
        }

        if (ma_device_start(&playbackDevice) != MA_SUCCESS) {
            ma_device_uninit(&playbackDevice);
            return false;
        }

        playbackActive = true;
        return true;
    }

    void stopCaptureDevice() {
        if (!captureActive) {
            return;
        }

        ma_device_stop(&captureDevice);
        ma_device_uninit(&captureDevice);
        captureActive = false;
    }

    void stopPlaybackDevice() {
        if (!playbackActive) {
            return;
        }

        ma_device_stop(&playbackDevice);
        ma_device_uninit(&playbackDevice);
        playbackActive = false;
    }
};

AudioEngine::AudioEngine() = default;

AudioEngine::~AudioEngine() {
    shutdown();
}

bool AudioEngine::init() {
    if (impl_ != nullptr) {
        return true;
    }

    auto impl = std::make_unique<Impl>();
    if (ma_context_init(nullptr, 0, nullptr, &impl->context) != MA_SUCCESS) {
        return false;
    }
    impl->contextReady = true;
    impl_ = impl.release();
    return true;
}

void AudioEngine::shutdown() {
    if (impl_ == nullptr) {
        return;
    }

    impl_->stopCaptureDevice();
    impl_->stopPlaybackDevice();

    if (impl_->contextReady) {
        ma_context_uninit(&impl_->context);
        impl_->contextReady = false;
    }

    delete impl_;
    impl_ = nullptr;
}

std::vector<AudioDeviceInfo> AudioEngine::listInputDevices() {
    if (impl_ == nullptr) {
        return {};
    }
    return impl_->listDevices(ma_device_type_capture);
}

std::vector<AudioDeviceInfo> AudioEngine::listOutputDevices() {
    if (impl_ == nullptr) {
        return {};
    }
    return impl_->listDevices(ma_device_type_playback);
}

bool AudioEngine::startCapture(int deviceIndex, uint32_t sampleRate) {
    if (impl_ == nullptr) {
        return false;
    }
    return impl_->startCaptureDevice(deviceIndex, sampleRate);
}

bool AudioEngine::startPlayback(int deviceIndex, uint32_t sampleRate) {
    if (impl_ == nullptr) {
        return false;
    }
    return impl_->startPlaybackDevice(deviceIndex, sampleRate);
}

void AudioEngine::stopCapture() {
    if (impl_ == nullptr) {
        return;
    }
    impl_->stopCaptureDevice();
}

void AudioEngine::stopPlayback() {
    if (impl_ == nullptr) {
        return;
    }
    impl_->stopPlaybackDevice();
}

void AudioEngine::clearCaptureBuffer() {
    if (impl_ == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(impl_->captureMutex);
    impl_->captureQueue.clear();
}

void AudioEngine::clearPlaybackBuffer() {
    if (impl_ == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(impl_->playbackMutex);
    impl_->playbackQueue.clear();
}

std::size_t AudioEngine::popCaptured(std::vector<float> &out, std::size_t maxSamples) {
    out.clear();
    if (impl_ == nullptr || maxSamples == 0) {
        return 0;
    }

    std::lock_guard<std::mutex> lock(impl_->captureMutex);
    const std::size_t n = std::min(maxSamples, impl_->captureQueue.size());
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        out.push_back(impl_->captureQueue.front());
        impl_->captureQueue.pop_front();
    }
    return n;
}

void AudioEngine::pushPlayback(const std::vector<float> &samples) {
    if (impl_ == nullptr || samples.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(impl_->playbackMutex);
    impl_->playbackQueue.insert(impl_->playbackQueue.end(), samples.begin(), samples.end());
    trimDeque(impl_->playbackQueue, kMaxPlaybackQueueSamples);
}

bool AudioEngine::captureRunning() const {
    return impl_ != nullptr && impl_->captureActive;
}

bool AudioEngine::playbackRunning() const {
    return impl_ != nullptr && impl_->playbackActive;
}
