#include "media_ffmpeg.h"

#include <algorithm>
#include <memory>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}

namespace {
struct AvFormatContextDeleter {
    void operator()(AVFormatContext *ctx) const {
        if (ctx != nullptr) {
            avformat_close_input(&ctx);
        }
    }
};

struct AvCodecContextDeleter {
    void operator()(AVCodecContext *ctx) const {
        if (ctx != nullptr) {
            avcodec_free_context(&ctx);
        }
    }
};

struct AvPacketDeleter {
    void operator()(AVPacket *pkt) const {
        if (pkt != nullptr) {
            av_packet_free(&pkt);
        }
    }
};

struct AvFrameDeleter {
    void operator()(AVFrame *frame) const {
        if (frame != nullptr) {
            av_frame_free(&frame);
        }
    }
};

struct SwrContextDeleter {
    void operator()(SwrContext *ctx) const {
        if (ctx != nullptr) {
            swr_free(&ctx);
        }
    }
};

std::string ffmpegError(int errnum) {
    char buf[256] = {};
    av_strerror(errnum, buf, sizeof(buf));
    return std::string(buf);
}

bool appendResampledFrame(SwrContext *swr,
                          const AVFrame *frame,
                          std::vector<float> &out,
                          std::string &error) {
    if (swr == nullptr || frame == nullptr) {
        error = "invalid resampler/frame";
        return false;
    }

    const int outSamples = av_rescale_rnd(swr_get_delay(swr, frame->sample_rate) + frame->nb_samples,
                                          frame->sample_rate,
                                          frame->sample_rate,
                                          AV_ROUND_UP);
    if (outSamples <= 0) {
        return true;
    }

    std::vector<float> tmp(static_cast<std::size_t>(outSamples));
    uint8_t *dstData[1] = {reinterpret_cast<uint8_t *>(tmp.data())};

    const int produced = swr_convert(swr,
                                     dstData,
                                     outSamples,
                                     const_cast<const uint8_t **>(frame->extended_data),
                                     frame->nb_samples);
    if (produced < 0) {
        error = "swr_convert failed: " + ffmpegError(produced);
        return false;
    }

    if (produced > 0) {
        out.insert(out.end(), tmp.begin(), tmp.begin() + produced);
    }

    return true;
}
} // namespace

bool decodeMediaAudioToMonoF32(const std::string &path,
                               uint32_t targetSampleRate,
                               std::vector<float> &outPcm,
                               std::string &error) {
    outPcm.clear();
    error.clear();

    AVFormatContext *rawFmt = nullptr;
    int rc = avformat_open_input(&rawFmt, path.c_str(), nullptr, nullptr);
    if (rc < 0) {
        error = "avformat_open_input failed: " + ffmpegError(rc);
        return false;
    }
    std::unique_ptr<AVFormatContext, AvFormatContextDeleter> fmt(rawFmt);

    rc = avformat_find_stream_info(fmt.get(), nullptr);
    if (rc < 0) {
        error = "avformat_find_stream_info failed: " + ffmpegError(rc);
        return false;
    }

    int audioStream = av_find_best_stream(fmt.get(), AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (audioStream < 0) {
        error = "no audio stream in media";
        return false;
    }

    AVStream *stream = fmt->streams[audioStream];
    if (stream == nullptr || stream->codecpar == nullptr) {
        error = "invalid audio stream";
        return false;
    }

    const AVCodec *codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (codec == nullptr) {
        error = "decoder not found for audio stream";
        return false;
    }

    AVCodecContext *rawCodec = avcodec_alloc_context3(codec);
    if (rawCodec == nullptr) {
        error = "avcodec_alloc_context3 failed";
        return false;
    }
    std::unique_ptr<AVCodecContext, AvCodecContextDeleter> codecCtx(rawCodec);

    rc = avcodec_parameters_to_context(codecCtx.get(), stream->codecpar);
    if (rc < 0) {
        error = "avcodec_parameters_to_context failed: " + ffmpegError(rc);
        return false;
    }

    rc = avcodec_open2(codecCtx.get(), codec, nullptr);
    if (rc < 0) {
        error = "avcodec_open2 failed: " + ffmpegError(rc);
        return false;
    }

    const uint32_t sampleRate = std::max<uint32_t>(8000, targetSampleRate);

    const AVChannelLayout outLayout = AV_CHANNEL_LAYOUT_MONO;

    SwrContext *rawSwr = nullptr;
    rc = swr_alloc_set_opts2(&rawSwr,
                             &outLayout,
                             AV_SAMPLE_FMT_FLT,
                             static_cast<int>(sampleRate),
                             &codecCtx->ch_layout,
                             codecCtx->sample_fmt,
                             codecCtx->sample_rate,
                             0,
                             nullptr);
    if (rc < 0 || rawSwr == nullptr) {
        error = "swr_alloc_set_opts2 failed: " + ffmpegError(rc);
        return false;
    }
    std::unique_ptr<SwrContext, SwrContextDeleter> swr(rawSwr);

    rc = swr_init(swr.get());
    if (rc < 0) {
        error = "swr_init failed: " + ffmpegError(rc);
        return false;
    }

    std::unique_ptr<AVPacket, AvPacketDeleter> pkt(av_packet_alloc());
    std::unique_ptr<AVFrame, AvFrameDeleter> frame(av_frame_alloc());
    if (!pkt || !frame) {
        error = "failed to allocate ffmpeg packet/frame";
        return false;
    }

    while ((rc = av_read_frame(fmt.get(), pkt.get())) >= 0) {
        if (pkt->stream_index != audioStream) {
            av_packet_unref(pkt.get());
            continue;
        }

        rc = avcodec_send_packet(codecCtx.get(), pkt.get());
        av_packet_unref(pkt.get());
        if (rc < 0) {
            error = "avcodec_send_packet failed: " + ffmpegError(rc);
            return false;
        }

        while ((rc = avcodec_receive_frame(codecCtx.get(), frame.get())) >= 0) {
            if (!appendResampledFrame(swr.get(), frame.get(), outPcm, error)) {
                return false;
            }
            av_frame_unref(frame.get());
        }

        if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) {
            continue;
        }
        if (rc < 0) {
            error = "avcodec_receive_frame failed: " + ffmpegError(rc);
            return false;
        }
    }

    rc = avcodec_send_packet(codecCtx.get(), nullptr);
    if (rc < 0) {
        error = "avcodec_send_packet(flush) failed: " + ffmpegError(rc);
        return false;
    }

    while ((rc = avcodec_receive_frame(codecCtx.get(), frame.get())) >= 0) {
        if (!appendResampledFrame(swr.get(), frame.get(), outPcm, error)) {
            return false;
        }
        av_frame_unref(frame.get());
    }

    if (outPcm.empty()) {
        error = "no decoded audio samples";
        return false;
    }

    return true;
}

