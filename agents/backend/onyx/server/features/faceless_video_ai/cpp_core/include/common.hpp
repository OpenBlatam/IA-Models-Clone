#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <functional>
#include <stdexcept>
#include <cstdint>

// FFmpeg includes
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
}

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace faceless_video {

// Custom exception types
class VideoProcessingError : public std::runtime_error {
public:
    explicit VideoProcessingError(const std::string& msg) 
        : std::runtime_error(msg) {}
};

class AudioProcessingError : public std::runtime_error {
public:
    explicit AudioProcessingError(const std::string& msg) 
        : std::runtime_error(msg) {}
};

class CodecError : public std::runtime_error {
public:
    explicit CodecError(const std::string& msg) 
        : std::runtime_error(msg) {}
};

// Common types
struct Resolution {
    int width;
    int height;
};

struct TimeRange {
    double start;
    double end;
    
    double duration() const { return end - start; }
};

struct VideoInfo {
    Resolution resolution;
    double duration;
    double fps;
    int64_t bitrate;
    std::string codec;
    std::string pixel_format;
};

struct AudioInfo {
    int sample_rate;
    int channels;
    int64_t bitrate;
    double duration;
    std::string codec;
    std::string sample_format;
};

// Progress callback type
using ProgressCallback = std::function<void(double progress)>;

// RAII wrapper for FFmpeg structures
template<typename T, auto Deleter>
class FFmpegPtr {
public:
    FFmpegPtr() : ptr_(nullptr) {}
    explicit FFmpegPtr(T* ptr) : ptr_(ptr) {}
    ~FFmpegPtr() { reset(); }
    
    // Move semantics
    FFmpegPtr(FFmpegPtr&& other) noexcept : ptr_(other.release()) {}
    FFmpegPtr& operator=(FFmpegPtr&& other) noexcept {
        reset(other.release());
        return *this;
    }
    
    // No copy
    FFmpegPtr(const FFmpegPtr&) = delete;
    FFmpegPtr& operator=(const FFmpegPtr&) = delete;
    
    T* get() const { return ptr_; }
    T* operator->() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }
    
    T** operator&() { return &ptr_; }
    
    T* release() {
        T* tmp = ptr_;
        ptr_ = nullptr;
        return tmp;
    }
    
    void reset(T* ptr = nullptr) {
        if (ptr_) {
            Deleter(&ptr_);
        }
        ptr_ = ptr;
    }

private:
    T* ptr_;
};

// Common FFmpeg RAII types
inline void free_format_context(AVFormatContext** ctx) {
    if (*ctx) avformat_close_input(ctx);
}

inline void free_codec_context(AVCodecContext** ctx) {
    if (*ctx) avcodec_free_context(ctx);
}

inline void free_frame(AVFrame** frame) {
    if (*frame) av_frame_free(frame);
}

inline void free_packet(AVPacket** pkt) {
    if (*pkt) av_packet_free(pkt);
}

inline void free_sws_context(SwsContext** ctx) {
    if (*ctx) sws_freeContext(*ctx);
    *ctx = nullptr;
}

using FormatContextPtr = FFmpegPtr<AVFormatContext, free_format_context>;
using CodecContextPtr = FFmpegPtr<AVCodecContext, free_codec_context>;
using FramePtr = FFmpegPtr<AVFrame, free_frame>;
using PacketPtr = FFmpegPtr<AVPacket, free_packet>;

// Utility functions
inline std::string av_error_string(int errnum) {
    char buf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(errnum, buf, sizeof(buf));
    return std::string(buf);
}

} // namespace faceless_video




