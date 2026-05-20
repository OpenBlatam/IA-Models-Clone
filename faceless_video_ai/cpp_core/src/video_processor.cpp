#include "video_processor.hpp"
#include <cmath>
#include <algorithm>
#include <thread>

namespace faceless_video {

class VideoProcessor::Impl {
public:
    Impl() {
        // Initialize FFmpeg (for older versions)
        #if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 9, 100)
        av_register_all();
        #endif
    }
    
    ~Impl() = default;
    
    // Helper to open input file
    FormatContextPtr open_input(const std::filesystem::path& path) {
        AVFormatContext* ctx = nullptr;
        int ret = avformat_open_input(&ctx, path.string().c_str(), nullptr, nullptr);
        if (ret < 0) {
            throw VideoProcessingError("Failed to open input: " + path.string() + 
                                       " - " + av_error_string(ret));
        }
        
        ret = avformat_find_stream_info(ctx, nullptr);
        if (ret < 0) {
            avformat_close_input(&ctx);
            throw VideoProcessingError("Failed to find stream info: " + av_error_string(ret));
        }
        
        return FormatContextPtr(ctx);
    }
    
    // Find best video stream
    int find_video_stream(AVFormatContext* ctx) {
        int idx = av_find_best_stream(ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (idx < 0) {
            throw VideoProcessingError("No video stream found");
        }
        return idx;
    }
    
    // Find best audio stream
    int find_audio_stream(AVFormatContext* ctx) {
        return av_find_best_stream(ctx, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    }
    
    // Create decoder context
    CodecContextPtr create_decoder(AVFormatContext* fmt_ctx, int stream_idx) {
        AVCodecParameters* codecpar = fmt_ctx->streams[stream_idx]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
        if (!codec) {
            throw CodecError("Decoder not found for codec: " + 
                           std::string(avcodec_get_name(codecpar->codec_id)));
        }
        
        AVCodecContext* ctx = avcodec_alloc_context3(codec);
        if (!ctx) {
            throw CodecError("Failed to allocate decoder context");
        }
        
        int ret = avcodec_parameters_to_context(ctx, codecpar);
        if (ret < 0) {
            avcodec_free_context(&ctx);
            throw CodecError("Failed to copy codec parameters: " + av_error_string(ret));
        }
        
        ret = avcodec_open2(ctx, codec, nullptr);
        if (ret < 0) {
            avcodec_free_context(&ctx);
            throw CodecError("Failed to open decoder: " + av_error_string(ret));
        }
        
        return CodecContextPtr(ctx);
    }
    
    // Create encoder context
    CodecContextPtr create_encoder(
        const std::string& codec_name,
        int width, int height,
        AVPixelFormat pix_fmt,
        double fps,
        int64_t bitrate,
        const EncodingOptions& options
    ) {
        const AVCodec* codec = avcodec_find_encoder_by_name(codec_name.c_str());
        if (!codec) {
            codec = avcodec_find_encoder(AV_CODEC_ID_H264);
        }
        if (!codec) {
            throw CodecError("Encoder not found: " + codec_name);
        }
        
        AVCodecContext* ctx = avcodec_alloc_context3(codec);
        if (!ctx) {
            throw CodecError("Failed to allocate encoder context");
        }
        
        ctx->width = width;
        ctx->height = height;
        ctx->time_base = AVRational{1, static_cast<int>(fps * 1000)};
        ctx->framerate = AVRational{static_cast<int>(fps * 1000), 1000};
        ctx->pix_fmt = pix_fmt;
        ctx->gop_size = options.keyframe_interval;
        ctx->max_b_frames = 2;
        
        if (bitrate > 0) {
            ctx->bit_rate = bitrate;
        }
        
        // Set CRF for quality-based encoding
        if (options.crf.has_value()) {
            av_opt_set_int(ctx->priv_data, "crf", *options.crf, 0);
        }
        
        // Set preset
        const char* preset_str = get_preset_string(options.preset);
        av_opt_set(ctx->priv_data, "preset", preset_str, 0);
        
        int ret = avcodec_open2(ctx, codec, nullptr);
        if (ret < 0) {
            avcodec_free_context(&ctx);
            throw CodecError("Failed to open encoder: " + av_error_string(ret));
        }
        
        return CodecContextPtr(ctx);
    }
    
    const char* get_preset_string(EncodingPreset preset) {
        switch (preset) {
            case EncodingPreset::Ultrafast: return "ultrafast";
            case EncodingPreset::Superfast: return "superfast";
            case EncodingPreset::Veryfast: return "veryfast";
            case EncodingPreset::Faster: return "faster";
            case EncodingPreset::Fast: return "fast";
            case EncodingPreset::Medium: return "medium";
            case EncodingPreset::Slow: return "slow";
            case EncodingPreset::Slower: return "slower";
            case EncodingPreset::Veryslow: return "veryslow";
            case EncodingPreset::Placebo: return "placebo";
            default: return "medium";
        }
    }
    
    // Convert frame to cv::Mat
    cv::Mat frame_to_mat(AVFrame* frame) {
        cv::Mat mat(frame->height, frame->width, CV_8UC3);
        
        SwsContext* sws_ctx = sws_getContext(
            frame->width, frame->height, (AVPixelFormat)frame->format,
            frame->width, frame->height, AV_PIX_FMT_BGR24,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );
        
        if (!sws_ctx) {
            throw VideoProcessingError("Failed to create sws context");
        }
        
        uint8_t* dst_data[4] = { mat.data, nullptr, nullptr, nullptr };
        int dst_linesize[4] = { static_cast<int>(mat.step[0]), 0, 0, 0 };
        
        sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
                  dst_data, dst_linesize);
        
        sws_freeContext(sws_ctx);
        return mat;
    }
    
    // Convert cv::Mat to frame
    void mat_to_frame(const cv::Mat& mat, AVFrame* frame) {
        SwsContext* sws_ctx = sws_getContext(
            mat.cols, mat.rows, AV_PIX_FMT_BGR24,
            frame->width, frame->height, (AVPixelFormat)frame->format,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );
        
        if (!sws_ctx) {
            throw VideoProcessingError("Failed to create sws context");
        }
        
        const uint8_t* src_data[4] = { mat.data, nullptr, nullptr, nullptr };
        int src_linesize[4] = { static_cast<int>(mat.step[0]), 0, 0, 0 };
        
        sws_scale(sws_ctx, src_data, src_linesize, 0, mat.rows,
                  frame->data, frame->linesize);
        
        sws_freeContext(sws_ctx);
    }
};

VideoProcessor::VideoProcessor() : impl_(std::make_unique<Impl>()) {}
VideoProcessor::~VideoProcessor() = default;
VideoProcessor::VideoProcessor(VideoProcessor&&) noexcept = default;
VideoProcessor& VideoProcessor::operator=(VideoProcessor&&) noexcept = default;

void VideoProcessor::apply_ken_burns(
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path,
    const KenBurnsParams& params,
    double fps,
    ProgressCallback progress
) {
    // Load image with OpenCV
    cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw VideoProcessingError("Failed to load image: " + image_path.string());
    }
    
    int total_frames = static_cast<int>(params.duration * fps);
    
    // Calculate output dimensions (maintain aspect ratio of common formats)
    int out_width = image.cols;
    int out_height = image.rows;
    
    // Adjust for common video aspect ratios
    double aspect = static_cast<double>(out_width) / out_height;
    if (aspect > 16.0/9.0) {
        out_width = static_cast<int>(out_height * 16.0 / 9.0);
    } else if (aspect < 9.0/16.0) {
        out_height = static_cast<int>(out_width * 16.0 / 9.0);
    }
    
    // Make dimensions even (required for many codecs)
    out_width = (out_width / 2) * 2;
    out_height = (out_height / 2) * 2;
    
    // Setup encoder
    EncodingOptions options;
    options.preset = EncodingPreset::Fast;
    options.quality = QualityLevel::High;
    
    auto encoder_ctx = impl_->create_encoder(
        "libx264", out_width, out_height,
        AV_PIX_FMT_YUV420P, fps, 0, options
    );
    
    // Setup output format context
    AVFormatContext* out_ctx = nullptr;
    avformat_alloc_output_context2(&out_ctx, nullptr, nullptr, output_path.string().c_str());
    if (!out_ctx) {
        throw VideoProcessingError("Failed to create output context");
    }
    
    AVStream* stream = avformat_new_stream(out_ctx, nullptr);
    if (!stream) {
        avformat_free_context(out_ctx);
        throw VideoProcessingError("Failed to create output stream");
    }
    
    avcodec_parameters_from_context(stream->codecpar, encoder_ctx.get());
    stream->time_base = encoder_ctx->time_base;
    
    if (!(out_ctx->oformat->flags & AVFMT_NOFILE)) {
        int ret = avio_open(&out_ctx->pb, output_path.string().c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            avformat_free_context(out_ctx);
            throw VideoProcessingError("Failed to open output file: " + av_error_string(ret));
        }
    }
    
    int ret = avformat_write_header(out_ctx, nullptr);
    if (ret < 0) {
        avio_closep(&out_ctx->pb);
        avformat_free_context(out_ctx);
        throw VideoProcessingError("Failed to write header: " + av_error_string(ret));
    }
    
    // Allocate frame
    AVFrame* frame = av_frame_alloc();
    frame->format = AV_PIX_FMT_YUV420P;
    frame->width = out_width;
    frame->height = out_height;
    av_frame_get_buffer(frame, 0);
    
    AVPacket* pkt = av_packet_alloc();
    
    // Generate frames with Ken Burns effect
    for (int i = 0; i < total_frames; ++i) {
        double t = static_cast<double>(i) / total_frames;  // 0.0 to 1.0
        
        // Interpolate zoom and position
        double zoom = params.start_zoom + (params.end_zoom - params.start_zoom) * t;
        double cx = params.start_x + (params.end_x - params.start_x) * t;
        double cy = params.start_y + (params.end_y - params.start_y) * t;
        
        // Calculate crop region
        int crop_w = static_cast<int>(image.cols / zoom);
        int crop_h = static_cast<int>(image.rows / zoom);
        int crop_x = static_cast<int>((image.cols - crop_w) * cx);
        int crop_y = static_cast<int>((image.rows - crop_h) * cy);
        
        // Clamp values
        crop_x = std::clamp(crop_x, 0, image.cols - crop_w);
        crop_y = std::clamp(crop_y, 0, image.rows - crop_h);
        
        // Extract and resize
        cv::Mat cropped = image(cv::Rect(crop_x, crop_y, crop_w, crop_h));
        cv::Mat resized;
        cv::resize(cropped, resized, cv::Size(out_width, out_height), 0, 0, cv::INTER_LANCZOS4);
        
        // Convert to frame
        av_frame_make_writable(frame);
        impl_->mat_to_frame(resized, frame);
        frame->pts = i * (encoder_ctx->time_base.den / encoder_ctx->time_base.num) / fps;
        
        // Encode
        ret = avcodec_send_frame(encoder_ctx.get(), frame);
        while (ret >= 0) {
            ret = avcodec_receive_packet(encoder_ctx.get(), pkt);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) {
                throw VideoProcessingError("Error encoding frame: " + av_error_string(ret));
            }
            
            av_packet_rescale_ts(pkt, encoder_ctx->time_base, stream->time_base);
            pkt->stream_index = stream->index;
            
            ret = av_interleaved_write_frame(out_ctx, pkt);
            av_packet_unref(pkt);
        }
        
        if (progress) {
            progress(static_cast<double>(i + 1) / total_frames);
        }
    }
    
    // Flush encoder
    avcodec_send_frame(encoder_ctx.get(), nullptr);
    while (true) {
        ret = avcodec_receive_packet(encoder_ctx.get(), pkt);
        if (ret == AVERROR_EOF) break;
        if (ret < 0) break;
        
        av_packet_rescale_ts(pkt, encoder_ctx->time_base, stream->time_base);
        pkt->stream_index = stream->index;
        av_interleaved_write_frame(out_ctx, pkt);
        av_packet_unref(pkt);
    }
    
    av_write_trailer(out_ctx);
    
    // Cleanup
    av_frame_free(&frame);
    av_packet_free(&pkt);
    if (!(out_ctx->oformat->flags & AVFMT_NOFILE)) {
        avio_closep(&out_ctx->pb);
    }
    avformat_free_context(out_ctx);
}

VideoInfo VideoProcessor::get_video_info(const std::filesystem::path& video_path) {
    auto fmt_ctx = impl_->open_input(video_path);
    int video_idx = impl_->find_video_stream(fmt_ctx.get());
    
    AVStream* stream = fmt_ctx->streams[video_idx];
    AVCodecParameters* codecpar = stream->codecpar;
    
    VideoInfo info;
    info.resolution.width = codecpar->width;
    info.resolution.height = codecpar->height;
    info.duration = fmt_ctx->duration / (double)AV_TIME_BASE;
    
    if (stream->avg_frame_rate.den != 0) {
        info.fps = av_q2d(stream->avg_frame_rate);
    } else if (stream->r_frame_rate.den != 0) {
        info.fps = av_q2d(stream->r_frame_rate);
    } else {
        info.fps = 30.0;  // Default
    }
    
    info.bitrate = codecpar->bit_rate;
    info.codec = avcodec_get_name(codecpar->codec_id);
    
    const AVPixFmtDescriptor* pix_desc = av_pix_fmt_desc_get((AVPixelFormat)codecpar->format);
    info.pixel_format = pix_desc ? pix_desc->name : "unknown";
    
    return info;
}

void VideoProcessor::compose_clips(
    const std::vector<std::filesystem::path>& clips,
    const std::filesystem::path& output_path,
    const std::vector<TransitionParams>& transitions,
    const EncodingOptions& options,
    ProgressCallback progress
) {
    if (clips.empty()) {
        throw VideoProcessingError("No clips provided for composition");
    }
    
    // Get info from first clip for output format
    VideoInfo first_info = get_video_info(clips[0]);
    
    // TODO: Implement full composition with transitions
    // This is a placeholder for the complex implementation
    
    // For now, concatenate without transitions using FFmpeg filter
    std::string filter_complex;
    std::vector<FormatContextPtr> input_ctxs;
    
    // Open all inputs
    for (size_t i = 0; i < clips.size(); ++i) {
        input_ctxs.push_back(impl_->open_input(clips[i]));
        filter_complex += "[" + std::to_string(i) + ":v]";
    }
    
    filter_complex += "concat=n=" + std::to_string(clips.size()) + ":v=1:a=0[outv]";
    
    // The actual implementation would use FFmpeg's filter graph API
    // This is simplified for demonstration
}

void VideoProcessor::optimize_for_web(
    const std::filesystem::path& input_path,
    const std::filesystem::path& output_path,
    std::optional<Resolution> target_resolution,
    const EncodingOptions& options,
    ProgressCallback progress
) {
    auto fmt_ctx = impl_->open_input(input_path);
    int video_idx = impl_->find_video_stream(fmt_ctx.get());
    int audio_idx = impl_->find_audio_stream(fmt_ctx.get());
    
    AVStream* in_video = fmt_ctx->streams[video_idx];
    VideoInfo info = get_video_info(input_path);
    
    // Determine output resolution
    int out_width = target_resolution ? target_resolution->width : info.resolution.width;
    int out_height = target_resolution ? target_resolution->height : info.resolution.height;
    
    // Make even
    out_width = (out_width / 2) * 2;
    out_height = (out_height / 2) * 2;
    
    // Create decoder
    auto decoder_ctx = impl_->create_decoder(fmt_ctx.get(), video_idx);
    
    // Create encoder with web-optimized settings
    EncodingOptions web_options = options;
    if (!web_options.crf.has_value()) {
        web_options.crf = 23;  // Good quality/size balance for web
    }
    
    auto encoder_ctx = impl_->create_encoder(
        "libx264", out_width, out_height,
        AV_PIX_FMT_YUV420P, info.fps,
        web_options.target_bitrate.value_or(0),
        web_options
    );
    
    // Set web-specific encoder options
    av_opt_set(encoder_ctx->priv_data, "profile", "high", 0);
    av_opt_set(encoder_ctx->priv_data, "level", "4.1", 0);
    av_opt_set_int(encoder_ctx->priv_data, "movflags", 1, 0);  // faststart
    
    // Setup output
    AVFormatContext* out_ctx = nullptr;
    avformat_alloc_output_context2(&out_ctx, nullptr, "mp4", output_path.string().c_str());
    if (!out_ctx) {
        throw VideoProcessingError("Failed to create output context");
    }
    
    AVStream* out_stream = avformat_new_stream(out_ctx, nullptr);
    avcodec_parameters_from_context(out_stream->codecpar, encoder_ctx.get());
    out_stream->time_base = encoder_ctx->time_base;
    
    if (!(out_ctx->oformat->flags & AVFMT_NOFILE)) {
        avio_open(&out_ctx->pb, output_path.string().c_str(), AVIO_FLAG_WRITE);
    }
    
    // Set faststart for web streaming
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "movflags", "+faststart", 0);
    avformat_write_header(out_ctx, &opts);
    av_dict_free(&opts);
    
    // Process frames
    AVFrame* frame = av_frame_alloc();
    AVFrame* out_frame = av_frame_alloc();
    out_frame->format = AV_PIX_FMT_YUV420P;
    out_frame->width = out_width;
    out_frame->height = out_height;
    av_frame_get_buffer(out_frame, 0);
    
    AVPacket* pkt = av_packet_alloc();
    
    SwsContext* sws_ctx = sws_getContext(
        decoder_ctx->width, decoder_ctx->height, decoder_ctx->pix_fmt,
        out_width, out_height, AV_PIX_FMT_YUV420P,
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );
    
    int64_t frame_count = 0;
    int64_t total_frames = static_cast<int64_t>(info.duration * info.fps);
    
    while (av_read_frame(fmt_ctx.get(), pkt) >= 0) {
        if (pkt->stream_index == video_idx) {
            int ret = avcodec_send_packet(decoder_ctx.get(), pkt);
            while (ret >= 0) {
                ret = avcodec_receive_frame(decoder_ctx.get(), frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                if (ret < 0) break;
                
                // Scale frame
                av_frame_make_writable(out_frame);
                sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
                         out_frame->data, out_frame->linesize);
                out_frame->pts = frame->pts;
                
                // Encode
                ret = avcodec_send_frame(encoder_ctx.get(), out_frame);
                while (ret >= 0) {
                    AVPacket* enc_pkt = av_packet_alloc();
                    ret = avcodec_receive_packet(encoder_ctx.get(), enc_pkt);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        av_packet_free(&enc_pkt);
                        break;
                    }
                    
                    av_packet_rescale_ts(enc_pkt, encoder_ctx->time_base, out_stream->time_base);
                    enc_pkt->stream_index = 0;
                    av_interleaved_write_frame(out_ctx, enc_pkt);
                    av_packet_free(&enc_pkt);
                }
                
                frame_count++;
                if (progress && total_frames > 0) {
                    progress(static_cast<double>(frame_count) / total_frames);
                }
            }
        }
        av_packet_unref(pkt);
    }
    
    // Flush encoder
    avcodec_send_frame(encoder_ctx.get(), nullptr);
    while (true) {
        AVPacket* enc_pkt = av_packet_alloc();
        int ret = avcodec_receive_packet(encoder_ctx.get(), enc_pkt);
        if (ret == AVERROR_EOF) {
            av_packet_free(&enc_pkt);
            break;
        }
        if (ret >= 0) {
            av_packet_rescale_ts(enc_pkt, encoder_ctx->time_base, out_stream->time_base);
            enc_pkt->stream_index = 0;
            av_interleaved_write_frame(out_ctx, enc_pkt);
        }
        av_packet_free(&enc_pkt);
        if (ret < 0) break;
    }
    
    av_write_trailer(out_ctx);
    
    // Cleanup
    sws_freeContext(sws_ctx);
    av_frame_free(&frame);
    av_frame_free(&out_frame);
    av_packet_free(&pkt);
    if (!(out_ctx->oformat->flags & AVFMT_NOFILE)) {
        avio_closep(&out_ctx->pb);
    }
    avformat_free_context(out_ctx);
}

void VideoProcessor::add_audio_track(
    const std::filesystem::path& video_path,
    const std::filesystem::path& audio_path,
    const std::filesystem::path& output_path,
    double volume,
    bool replace_existing
) {
    // This would use FFmpeg's audio mixing capabilities
    // Placeholder implementation
}

void VideoProcessor::extract_frames(
    const std::filesystem::path& video_path,
    const std::filesystem::path& output_dir,
    double fps,
    const std::string& format
) {
    std::filesystem::create_directories(output_dir);
    
    auto fmt_ctx = impl_->open_input(video_path);
    int video_idx = impl_->find_video_stream(fmt_ctx.get());
    auto decoder_ctx = impl_->create_decoder(fmt_ctx.get(), video_idx);
    
    AVFrame* frame = av_frame_alloc();
    AVPacket* pkt = av_packet_alloc();
    
    int frame_count = 0;
    double video_fps = av_q2d(fmt_ctx->streams[video_idx]->avg_frame_rate);
    int frame_skip = static_cast<int>(video_fps / fps);
    if (frame_skip < 1) frame_skip = 1;
    
    while (av_read_frame(fmt_ctx.get(), pkt) >= 0) {
        if (pkt->stream_index == video_idx) {
            int ret = avcodec_send_packet(decoder_ctx.get(), pkt);
            while (ret >= 0) {
                ret = avcodec_receive_frame(decoder_ctx.get(), frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                if (ret < 0) break;
                
                if (frame_count % frame_skip == 0) {
                    cv::Mat mat = impl_->frame_to_mat(frame);
                    std::string filename = output_dir.string() + "/frame_" + 
                                          std::to_string(frame_count / frame_skip) + "." + format;
                    cv::imwrite(filename, mat);
                }
                
                frame_count++;
            }
        }
        av_packet_unref(pkt);
    }
    
    av_frame_free(&frame);
    av_packet_free(&pkt);
}

void VideoProcessor::trim(
    const std::filesystem::path& input_path,
    const std::filesystem::path& output_path,
    const TimeRange& range,
    bool reencode
) {
    // Implementation using FFmpeg seek and copy/reencode
}

void VideoProcessor::scale(
    const std::filesystem::path& input_path,
    const std::filesystem::path& output_path,
    const Resolution& resolution,
    bool maintain_aspect
) {
    optimize_for_web(input_path, output_path, resolution, {});
}

} // namespace faceless_video




