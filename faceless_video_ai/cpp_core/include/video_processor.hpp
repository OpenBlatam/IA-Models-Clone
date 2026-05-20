#pragma once

#include "common.hpp"
#include <filesystem>

namespace faceless_video {

// Encoding presets
enum class EncodingPreset {
    Ultrafast,
    Superfast,
    Veryfast,
    Faster,
    Fast,
    Medium,
    Slow,
    Slower,
    Veryslow,
    Placebo
};

// Quality settings
enum class QualityLevel {
    Low,      // CRF 28-32
    Medium,   // CRF 23-27
    High,     // CRF 18-22
    Maximum   // CRF 15-17
};

// Codec options
struct EncodingOptions {
    EncodingPreset preset = EncodingPreset::Medium;
    QualityLevel quality = QualityLevel::High;
    std::optional<int64_t> target_bitrate;
    std::optional<int64_t> max_bitrate;
    std::optional<int> crf;
    bool two_pass = false;
    std::string pixel_format = "yuv420p";
    int keyframe_interval = 250;
    
    // Audio options
    int audio_bitrate = 192000;
    int audio_sample_rate = 48000;
};

// Ken Burns effect parameters
struct KenBurnsParams {
    double start_zoom = 1.0;
    double end_zoom = 1.2;
    double start_x = 0.5;  // Center position (0-1)
    double start_y = 0.5;
    double end_x = 0.5;
    double end_y = 0.5;
    double duration = 5.0;
};

// Transition types
enum class TransitionType {
    None,
    Fade,
    CrossFade,
    Wipe,
    Slide,
    Zoom,
    Dissolve,
    CircleWipe,
    DiagonalWipe
};

struct TransitionParams {
    TransitionType type = TransitionType::CrossFade;
    double duration = 0.5;
    std::string easing = "easeInOut";
};

class VideoProcessor {
public:
    VideoProcessor();
    ~VideoProcessor();
    
    // Disable copy
    VideoProcessor(const VideoProcessor&) = delete;
    VideoProcessor& operator=(const VideoProcessor&) = delete;
    
    // Enable move
    VideoProcessor(VideoProcessor&&) noexcept;
    VideoProcessor& operator=(VideoProcessor&&) noexcept;
    
    /**
     * Apply Ken Burns effect to a still image
     * 
     * @param image_path Path to input image
     * @param output_path Path for output video
     * @param params Ken Burns parameters
     * @param fps Output frame rate
     * @param progress Optional progress callback
     */
    void apply_ken_burns(
        const std::filesystem::path& image_path,
        const std::filesystem::path& output_path,
        const KenBurnsParams& params,
        double fps = 30.0,
        ProgressCallback progress = nullptr
    );
    
    /**
     * Compose multiple clips into a single video
     * 
     * @param clips Vector of input video paths
     * @param output_path Path for output video
     * @param transitions Transitions between clips
     * @param options Encoding options
     * @param progress Optional progress callback
     */
    void compose_clips(
        const std::vector<std::filesystem::path>& clips,
        const std::filesystem::path& output_path,
        const std::vector<TransitionParams>& transitions,
        const EncodingOptions& options = {},
        ProgressCallback progress = nullptr
    );
    
    /**
     * Optimize video for web delivery
     * 
     * @param input_path Input video path
     * @param output_path Output video path
     * @param target_resolution Target resolution (optional)
     * @param options Encoding options
     * @param progress Optional progress callback
     */
    void optimize_for_web(
        const std::filesystem::path& input_path,
        const std::filesystem::path& output_path,
        std::optional<Resolution> target_resolution = std::nullopt,
        const EncodingOptions& options = {},
        ProgressCallback progress = nullptr
    );
    
    /**
     * Add audio track to video
     * 
     * @param video_path Input video path
     * @param audio_path Audio track path
     * @param output_path Output video path
     * @param volume Audio volume (0.0 - 2.0)
     * @param mix_mode How to mix with existing audio
     */
    void add_audio_track(
        const std::filesystem::path& video_path,
        const std::filesystem::path& audio_path,
        const std::filesystem::path& output_path,
        double volume = 1.0,
        bool replace_existing = false
    );
    
    /**
     * Extract frames from video
     * 
     * @param video_path Input video path
     * @param output_dir Output directory for frames
     * @param fps Frames per second to extract
     * @param format Output format (png, jpg)
     */
    void extract_frames(
        const std::filesystem::path& video_path,
        const std::filesystem::path& output_dir,
        double fps = 1.0,
        const std::string& format = "png"
    );
    
    /**
     * Get video information
     * 
     * @param video_path Path to video
     * @return VideoInfo structure
     */
    VideoInfo get_video_info(const std::filesystem::path& video_path);
    
    /**
     * Trim video to specified range
     * 
     * @param input_path Input video path
     * @param output_path Output video path
     * @param range Time range to keep
     * @param reencode Whether to re-encode (slower but more accurate)
     */
    void trim(
        const std::filesystem::path& input_path,
        const std::filesystem::path& output_path,
        const TimeRange& range,
        bool reencode = false
    );
    
    /**
     * Scale video to target resolution
     * 
     * @param input_path Input video path
     * @param output_path Output video path
     * @param resolution Target resolution
     * @param maintain_aspect Whether to maintain aspect ratio
     */
    void scale(
        const std::filesystem::path& input_path,
        const std::filesystem::path& output_path,
        const Resolution& resolution,
        bool maintain_aspect = true
    );

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace faceless_video




