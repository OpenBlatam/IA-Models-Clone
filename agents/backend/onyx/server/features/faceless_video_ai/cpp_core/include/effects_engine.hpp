#pragma once

#include "common.hpp"
#include <filesystem>
#include <variant>

namespace faceless_video {

// Effect types
enum class EffectType {
    // Color adjustments
    Brightness,
    Contrast,
    Saturation,
    Hue,
    Gamma,
    ColorBalance,
    
    // Filters
    Blur,
    Sharpen,
    EdgeDetect,
    Emboss,
    
    // Artistic
    Vignette,
    FilmGrain,
    ChromaticAberration,
    Glitch,
    
    // Transformations
    Rotate,
    Flip,
    Mirror,
    
    // Overlays
    TextOverlay,
    ImageOverlay,
    Watermark
};

// Effect parameters
struct BrightnessParams {
    double value = 0.0;  // -1.0 to 1.0
};

struct ContrastParams {
    double value = 1.0;  // 0.0 to 3.0
};

struct SaturationParams {
    double value = 1.0;  // 0.0 to 3.0
};

struct BlurParams {
    int radius = 5;
    std::string type = "gaussian";  // gaussian, box, motion
    std::optional<double> sigma;
};

struct SharpenParams {
    double amount = 1.0;
    double radius = 1.0;
    double threshold = 0.0;
};

struct VignetteParams {
    double intensity = 0.5;
    double radius = 0.5;
    double softness = 0.5;
};

struct FilmGrainParams {
    double intensity = 0.3;
    double size = 1.0;
    bool colored = false;
};

struct GlitchParams {
    double intensity = 0.5;
    double frequency = 0.1;
    bool rgb_shift = true;
    bool scanlines = false;
};

struct TextOverlayParams {
    std::string text;
    std::string font_path;
    int font_size = 48;
    uint32_t color = 0xFFFFFFFF;  // RGBA
    double x = 0.5;  // Relative position
    double y = 0.9;
    std::string alignment = "center";
    std::optional<uint32_t> background_color;
    std::optional<uint32_t> outline_color;
    int outline_width = 0;
};

struct ImageOverlayParams {
    std::filesystem::path image_path;
    double x = 0.0;  // Relative position
    double y = 0.0;
    double scale = 1.0;
    double opacity = 1.0;
    std::optional<TimeRange> time_range;
};

struct WatermarkParams {
    std::variant<std::string, std::filesystem::path> content;  // Text or image path
    std::string position = "bottom-right";  // top-left, top-right, bottom-left, bottom-right, center
    double opacity = 0.5;
    double scale = 1.0;
    int margin = 20;
};

// Generic effect configuration
struct EffectConfig {
    EffectType type;
    std::variant<
        BrightnessParams,
        ContrastParams,
        SaturationParams,
        BlurParams,
        SharpenParams,
        VignetteParams,
        FilmGrainParams,
        GlitchParams,
        TextOverlayParams,
        ImageOverlayParams,
        WatermarkParams
    > params;
    std::optional<TimeRange> time_range;  // When to apply (null = entire video)
    std::optional<std::string> easing;    // For animated effects
};

class EffectsEngine {
public:
    EffectsEngine();
    ~EffectsEngine();
    
    // Disable copy
    EffectsEngine(const EffectsEngine&) = delete;
    EffectsEngine& operator=(const EffectsEngine&) = delete;
    
    /**
     * Apply a single effect to a frame
     * 
     * @param frame Input frame (modified in place)
     * @param config Effect configuration
     * @param time Current time in video (for animated effects)
     */
    void apply_effect(cv::Mat& frame, const EffectConfig& config, double time = 0.0);
    
    /**
     * Apply multiple effects to a frame
     * 
     * @param frame Input frame (modified in place)
     * @param effects Vector of effect configurations
     * @param time Current time in video
     */
    void apply_effects(cv::Mat& frame, const std::vector<EffectConfig>& effects, double time = 0.0);
    
    /**
     * Apply effects to entire video
     * 
     * @param input_path Input video path
     * @param output_path Output video path
     * @param effects Effects to apply
     * @param progress Optional progress callback
     */
    void process_video(
        const std::filesystem::path& input_path,
        const std::filesystem::path& output_path,
        const std::vector<EffectConfig>& effects,
        ProgressCallback progress = nullptr
    );
    
    /**
     * Apply color grading LUT
     * 
     * @param frame Input frame
     * @param lut_path Path to LUT file (.cube, .3dl)
     * @param intensity LUT intensity (0.0 - 1.0)
     */
    void apply_lut(cv::Mat& frame, const std::filesystem::path& lut_path, double intensity = 1.0);
    
    /**
     * Add animated text with effects
     * 
     * @param input_path Input video path
     * @param output_path Output video path
     * @param text_configs Text overlay configurations with timing
     * @param progress Optional progress callback
     */
    void add_animated_text(
        const std::filesystem::path& input_path,
        const std::filesystem::path& output_path,
        const std::vector<TextOverlayParams>& text_configs,
        ProgressCallback progress = nullptr
    );
    
    /**
     * Generate video thumbnail
     * 
     * @param video_path Input video path
     * @param output_path Output image path
     * @param time Time to capture (seconds)
     * @param resolution Output resolution
     */
    void generate_thumbnail(
        const std::filesystem::path& video_path,
        const std::filesystem::path& output_path,
        double time = 0.0,
        std::optional<Resolution> resolution = std::nullopt
    );

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    
    // Effect implementations
    void apply_brightness(cv::Mat& frame, const BrightnessParams& params);
    void apply_contrast(cv::Mat& frame, const ContrastParams& params);
    void apply_saturation(cv::Mat& frame, const SaturationParams& params);
    void apply_blur(cv::Mat& frame, const BlurParams& params);
    void apply_sharpen(cv::Mat& frame, const SharpenParams& params);
    void apply_vignette(cv::Mat& frame, const VignetteParams& params);
    void apply_film_grain(cv::Mat& frame, const FilmGrainParams& params);
    void apply_glitch(cv::Mat& frame, const GlitchParams& params, double time);
    void apply_text_overlay(cv::Mat& frame, const TextOverlayParams& params);
    void apply_image_overlay(cv::Mat& frame, const ImageOverlayParams& params);
    void apply_watermark(cv::Mat& frame, const WatermarkParams& params);
};

} // namespace faceless_video




