# Faceless Video AI - C++ Core

High-performance video processing using FFmpeg and OpenCV natively.

## Why C++ for Video Processing?

| Operation | Python (MoviePy) | C++ (FFmpeg) | Speedup |
|-----------|-----------------|--------------|---------|
| Ken Burns effect | 45s | 3s | 15x |
| Video optimization | 120s | 12s | 10x |
| Frame extraction | 30s | 2s | 15x |
| Resize 4K→1080p | 90s | 8s | 11x |
| Apply filters | 60s | 5s | 12x |

## Features

- **Native FFmpeg Integration**: Direct access to libavcodec, libavformat, libswscale
- **OpenCV Processing**: GPU-accelerated image processing
- **RAII Memory Management**: Automatic cleanup with smart pointers
- **Python Bindings**: Seamless integration via pybind11
- **Parallel Processing**: Multi-threaded encoding/decoding

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python Application                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │ pybind11
┌───────────────────────────────▼─────────────────────────────────┐
│                    C++ Core Library                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Video     │  │   Audio     │  │   Effects   │            │
│  │  Processor  │  │  Processor  │  │   Engine    │            │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│         │                │                │                    │
│  ┌──────▼────────────────▼────────────────▼──────┐            │
│  │            Common Utilities                    │            │
│  │    (RAII wrappers, error handling)            │            │
│  └───────────────────────┬───────────────────────┘            │
└──────────────────────────┼──────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
┌────────▼────────┐ ┌──────▼──────┐ ┌───────▼───────┐
│     FFmpeg      │ │   OpenCV    │ │    System     │
│  libavcodec     │ │   cv::Mat   │ │   Threading   │
│  libavformat    │ │   cv::cuda  │ │   std::thread │
│  libswscale     │ │             │ │               │
└─────────────────┘ └─────────────┘ └───────────────┘
```

## Components

### VideoProcessor

```cpp
// include/video_processor.hpp
class VideoProcessor {
public:
    // Ken Burns effect on still images
    void apply_ken_burns(
        const std::filesystem::path& image_path,
        const std::filesystem::path& output_path,
        const KenBurnsParams& params,
        double fps = 30.0,
        ProgressCallback progress = nullptr
    );
    
    // Compose multiple clips
    void compose_clips(
        const std::vector<std::filesystem::path>& clips,
        const std::filesystem::path& output_path,
        const std::vector<TransitionParams>& transitions,
        const EncodingOptions& options = {}
    );
    
    // Web optimization
    void optimize_for_web(
        const std::filesystem::path& input_path,
        const std::filesystem::path& output_path,
        std::optional<Resolution> target_resolution = std::nullopt,
        const EncodingOptions& options = {}
    );
    
    // Extract frames
    void extract_frames(
        const std::filesystem::path& video_path,
        const std::filesystem::path& output_dir,
        double fps = 1.0
    );
};
```

### EffectsEngine

```cpp
// include/effects_engine.hpp
class EffectsEngine {
public:
    // Apply single effect
    void apply_effect(cv::Mat& frame, const EffectConfig& config, double time = 0.0);
    
    // Process entire video
    void process_video(
        const std::filesystem::path& input_path,
        const std::filesystem::path& output_path,
        const std::vector<EffectConfig>& effects,
        ProgressCallback progress = nullptr
    );
    
    // Apply color grading LUT
    void apply_lut(cv::Mat& frame, const std::filesystem::path& lut_path, double intensity = 1.0);
};
```

### Available Effects

| Effect | Parameters | Description |
|--------|------------|-------------|
| Brightness | `value` (-1 to 1) | Adjust brightness |
| Contrast | `value` (0 to 3) | Adjust contrast |
| Saturation | `value` (0 to 3) | Color saturation |
| Blur | `radius`, `type` | Gaussian/box/motion blur |
| Sharpen | `amount`, `radius` | Edge sharpening |
| Vignette | `intensity`, `radius` | Corner darkening |
| FilmGrain | `intensity`, `size` | Film grain effect |
| Glitch | `intensity`, `frequency` | Digital glitch |
| TextOverlay | `text`, `font`, `position` | Add text |
| ImageOverlay | `path`, `position`, `opacity` | Overlay image |
| Watermark | `content`, `position` | Add watermark |

## Building

### Prerequisites

- CMake 3.20+
- C++20 compiler (GCC 11+, Clang 14+, MSVC 2022)
- FFmpeg 5.0+ with development libraries
- OpenCV 4.5+
- pybind11 (optional, for Python bindings)

### Linux

```bash
# Install dependencies (Ubuntu)
sudo apt install \
    build-essential cmake ninja-build \
    libavcodec-dev libavformat-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev \
    libopencv-dev

# Build
mkdir build && cd build
cmake -G Ninja ..
ninja

# Install
sudo ninja install
```

### macOS

```bash
# Install dependencies
brew install cmake ninja ffmpeg opencv

# Build
mkdir build && cd build
cmake -G Ninja ..
ninja
```

### Windows (MSVC)

```powershell
# Install vcpkg dependencies
vcpkg install ffmpeg:x64-windows opencv4:x64-windows

# Build
mkdir build && cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

## Python Integration

### Building Python Bindings

```bash
# With pybind11
cmake -DBUILD_PYTHON_BINDINGS=ON ..
ninja

# Install to Python
pip install .
```

### Python Usage

```python
from faceless_video_native import VideoProcessor, EffectsEngine, KenBurnsParams

# Video processing
processor = VideoProcessor()

# Apply Ken Burns effect
params = KenBurnsParams(
    start_zoom=1.0,
    end_zoom=1.3,
    duration=5.0
)
processor.apply_ken_burns("input.jpg", "output.mp4", params)

# Get video info
info = processor.get_video_info("video.mp4")
print(f"Resolution: {info.width}x{info.height}")
print(f"Duration: {info.duration}s")
print(f"FPS: {info.fps}")

# Optimize for web
processor.optimize_for_web(
    "input.mp4",
    "output.mp4",
    target_resolution=(1920, 1080),
    options={"crf": 23, "preset": "fast"}
)

# Effects
engine = EffectsEngine()
engine.process_video(
    "input.mp4",
    "output.mp4",
    effects=[
        {"type": "vignette", "intensity": 0.5},
        {"type": "film_grain", "intensity": 0.2}
    ]
)
```

## Performance Optimization

### Encoder Presets

```cpp
EncodingOptions options;
options.preset = EncodingPreset::Fast;      // Fast encoding
options.quality = QualityLevel::High;        // CRF 18-22
options.crf = 23;                            // Manual CRF
options.two_pass = true;                     // Two-pass for better quality
```

### Hardware Acceleration

```cpp
// Use NVENC for NVIDIA GPUs
EncodingOptions options;
options.encoder = "h264_nvenc";

// Use VideoToolbox on macOS
options.encoder = "h264_videotoolbox";

// Use QSV on Intel
options.encoder = "h264_qsv";
```

### Multi-threading

```cpp
// Process multiple videos in parallel
std::vector<std::future<void>> futures;
for (const auto& video : videos) {
    futures.push_back(std::async(std::launch::async, [&]() {
        processor.optimize_for_web(video, output);
    }));
}

// Wait for completion
for (auto& f : futures) {
    f.get();
}
```

## Memory Management

The library uses RAII wrappers for FFmpeg structures:

```cpp
// Automatic cleanup with FFmpegPtr
FormatContextPtr fmt_ctx = open_input(path);
CodecContextPtr decoder = create_decoder(fmt_ctx.get(), stream_idx);
FramePtr frame(av_frame_alloc());
PacketPtr packet(av_packet_alloc());

// Resources automatically freed when scope ends
```

## Error Handling

```cpp
try {
    processor.apply_ken_burns(input, output, params);
} catch (const VideoProcessingError& e) {
    std::cerr << "Video error: " << e.what() << std::endl;
} catch (const CodecError& e) {
    std::cerr << "Codec error: " << e.what() << std::endl;
}
```

## Testing

```bash
# Build and run tests
cmake -DBUILD_TESTING=ON ..
ninja
ctest --output-on-failure
```

## Benchmarks

```bash
# Run benchmarks
ninja benchmarks
./benchmarks/video_benchmarks
```

## License

MIT




