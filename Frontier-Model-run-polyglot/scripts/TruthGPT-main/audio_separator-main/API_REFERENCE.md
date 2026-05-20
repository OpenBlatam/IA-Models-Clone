# API Reference

Complete API reference for Audio Separator.

## Core Classes

### AudioSeparator

Main class for audio source separation.

```python
from audio_separator import AudioSeparator

separator = AudioSeparator(
    model_type="demucs",
    model_kwargs={},
    sample_rate=44100
)
```

#### Methods

- `separate_file(audio_path, output_dir, save_outputs=True)` - Separate audio file
- `separate_audio(audio, return_tensors=False)` - Separate audio data directly

### BatchSeparator

Batch processing for multiple files.

```python
from audio_separator import BatchSeparator

batch_separator = BatchSeparator(
    model_type="demucs",
    model_kwargs={},
    sample_rate=44100
)
```

#### Methods

- `separate_files(audio_paths, output_dir, show_progress=True)` - Process multiple files
- `separate_directory(input_dir, output_dir, extensions, recursive)` - Process directory

## Model Builder

### build_audio_separator_model

Build a separator model.

```python
from audio_separator import build_audio_separator_model

model = build_audio_separator_model(
    model_type="demucs",
    **kwargs
)
```

## Utilities

### Audio Analysis

```python
from audio_separator.utils.audio_analysis import (
    analyze_audio,
    detect_silence,
    calculate_loudness,
    detect_beats,
    extract_features
)
```

### Audio Enhancement

```python
from audio_separator.utils.audio_enhancement import (
    denoise_audio,
    normalize_audio_peak,
    normalize_audio_rms,
    apply_fade,
    apply_compression
)
```

### Format Conversion

```python
from audio_separator.utils.format_converter import (
    convert_format,
    batch_convert
)
```

### Quality Metrics

```python
from audio_separator.utils.quality_metrics import (
    calculate_separation_quality,
    assess_audio_quality,
    compare_separations
)
```

### Visualization

```python
from audio_separator.utils.visualization import (
    plot_waveform,
    plot_spectrogram,
    plot_separation_comparison,
    create_separation_report
)
```

### Performance

```python
from audio_separator.utils.performance import (
    timeit,
    timer,
    PerformanceMonitor,
    profile_memory
)
```

### Parallel Processing

```python
from audio_separator.utils.parallel_processing import (
    process_parallel,
    batch_process_files,
    chunk_audio_processing
)
```

## Exceptions

```python
from audio_separator.exceptions import (
    AudioSeparatorError,
    AudioProcessingError,
    AudioFormatError,
    AudioModelError,
    AudioValidationError,
    AudioIOError,
    AudioInitializationError,
    AudioConfigurationError
)
```

## Configuration

```python
from audio_separator.config import (
    AudioConfig,
    SeparationConfig,
    ModelConfig,
    ProcessingConfig,
    OutputConfig,
    AudioSeparatorConfig
)
```

## Evaluation Metrics

```python
from audio_separator.eval.metrics import (
    calculate_sdr,
    calculate_sir,
    calculate_sar,
    calculate_isdr,
    evaluate_separation
)
```

## Complete Example

```python
from audio_separator import AudioSeparator
from audio_separator.utils.audio_analysis import analyze_audio
from audio_separator.utils.quality_metrics import calculate_separation_quality

# Initialize separator
separator = AudioSeparator(model_type="demucs")

# Separate audio
results = separator.separate_file("song.mp3", output_dir="output")

# Analyze original
from audio_separator.processor.audio_loader import AudioLoader
loader = AudioLoader()
audio, sr = loader.load("song.mp3")
analysis = analyze_audio(audio, sample_rate=sr)

# Assess quality
from audio_separator.utils.quality_metrics import calculate_separation_quality
quality = calculate_separation_quality(results, audio)

print("Separation complete!")
print(f"Analysis: {analysis}")
print(f"Quality: {quality}")
```

