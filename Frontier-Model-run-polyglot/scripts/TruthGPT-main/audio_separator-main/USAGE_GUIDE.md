# Usage Guide

Complete guide for using Audio Separator.

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from audio_separator import AudioSeparator

separator = AudioSeparator(model_type="demucs")
results = separator.separate_file("song.mp3", output_dir="separated")
```

## Common Use Cases

### 1. Separate a Single File

```python
from audio_separator import AudioSeparator

separator = AudioSeparator(model_type="demucs")
results = separator.separate_file("song.mp3")
```

### 2. Batch Processing

```python
from audio_separator import BatchSeparator

batch_separator = BatchSeparator(model_type="demucs")
results = batch_separator.separate_directory("input/", "output/")
```

### 3. Using Different Models

```python
# Demucs (recommended)
separator = AudioSeparator(model_type="demucs")

# Spleeter
separator = AudioSeparator(model_type="spleeter", model_kwargs={"stems": 4})

# Hybrid
separator = AudioSeparator(
    model_type="hybrid",
    model_kwargs={"models": ["demucs", "spleeter"]}
)
```

### 4. Audio Enhancement

```python
from audio_separator.utils.audio_enhancement import (
    denoise_audio,
    normalize_audio_peak,
    apply_fade
)

# Load audio
from audio_separator.processor.audio_loader import AudioLoader
loader = AudioLoader()
audio, sr = loader.load("input.wav")

# Enhance
enhanced = denoise_audio(audio, method="simple", strength=0.5)
enhanced = normalize_audio_peak(enhanced, target_peak=0.95)
enhanced = apply_fade(enhanced, fade_in=0.5, fade_out=1.0, sample_rate=sr)

# Save
from audio_separator.processor.audio_saver import AudioSaver
saver = AudioSaver()
saver.save(enhanced, "output_enhanced.wav", sample_rate=sr)
```

### 5. Audio Analysis

```python
from audio_separator.utils.audio_analysis import (
    analyze_audio,
    detect_silence,
    calculate_loudness
)

loader = AudioLoader()
audio, sr = loader.load("input.wav")

# Analyze
analysis = analyze_audio(audio, sample_rate=sr)
silence_regions = detect_silence(audio, sample_rate=sr)
loudness = calculate_loudness(audio, sample_rate=sr)
```

### 6. Quality Assessment

```python
from audio_separator.utils.quality_metrics import (
    calculate_separation_quality,
    assess_audio_quality
)

separator = AudioSeparator(model_type="demucs")
results = separator.separate_file("input.wav")

loader = AudioLoader()
mixture, sr = loader.load("input.wav")

# Load separated sources
separated = {}
for source_name, source_path in results.items():
    audio, _ = loader.load(source_path)
    separated[source_name] = audio

# Assess quality
quality = calculate_separation_quality(separated, mixture)
```

### 7. Audio Mixing

```python
from audio_separator.utils.audio_merger import merge_sources, create_mix

# Load separated sources
loader = AudioLoader()
sources = {}
for source_name, source_path in results.items():
    audio, sr = loader.load(source_path)
    sources[source_name] = audio

# Merge with custom volumes
volumes = {"vocals": 1.0, "drums": 0.8, "bass": 0.9, "other": 0.7}
merged = merge_sources(sources, volumes=volumes)

# Or create custom mix
mix_config = {
    "vocals_volume": 1.2,
    "drums_volume": 0.9,
    "fade_in": 0.5,
    "fade_out": 1.0
}
custom_mix = create_mix(sources, mix_config, sample_rate=44100)
```

### 8. Export/Import

```python
from audio_separator.utils.export_utils import (
    export_separation_metadata,
    export_separation_report,
    import_separation_metadata
)

# Export
export_separation_metadata(results, "metadata.json")
export_separation_report(results, analysis=analysis, quality=quality)

# Import
metadata = import_separation_metadata("metadata.json")
```

### 9. Backup/Restore

```python
from audio_separator.utils.backup_utils import (
    backup_separation_results,
    restore_separation_results,
    list_backups
)

# Backup
backup_path = backup_separation_results("separated", "backups")

# List backups
backups = list_backups("backups")

# Restore
restore_separation_results(backups[0]['path'], "restored")
```

### 10. Visualization

```python
from audio_separator.utils.visualization import (
    plot_waveform,
    plot_spectrogram,
    create_separation_report
)

# Plot waveform
plot_waveform(audio, sample_rate=sr, save_path="waveform.png")

# Plot spectrogram
plot_spectrogram(audio, sample_rate=sr, save_path="spectrogram.png")

# Create full report
create_separation_report("input.wav", results, "report/")
```

## Command Line Usage

### Separate File

```bash
audio-separator separate song.mp3 -o output/
```

### Batch Process

```bash
audio-separator batch input_dir/ -o output_dir/ -r
```

### Convert Format

```bash
python scripts/convert_audio.py input.wav -f mp3 -o output.mp3
```

### Analyze Audio

```bash
python scripts/analyze_audio.py input.wav -o analysis.json
```

### Backup Results

```bash
python scripts/backup_results.py backup separated/ -o backups/
```

## Advanced Usage

### Custom Configuration

```python
from audio_separator.config import AudioSeparatorConfig

config = AudioSeparatorConfig()
config.audio.sample_rate = 48000
config.separation.num_sources = 4
config.model.model_type = "demucs"
```

### Parallel Processing

```python
from audio_separator.utils.parallel_processing import process_parallel

def process_file(file_path):
    separator = AudioSeparator()
    return separator.separate_file(file_path)

results = process_parallel(audio_files, process_file, max_workers=4)
```

### Performance Monitoring

```python
from audio_separator.utils.performance import timer, PerformanceMonitor

with timer("Separation"):
    results = separator.separate_file("input.wav")

monitor = PerformanceMonitor()
monitor.start("operation")
# ... do work ...
monitor.stop("operation")
stats = monitor.get_stats("operation")
```

## Tips and Best Practices

1. **Use appropriate model**: Demucs for quality, Spleeter for speed
2. **Enable caching**: For repeated processing of same files
3. **Use batch processing**: For multiple files
4. **Monitor performance**: Use performance utilities
5. **Backup results**: Use backup utilities for important work
6. **Validate inputs**: Check file formats and parameters
7. **Handle errors**: Use try-except for robust code
8. **Use logging**: Enable logging for debugging

## Troubleshooting

### Common Issues

1. **Model not found**: Install required model library
2. **CUDA out of memory**: Use CPU or reduce batch size
3. **Format not supported**: Check file extension and install libraries
4. **Slow processing**: Enable GPU, use caching, or optimize batch size

### Getting Help

- Check documentation in `docs/`
- Review examples in `examples/`
- Check error messages and logs
- Open an issue on GitHub

