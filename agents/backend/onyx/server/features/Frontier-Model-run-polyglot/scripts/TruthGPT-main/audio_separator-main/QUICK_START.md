# Quick Start Guide

Get started with Audio Separator in minutes!

## Installation

```bash
# Basic installation
pip install -e .

# With all dependencies
pip install -e ".[all]"
```

## Basic Usage

### Python API

```python
from audio_separator import AudioSeparator

# Initialize separator
separator = AudioSeparator(model_type="demucs")

# Separate an audio file
results = separator.separate_file("song.mp3", output_dir="separated")

# Results contain paths to separated sources
for source_name, output_path in results.items():
    print(f"{source_name}: {output_path}")
```

### Command Line

```bash
# Separate a single file
audio-separator separate song.mp3 -o output/

# Batch process directory
audio-separator batch input_dir/ -o output_dir/ -r

# Show device information
audio-separator info --device
```

## Examples

### Example 1: Basic Separation

```python
from audio_separator import AudioSeparator

separator = AudioSeparator(model_type="demucs")
results = separator.separate_file("song.mp3")
```

### Example 2: Batch Processing

```python
from audio_separator import BatchSeparator

batch_separator = BatchSeparator(model_type="demucs")
results = batch_separator.separate_directory("input/", "output/")
```

### Example 3: Using Different Models

```python
# Demucs (recommended)
separator = AudioSeparator(model_type="demucs")

# Spleeter
separator = AudioSeparator(model_type="spleeter", model_kwargs={"stems": 4})

# Hybrid (combines multiple models)
separator = AudioSeparator(
    model_type="hybrid",
    model_kwargs={"models": ["demucs", "spleeter"]}
)
```

### Example 4: Custom Configuration

```python
from audio_separator import AudioSeparator
from audio_separator.config import AudioSeparatorConfig

config = AudioSeparatorConfig()
config.audio.sample_rate = 48000
config.separation.num_sources = 4

separator = AudioSeparator(
    model_type="demucs",
    sample_rate=config.audio.sample_rate
)
```

## Supported Formats

### Input Formats
- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)
- M4A (.m4a)
- OGG (.ogg)
- AAC (.aac)
- WMA (.wma)

### Output Formats
- WAV (.wav) - Recommended
- MP3 (.mp3)
- FLAC (.flac)
- OGG (.ogg)

## Troubleshooting

### Common Issues

1. **Model not found**: Install the required model library
   ```bash
   pip install demucs  # For Demucs
   pip install spleeter  # For Spleeter
   ```

2. **CUDA out of memory**: Use CPU or reduce batch size
   ```python
   separator = AudioSeparator(model_type="demucs")
   # Models will automatically use CPU if CUDA is not available
   ```

3. **File format not supported**: Check file extension and install required libraries
   ```bash
   pip install librosa soundfile
   ```

## Next Steps

- Read the [README.md](README.md) for detailed documentation
- Check [examples/](examples/) for more examples
- See [IMPROVEMENTS.md](IMPROVEMENTS.md) for advanced features
- Review [STRUCTURE.md](STRUCTURE.md) for architecture details

## Getting Help

- Check the documentation in `docs/`
- Review examples in `examples/`
- Open an issue on GitHub
- Check the [FAQ](README.md#faq) section

