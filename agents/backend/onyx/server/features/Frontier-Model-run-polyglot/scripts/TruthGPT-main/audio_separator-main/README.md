# Audio Separator

Advanced Audio Source Separation Framework

Audio Separator is a unified framework for separating audio into multiple sources (vocals, drums, bass, other instruments, etc.). It supports multiple state-of-the-art separation models and provides a simple, consistent API.

## Features

- **Multiple Models**: Support for Demucs, Spleeter, LALAL.AI, and hybrid models
- **Easy to Use**: Simple Python API for audio separation
- **Batch Processing**: Process multiple files or entire directories
- **Flexible**: Works with various audio formats (WAV, MP3, FLAC, etc.)
- **Extensible**: Easy to add new separation models

## Installation

```bash
# Basic installation
pip install -e .

# With all dependencies
pip install -e ".[all]"

# For specific models
pip install -e ".[demucs]"  # For Demucs
pip install -e ".[spleeter]"  # For Spleeter
```

## Quick Start

### Basic Usage

```python
from audio_separator import AudioSeparator

# Initialize separator
separator = AudioSeparator(model_type="demucs")

# Separate an audio file
results = separator.separate_file("song.mp3", output_dir="separated")

# Results will contain paths to separated sources:
# {
#     "vocals": "separated/song_vocals.wav",
#     "drums": "separated/song_drums.wav",
#     "bass": "separated/song_bass.wav",
#     "other": "separated/song_other.wav"
# }
```

### Batch Processing

```python
from audio_separator import BatchSeparator

# Initialize batch separator
batch_separator = BatchSeparator(model_type="demucs")

# Process multiple files
audio_files = ["song1.mp3", "song2.mp3", "song3.mp3"]
results = batch_separator.separate_files(audio_files, output_dir="output")

# Or process entire directory
results = batch_separator.separate_directory("input_folder", "output_folder")
```

### Using Different Models

```python
# Demucs (default, recommended)
separator = AudioSeparator(model_type="demucs", model_kwargs={"variant": "htdemucs"})

# Spleeter
separator = AudioSeparator(model_type="spleeter", model_kwargs={"stems": 4})

# Hybrid (combines multiple models)
separator = AudioSeparator(
    model_type="hybrid",
    model_kwargs={"models": ["demucs", "spleeter"]}
)

# LALAL.AI (requires API key)
separator = AudioSeparator(
    model_type="lalal",
    model_kwargs={"api_key": "your_api_key"}
)
```

## Supported Models

### Demucs
- **Variant**: `htdemucs` (default), `htdemucs_ft`, `mdx`
- **Sources**: Vocals, Drums, Bass, Other
- **Best for**: High-quality separation, music

### Spleeter
- **Stems**: 2 (vocals/accompaniment) or 4 (vocals/drums/bass/other)
- **Best for**: Fast separation, good quality

### LALAL.AI
- **Sources**: Vocals, Instrumental
- **Best for**: API-based separation, cloud processing
- **Note**: Requires API key

### Hybrid
- **Combines**: Multiple models for improved results
- **Best for**: Maximum quality, ensemble methods

## Examples

See the `examples/` directory for more detailed examples:

- `basic_separation.py` - Basic usage examples
- `batch_processing.py` - Batch processing examples
- `model_comparison.py` - Compare different models
- `custom_separation.py` - Advanced customization

## API Reference

### AudioSeparator

Main class for audio separation.

```python
AudioSeparator(
    model_type: str = "demucs",
    model_kwargs: Optional[Dict] = None,
    sample_rate: int = 44100
)
```

**Methods:**
- `separate_file(audio_path, output_dir, save_outputs=True)` - Separate an audio file
- `separate_audio(audio, return_tensors=False)` - Separate audio data directly

### BatchSeparator

Batch processing for multiple files.

```python
BatchSeparator(
    model_type: str = "demucs",
    model_kwargs: Optional[Dict] = None,
    sample_rate: int = 44100
)
```

**Methods:**
- `separate_files(audio_paths, output_dir, show_progress=True)` - Process multiple files
- `separate_directory(input_dir, output_dir, extensions, recursive)` - Process directory

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy
- Librosa or SoundFile (for audio I/O)

### Model-Specific Requirements

- **Demucs**: `pip install demucs`
- **Spleeter**: `pip install spleeter`
- **LALAL.AI**: `pip install requests` (API-based, no local model)

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Citation

If you use Audio Separator in your research, please cite:

```bibtex
@software{audio_separator,
  title={Audio Separator: Advanced Audio Source Separation Framework},
  author={Blatam Academy},
  year={2024},
  url={https://github.com/blatam-academy/audio-separator}
}
```

## Acknowledgments

This framework integrates and builds upon:
- [Demucs](https://github.com/facebookresearch/demucs) by Meta AI Research
- [Spleeter](https://github.com/deezer/spleeter) by Deezer Research
- [LALAL.AI](https://www.lalal.ai/) for API-based separation

