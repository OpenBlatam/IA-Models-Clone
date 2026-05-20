# Features Overview

Complete list of features in Audio Separator.

## Core Features

### 1. Audio Separation Models
- **Demucs**: State-of-the-art music source separation
- **Spleeter**: Fast and efficient separation
- **LALAL.AI**: API-based cloud separation
- **Hybrid**: Ensemble of multiple models for best results

### 2. Audio Processing
- **Preprocessing**: Resampling, normalization, silence trimming
- **Postprocessing**: Denoising, normalization, format conversion
- **Enhancement**: Denoising, compression, fade in/out
- **Format Conversion**: Convert between audio formats

### 3. Batch Processing
- Process multiple files
- Process entire directories
- Recursive directory processing
- Progress tracking

### 4. Configuration Management
- Structured configuration with dataclasses
- Separate configs for audio, separation, model, processing, output
- Serialization to/from dictionaries
- Validation

### 5. Device Management
- Automatic GPU/CPU detection
- CUDA support
- MPS support (Apple Silicon)
- Device information

### 6. Error Handling
- Custom exception hierarchy
- Informative error messages
- Error codes for programmatic handling
- Detailed error context

### 7. Logging
- Structured logging
- Configurable log levels
- Context information
- File and console output

### 8. Validation
- File validation
- Format validation
- Parameter validation
- Audio array validation

### 9. Caching
- Model output caching
- Automatic cache management
- Cache size monitoring
- Cache cleanup

### 10. Performance Monitoring
- Function timing decorators
- Context managers for timing
- Performance metrics
- Memory profiling

## Utilities

### Audio Utilities
- Get audio information
- Resample audio
- Convert to mono
- Normalize audio

### Format Conversion
- Convert between formats
- Batch conversion
- Sample rate conversion
- Channel conversion

### Audio Enhancement
- Denoising (simple, spectral, wiener)
- Peak normalization
- RMS normalization
- Fade in/out
- Compression

### Validation Utilities
- File validation
- Format validation
- Parameter validation
- Audio validation

### Device Utilities
- Device detection
- Device information
- Tensor movement

### Progress Utilities
- Progress bars
- Progress tracking
- Context managers

### Cache Utilities
- Cache management
- Cache statistics
- Cache cleanup

### Performance Utilities
- Timing decorators
- Performance monitoring
- Memory profiling

## Command-Line Interface

### Commands
- `separate`: Separate a single audio file
- `batch`: Batch process multiple files
- `info`: Show system information

### Options
- Model selection
- Output directory
- Sample rate
- Format options
- Recursive processing
- Progress display

## Evaluation

### Metrics
- SDR (Signal-to-Distortion Ratio)
- SIR (Signal-to-Interference Ratio)
- SAR (Signal-to-Artifacts Ratio)
- ISDR (Improvement in Signal-to-Distortion Ratio)

### Evaluation Tools
- Evaluation scripts
- Metric calculation
- Comparison tools

## Documentation

### Guides
- Quick Start Guide
- Architecture Documentation
- API Reference
- Examples

### Reference
- Changelog
- Contributing Guidelines
- Feature List
- Improvement Logs

## Testing

### Test Coverage
- Exception handling
- Validation utilities
- Model builder
- Device utilities
- Format conversion
- Audio enhancement

### Test Types
- Unit tests
- Integration tests
- Performance tests

## Examples

### Basic Examples
- Basic separation
- Batch processing
- Model comparison
- Error handling

### Advanced Examples
- Custom configuration
- Audio enhancement
- Performance monitoring
- Format conversion

## Integration

### Supported Libraries
- PyTorch
- NumPy
- Librosa
- SoundFile
- SciPy
- tqdm

### Supported Formats
- Input: WAV, MP3, FLAC, M4A, OGG, AAC, WMA
- Output: WAV, MP3, FLAC, OGG

## Performance

### Optimizations
- GPU acceleration
- Batch processing
- Caching
- Efficient memory usage

### Monitoring
- Performance metrics
- Memory profiling
- Timing information

## Extensibility

### Adding Models
- Base model interface
- Model registration
- Custom model support

### Adding Processors
- Processor interface
- Custom processors
- Pipeline support

### Adding Utilities
- Utility functions
- Helper classes
- Extension points

