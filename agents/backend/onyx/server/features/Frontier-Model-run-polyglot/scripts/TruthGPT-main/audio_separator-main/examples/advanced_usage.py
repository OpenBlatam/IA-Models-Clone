"""
Advanced usage examples for audio separator.
"""

from pathlib import Path
from audio_separator import AudioSeparator, BatchSeparator
from audio_separator.config import AudioSeparatorConfig
from audio_separator.utils.device_utils import get_device_info
from audio_separator.utils.cache_utils import CacheManager
from audio_separator.utils.progress_utils import ProgressTracker


def example_with_config():
    """Example using configuration object."""
    print("Example 1: Using Configuration")
    print("-" * 50)
    
    # Create custom configuration
    config = AudioSeparatorConfig()
    config.audio.sample_rate = 48000
    config.separation.num_sources = 4
    config.model.model_type = "demucs"
    config.output.output_format = "wav"
    
    print(f"Sample Rate: {config.audio.sample_rate}")
    print(f"Number of Sources: {config.separation.num_sources}")
    print(f"Model Type: {config.model.model_type}")
    print()


def example_device_info():
    """Example showing device information."""
    print("Example 2: Device Information")
    print("-" * 50)
    
    device_info = get_device_info()
    print(f"CUDA Available: {device_info['cuda']}")
    print(f"MPS Available: {device_info['mps']}")
    
    if device_info['cuda']:
        print(f"CUDA Devices: {len(device_info['cuda_devices'])}")
        for device in device_info['cuda_devices']:
            print(f"  - {device['name']} ({device['memory']})")
    print()


def example_with_cache():
    """Example using caching."""
    print("Example 3: Using Cache")
    print("-" * 50)
    
    cache = CacheManager()
    
    # Check cache size
    cache_size = cache.get_size()
    print(f"Cache size: {cache_size / 1024 / 1024:.2f} MB")
    
    # Clear cache if needed
    # deleted = cache.clear()
    # print(f"Cleared {deleted} cache files")
    print()


def example_batch_with_progress():
    """Example of batch processing with progress tracking."""
    print("Example 4: Batch Processing with Progress")
    print("-" * 50)
    
    batch_separator = BatchSeparator(model_type="demucs")
    
    audio_files = [
        "song1.mp3",
        "song2.mp3",
        "song3.mp3"
    ]
    
    # Process with progress tracking
    with ProgressTracker(len(audio_files), "Processing files") as tracker:
        for audio_file in audio_files:
            try:
                results = batch_separator.separator.separate_file(audio_file)
                tracker.update(1, f"Completed: {Path(audio_file).name}")
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                tracker.update(1, f"Failed: {Path(audio_file).name}")
    print()


def example_custom_separation():
    """Example of custom separation workflow."""
    print("Example 5: Custom Separation Workflow")
    print("-" * 50)
    
    separator = AudioSeparator(model_type="demucs")
    
    audio_path = "example_song.mp3"
    
    # Step 1: Load audio
    from audio_separator.processor.audio_loader import AudioLoader
    loader = AudioLoader()
    audio, sr = loader.load(audio_path)
    print(f"Loaded audio: {len(audio)} samples at {sr} Hz")
    
    # Step 2: Separate
    separated = separator.separate_audio(audio, return_tensors=False)
    print(f"Separated into {len(separated)} sources")
    
    # Step 3: Process individual sources
    for source_name, source_audio in separated.items():
        print(f"  {source_name}: {source_audio.shape}")
    
    print()


def example_error_handling():
    """Example of proper error handling."""
    print("Example 6: Error Handling")
    print("-" * 50)
    
    from audio_separator.exceptions import (
        AudioIOError,
        AudioProcessingError,
        AudioValidationError
    )
    
    separator = AudioSeparator(model_type="demucs")
    
    # Try to separate non-existent file
    try:
        separator.separate_file("nonexistent.mp3")
    except AudioIOError as e:
        print(f"IO Error: {e.message}")
        print(f"Component: {e.component}")
        print(f"Error Code: {e.error_code}")
    except AudioProcessingError as e:
        print(f"Processing Error: {e.message}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    
    print()


if __name__ == "__main__":
    example_with_config()
    example_device_info()
    example_with_cache()
    example_batch_with_progress()
    example_custom_separation()
    example_error_handling()

