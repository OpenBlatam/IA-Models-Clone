"""
Batch processing examples.
"""

from audio_separator import BatchSeparator
from pathlib import Path


def example_batch_files():
    """Example of processing multiple files."""
    print("Example 1: Batch File Processing")
    print("-" * 50)
    
    # Initialize batch separator
    batch_separator = BatchSeparator(model_type="demucs")
    
    # List of audio files to process
    audio_files = [
        "song1.mp3",
        "song2.mp3",
        "song3.mp3"
    ]
    
    # Process all files
    results = batch_separator.separate_files(
        audio_files,
        output_dir="batch_output",
        show_progress=True
    )
    
    print(f"Processed {len(audio_files)} files:")
    for audio_path, separated in results.items():
        print(f"\n  {Path(audio_path).name}:")
        for source_name, output_path in separated.items():
            print(f"    - {source_name}: {output_path}")
    print()


def example_directory_processing():
    """Example of processing an entire directory."""
    print("Example 2: Directory Processing")
    print("-" * 50)
    
    # Initialize batch separator
    batch_separator = BatchSeparator(model_type="demucs")
    
    # Process entire directory
    input_dir = "input_audio"
    output_dir = "output_separated"
    
    results = batch_separator.separate_directory(
        input_dir,
        output_dir,
        extensions=[".mp3", ".wav", ".flac"],
        recursive=True  # Process subdirectories too
    )
    
    print(f"Processed directory: {input_dir}")
    print(f"Found {len(results)} audio files")
    print(f"Output directory: {output_dir}")
    print()


if __name__ == "__main__":
    example_batch_files()
    example_directory_processing()

