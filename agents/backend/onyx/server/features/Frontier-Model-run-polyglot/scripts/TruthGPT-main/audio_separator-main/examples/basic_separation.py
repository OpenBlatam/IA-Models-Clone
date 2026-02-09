"""
Basic audio separation examples.
Refactored to use constants and improve consistency.
"""

from audio_separator import AudioSeparator
from audio_separator.separator.constants import DEFAULT_MODEL_TYPE


def example_basic_separation():
    """Basic example of separating an audio file."""
    print("Example 1: Basic Audio Separation")
    print("-" * 50)
    
    # Initialize separator with default model (Demucs)
    separator = AudioSeparator(model_type=DEFAULT_MODEL_TYPE)
    
    # Separate audio file
    audio_path = "example_song.mp3"
    results = separator.separate_file(audio_path, output_dir="separated")
    
    print(f"Separated {audio_path} into:")
    for source_name, output_path in results.items():
        print(f"  - {source_name}: {output_path}")
    print()


def example_spleeter_separation():
    """Example using Spleeter model."""
    print("Example 2: Spleeter Separation")
    print("-" * 50)
    
    # Initialize with Spleeter
    separator = AudioSeparator(
        model_type="spleeter",
        model_kwargs={"stems": 4}
    )
    
    audio_path = "example_song.mp3"
    results = separator.separate_file(audio_path, output_dir="separated_spleeter")
    
    print(f"Separated {audio_path} using Spleeter:")
    for source_name, output_path in results.items():
        print(f"  - {source_name}: {output_path}")
    print()


def example_hybrid_separation():
    """Example using hybrid model."""
    print("Example 3: Hybrid Model Separation")
    print("-" * 50)
    
    # Initialize with hybrid model (combines multiple models)
    separator = AudioSeparator(
        model_type="hybrid",
        model_kwargs={
            "models": ["demucs", "spleeter"],
            "weights": [0.6, 0.4]  # Weight for each model
        }
    )
    
    audio_path = "example_song.mp3"
    results = separator.separate_file(audio_path, output_dir="separated_hybrid")
    
    print(f"Separated {audio_path} using hybrid model:")
    for source_name, output_path in results.items():
        print(f"  - {source_name}: {output_path}")
    print()


if __name__ == "__main__":
    # Run examples
    example_basic_separation()
    example_spleeter_separation()
    example_hybrid_separation()

