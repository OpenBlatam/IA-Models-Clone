"""
Examples of quality assessment features.
"""

from audio_separator import AudioSeparator
from audio_separator.processor.audio_loader import AudioLoader
from audio_separator.utils.quality_metrics import (
    calculate_separation_quality,
    assess_audio_quality,
    compare_separations
)


def example_separation_quality():
    """Example of separation quality assessment."""
    print("Example 1: Separation Quality Assessment")
    print("-" * 50)
    
    separator = AudioSeparator(model_type="demucs")
    results = separator.separate_file("input.wav", output_dir="separated")
    
    loader = AudioLoader()
    mixture, sr = loader.load("input.wav")
    
    # Load separated sources
    separated = {}
    for source_name, source_path in results.items():
        source_audio, _ = loader.load(source_path, sample_rate=sr)
        separated[source_name] = source_audio
    
    # Calculate quality metrics
    quality = calculate_separation_quality(separated, mixture)
    
    print("Separation Quality Metrics:")
    for source_name, metrics in quality.items():
        print(f"\n  {source_name}:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value:.4f}")
    print()


def example_audio_quality():
    """Example of audio quality assessment."""
    print("Example 2: Audio Quality Assessment")
    print("-" * 50)
    
    loader = AudioLoader()
    audio, sr = loader.load("input.wav")
    
    quality = assess_audio_quality(audio, sample_rate=sr)
    
    print("Audio Quality Metrics:")
    for key, value in quality.items():
        print(f"  {key}: {value:.4f}")
    print()


def example_comparison():
    """Example of comparing two separations."""
    print("Example 3: Separation Comparison")
    print("-" * 50)
    
    # Separate with different models
    separator1 = AudioSeparator(model_type="demucs")
    separator2 = AudioSeparator(model_type="spleeter")
    
    results1 = separator1.separate_file("input.wav", output_dir="separated_demucs")
    results2 = separator2.separate_file("input.wav", output_dir="separated_spleeter")
    
    loader = AudioLoader()
    
    # Load separations
    separation1 = {}
    separation2 = {}
    
    for source_name in ["vocals", "drums", "bass", "other"]:
        if source_name in results1:
            audio, sr = loader.load(results1[source_name])
            separation1[source_name] = audio
        
        if source_name in results2:
            audio, sr = loader.load(results2[source_name])
            separation2[source_name] = audio
    
    # Compare
    comparison = compare_separations(separation1, separation2)
    
    print("Separation Comparison:")
    for key, value in comparison.items():
        print(f"  {key}: {value:.4f}")
    print()


if __name__ == "__main__":
    example_separation_quality()
    example_audio_quality()
    example_comparison()

