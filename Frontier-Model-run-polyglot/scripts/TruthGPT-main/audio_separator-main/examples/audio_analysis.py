"""
Examples of audio analysis features.
"""

from audio_separator.processor.audio_loader import AudioLoader
from audio_separator.utils.audio_analysis import (
    analyze_audio,
    detect_silence,
    calculate_loudness,
    detect_beats,
    extract_features
)


def example_basic_analysis():
    """Example of basic audio analysis."""
    print("Example 1: Basic Audio Analysis")
    print("-" * 50)
    
    loader = AudioLoader()
    audio, sr = loader.load("input.wav")
    
    analysis = analyze_audio(audio, sample_rate=sr)
    
    print("Audio Statistics:")
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()


def example_silence_detection():
    """Example of silence detection."""
    print("Example 2: Silence Detection")
    print("-" * 50)
    
    loader = AudioLoader()
    audio, sr = loader.load("input.wav")
    
    silence_regions = detect_silence(
        audio,
        threshold=0.01,
        min_duration=0.1,
        sample_rate=sr
    )
    
    print(f"Found {len(silence_regions)} silence regions:")
    for start, end in silence_regions:
        print(f"  {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
    print()


def example_loudness():
    """Example of loudness calculation."""
    print("Example 3: Loudness Calculation")
    print("-" * 50)
    
    loader = AudioLoader()
    audio, sr = loader.load("input.wav")
    
    loudness = calculate_loudness(audio, sample_rate=sr)
    
    print("Loudness Metrics:")
    for key, value in loudness.items():
        print(f"  {key}: {value:.2f} dB")
    print()


def example_beat_detection():
    """Example of beat detection."""
    print("Example 4: Beat Detection")
    print("-" * 50)
    
    loader = AudioLoader()
    audio, sr = loader.load("input.wav")
    
    beat_times, tempo = detect_beats(audio, sample_rate=sr)
    
    print(f"Detected tempo: {tempo:.1f} BPM")
    print(f"Found {len(beat_times)} beats")
    if len(beat_times) > 0:
        print(f"First 5 beats: {beat_times[:5]}")
    print()


def example_feature_extraction():
    """Example of feature extraction."""
    print("Example 5: Feature Extraction")
    print("-" * 50)
    
    loader = AudioLoader()
    audio, sr = loader.load("input.wav")
    
    features = extract_features(audio, sample_rate=sr)
    
    print("Extracted Features:")
    for feature_name, feature_data in features.items():
        print(f"  {feature_name}: shape {feature_data.shape}")
    print()


if __name__ == "__main__":
    example_basic_analysis()
    example_silence_detection()
    example_loudness()
    example_beat_detection()
    example_feature_extraction()

