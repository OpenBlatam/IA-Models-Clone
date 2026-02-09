"""
Examples of audio enhancement features.
"""

from audio_separator import AudioSeparator
from audio_separator.utils.audio_enhancement import (
    denoise_audio,
    normalize_audio_peak,
    normalize_audio_rms,
    apply_fade,
    apply_compression
)
from audio_separator.processor.audio_loader import AudioLoader
from audio_separator.processor.audio_saver import AudioSaver
import numpy as np


def example_denoising():
    """Example of audio denoising."""
    print("Example 1: Audio Denoising")
    print("-" * 50)
    
    loader = AudioLoader()
    saver = AudioSaver()
    
    # Load audio
    audio, sr = loader.load("input.wav")
    print(f"Loaded audio: {len(audio)} samples")
    
    # Denoise
    denoised = denoise_audio(audio, method="simple", strength=0.5)
    print(f"Denoised audio: {len(denoised)} samples")
    
    # Save
    saver.save(denoised, "output_denoised.wav", sample_rate=sr)
    print("Saved denoised audio")
    print()


def example_normalization():
    """Example of audio normalization."""
    print("Example 2: Audio Normalization")
    print("-" * 50)
    
    loader = AudioLoader()
    saver = AudioSaver()
    
    # Load audio
    audio, sr = loader.load("input.wav")
    
    # Normalize to peak
    normalized_peak = normalize_audio_peak(audio, target_peak=0.95)
    saver.save(normalized_peak, "output_normalized_peak.wav", sample_rate=sr)
    print("Saved peak-normalized audio")
    
    # Normalize to RMS
    normalized_rms = normalize_audio_rms(audio, target_rms=0.1)
    saver.save(normalized_rms, "output_normalized_rms.wav", sample_rate=sr)
    print("Saved RMS-normalized audio")
    print()


def example_fade():
    """Example of applying fade in/out."""
    print("Example 3: Fade In/Out")
    print("-" * 50)
    
    loader = AudioLoader()
    saver = AudioSaver()
    
    # Load audio
    audio, sr = loader.load("input.wav")
    
    # Apply fade
    faded = apply_fade(audio, fade_in=0.5, fade_out=1.0, sample_rate=sr)
    saver.save(faded, "output_faded.wav", sample_rate=sr)
    print("Saved audio with fade in/out")
    print()


def example_compression():
    """Example of audio compression."""
    print("Example 4: Audio Compression")
    print("-" * 50)
    
    loader = AudioLoader()
    saver = AudioSaver()
    
    # Load audio
    audio, sr = loader.load("input.wav")
    
    # Apply compression
    compressed = apply_compression(
        audio,
        threshold=0.7,
        ratio=4.0,
        attack=0.003,
        release=0.1,
        sample_rate=sr
    )
    saver.save(compressed, "output_compressed.wav", sample_rate=sr)
    print("Saved compressed audio")
    print()


def example_enhanced_separation():
    """Example of separation with enhancement."""
    print("Example 5: Enhanced Separation")
    print("-" * 50)
    
    separator = AudioSeparator(model_type="demucs")
    
    # Separate
    results = separator.separate_file("input.wav", output_dir="separated")
    
    # Enhance each separated source
    loader = AudioLoader()
    saver = AudioSaver()
    
    for source_name, source_path in results.items():
        # Load separated source
        audio, sr = loader.load(source_path)
        
        # Enhance
        enhanced = denoise_audio(audio, method="simple", strength=0.3)
        enhanced = normalize_audio_peak(enhanced, target_peak=0.95)
        enhanced = apply_fade(enhanced, fade_in=0.1, fade_out=0.1, sample_rate=sr)
        
        # Save enhanced version
        enhanced_path = source_path.replace(".wav", "_enhanced.wav")
        saver.save(enhanced, enhanced_path, sample_rate=sr)
        print(f"Enhanced {source_name}: {enhanced_path}")
    
    print()


if __name__ == "__main__":
    example_denoising()
    example_normalization()
    example_fade()
    example_compression()
    example_enhanced_separation()

