"""
Examples of audio mixing and merging.
"""

from audio_separator import AudioSeparator
from audio_separator.processor.audio_loader import AudioLoader
from audio_separator.processor.audio_saver import AudioSaver
from audio_separator.utils.audio_merger import (
    merge_sources,
    create_mix,
    blend_audio
)


def example_merge_sources():
    """Example of merging separated sources."""
    print("Example 1: Merging Sources")
    print("-" * 50)
    
    separator = AudioSeparator(model_type="demucs")
    results = separator.separate_file("input.wav", output_dir="separated")
    
    loader = AudioLoader()
    saver = AudioSaver()
    
    # Load separated sources
    sources = {}
    for source_name, source_path in results.items():
        audio, sr = loader.load(source_path)
        sources[source_name] = audio
    
    # Merge with custom volumes
    volumes = {
        "vocals": 1.0,
        "drums": 0.8,
        "bass": 0.9,
        "other": 0.7
    }
    
    merged = merge_sources(sources, volumes=volumes, normalize=True)
    
    # Save merged audio
    saver.save(merged, "output_merged.wav", sample_rate=44100)
    print("Saved merged audio")
    print()


def example_custom_mix():
    """Example of creating custom mix."""
    print("Example 2: Custom Mix")
    print("-" * 50)
    
    separator = AudioSeparator(model_type="demucs")
    results = separator.separate_file("input.wav", output_dir="separated")
    
    loader = AudioLoader()
    saver = AudioSaver()
    
    # Load separated sources
    sources = {}
    for source_name, source_path in results.items():
        audio, sr = loader.load(source_path)
        sources[source_name] = audio
    
    # Create custom mix
    mix_config = {
        "vocals_volume": 1.2,
        "drums_volume": 0.9,
        "bass_volume": 1.0,
        "other_volume": 0.8,
        "fade_in": 0.5,
        "fade_out": 1.0
    }
    
    custom_mix = create_mix(sources, mix_config, sample_rate=44100)
    
    # Save custom mix
    saver.save(custom_mix, "output_custom_mix.wav", sample_rate=44100)
    print("Saved custom mix")
    print()


def example_blend():
    """Example of blending two audio files."""
    print("Example 3: Audio Blending")
    print("-" * 50)
    
    loader = AudioLoader()
    saver = AudioSaver()
    
    # Load two audio files
    audio1, sr1 = loader.load("input1.wav")
    audio2, sr2 = loader.load("input2.wav")
    
    # Blend with 50/50 ratio
    blended = blend_audio(audio1, audio2, blend_ratio=0.5)
    
    # Save blended audio
    saver.save(blended, "output_blended.wav", sample_rate=sr1)
    print("Saved blended audio")
    print()


if __name__ == "__main__":
    example_merge_sources()
    example_custom_mix()
    example_blend()

