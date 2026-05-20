"""
Examples of visualization features.
"""

from audio_separator import AudioSeparator
from audio_separator.utils.visualization import (
    plot_waveform,
    plot_spectrogram,
    plot_separation_comparison,
    create_separation_report
)
from audio_separator.processor.audio_loader import AudioLoader


def example_waveform_plot():
    """Example of plotting waveform."""
    print("Example 1: Waveform Plot")
    print("-" * 50)
    
    loader = AudioLoader()
    audio, sr = loader.load("input.wav")
    
    plot_waveform(
        audio,
        sample_rate=sr,
        title="Example Audio Waveform",
        save_path="output_waveform.png"
    )
    print("Saved waveform plot")
    print()


def example_spectrogram_plot():
    """Example of plotting spectrogram."""
    print("Example 2: Spectrogram Plot")
    print("-" * 50)
    
    loader = AudioLoader()
    audio, sr = loader.load("input.wav")
    
    plot_spectrogram(
        audio,
        sample_rate=sr,
        title="Example Audio Spectrogram",
        save_path="output_spectrogram.png"
    )
    print("Saved spectrogram plot")
    print()


def example_separation_visualization():
    """Example of visualizing separation results."""
    print("Example 3: Separation Visualization")
    print("-" * 50)
    
    separator = AudioSeparator(model_type="demucs")
    results = separator.separate_file("input.wav", output_dir="separated")
    
    # Create visual report
    create_separation_report(
        "input.wav",
        results,
        "report",
        sample_rate=44100
    )
    print("Created separation report")
    print()


if __name__ == "__main__":
    example_waveform_plot()
    example_spectrogram_plot()
    example_separation_visualization()

