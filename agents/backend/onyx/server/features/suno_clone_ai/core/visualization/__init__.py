"""
Visualization Module

Provides:
- Training visualization
- Model architecture visualization
- Metrics plotting
- Audio visualization
"""

from .training_plots import (
    TrainingPlotter,
    plot_training_history,
    plot_loss_curves,
    plot_metrics
)

from .model_visualizer import (
    ModelVisualizer,
    visualize_model_architecture,
    plot_attention_weights
)

from .audio_visualizer import (
    AudioVisualizer,
    plot_waveform,
    plot_spectrogram,
    plot_mel_spectrogram
)

__all__ = [
    # Training plots
    "TrainingPlotter",
    "plot_training_history",
    "plot_loss_curves",
    "plot_metrics",
    # Model visualization
    "ModelVisualizer",
    "visualize_model_architecture",
    "plot_attention_weights",
    # Audio visualization
    "AudioVisualizer",
    "plot_waveform",
    "plot_spectrogram",
    "plot_mel_spectrogram"
]



