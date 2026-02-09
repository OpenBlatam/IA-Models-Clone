"""
Visualization utilities for audio analysis.
"""

from typing import Optional, List
import numpy as np

from ..logger import logger


def plot_waveform(
    audio: np.ndarray,
    sample_rate: int = 44100,
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Plot audio waveform.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        title: Plot title
        save_path: Path to save plot (None to show)
    """
    try:
        import matplotlib.pyplot as plt
        
        time_axis = np.arange(len(audio)) / sample_rate
        
        plt.figure(figsize=(12, 4))
        plt.plot(time_axis, audio)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.title(title or "Audio Waveform")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved waveform plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        logger.warning("matplotlib not available, cannot plot waveform")


def plot_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 44100,
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Plot audio spectrogram.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        title: Plot title
        save_path: Path to save plot (None to show)
    """
    try:
        import matplotlib.pyplot as plt
        import librosa
        
        # Compute spectrogram
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        db_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # Plot
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(
            db_magnitude,
            sr=sample_rate,
            x_axis='time',
            y_axis='hz',
            hop_length=512
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(title or "Audio Spectrogram")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved spectrogram plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        logger.warning("matplotlib/librosa not available, cannot plot spectrogram")


def plot_separation_comparison(
    original: np.ndarray,
    separated: dict,
    sample_rate: int = 44100,
    save_path: Optional[str] = None
):
    """
    Plot comparison of original and separated audio.
    
    Args:
        original: Original audio array
        separated: Dictionary of separated sources
        sample_rate: Sample rate
        save_path: Path to save plot (None to show)
    """
    try:
        import matplotlib.pyplot as plt
        
        num_sources = len(separated)
        fig, axes = plt.subplots(num_sources + 1, 1, figsize=(12, 3 * (num_sources + 1)))
        
        if num_sources == 0:
            axes = [axes]
        
        time_axis = np.arange(len(original)) / sample_rate
        
        # Plot original
        axes[0].plot(time_axis, original)
        axes[0].set_title("Original Audio")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)
        
        # Plot separated sources
        for i, (source_name, source_audio) in enumerate(separated.items(), 1):
            source_time = np.arange(len(source_audio)) / sample_rate
            axes[i].plot(source_time, source_audio)
            axes[i].set_title(f"Separated: {source_name}")
            axes[i].set_ylabel("Amplitude")
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel("Time (seconds)")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        logger.warning("matplotlib not available, cannot plot comparison")


def create_separation_report(
    audio_path: str,
    separated: dict,
    output_dir: str,
    sample_rate: int = 44100
):
    """
    Create a visual report of separation results.
    
    Args:
        audio_path: Path to original audio
        separated: Dictionary of separated sources
        output_dir: Output directory for report
    """
    from pathlib import Path
    from ..processor.audio_loader import AudioLoader
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loader = AudioLoader()
    original, sr = loader.load(audio_path, sample_rate=sample_rate)
    
    audio_name = Path(audio_path).stem
    
    # Plot original waveform
    plot_waveform(
        original,
        sample_rate=sr,
        title=f"Original: {audio_name}",
        save_path=str(output_dir / f"{audio_name}_original_waveform.png")
    )
    
    # Plot original spectrogram
    plot_spectrogram(
        original,
        sample_rate=sr,
        title=f"Original: {audio_name}",
        save_path=str(output_dir / f"{audio_name}_original_spectrogram.png")
    )
    
    # Plot each separated source
    separated_audio = {}
    for source_name, source_path in separated.items():
        source_audio, _ = loader.load(source_path, sample_rate=sr)
        separated_audio[source_name] = source_audio
        
        plot_waveform(
            source_audio,
            sample_rate=sr,
            title=f"{source_name}: {audio_name}",
            save_path=str(output_dir / f"{audio_name}_{source_name}_waveform.png")
        )
    
    # Plot comparison
    plot_separation_comparison(
        original,
        separated_audio,
        sample_rate=sr,
        save_path=str(output_dir / f"{audio_name}_comparison.png")
    )
    
    logger.info(f"Created separation report in {output_dir}")

