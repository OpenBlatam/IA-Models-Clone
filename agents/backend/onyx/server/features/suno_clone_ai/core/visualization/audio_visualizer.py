"""
Audio Visualization

Utilities for visualizing audio waveforms and spectrograms.
"""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available for plotting")

# Try to import librosa for spectrograms
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa not available for spectrograms")


class AudioVisualizer:
    """Visualize audio waveforms and spectrograms."""
    
    def __init__(self, figsize: tuple = (12, 6)):
        """
        Initialize audio visualizer.
        
        Args:
            figsize: Figure size
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for visualization")
        
        self.figsize = figsize
    
    def plot_waveform(
        self,
        audio: np.ndarray,
        sample_rate: int = 32000,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> None:
        """
        Plot audio waveform.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            save_path: Path to save plot
            show: Whether to show plot
        """
        duration = len(audio) / sample_rate
        time_axis = np.linspace(0, duration, len(audio))
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(time_axis, audio)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title('Audio Waveform', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved waveform plot: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_spectrogram(
        self,
        audio: np.ndarray,
        sample_rate: int = 32000,
        n_fft: int = 2048,
        hop_length: int = 512,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> None:
        """
        Plot spectrogram.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            n_fft: FFT window size
            hop_length: Hop length
            save_path: Path to save plot
            show: Whether to show plot
        """
        if not LIBROSA_AVAILABLE:
            logger.warning("Librosa not available, using basic FFT")
            # Basic FFT-based spectrogram
            stft = np.abs(np.fft.rfft(audio, n=n_fft))
            freqs = np.fft.rfftfreq(n_fft, 1/sample_rate)
            time_axis = np.linspace(0, len(audio)/sample_rate, len(stft))
            
            fig, ax = plt.subplots(figsize=self.figsize)
            im = ax.imshow(
                stft.reshape(-1, 1).T,
                aspect='auto',
                origin='lower',
                cmap='viridis'
            )
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Frequency (Hz)', fontsize=12)
            ax.set_title('Spectrogram', fontsize=14)
            plt.colorbar(im, ax=ax, label='Magnitude')
        else:
            # Use librosa for proper spectrogram
            D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(D)
            magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
            
            fig, ax = plt.subplots(figsize=self.figsize)
            im = ax.imshow(
                magnitude_db,
                aspect='auto',
                origin='lower',
                cmap='viridis',
                interpolation='nearest'
            )
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Frequency (Hz)', fontsize=12)
            ax.set_title('Spectrogram', fontsize=14)
            plt.colorbar(im, ax=ax, label='Magnitude (dB)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved spectrogram plot: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_mel_spectrogram(
        self,
        audio: np.ndarray,
        sample_rate: int = 32000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> None:
        """
        Plot mel spectrogram.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length
            save_path: Path to save plot
            show: Whether to show plot
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("Librosa required for mel spectrogram")
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(
            mel_spec_db,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Mel Frequency', fontsize=12)
        ax.set_title('Mel Spectrogram', fontsize=14)
        plt.colorbar(im, ax=ax, label='Magnitude (dB)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved mel spectrogram plot: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def plot_waveform(
    audio: np.ndarray,
    sample_rate: int = 32000,
    save_path: Optional[str] = None
) -> None:
    """Convenience function to plot waveform."""
    visualizer = AudioVisualizer()
    visualizer.plot_waveform(audio, sample_rate, save_path)


def plot_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 32000,
    save_path: Optional[str] = None
) -> None:
    """Convenience function to plot spectrogram."""
    visualizer = AudioVisualizer()
    visualizer.plot_spectrogram(audio, sample_rate, save_path=save_path)


def plot_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 32000,
    save_path: Optional[str] = None
) -> None:
    """Convenience function to plot mel spectrogram."""
    visualizer = AudioVisualizer()
    visualizer.plot_mel_spectrogram(audio, sample_rate, save_path=save_path)



