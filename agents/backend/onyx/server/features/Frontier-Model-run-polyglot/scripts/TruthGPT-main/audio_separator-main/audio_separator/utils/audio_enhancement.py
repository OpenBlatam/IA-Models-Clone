"""
Audio enhancement utilities.

Refactored to consolidate functions into AudioEnhancer class.
"""

import numpy as np
from typing import Optional

from ..core.base_component import BaseComponent
from ..separator.base_separator import DEFAULT_SAMPLE_RATE
from .audio_helpers import (
    normalize_by_peak,
    normalize_by_rms,
    calculate_peak,
    calculate_rms
)
from ..logger import logger


class AudioEnhancer(BaseComponent):
    """
    Audio enhancement utilities.
    
    Responsibilities:
    - Denoise audio
    - Normalize audio (peak, RMS)
    - Apply fades
    - Apply compression
    
    Single Responsibility: Handle all audio enhancement operations.
    """
    
    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, name: Optional[str] = None):
        """
        Initialize audio enhancer.
        
        Args:
            sample_rate: Sample rate for audio operations
            name: Component name (defaults to class name)
        """
        super().__init__(name=name or "AudioEnhancer")
        self.sample_rate = sample_rate
        self.initialize()
    
    def _do_initialize(self, **kwargs):
        """No initialization needed."""
        pass
    
    def denoise(
        self,
        audio: np.ndarray,
        method: str = "simple",
        strength: float = 0.5
    ) -> np.ndarray:
        """
        Denoise audio using various methods.
        
        Args:
            audio: Input audio array
            method: Denoising method ('simple', 'spectral', 'wiener')
            strength: Denoising strength (0.0-1.0)
            
        Returns:
            Denoised audio array
        """
        if method == "simple":
            return self._simple_denoise(audio, strength)
        elif method == "spectral":
            return self._spectral_denoise(audio, strength)
        elif method == "wiener":
            return self._wiener_denoise(audio, strength)
        else:
            logger.warning(f"Unknown denoising method: {method}, using simple")
            return self._simple_denoise(audio, strength)
    
    def normalize_peak(self, audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """
        Normalize audio to target peak level.
        
        Uses helper from audio_helpers to eliminate duplication.
        
        Args:
            audio: Input audio array
            target_peak: Target peak level (0.0-1.0)
            
        Returns:
            Normalized audio array
        """
        return normalize_by_peak(audio, target_peak)
    
    def normalize_rms(self, audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        """
        Normalize audio to target RMS level.
        
        Uses helper from audio_helpers to eliminate duplication.
        
        Args:
            audio: Input audio array
            target_rms: Target RMS level
            
        Returns:
            Normalized audio array
        """
        return normalize_by_rms(audio, target_rms)
    
    def apply_fade(
        self,
        audio: np.ndarray,
        fade_in: float = 0.0,
        fade_out: float = 0.0
    ) -> np.ndarray:
        """
        Apply fade in/out to audio.
        
        Args:
            audio: Input audio array
            fade_in: Fade in duration in seconds
            fade_out: Fade out duration in seconds
            
        Returns:
            Audio with fade applied
        """
        result = audio.copy()
        
        if fade_in > 0:
            fade_samples = int(fade_in * self.sample_rate)
            fade_samples = min(fade_samples, len(result))
            fade_curve = np.linspace(0, 1, fade_samples)
            if result.ndim == 1:
                result[:fade_samples] *= fade_curve
            else:
                result[:, :fade_samples] *= fade_curve
        
        if fade_out > 0:
            fade_samples = int(fade_out * self.sample_rate)
            fade_samples = min(fade_samples, len(result))
            fade_curve = np.linspace(1, 0, fade_samples)
            if result.ndim == 1:
                result[-fade_samples:] *= fade_curve
            else:
                result[:, -fade_samples:] *= fade_curve
        
        return result
    
    def apply_compression(
        self,
        audio: np.ndarray,
        threshold: float = 0.7,
        ratio: float = 4.0,
        attack: float = 0.003,
        release: float = 0.1
    ) -> np.ndarray:
        """
        Apply audio compression.
        
        Args:
            audio: Input audio array
            threshold: Compression threshold (0.0-1.0)
            ratio: Compression ratio
            attack: Attack time in seconds
            release: Release time in seconds
            
        Returns:
            Compressed audio array
        """
        compressed = audio.copy()
        envelope = np.abs(audio)
        over_threshold = envelope > threshold
        
        if np.any(over_threshold):
            excess = envelope[over_threshold] - threshold
            compressed_excess = threshold + excess / ratio
            gain = compressed_excess / envelope[over_threshold]
            
            attack_samples = int(attack * self.sample_rate)
            release_samples = int(release * self.sample_rate)
            
            gain_smooth = np.ones_like(envelope)
            gain_smooth[over_threshold] = gain
            
            # Smooth gain changes
            for i in range(1, len(gain_smooth)):
                if gain_smooth[i] < gain_smooth[i-1]:
                    gain_smooth[i] = gain_smooth[i-1] + (gain_smooth[i] - gain_smooth[i-1]) / attack_samples
                else:
                    gain_smooth[i] = gain_smooth[i-1] + (gain_smooth[i] - gain_smooth[i-1]) / release_samples
            
            compressed = audio * gain_smooth
        
        return compressed
    
    # Private helper methods
    def _simple_denoise(self, audio: np.ndarray, strength: float) -> np.ndarray:
        """Simple high-pass filter denoising."""
        window_size = max(1, int(0.001 * len(audio)))  # 1ms window
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            if audio.ndim == 1:
                audio = np.convolve(audio, kernel, mode='same')
            else:
                audio = np.apply_along_axis(
                    lambda x: np.convolve(x, kernel, mode='same'),
                    axis=-1,
                    arr=audio
                )
        return audio * (1 - strength * 0.1)
    
    def _spectral_denoise(self, audio: np.ndarray, strength: float) -> np.ndarray:
        """Spectral subtraction denoising."""
        try:
            import librosa
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            noise_estimate = np.mean(magnitude[:, :5], axis=1, keepdims=True)
            enhanced_magnitude = magnitude - strength * noise_estimate
            enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            audio = librosa.istft(enhanced_stft)
            return audio
        except ImportError:
            logger.warning("librosa/scipy not available, using simple denoising")
            return self._simple_denoise(audio, strength)
    
    def _wiener_denoise(self, audio: np.ndarray, strength: float) -> np.ndarray:
        """Wiener filter denoising."""
        try:
            from scipy import signal
            filtered = signal.wiener(audio, mysize=5)
            return audio * (1 - strength) + filtered * strength
        except ImportError:
            logger.warning("scipy not available, using simple denoising")
            return self._simple_denoise(audio, strength)


# ════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def denoise_audio(
    audio: np.ndarray,
    method: str = "simple",
    strength: float = 0.5
) -> np.ndarray:
    """
    Denoise audio (backward compatibility).
    
    Args:
        audio: Input audio array
        method: Denoising method ('simple', 'spectral', 'wiener')
        strength: Denoising strength (0.0-1.0)
        
    Returns:
        Denoised audio array
    """
    enhancer = AudioEnhancer()
    return enhancer.denoise(audio, method, strength)


def normalize_audio_peak(
    audio: np.ndarray,
    target_peak: float = 0.95
) -> np.ndarray:
    """
    Normalize audio to target peak (backward compatibility).
    
    Uses helper from audio_helpers directly to avoid unnecessary object creation.
    
    Args:
        audio: Input audio array
        target_peak: Target peak level (0.0-1.0)
        
    Returns:
        Normalized audio array
    """
    return normalize_by_peak(audio, target_peak)


def normalize_audio_rms(
    audio: np.ndarray,
    target_rms: float = 0.1
) -> np.ndarray:
    """
    Normalize audio to target RMS (backward compatibility).
    
    Uses helper from audio_helpers directly to avoid unnecessary object creation.
    
    Args:
        audio: Input audio array
        target_rms: Target RMS level
        
    Returns:
        Normalized audio array
    """
    return normalize_by_rms(audio, target_rms)


def apply_fade(
    audio: np.ndarray,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> np.ndarray:
    """
    Apply fade (backward compatibility).
    
    Args:
        audio: Input audio array
        fade_in: Fade in duration in seconds
        fade_out: Fade out duration in seconds
        sample_rate: Sample rate
        
    Returns:
        Audio with fade applied
    """
    enhancer = AudioEnhancer(sample_rate=sample_rate)
    return enhancer.apply_fade(audio, fade_in, fade_out)


def apply_compression(
    audio: np.ndarray,
    threshold: float = 0.7,
    ratio: float = 4.0,
    attack: float = 0.003,
    release: float = 0.1,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> np.ndarray:
    """
    Apply compression (backward compatibility).
    
    Args:
        audio: Input audio array
        threshold: Compression threshold (0.0-1.0)
        ratio: Compression ratio
        attack: Attack time in seconds
        release: Release time in seconds
        sample_rate: Sample rate
        
    Returns:
        Compressed audio array
    """
    enhancer = AudioEnhancer(sample_rate=sample_rate)
    return enhancer.apply_compression(audio, threshold, ratio, attack, release)
