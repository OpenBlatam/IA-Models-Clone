"""
Advanced Audio Processing
"""

import numpy as np
import torch
import torchaudio
from typing import Optional, Dict, Any, Tuple, List
import logging
from scipy import signal
import librosa

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Advanced audio processing utilities"""
    
    def __init__(self, sample_rate: int = 32000):
        """
        Initialize audio processor
        
        Args:
            sample_rate: Sample rate
        """
        self.sample_rate = sample_rate
        logger.info(f"AudioProcessor initialized: sample_rate={sample_rate}")
    
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1]
        
        Args:
            audio: Audio array
        
        Returns:
            Normalized audio
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio / max_val
        return audio
    
    def apply_gain(self, audio: np.ndarray, gain_db: float) -> np.ndarray:
        """
        Apply gain to audio
        
        Args:
            audio: Audio array
            gain_db: Gain in dB
        
        Returns:
            Audio with gain applied
        """
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear
    
    def apply_compressor(
        self,
        audio: np.ndarray,
        threshold: float = -12.0,
        ratio: float = 4.0,
        attack: float = 0.003,
        release: float = 0.1
    ) -> np.ndarray:
        """
        Apply audio compression
        
        Args:
            audio: Audio array
            threshold: Threshold in dB
            ratio: Compression ratio
            attack: Attack time in seconds
            release: Release time in seconds
        
        Returns:
            Compressed audio
        """
        # Simple compressor implementation
        threshold_linear = 10 ** (threshold / 20)
        attack_samples = int(attack * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        
        compressed = np.copy(audio)
        envelope = np.abs(audio)
        
        for i in range(1, len(envelope)):
            if envelope[i] > threshold_linear:
                # Compression
                excess = envelope[i] - threshold_linear
                reduced = excess / ratio
                envelope[i] = threshold_linear + reduced
            else:
                # Release
                envelope[i] = envelope[i-1] * 0.99 + envelope[i] * 0.01
        
        # Apply envelope
        compressed = np.sign(audio) * envelope
        
        return compressed
    
    def apply_reverb(
        self,
        audio: np.ndarray,
        room_size: float = 0.5,
        damping: float = 0.5
    ) -> np.ndarray:
        """
        Apply reverb effect
        
        Args:
            audio: Audio array
            room_size: Room size (0-1)
            damping: Damping (0-1)
        
        Returns:
            Audio with reverb
        """
        # Simple reverb using delay and feedback
        delay_samples = int(0.03 * self.sample_rate * room_size)
        reverb = np.zeros_like(audio)
        
        for i in range(delay_samples, len(audio)):
            reverb[i] = audio[i] + audio[i - delay_samples] * damping * 0.3
        
        return reverb
    
    def apply_eq(
        self,
        audio: np.ndarray,
        low_gain: float = 0.0,
        mid_gain: float = 0.0,
        high_gain: float = 0.0
    ) -> np.ndarray:
        """
        Apply equalization
        
        Args:
            audio: Audio array
            low_gain: Low frequency gain in dB
            mid_gain: Mid frequency gain in dB
            high_gain: High frequency gain in dB
        
        Returns:
            Equalized audio
        """
        # Simple EQ using filters
        nyquist = self.sample_rate / 2
        
        # Low shelf
        if low_gain != 0:
            low_cutoff = 200
            b, a = signal.iirfilter(2, low_cutoff/nyquist, btype='low', ftype='butter')
            low_filtered = signal.filtfilt(b, a, audio)
            audio = audio + low_filtered * (10 ** (low_gain / 20) - 1)
        
        # High shelf
        if high_gain != 0:
            high_cutoff = 5000
            b, a = signal.iirfilter(2, high_cutoff/nyquist, btype='high', ftype='butter')
            high_filtered = signal.filtfilt(b, a, audio)
            audio = audio + high_filtered * (10 ** (high_gain / 20) - 1)
        
        return audio
    
    def mix_audio(
        self,
        tracks: List[np.ndarray],
        volumes: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Mix multiple audio tracks
        
        Args:
            tracks: List of audio arrays
            volumes: Optional volume levels (0-1)
        
        Returns:
            Mixed audio
        """
        if not tracks:
            return np.array([])
        
        # Normalize lengths
        max_len = max(len(track) for track in tracks)
        mixed = np.zeros(max_len)
        
        volumes = volumes or [1.0] * len(tracks)
        
        for track, volume in zip(tracks, volumes):
            padded = np.pad(track, (0, max_len - len(track)), mode='constant')
            mixed += padded * volume
        
        # Normalize to prevent clipping
        max_val = np.abs(mixed).max()
        if max_val > 1.0:
            mixed = mixed / max_val
        
        return mixed
    
    def fade_in_out(
        self,
        audio: np.ndarray,
        fade_in_duration: float = 0.5,
        fade_out_duration: float = 0.5
    ) -> np.ndarray:
        """
        Apply fade in/out
        
        Args:
            audio: Audio array
            fade_in_duration: Fade in duration in seconds
            fade_out_duration: Fade out duration in seconds
        
        Returns:
            Audio with fades
        """
        fade_in_samples = int(fade_in_duration * self.sample_rate)
        fade_out_samples = int(fade_out_duration * self.sample_rate)
        
        result = np.copy(audio)
        
        # Fade in
        if fade_in_samples > 0:
            fade_in_curve = np.linspace(0, 1, fade_in_samples)
            result[:fade_in_samples] *= fade_in_curve
        
        # Fade out
        if fade_out_samples > 0:
            fade_out_curve = np.linspace(1, 0, fade_out_samples)
            result[-fade_out_samples:] *= fade_out_curve
        
        return result
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract audio features
        
        Args:
            audio: Audio array
        
        Returns:
            Dictionary of features
        """
        features = {
            "duration": len(audio) / self.sample_rate,
            "rms": np.sqrt(np.mean(audio**2)),
            "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(audio)[0]),
            "spectral_centroid": np.mean(librosa.feature.spectral_centroid(audio, sr=self.sample_rate)[0]),
            "spectral_rolloff": np.mean(librosa.feature.spectral_rolloff(audio, sr=self.sample_rate)[0]),
        }
        
        return features

