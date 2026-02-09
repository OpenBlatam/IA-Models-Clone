"""
Advanced Audio Processing Optimizations

Fast audio processing with:
- Numba JIT compilation
- Vectorized operations
- Memory-efficient processing
- GPU acceleration where possible
"""

import logging
import numpy as np
from typing import Optional, Tuple
import torch

logger = logging.getLogger(__name__)

# Try to import numba
try:
    from numba import jit, prange, types
    from numba import float32, float64
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@jit(nopython=True, parallel=True) if USE_NUMBA else lambda x: x
def fast_normalize(audio: np.ndarray) -> np.ndarray:
    """Fast normalization with numba."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


@jit(nopython=True, parallel=True) if USE_NUMBA else lambda x: x
def fast_fade_in_out(
    audio: np.ndarray,
    fade_samples: int
) -> np.ndarray:
    """Fast fade in/out with numba."""
    result = audio.copy()
    n = len(audio)
    
    # Fade in
    for i in prange(min(fade_samples, n)):
        factor = i / fade_samples
        result[i] *= factor
    
    # Fade out
    for i in prange(max(0, n - fade_samples), n):
        factor = (n - i) / fade_samples
        result[i] *= factor
    
    return result


@jit(nopython=True) if USE_NUMBA else lambda x: x
def fast_resample_linear(
    audio: np.ndarray,
    original_rate: int,
    target_rate: int
) -> np.ndarray:
    """Fast linear resampling."""
    if original_rate == target_rate:
        return audio
    
    ratio = original_rate / target_rate
    new_length = int(len(audio) / ratio)
    result = np.zeros(new_length, dtype=audio.dtype)
    
    for i in range(new_length):
        src_index = i * ratio
        src_index_int = int(src_index)
        src_index_frac = src_index - src_index_int
        
        if src_index_int + 1 < len(audio):
            result[i] = (
                audio[src_index_int] * (1 - src_index_frac) +
                audio[src_index_int + 1] * src_index_frac
            )
        else:
            result[i] = audio[src_index_int]
    
    return result


class FastAudioProcessor:
    """
    Fast audio processing with optimizations.
    """
    
    @staticmethod
    def normalize(audio: np.ndarray, target_max: float = 1.0) -> np.ndarray:
        """
        Fast normalization.
        
        Args:
            audio: Audio array
            target_max: Target maximum value
            
        Returns:
            Normalized audio
        """
        if USE_NUMBA:
            normalized = fast_normalize(audio)
            return normalized * target_max
        else:
            max_val = np.abs(audio).max()
            if max_val > 0:
                return (audio / max_val) * target_max
            return audio
    
    @staticmethod
    def apply_fade(
        audio: np.ndarray,
        fade_duration_ms: float = 100.0,
        sample_rate: int = 32000
    ) -> np.ndarray:
        """
        Apply fade in/out.
        
        Args:
            audio: Audio array
            fade_duration_ms: Fade duration in milliseconds
            sample_rate: Sample rate
            
        Returns:
            Audio with fade
        """
        fade_samples = int(fade_duration_ms * sample_rate / 1000.0)
        
        if USE_NUMBA:
            return fast_fade_in_out(audio, fade_samples)
        else:
            result = audio.copy()
            n = len(audio)
            
            # Fade in
            fade_in_end = min(fade_samples, n)
            fade_in = np.linspace(0, 1, fade_in_end)
            result[:fade_in_end] *= fade_in
            
            # Fade out
            fade_out_start = max(0, n - fade_samples)
            fade_out = np.linspace(1, 0, n - fade_out_start)
            result[fade_out_start:] *= fade_out
            
            return result
    
    @staticmethod
    def resample(
        audio: np.ndarray,
        original_rate: int,
        target_rate: int
    ) -> np.ndarray:
        """
        Fast resampling.
        
        Args:
            audio: Audio array
            original_rate: Original sample rate
            target_rate: Target sample rate
            
        Returns:
            Resampled audio
        """
        if original_rate == target_rate:
            return audio
        
        if USE_NUMBA:
            return fast_resample_linear(audio, original_rate, target_rate)
        else:
            # Fallback to scipy if available
            try:
                from scipy import signal
                num_samples = int(len(audio) * target_rate / original_rate)
                return signal.resample(audio, num_samples)
            except ImportError:
                # Simple linear interpolation
                ratio = original_rate / target_rate
                indices = np.arange(0, len(audio), ratio)
                indices = indices.astype(int)
                indices = np.clip(indices, 0, len(audio) - 1)
                return audio[indices]
    
    @staticmethod
    def trim_silence(
        audio: np.ndarray,
        threshold: float = 0.01,
        frame_length: int = 2048
    ) -> np.ndarray:
        """
        Trim silence from beginning and end.
        
        Args:
            audio: Audio array
            threshold: Silence threshold
            frame_length: Frame length for analysis
            
        Returns:
            Trimmed audio
        """
        # Find non-silent frames
        frames = len(audio) // frame_length
        energy = np.array([
            np.abs(audio[i * frame_length:(i + 1) * frame_length]).mean()
            for i in range(frames)
        ])
        
        # Find start and end
        non_silent = np.where(energy > threshold)[0]
        if len(non_silent) == 0:
            return audio
        
        start_frame = non_silent[0]
        end_frame = non_silent[-1] + 1
        
        start_sample = start_frame * frame_length
        end_sample = end_frame * frame_length
        
        return audio[start_sample:end_sample]
    
    @staticmethod
    def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
        """
        Apply gain in dB.
        
        Args:
            audio: Audio array
            gain_db: Gain in decibels
            
        Returns:
            Audio with gain applied
        """
        gain_linear = 10 ** (gain_db / 20.0)
        return audio * gain_linear
    
    @staticmethod
    def mix_tracks_fast(
        tracks: list[np.ndarray],
        volumes: Optional[list[float]] = None
    ) -> np.ndarray:
        """
        Fast mixing of multiple tracks.
        
        Args:
            tracks: List of audio tracks
            volumes: Optional volume levels
            
        Returns:
            Mixed audio
        """
        if not tracks:
            return np.array([])
        
        # Find maximum length
        max_length = max(len(track) for track in tracks)
        
        # Normalize volumes
        if volumes is None:
            volumes = [1.0] * len(tracks)
        
        # Mix
        mixed = np.zeros(max_length, dtype=np.float32)
        for track, volume in zip(tracks, volumes):
            track_padded = np.pad(
                track,
                (0, max_length - len(track)),
                mode='constant'
            )
            mixed += track_padded * volume
        
        # Normalize to prevent clipping
        max_val = np.abs(mixed).max()
        if max_val > 1.0:
            mixed = mixed / max_val
        
        return mixed


class GPUAudioProcessor:
    """
    GPU-accelerated audio processing.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize GPU processor.
        
        Args:
            device: CUDA device
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def normalize_gpu(self, audio: np.ndarray) -> np.ndarray:
        """Normalize on GPU."""
        if self.device == "cpu":
            return FastAudioProcessor.normalize(audio)
        
        audio_tensor = torch.from_numpy(audio).to(self.device)
        max_val = torch.abs(audio_tensor).max()
        if max_val > 0:
            audio_tensor = audio_tensor / max_val
        
        return audio_tensor.cpu().numpy()
    
    def apply_fade_gpu(
        self,
        audio: np.ndarray,
        fade_samples: int
    ) -> np.ndarray:
        """Apply fade on GPU."""
        if self.device == "cpu":
            return FastAudioProcessor.apply_fade(audio, fade_samples * 1000 / 32000)
        
        audio_tensor = torch.from_numpy(audio).to(self.device)
        n = len(audio)
        
        # Create fade curves
        fade_in = torch.linspace(0, 1, min(fade_samples, n), device=self.device)
        fade_out = torch.linspace(1, 0, min(fade_samples, n), device=self.device)
        
        # Apply
        audio_tensor[:len(fade_in)] *= fade_in
        audio_tensor[-len(fade_out):] *= fade_out
        
        return audio_tensor.cpu().numpy()













