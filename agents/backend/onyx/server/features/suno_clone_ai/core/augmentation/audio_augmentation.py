"""
Audio Augmentation

Utilities for audio data augmentation.
"""

import logging
import numpy as np
import torch
from typing import Optional, List, Callable

logger = logging.getLogger(__name__)

# Try to import librosa for audio augmentation
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa not available for audio augmentation")


class AudioAugmenter:
    """Base class for audio augmentation."""
    
    def __call__(self, audio: np.ndarray, sample_rate: int = 32000) -> np.ndarray:
        """
        Apply augmentation.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            Augmented audio
        """
        raise NotImplementedError


class TimeStretch(AudioAugmenter):
    """Time stretching augmentation."""
    
    def __init__(self, rate: float = 1.0):
        """
        Initialize time stretch.
        
        Args:
            rate: Stretch rate (1.0 = no change)
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("Librosa required for time stretching")
        self.rate = rate
    
    def __call__(self, audio: np.ndarray, sample_rate: int = 32000) -> np.ndarray:
        """Apply time stretching."""
        return librosa.effects.time_stretch(audio, rate=self.rate)


class PitchShift(AudioAugmenter):
    """Pitch shifting augmentation."""
    
    def __init__(self, n_steps: float = 0.0):
        """
        Initialize pitch shift.
        
        Args:
            n_steps: Number of semitones to shift
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("Librosa required for pitch shifting")
        self.n_steps = n_steps
    
    def __call__(self, audio: np.ndarray, sample_rate: int = 32000) -> np.ndarray:
        """Apply pitch shifting."""
        return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=self.n_steps)


class AddNoise(AudioAugmenter):
    """Add noise augmentation."""
    
    def __init__(self, noise_factor: float = 0.01):
        """
        Initialize noise addition.
        
        Args:
            noise_factor: Noise factor (0.0 = no noise)
        """
        self.noise_factor = noise_factor
    
    def __call__(self, audio: np.ndarray, sample_rate: int = 32000) -> np.ndarray:
        """Add noise."""
        noise = np.random.normal(0, self.noise_factor, audio.shape)
        return audio + noise


class TimeMasking(AudioAugmenter):
    """Time masking augmentation."""
    
    def __init__(self, max_mask_length: int = 100):
        """
        Initialize time masking.
        
        Args:
            max_mask_length: Maximum mask length in samples
        """
        self.max_mask_length = max_mask_length
    
    def __call__(self, audio: np.ndarray, sample_rate: int = 32000) -> np.ndarray:
        """Apply time masking."""
        mask_length = np.random.randint(0, self.max_mask_length)
        mask_start = np.random.randint(0, len(audio) - mask_length)
        audio[mask_start:mask_start + mask_length] = 0
        return audio


class FrequencyMasking(AudioAugmenter):
    """Frequency masking augmentation (applied to spectrogram)."""
    
    def __init__(self, max_mask_freq: int = 10):
        """
        Initialize frequency masking.
        
        Args:
            max_mask_freq: Maximum frequency bins to mask
        """
        self.max_mask_freq = max_mask_freq
    
    def __call__(self, audio: np.ndarray, sample_rate: int = 32000) -> np.ndarray:
        """
        Apply frequency masking.
        
        Note: This is a placeholder. For proper frequency masking,
        apply to spectrogram and convert back.
        """
        # For now, return original audio
        # Proper implementation would work on spectrogram
        return audio


def create_audio_augmentation_pipeline(
    augmentations: List[AudioAugmenter],
    p: float = 1.0
) -> Callable:
    """
    Create augmentation pipeline.
    
    Args:
        augmentations: List of augmentation functions
        p: Probability of applying augmentation
        
    Returns:
        Augmentation function
    """
    def augment(audio: np.ndarray, sample_rate: int = 32000) -> np.ndarray:
        if np.random.random() < p:
            for aug in augmentations:
                audio = aug(audio, sample_rate)
        return audio
    
    return augment



