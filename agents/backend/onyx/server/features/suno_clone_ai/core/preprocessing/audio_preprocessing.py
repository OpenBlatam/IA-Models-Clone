"""
Audio Preprocessing

Utilities for preprocessing audio data.
"""

import logging
import numpy as np
from typing import Optional, Callable, List

logger = logging.getLogger(__name__)

# Try to import librosa
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa not available for audio preprocessing")


class AudioPreprocessor:
    """Preprocess audio data."""
    
    def __init__(self, target_sample_rate: int = 32000):
        """
        Initialize audio preprocessor.
        
        Args:
            target_sample_rate: Target sample rate
        """
        self.target_sample_rate = target_sample_rate
    
    def normalize(
        self,
        audio: np.ndarray,
        method: str = "peak"
    ) -> np.ndarray:
        """
        Normalize audio.
        
        Args:
            audio: Audio array
            method: Normalization method ('peak', 'rms')
            
        Returns:
            Normalized audio
        """
        if method == "peak":
            max_val = np.abs(audio).max()
            if max_val > 0:
                return audio / max_val
            return audio
        elif method == "rms":
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                return audio / rms
            return audio
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def resample(
        self,
        audio: np.ndarray,
        original_sample_rate: int
    ) -> np.ndarray:
        """
        Resample audio.
        
        Args:
            audio: Audio array
            original_sample_rate: Original sample rate
            
        Returns:
            Resampled audio
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("Librosa required for resampling")
        
        if original_sample_rate == self.target_sample_rate:
            return audio
        
        return librosa.resample(
            audio,
            orig_sr=original_sample_rate,
            target_sr=self.target_sample_rate
        )
    
    def trim(
        self,
        audio: np.ndarray,
        top_db: float = 60.0
    ) -> np.ndarray:
        """
        Trim silence from audio.
        
        Args:
            audio: Audio array
            top_db: Top dB for trimming
            
        Returns:
            Trimmed audio
        """
        if not LIBROSA_AVAILABLE:
            logger.warning("Librosa not available, skipping trim")
            return audio
        
        return librosa.effects.trim(audio, top_db=top_db)[0]
    
    def pad(
        self,
        audio: np.ndarray,
        target_length: int,
        mode: str = "constant"
    ) -> np.ndarray:
        """
        Pad audio to target length.
        
        Args:
            audio: Audio array
            target_length: Target length
            mode: Padding mode
            
        Returns:
            Padded audio
        """
        current_length = len(audio)
        
        if current_length >= target_length:
            return audio[:target_length]
        
        pad_length = target_length - current_length
        
        if mode == "constant":
            return np.pad(audio, (0, pad_length), mode='constant')
        elif mode == "reflect":
            return np.pad(audio, (0, pad_length), mode='reflect')
        else:
            raise ValueError(f"Unknown padding mode: {mode}")


def normalize_audio(
    audio: np.ndarray,
    method: str = "peak"
) -> np.ndarray:
    """Convenience function to normalize audio."""
    preprocessor = AudioPreprocessor()
    return preprocessor.normalize(audio, method)


def resample_audio(
    audio: np.ndarray,
    original_sample_rate: int,
    target_sample_rate: int = 32000
) -> np.ndarray:
    """Convenience function to resample audio."""
    preprocessor = AudioPreprocessor(target_sample_rate)
    return preprocessor.resample(audio, original_sample_rate)


def trim_audio(
    audio: np.ndarray,
    top_db: float = 60.0
) -> np.ndarray:
    """Convenience function to trim audio."""
    preprocessor = AudioPreprocessor()
    return preprocessor.trim(audio, top_db)


def create_audio_preprocessing_pipeline(
    steps: List[str],
    target_sample_rate: int = 32000
) -> Callable:
    """
    Create audio preprocessing pipeline.
    
    Args:
        steps: List of preprocessing steps
        target_sample_rate: Target sample rate
        
    Returns:
        Preprocessing function
    """
    preprocessor = AudioPreprocessor(target_sample_rate)
    
    def preprocess(audio: np.ndarray, original_sample_rate: int = 32000) -> np.ndarray:
        for step in steps:
            if step == "normalize":
                audio = preprocessor.normalize(audio)
            elif step == "resample":
                audio = preprocessor.resample(audio, original_sample_rate)
            elif step == "trim":
                audio = preprocessor.trim(audio)
            else:
                logger.warning(f"Unknown preprocessing step: {step}")
        
        return audio
    
    return preprocess



