"""
Audio utility functions.
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np


def get_audio_info(audio_path: str) -> dict:
    """
    Get information about an audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with audio information
    """
    try:
        import librosa
        
        duration = librosa.get_duration(path=audio_path)
        sr = librosa.get_samplerate(audio_path)
        
        # Load a small sample to get channels
        y, _ = librosa.load(audio_path, sr=sr, duration=0.1, mono=False)
        channels = 1 if y.ndim == 1 else y.shape[0]
        
        return {
            "duration": duration,
            "sample_rate": sr,
            "channels": channels,
            "file_size": Path(audio_path).stat().st_size
        }
    except ImportError:
        raise ImportError("librosa required for audio info")
    except Exception as e:
        raise RuntimeError(f"Error getting audio info: {str(e)}")


def resample_audio(
    audio: np.ndarray,
    original_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Audio array
        original_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio
    """
    if original_sr == target_sr:
        return audio
    
    try:
        import librosa
        return librosa.resample(
            audio.astype(np.float32),
            orig_sr=original_sr,
            target_sr=target_sr
        )
    except ImportError:
        raise ImportError("librosa required for resampling")


def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert stereo audio to mono.
    
    Args:
        audio: Audio array (can be mono or stereo)
        
    Returns:
        Mono audio array
    """
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        return np.mean(audio, axis=0)
    else:
        raise ValueError(f"Unexpected audio shape: {audio.shape}")


def normalize_audio(audio: np.ndarray, target_max: float = 1.0) -> np.ndarray:
    """
    Normalize audio to target maximum value.
    
    Args:
        audio: Audio array
        target_max: Target maximum value (default: 1.0)
        
    Returns:
        Normalized audio
    """
    max_val = np.abs(audio).max()
    if max_val > 0:
        return audio * (target_max / max_val)
    return audio

