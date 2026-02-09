"""
Common audio processing helper functions.

This module consolidates frequently used audio operations to eliminate
duplication across the codebase.

Refactored to:
- Consolidate audio padding logic
- Centralize audio calculations (RMS, peak, dB)
- Provide reusable normalization helpers
"""

from typing import Union
import numpy as np


def pad_audio_to_length(
    audio: np.ndarray,
    target_length: int,
    pad_mode: str = 'constant'
) -> np.ndarray:
    """
    Pad audio array to target length.
    
    Args:
        audio: Audio array to pad
        target_length: Target length in samples
        pad_mode: Padding mode (see numpy.pad documentation)
    
    Returns:
        Padded audio array
    
    Raises:
        ValueError: If target_length is less than current length
    """
    current_length = len(audio)
    
    if current_length == target_length:
        return audio
    
    if current_length > target_length:
        return audio[:target_length]
    
    # Pad to target length
    pad_length = target_length - current_length
    return np.pad(audio, (0, pad_length), mode=pad_mode)


def calculate_rms(audio: np.ndarray) -> float:
    """
    Calculate RMS (Root Mean Square) of audio.
    
    Args:
        audio: Audio array
    
    Returns:
        RMS value
    """
    return float(np.sqrt(np.mean(audio ** 2)))


def calculate_peak(audio: np.ndarray) -> float:
    """
    Calculate peak amplitude of audio.
    
    Args:
        audio: Audio array
    
    Returns:
        Peak amplitude value
    """
    return float(np.abs(audio).max())


def amplitude_to_db(amplitude: float, reference: float = 1.0) -> float:
    """
    Convert amplitude to decibels.
    
    Args:
        amplitude: Amplitude value
        reference: Reference amplitude (default 1.0)
    
    Returns:
        Value in decibels
    """
    if amplitude <= 0:
        return -np.inf
    return float(20 * np.log10(amplitude / reference + 1e-10))


def db_to_amplitude(db: float, reference: float = 1.0) -> float:
    """
    Convert decibels to amplitude.
    
    Args:
        db: Value in decibels
        reference: Reference amplitude (default 1.0)
    
    Returns:
        Amplitude value
    """
    return float(reference * (10 ** (db / 20)))


def normalize_by_peak(
    audio: np.ndarray,
    target_peak: float = 1.0
) -> np.ndarray:
    """
    Normalize audio to target peak level.
    
    Args:
        audio: Input audio array
        target_peak: Target peak level (0.0-1.0)
    
    Returns:
        Normalized audio array
    """
    peak = calculate_peak(audio)
    if peak > 0:
        return audio * (target_peak / peak)
    return audio


def normalize_by_rms(
    audio: np.ndarray,
    target_rms: float = 0.1
) -> np.ndarray:
    """
    Normalize audio to target RMS level.
    
    Args:
        audio: Input audio array
        target_rms: Target RMS level
    
    Returns:
        Normalized audio array
    """
    rms = calculate_rms(audio)
    if rms > 0:
        return audio * (target_rms / rms)
    return audio


def ensure_same_length(
    *audio_arrays: np.ndarray,
    pad_mode: str = 'constant'
) -> tuple[np.ndarray, ...]:
    """
    Ensure multiple audio arrays have the same length by padding.
    
    Args:
        *audio_arrays: Variable number of audio arrays
        pad_mode: Padding mode (see numpy.pad documentation)
    
    Returns:
        Tuple of audio arrays with same length
    """
    if not audio_arrays:
        return tuple()
    
    max_length = max(len(audio) for audio in audio_arrays)
    return tuple(
        pad_audio_to_length(audio, max_length, pad_mode)
        for audio in audio_arrays
    )

