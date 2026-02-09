"""
Common audio processing utilities.
Refactored to eliminate duplication.
"""

from typing import Union
import numpy as np
import torch

from ..logger import logger


def normalize_audio_peak(
    audio: np.ndarray,
    target_peak: float = 1.0,
    check_clipping: bool = True
) -> np.ndarray:
    """
    Normalize audio to target peak level.
    
    Args:
        audio: Input audio array
        target_peak: Target peak level (default 1.0)
        check_clipping: Warn if clipping occurs
        
    Returns:
        Normalized audio array
    """
    max_val = np.abs(audio).max()
    
    if max_val == 0:
        return audio
    
    normalized = audio * (target_peak / max_val)
    
    if check_clipping and np.abs(normalized).max() > 1.0:
        logger.warning("Audio clipping detected after normalization")
    
    return normalized


def normalize_audio_rms(
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
    rms = np.sqrt(np.mean(audio ** 2))
    
    if rms == 0:
        return audio
    
    return audio * (target_rms / rms)


def to_numpy(audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert audio to numpy array.
    
    Args:
        audio: Input audio (numpy array or torch tensor)
        
    Returns:
        Numpy array
    """
    if isinstance(audio, torch.Tensor):
        return audio.detach().cpu().numpy()
    return audio


def to_tensor(
    audio: Union[np.ndarray, torch.Tensor],
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Convert audio to torch tensor.
    
    Args:
        audio: Input audio (numpy array or torch tensor)
        dtype: Target tensor dtype
        
    Returns:
        Torch tensor
    """
    if isinstance(audio, np.ndarray):
        return torch.from_numpy(audio).to(dtype)
    return audio.to(dtype)


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio to mono if needed.
    
    Args:
        audio: Input audio array
        
    Returns:
        Mono audio array
    """
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        return np.mean(audio, axis=0)
    else:
        raise ValueError(f"Invalid audio dimensions: {audio.ndim}")


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio to stereo if needed.
    
    Args:
        audio: Input audio array
        
    Returns:
        Stereo audio array
    """
    if audio.ndim == 1:
        return np.stack([audio, audio])
    elif audio.ndim == 2:
        return audio
    else:
        raise ValueError(f"Invalid audio dimensions: {audio.ndim}")
