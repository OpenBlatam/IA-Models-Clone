"""
Audio saving utilities.
Refactored to use constants and improve organization.
"""

from pathlib import Path
from typing import Union
import numpy as np
import torch

from .constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_AUDIO_FORMAT,
    ERROR_CODE_SAVE_FAILED,
    ERROR_CODE_UNSUPPORTED_FORMAT,
    INT16_MAX,
    INT32_MAX,
    AUDIO_CLIP_MIN,
    AUDIO_CLIP_MAX
)
from ..exceptions import AudioIOError, AudioFormatError
from ..logger import logger


class AudioSaver:
    """Save audio files in various formats."""
    
    @staticmethod
    def save(
        audio: Union[np.ndarray, torch.Tensor],
        output_path: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        format: str = DEFAULT_AUDIO_FORMAT
    ):
        """
        Save audio to file.
        
        Args:
            audio: Audio array or tensor
            output_path: Output file path
            sample_rate: Sample rate
            format: Audio format (wav, mp3, flac, etc.)
        """
        output_path = Path(output_path)
        
        # Create output directory if needed
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise AudioIOError(
                f"Failed to create output directory: {str(e)}",
                component="AudioSaver",
                error_code=ERROR_CODE_SAVE_FAILED
            ) from e
        
        # Convert and prepare audio
        audio = AudioSaver._prepare_audio(audio)
        
        logger.debug(f"Saving audio: {output_path}, format={format}, sr={sample_rate}")
        
        # Try different audio libraries
        try:
            AudioSaver._save_with_soundfile(audio, output_path, sample_rate, format)
            return
        except ImportError:
            pass
        
        try:
            if format.lower() == 'wav':
                AudioSaver._save_with_scipy(audio, output_path, sample_rate)
                return
        except ImportError:
            pass
        
        # Fallback: try librosa/soundfile combination
        try:
            if format.lower() == 'wav':
                AudioSaver._save_with_soundfile(audio, output_path, sample_rate)
                return
        except ImportError:
            pass
        
        raise AudioIOError(
            "No audio library found. Install one of: soundfile, scipy, librosa",
            component="AudioSaver",
            error_code=ERROR_CODE_SAVE_FAILED
        )
    
    @staticmethod
    def _prepare_audio(audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Prepare audio for saving.
        
        Args:
            audio: Input audio (numpy array or tensor)
            
        Returns:
            Prepared numpy array
        """
        # Convert tensor to numpy
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        
        # Ensure correct shape
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)  # (1, samples) for mono
        elif audio.ndim == 3:
            audio = audio.squeeze(0)  # Remove batch dimension
        
        # Normalize if needed
        audio = AudioSaver._normalize_audio_dtype(audio)
        
        # Clip to valid range
        audio = np.clip(audio, AUDIO_CLIP_MIN, AUDIO_CLIP_MAX)
        
        return audio
    
    @staticmethod
    def _normalize_audio_dtype(audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio dtype to float32.
        
        Args:
            audio: Input audio array
            
        Returns:
            Audio array as float32
        """
        if audio.dtype == np.float32:
            return audio
        elif audio.dtype == np.int16:
            return audio.astype(np.float32) / INT16_MAX
        elif audio.dtype == np.int32:
            return audio.astype(np.float32) / INT32_MAX
        else:
            return audio.astype(np.float32)
    
    @staticmethod
    def _save_with_soundfile(
        audio: np.ndarray,
        output_path: Path,
        sample_rate: int,
        format: str = "wav"
    ) -> None:
        """Save audio using soundfile library."""
        import soundfile as sf
        sf.write(str(output_path), audio.T, sample_rate, format=format)
    
    @staticmethod
    def _save_with_scipy(
        audio: np.ndarray,
        output_path: Path,
        sample_rate: int
    ) -> None:
        """Save audio using scipy library (WAV only)."""
        from scipy.io import wavfile
        # Convert to int16 for scipy
        audio_int16 = (audio * (INT16_MAX - 1)).astype(np.int16)
        wavfile.write(str(output_path), sample_rate, audio_int16.T)

