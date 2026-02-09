"""
Audio loading utilities.
Refactored to use constants.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch

from .constants import (
    ERROR_CODE_FILE_NOT_FOUND,
    ERROR_CODE_NOT_A_FILE,
    ERROR_CODE_LIBROSA_LOAD_FAILED,
    ERROR_CODE_SOUNDFILE_LOAD_FAILED,
    ERROR_CODE_SCIPY_LOAD_FAILED,
    ERROR_CODE_NO_AUDIO_LIBRARY,
    ERROR_CODE_AUDIO_LOAD_FAILED
)
from ..exceptions import AudioIOError, AudioFormatError
from ..logger import logger


class AudioLoader:
    """Load audio files in various formats."""
    
    @staticmethod
    def load(
        audio_path: str,
        sample_rate: Optional[int] = None,
        mono: bool = False
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate (None to keep original)
            mono: Convert to mono if True
            
        Returns:
            Tuple of (audio_array, sample_rate)
            
        Raises:
            AudioIOError: If file cannot be read
            AudioFormatError: If format is not supported
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise AudioIOError(
                f"Audio file not found: {audio_path}",
                component="AudioLoader",
                error_code=ERROR_CODE_FILE_NOT_FOUND
            )
        
        if not audio_path.is_file():
            raise AudioIOError(
                f"Path is not a file: {audio_path}",
                component="AudioLoader",
                error_code=ERROR_CODE_NOT_A_FILE
            )
        
        logger.debug(f"Loading audio file: {audio_path}")
        
        # Try different audio libraries
        try:
            import librosa
            try:
                audio, sr = librosa.load(
                    str(audio_path),
                    sr=sample_rate,
                    mono=mono
                )
                logger.debug(f"Loaded audio: shape={audio.shape}, sr={sr}")
                return audio, sr
            except Exception as e:
                raise AudioFormatError(
                    f"Failed to load audio with librosa: {str(e)}",
                    component="AudioLoader",
                    error_code=ERROR_CODE_LIBROSA_LOAD_FAILED
                ) from e
        except ImportError:
            pass
        
        try:
            import soundfile as sf
            try:
                audio, sr = sf.read(str(audio_path))
                if sample_rate and sr != sample_rate:
                    try:
                        import librosa
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                        sr = sample_rate
                    except ImportError:
                        logger.warning("librosa not available for resampling")
                if mono and audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                logger.debug(f"Loaded audio: shape={audio.shape}, sr={sr}")
                return audio, sr
            except Exception as e:
                raise AudioFormatError(
                    f"Failed to load audio with soundfile: {str(e)}",
                    component="AudioLoader",
                    error_code=ERROR_CODE_SOUNDFILE_LOAD_FAILED
                ) from e
        except ImportError:
            pass
        
        try:
            from scipy.io import wavfile
            try:
                sr, audio = wavfile.read(str(audio_path))
                if sample_rate and sr != sample_rate:
                    try:
                        import librosa
                        audio = librosa.resample(audio.astype(np.float32), 
                                                orig_sr=sr, target_sr=sample_rate)
                        sr = sample_rate
                    except ImportError:
                        logger.warning("librosa not available for resampling")
                if mono and audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                audio = audio.astype(np.float32)
                logger.debug(f"Loaded audio: shape={audio.shape}, sr={sr}")
                return audio, sr
            except Exception as e:
                raise AudioFormatError(
                    f"Failed to load audio with scipy: {str(e)}",
                    component="AudioLoader",
                    error_code=ERROR_CODE_SCIPY_LOAD_FAILED
                ) from e
        except ImportError:
            pass
        
        raise AudioIOError(
            "No audio library found. Install one of: librosa, soundfile, scipy",
            component="AudioLoader",
            error_code=ERROR_CODE_NO_AUDIO_LIBRARY
        )
    
    @staticmethod
    def load_as_tensor(
        audio_path: str,
        sample_rate: Optional[int] = None,
        mono: bool = False
    ) -> Tuple[torch.Tensor, int]:
        """
        Load audio as PyTorch tensor.
        
        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate
            mono: Convert to mono
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        audio, sr = AudioLoader.load(audio_path, sample_rate, mono)
        audio_tensor = torch.from_numpy(audio).float()
        return audio_tensor, sr

