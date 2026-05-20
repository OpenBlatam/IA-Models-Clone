"""
Audio preprocessing utilities.
Refactored to use base processor.
"""

from typing import Optional, Union
import numpy as np
import torch

from .base_processor import BaseAudioProcessor
from .constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_NORMALIZE,
    DEFAULT_TRIM_SILENCE,
    ERROR_CODE_PREPROCESS_FAILED
)
from .audio_utils import normalize_audio_peak
from ..exceptions import AudioValidationError, AudioProcessingError
from ..logger import logger


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_SILENCE_THRESHOLD = 0.01


class AudioPreprocessor(BaseAudioProcessor):
    """Preprocess audio for model input."""
    
    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        normalize: bool = DEFAULT_NORMALIZE,
        trim_silence: bool = DEFAULT_TRIM_SILENCE
    ):
        """
        Initialize preprocessor.
        
        Args:
            sample_rate: Target sample rate
            normalize: Normalize audio to [-1, 1]
            trim_silence: Trim leading/trailing silence
        """
        super().__init__(sample_rate=sample_rate, normalize=normalize, name="AudioPreprocessor")
        self.trim_silence = trim_silence
        self.initialize()
    
    def process(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        original_sr: Optional[int] = None
    ) -> torch.Tensor:
        """
        Preprocess audio.
        
        Args:
            audio: Input audio
            original_sr: Original sample rate (for resampling)
            
        Returns:
            Preprocessed tensor
            
        Raises:
            AudioValidationError: If audio is invalid
            AudioProcessingError: If processing fails
        """
        try:
            # Validate audio
            self.validate_audio(audio)
            
            # Convert to numpy if tensor
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            
            # Resample if needed
            if original_sr and original_sr != self.sample_rate:
                audio = self._resample(audio, original_sr, self.sample_rate)
            
            # Trim silence
            if self.trim_silence:
                audio = self._trim_silence(audio)
            
            # Normalize
            if self.normalize:
                audio = normalize_audio_peak(audio)
            
            # Convert to tensor and normalize shape
            audio_tensor = torch.from_numpy(audio).float()
            audio_tensor = self.normalize_shape(audio_tensor)
            
            return audio_tensor
            
        except (AudioValidationError, ImportError):
            raise
        except Exception as e:
            raise AudioProcessingError(
                f"Preprocessing failed: {str(e)}",
                component=self.name,
                error_code=ERROR_CODE_PREPROCESS_FAILED
            ) from e
    
    def _resample(
        self,
        audio: np.ndarray,
        original_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        Resample audio.
        
        Args:
            audio: Input audio
            original_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio
        """
        try:
            import librosa
            return librosa.resample(
                audio.astype(np.float32),
                orig_sr=original_sr,
                target_sr=target_sr
            )
        except ImportError:
            raise ImportError("librosa required for resampling")
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1].
        
        Delegates to base class method to eliminate duplication.
        """
        return self._normalize_audio(audio, check_clipping=False)
    
    def _trim_silence(self, audio: np.ndarray, threshold: float = DEFAULT_SILENCE_THRESHOLD) -> np.ndarray:
        """
        Trim leading and trailing silence.
        
        Args:
            audio: Input audio array
            threshold: Silence threshold (default: DEFAULT_SILENCE_THRESHOLD)
            
        Returns:
            Trimmed audio array
        """
        # Find non-silent regions
        non_silent = np.abs(audio) > threshold
        
        if non_silent.any():
            start = np.argmax(non_silent)
            end = len(audio) - np.argmax(non_silent[::-1])
            audio = audio[start:end]
        
        return audio

