"""
Base model for audio source separation.

Refactored to:
- Extract constants and error codes
- Improve validation logic
- Enhance type hints and documentation
- Better error handling
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np

from ..exceptions import AudioModelError, AudioValidationError
from ..logger import logger
from .constants import (
    DEFAULT_NUM_SOURCES,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_N_FFT,
    DEFAULT_HOP_LENGTH,
    ERROR_CODE_INVALID_NUM_SOURCES,
    ERROR_CODE_INVALID_SAMPLE_RATE,
    ERROR_CODE_INVALID_N_FFT,
    ERROR_CODE_INVALID_HOP_LENGTH,
    ERROR_CODE_EMPTY_AUDIO,
    ERROR_CODE_INVALID_AUDIO_TYPE,
    ERROR_CODE_INVALID_AUDIO_DIMENSIONS,
    ERROR_CODE_EMPTY_SEPARATED,
    ERROR_CODE_INVALID_TENSOR_TYPE
)

# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def _validate_positive_int(value: int, name: str, error_code: str) -> None:
    """
    Validate that a value is a positive integer.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        error_code: Error code to use
        
    Raises:
        AudioValidationError: If value is invalid
    """
    if value <= 0:
        raise AudioValidationError(
            f"{name} must be > 0, got {value}",
            component="BaseSeparatorModel",
            error_code=error_code
        )


def _validate_num_sources(num_sources: int) -> None:
    """Validate number of sources."""
    if num_sources < 1:
        raise AudioValidationError(
            f"num_sources must be >= 1, got {num_sources}",
            component="BaseSeparatorModel",
            error_code=ERROR_CODE_INVALID_NUM_SOURCES
        )


# ════════════════════════════════════════════════════════════════════════════
# BASE SEPARATOR MODEL
# ════════════════════════════════════════════════════════════════════════════

class BaseSeparatorModel(nn.Module, ABC):
    """
    Base class for audio source separation models.
    
    This class defines the interface that all separator models must implement.
    It provides common functionality for:
    - Parameter validation
    - Audio preprocessing (numpy/torch conversion, shape handling)
    - Audio postprocessing (tensor to numpy conversion)
    
    Subclasses must implement:
    - forward(): Model forward pass for separation
    - separate(): High-level separation interface
    """
    
    def __init__(
        self,
        num_sources: int = DEFAULT_NUM_SOURCES,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        **kwargs
    ):
        """
        Initialize the base separator model.
        
        Args:
            num_sources: Number of sources to separate (e.g., 4 for vocals, drums, bass, other)
            sample_rate: Audio sample rate in Hz
            n_fft: FFT window size for STFT
            hop_length: Hop length for STFT
            **kwargs: Additional model-specific parameters
            
        Raises:
            AudioValidationError: If parameters are invalid
        """
        super().__init__()
        
        # Validate parameters
        _validate_num_sources(num_sources)
        _validate_positive_int(sample_rate, "sample_rate", ERROR_CODE_INVALID_SAMPLE_RATE)
        _validate_positive_int(n_fft, "n_fft", ERROR_CODE_INVALID_N_FFT)
        _validate_positive_int(hop_length, "hop_length", ERROR_CODE_INVALID_HOP_LENGTH)
        
        # Store parameters
        self.num_sources = num_sources
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        logger.debug(
            f"Initialized {self.__class__.__name__} with "
            f"num_sources={num_sources}, sample_rate={sample_rate}, "
            f"n_fft={n_fft}, hop_length={hop_length}"
        )
    
    @abstractmethod
    def forward(
        self, 
        audio: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to separate audio sources.
        
        This is the core model inference method.
        
        Args:
            audio: Input audio tensor of shape (batch, channels, samples)
            
        Returns:
            Dictionary mapping source names to separated audio tensors
            Each tensor has shape (batch, channels, samples)
        """
        pass
    
    @abstractmethod
    def separate(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Separate audio file into multiple sources.
        
        This is the high-level interface for audio separation.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save separated sources (None for default)
            **kwargs: Additional arguments for separation
            
        Returns:
            Dictionary mapping source names to output file paths
        """
        pass
    
    def preprocess(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess audio for model input.
        
        Handles:
        - Conversion from numpy to torch
        - Type conversion to float32
        - Shape normalization to (batch, channels, samples)
        
        Args:
            audio: Raw audio array or tensor
            
        Returns:
            Preprocessed tensor of shape (batch, channels, samples)
            
        Raises:
            AudioValidationError: If audio format is invalid
        """
        # Convert to torch tensor if needed
        if isinstance(audio, np.ndarray):
            if audio.size == 0:
                raise AudioValidationError(
                    "Audio array is empty",
                    component="BaseSeparatorModel",
                    error_code=ERROR_CODE_EMPTY_AUDIO
                )
            audio = torch.from_numpy(audio).float()
        elif isinstance(audio, torch.Tensor):
            audio = audio.float()
        else:
            raise AudioValidationError(
                f"Audio must be numpy array or torch tensor, got {type(audio)}",
                component="BaseSeparatorModel",
                error_code=ERROR_CODE_INVALID_AUDIO_TYPE
            )
        
        # Normalize shape to (batch, channels, samples)
        audio = self._normalize_audio_shape(audio)
        
        return audio
    
    def _normalize_audio_shape(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Normalize audio tensor to shape (batch, channels, samples).
        
        Args:
            audio: Input tensor with 1-3 dimensions
            
        Returns:
            Tensor with shape (batch, channels, samples)
            
        Raises:
            AudioValidationError: If dimensions are invalid
        """
        dims = audio.dim()
        
        if dims == 1:
            # Mono audio: (samples) -> (1, 1, samples)
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif dims == 2:
            # Stereo or (samples, channels): -> (1, channels, samples)
            if audio.shape[0] > audio.shape[1]:
                # Assume (samples, channels), transpose to (channels, samples)
                audio = audio.T
            audio = audio.unsqueeze(0)
        elif dims == 3:
            # Already in correct format (batch, channels, samples)
            pass
        else:
            raise AudioValidationError(
                f"Audio must have 1-3 dimensions, got {dims}",
                component="BaseSeparatorModel",
                error_code=ERROR_CODE_INVALID_AUDIO_DIMENSIONS
            )
        
        return audio
    
    def postprocess(
        self, 
        separated: Dict[str, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """
        Postprocess model outputs to numpy arrays.
        
        Handles:
        - Tensor to numpy conversion
        - Batch dimension removal
        - Mono channel dimension removal
        
        Args:
            separated: Dictionary of separated tensors
            
        Returns:
            Dictionary of numpy arrays ready for audio I/O
            
        Raises:
            AudioValidationError: If separated dictionary is empty or invalid
        """
        if not separated:
            raise AudioValidationError(
                "Separated dictionary is empty",
                component="BaseSeparatorModel",
                error_code=ERROR_CODE_EMPTY_SEPARATED
            )
        
        result = {}
        for source_name, tensor in separated.items():
            if not isinstance(tensor, torch.Tensor):
                raise AudioValidationError(
                    f"Expected torch.Tensor for source '{source_name}', got {type(tensor)}",
                    component="BaseSeparatorModel",
                    error_code=ERROR_CODE_INVALID_TENSOR_TYPE
                )
            
            # Convert to numpy and remove batch/channel dimensions if needed
            audio = tensor.detach().cpu().numpy()
            audio = self._remove_extra_dimensions(audio)
            
            result[source_name] = audio
        
        logger.debug(f"Postprocessed {len(result)} sources")
        return result
    
    def _remove_extra_dimensions(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove batch and mono channel dimensions from audio array.
        
        Args:
            audio: Audio array with possible batch/channel dimensions
            
        Returns:
            Audio array with extra dimensions removed
        """
        # Remove batch dimension if present
        if audio.ndim == 3:
            audio = audio.squeeze(0)
        
        # Remove channel dimension if mono
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio.squeeze(0)
        
        return audio
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"{self.__class__.__name__}("
            f"num_sources={self.num_sources}, "
            f"sample_rate={self.sample_rate}, "
            f"n_fft={self.n_fft}, "
            f"hop_length={self.hop_length})"
        )
