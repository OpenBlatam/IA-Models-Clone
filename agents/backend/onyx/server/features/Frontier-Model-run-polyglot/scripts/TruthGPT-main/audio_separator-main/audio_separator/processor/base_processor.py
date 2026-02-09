"""
Base processor for audio processing operations.
Refactored to reduce duplication.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np
import torch

from ..core.base_component import BaseComponent
from ..exceptions import AudioValidationError, AudioProcessingError
from ..logger import logger


class BaseAudioProcessor(BaseComponent, ABC):
    """
    Base class for audio processors.
    
    Provides common functionality:
    - Audio validation
    - Shape normalization
    - Error handling
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        normalize: bool = True,
        name: Optional[str] = None
    ):
        """
        Initialize base processor.
        
        Args:
            sample_rate: Target sample rate
            normalize: Whether to normalize audio
            name: Processor name
        """
        super().__init__(name=name)
        self.sample_rate = sample_rate
        self.normalize = normalize
    
    def _do_initialize(self, **kwargs):
        """Initialize processor (no-op by default)."""
        pass
    
    def validate_audio(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        allow_empty: bool = False
    ) -> None:
        """
        Validate audio input.
        
        Args:
            audio: Audio to validate
            allow_empty: Allow empty audio
            
        Raises:
            AudioValidationError: If audio is invalid
        """
        if not isinstance(audio, (np.ndarray, torch.Tensor)):
            raise AudioValidationError(
                f"Audio must be numpy array or torch tensor, got {type(audio)}",
                component=self.name,
                error_code="INVALID_AUDIO_TYPE"
            )
        
        if isinstance(audio, np.ndarray):
            if not allow_empty and audio.size == 0:
                raise AudioValidationError(
                    "Audio array is empty",
                    component=self.name,
                    error_code="EMPTY_AUDIO"
                )
            if np.any(np.isnan(audio)):
                raise AudioValidationError(
                    "Audio contains NaN values",
                    component=self.name,
                    error_code="NAN_IN_AUDIO"
                )
            if np.any(np.isinf(audio)):
                raise AudioValidationError(
                    "Audio contains Inf values",
                    component=self.name,
                    error_code="INF_IN_AUDIO"
                )
        elif isinstance(audio, torch.Tensor):
            if not allow_empty and audio.numel() == 0:
                raise AudioValidationError(
                    "Audio tensor is empty",
                    component=self.name,
                    error_code="EMPTY_AUDIO"
                )
    
    def normalize_shape(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        target_shape: str = "batch_channels_samples"
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize audio shape to target format.
        
        Args:
            audio: Input audio
            target_shape: Target shape format
            
        Returns:
            Audio with normalized shape
        """
        if target_shape == "batch_channels_samples":
            if isinstance(audio, np.ndarray):
                if audio.ndim == 1:
                    audio = audio.reshape(1, 1, -1)
                elif audio.ndim == 2:
                    if audio.shape[0] > audio.shape[1]:
                        audio = audio.T
                    audio = audio.reshape(1, *audio.shape)
                elif audio.ndim == 3:
                    pass  # Already correct
                else:
                    raise AudioValidationError(
                        f"Invalid audio dimensions: {audio.ndim}",
                        component=self.name,
                        error_code="INVALID_DIMENSIONS"
                    )
            elif isinstance(audio, torch.Tensor):
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0).unsqueeze(0)
                elif audio.dim() == 2:
                    if audio.shape[0] > audio.shape[1]:
                        audio = audio.T
                    audio = audio.unsqueeze(0)
                elif audio.dim() == 3:
                    pass  # Already correct
                else:
                    raise AudioValidationError(
                        f"Invalid audio dimensions: {audio.dim()}",
                        component=self.name,
                        error_code="INVALID_DIMENSIONS"
                    )
        
        return audio
    
    def _normalize_audio(
        self,
        audio: np.ndarray,
        check_clipping: bool = False
    ) -> np.ndarray:
        """
        Normalize audio to [-1, 1].
        
        Single Responsibility: Normalize audio amplitude.
        Eliminates duplication between preprocessor and postprocessor.
        
        Args:
            audio: Input audio array
            check_clipping: Only normalize if audio exceeds 1.0 (for postprocessing)
            
        Returns:
            Normalized audio array
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            if check_clipping:
                # Only normalize if clipping would occur (postprocessing)
                if max_val > 1.0:
                    audio = audio / max_val
            else:
                # Always normalize (preprocessing)
                audio = audio / max_val
        return audio
    
    @abstractmethod
    def process(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Process audio.
        
        Args:
            audio: Input audio
            **kwargs: Additional parameters
            
        Returns:
            Processed audio
        """
        pass

