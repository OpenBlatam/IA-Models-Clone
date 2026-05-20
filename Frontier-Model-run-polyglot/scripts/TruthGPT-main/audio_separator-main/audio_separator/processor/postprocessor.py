"""
Audio postprocessing utilities.
Refactored to use base processor.
"""

from typing import Dict
import numpy as np
import torch

from .base_processor import BaseAudioProcessor
from .constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_NORMALIZE,
    DEFAULT_DENOISE,
    ERROR_CODE_POSTPROCESS_FAILED,
    ERROR_CODE_EMPTY_SEPARATED,
    ERROR_CODE_INVALID_TENSOR_TYPE
)
from .audio_utils import normalize_audio_peak
from ..exceptions import AudioValidationError, AudioProcessingError
from ..logger import logger


class AudioPostprocessor(BaseAudioProcessor):
    """Postprocess model outputs."""
    
    def __init__(
        self,
        normalize: bool = DEFAULT_NORMALIZE,
        denoise: bool = DEFAULT_DENOISE,
        sample_rate: int = DEFAULT_SAMPLE_RATE
    ):
        """
        Initialize postprocessor.
        
        Args:
            normalize: Normalize output audio
            denoise: Apply denoising
            sample_rate: Sample rate
        """
        super().__init__(sample_rate=sample_rate, normalize=normalize, name="AudioPostprocessor")
        self.denoise = denoise
        self.initialize()
    
    def process(
        self,
        separated: Dict[str, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """
        Postprocess separated audio sources.
        
        Args:
            separated: Dictionary of separated tensors
            
        Returns:
            Dictionary of numpy arrays
            
        Raises:
            AudioValidationError: If separated dictionary is invalid
            AudioProcessingError: If postprocessing fails
        """
        if not separated:
            raise AudioValidationError(
                "Separated dictionary is empty",
                component=self.name,
                error_code=ERROR_CODE_EMPTY_SEPARATED
            )
        
        try:
            result = {}
            
            for source_name, tensor in separated.items():
                # Validate tensor
                if not isinstance(tensor, torch.Tensor):
                    raise AudioValidationError(
                        f"Expected torch.Tensor for source '{source_name}', got {type(tensor)}",
                        component=self.name,
                        error_code=ERROR_CODE_INVALID_TENSOR_TYPE
                    )
                
                # Convert to numpy
                audio = tensor.detach().cpu().numpy()
                
                # Remove batch/channel dimensions if needed
                if audio.ndim == 3:
                    audio = audio.squeeze(0)  # Remove batch
                if audio.ndim == 2 and audio.shape[0] == 1:
                    audio = audio.squeeze(0)  # Remove channel if mono
                
                # Denoise if requested
                if self.denoise:
                    audio = self._denoise(audio)
                
                # Normalize
                if self.normalize:
                    audio = normalize_audio_peak(audio, check_clipping=False)
                
                result[source_name] = audio
            
            logger.debug(f"Postprocessed {len(result)} sources")
            return result
            
        except (AudioValidationError,):
            raise
        except Exception as e:
            raise AudioProcessingError(
                f"Postprocessing failed: {str(e)}",
                component=self.name,
                error_code=ERROR_CODE_POSTPROCESS_FAILED
            ) from e
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1].
        
        Delegates to base class method to eliminate duplication.
        Only normalizes if clipping would occur (check_clipping=True).
        """
        return self._normalize_audio(audio, check_clipping=True)
    
    def _denoise(self, audio: np.ndarray) -> np.ndarray:
        """Simple denoising (can be enhanced with more sophisticated methods)."""
        # Simple high-pass filter to remove low-frequency noise
        # In practice, you might use more sophisticated methods
        return audio

