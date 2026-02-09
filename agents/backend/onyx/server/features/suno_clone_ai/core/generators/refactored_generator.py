"""
Refactored Music Generator

Refactored version of MusicGenerator following deep learning best practices:
- Uses shared utilities
- Better error handling
- Improved logging
- Consistent with base generator interface
"""

import logging
from typing import Optional, Dict, Any
import torch
import torchaudio
import numpy as np
from pathlib import Path

from .transformers_generator import TransformersMusicGenerator
from ..utils.model_utils import check_for_nan_inf
from ...config.settings import settings

logger = logging.getLogger(__name__)


class RefactoredMusicGenerator(TransformersMusicGenerator):
    """
    Refactored music generator with enhanced features.
    
    Extends TransformersMusicGenerator with:
    - Audio saving functionality
    - Better integration with settings
    - Enhanced error handling
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_mixed_precision: bool = True,
        use_compile: bool = True,
        compile_mode: str = "reduce-overhead"
    ):
        """
        Initialize refactored music generator.
        
        Args:
            model_name: Model name (uses settings if None)
            use_mixed_precision: Enable mixed precision
            use_compile: Compile model for speed
            compile_mode: Compilation mode
        """
        model_name = model_name or settings.music_model
        
        super().__init__(
            model_name=model_name,
            use_mixed_precision=use_mixed_precision,
            use_compile=use_compile,
            compile_mode=compile_mode
        )
    
    def generate_from_text(
        self,
        text: str,
        duration: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate music from text (backward compatibility method).
        
        Args:
            text: Song description or lyrics
            duration: Duration in seconds
            guidance_scale: Guidance scale
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            **kwargs: Additional parameters
            
        Returns:
            Generated audio array
        """
        # Use default values from settings
        duration = duration or settings.default_duration
        guidance_scale = guidance_scale or settings.cfg_coef
        temperature = temperature or settings.temperature
        top_k = top_k or settings.top_k
        top_p = top_p or settings.top_p
        
        # Validate duration
        if duration <= 0 or duration > settings.max_audio_length:
            raise ValueError(
                f"Duration must be between 1 and {settings.max_audio_length} seconds"
            )
        
        # Generate using parent method
        audio = self.generate(
            prompt=text,
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            guidance_scale=guidance_scale,
            **kwargs
        )
        
        return audio
    
    def save_audio(
        self,
        audio: np.ndarray,
        output_path: str,
        sample_rate: Optional[int] = None
    ) -> str:
        """
        Save generated audio to file with proper error handling.
        
        Args:
            audio: Audio array
            output_path: Path where to save the file
            sample_rate: Sample rate of the audio
            
        Returns:
            Path of saved file
            
        Raises:
            ValueError: If audio is invalid
        """
        try:
            # Validate audio
            if audio is None or len(audio) == 0:
                raise ValueError("Audio array is empty")
            
            if np.isnan(audio).any() or np.isinf(audio).any():
                raise ValueError("Audio contains NaN or Inf values")
            
            sample_rate = sample_rate or settings.sample_rate
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Normalize audio to prevent clipping
            max_val = torch.abs(audio_tensor).max()
            if max_val > 1.0:
                audio_tensor = audio_tensor / max_val
                logger.warning("Audio normalized to prevent clipping")
            
            # Save file
            torchaudio.save(
                str(output_path),
                audio_tensor,
                sample_rate,
                format="wav"
            )
            
            logger.info(f"Audio saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving audio: {e}", exc_info=True)
            raise
    
    def generate_and_save(
        self,
        text: str,
        output_path: str,
        duration: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate music and save directly.
        
        Args:
            text: Song description
            output_path: Path where to save
            duration: Duration in seconds
            **kwargs: Additional parameters
            
        Returns:
            Path of saved file
        """
        audio = self.generate_from_text(text, duration=duration, **kwargs)
        return self.save_audio(audio, output_path)


# Global instance for backward compatibility
_refactored_generator: Optional[RefactoredMusicGenerator] = None


def get_refactored_music_generator() -> RefactoredMusicGenerator:
    """
    Get the global refactored music generator instance.
    
    Returns:
        RefactoredMusicGenerator instance
    """
    global _refactored_generator
    if _refactored_generator is None:
        _refactored_generator = RefactoredMusicGenerator()
    return _refactored_generator

