"""
Input Validation

Validates user inputs for generation, training, etc.
"""

import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class InputValidator:
    """Validates various types of inputs."""
    
    @staticmethod
    def validate_prompt(
        prompt: str,
        max_length: int = 512,
        min_length: int = 1
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate text prompt.
        
        Args:
            prompt: Text prompt
            max_length: Maximum prompt length
            min_length: Minimum prompt length
            
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(prompt, str):
            return False, "Prompt must be a string"
        
        if len(prompt.strip()) < min_length:
            return False, f"Prompt too short (min {min_length} characters)"
        
        if len(prompt) > max_length:
            return False, f"Prompt too long (max {max_length} characters)"
        
        return True, None
    
    @staticmethod
    def validate_audio(
        audio: np.ndarray,
        sample_rate: int = 32000,
        max_duration: float = 30.0,
        min_duration: float = 0.1
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate audio array.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            max_duration: Maximum duration in seconds
            min_duration: Minimum duration in seconds
            
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(audio, np.ndarray):
            return False, "Audio must be numpy array"
        
        if audio.ndim != 1:
            return False, "Audio must be 1D array"
        
        if len(audio) == 0:
            return False, "Audio cannot be empty"
        
        duration = len(audio) / sample_rate
        
        if duration < min_duration:
            return False, f"Audio too short (min {min_duration}s)"
        
        if duration > max_duration:
            return False, f"Audio too long (max {max_duration}s)"
        
        # Check for NaN/Inf
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            return False, "Audio contains NaN or Inf values"
        
        return True, None
    
    @staticmethod
    def validate_generation_params(
        duration: Optional[float] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate generation parameters.
        
        Args:
            duration: Generation duration
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional parameters
            
        Returns:
            (is_valid, error_message)
        """
        if duration is not None:
            if duration <= 0:
                return False, "Duration must be positive"
            if duration > 300:  # 5 minutes max
                return False, "Duration too long (max 300s)"
        
        if temperature is not None:
            if temperature < 0:
                return False, "Temperature must be non-negative"
            if temperature > 10:
                return False, "Temperature too high (max 10)"
        
        if top_p is not None:
            if top_p <= 0 or top_p > 1:
                return False, "top_p must be in (0, 1]"
        
        return True, None


def validate_prompt(prompt: str, **kwargs) -> Tuple[bool, Optional[str]]:
    """Convenience function for prompt validation."""
    return InputValidator.validate_prompt(prompt, **kwargs)


def validate_audio(audio: np.ndarray, **kwargs) -> Tuple[bool, Optional[str]]:
    """Convenience function for audio validation."""
    return InputValidator.validate_audio(audio, **kwargs)



