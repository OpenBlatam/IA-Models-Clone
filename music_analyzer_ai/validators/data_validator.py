"""
Data Validators
"""

from typing import Any, Dict, List
import numpy as np

from .validator import BaseValidator, ValidationResult


class AudioDataValidator(BaseValidator):
    """
    Validator for audio data
    """
    
    def __init__(self, min_length: float = 0.0, max_length: float = None, sample_rate: int = None):
        super().__init__("AudioDataValidator")
        self.min_length = min_length
        self.max_length = max_length
        self.sample_rate = sample_rate
    
    def validate(self, data: Any) -> ValidationResult:
        """Validate audio data"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if isinstance(data, dict):
            audio = data.get("audio")
            sr = data.get("sample_rate", self.sample_rate)
        else:
            audio = data
            sr = self.sample_rate
        
        if audio is None:
            self._add_error(result, "Audio data is None")
            return result
        
        # Convert to numpy if needed
        if not isinstance(audio, np.ndarray):
            self._add_error(result, "Audio must be numpy array")
            return result
        
        # Check length
        if sr:
            duration = len(audio) / sr
            if self.min_length and duration < self.min_length:
                self._add_error(result, f"Audio too short: {duration:.2f}s < {self.min_length}s")
            
            if self.max_length and duration > self.max_length:
                self._add_warning(result, f"Audio too long: {duration:.2f}s > {self.max_length}s")
        
        # Check for silence
        if np.all(audio == 0):
            self._add_warning(result, "Audio appears to be silent")
        
        return result


class TrackIDValidator(BaseValidator):
    """
    Validator for track IDs
    """
    
    def validate(self, data: Any) -> ValidationResult:
        """Validate track ID"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if not isinstance(data, str):
            self._add_error(result, "Track ID must be a string")
            return result
        
        if not data or len(data.strip()) == 0:
            self._add_error(result, "Track ID cannot be empty")
        
        if len(data) > 100:
            self._add_warning(result, "Track ID is unusually long")
        
        return result








