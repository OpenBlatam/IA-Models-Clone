"""
Data Validation System
Validate input data before processing
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DataValidator:
    """
    Validate data before model inference
    """
    
    def __init__(self):
        self.validation_rules: Dict[str, List[Callable]] = {}
    
    def add_validation_rule(
        self,
        field_name: str,
        rule: Callable,
        error_message: str
    ):
        """Add validation rule for a field"""
        if field_name not in self.validation_rules:
            self.validation_rules[field_name] = []
        
        self.validation_rules[field_name].append({
            "rule": rule,
            "error_message": error_message
        })
    
    def validate(self, data: Any, data_type: str = "features") -> Tuple[bool, Optional[str]]:
        """Validate data"""
        if data_type == "features":
            return self._validate_features(data)
        elif data_type == "audio":
            return self._validate_audio(data)
        else:
            return True, None
    
    def _validate_features(self, features: np.ndarray) -> Tuple[bool, Optional[str]]:
        """Validate feature array"""
        # Check if numpy array
        if not isinstance(features, np.ndarray):
            return False, "Features must be numpy array"
        
        # Check shape
        if len(features.shape) != 1 and len(features.shape) != 2:
            return False, f"Invalid feature shape: {features.shape}"
        
        # Check for NaN
        if np.isnan(features).any():
            return False, "Features contain NaN values"
        
        # Check for Inf
        if np.isinf(features).any():
            return False, "Features contain Inf values"
        
        # Check value range (optional)
        if features.max() > 1e6 or features.min() < -1e6:
            logger.warning("Features have very large values")
        
        return True, None
    
    def _validate_audio(self, audio: Any) -> Tuple[bool, Optional[str]]:
        """Validate audio data"""
        # Check if audio file path or array
        if isinstance(audio, str):
            from pathlib import Path
            if not Path(audio).exists():
                return False, f"Audio file not found: {audio}"
        elif isinstance(audio, np.ndarray):
            if len(audio.shape) != 1:
                return False, "Audio must be 1D array"
            if len(audio) == 0:
                return False, "Audio array is empty"
        else:
            return False, "Audio must be file path or numpy array"
        
        return True, None


class ModelInputValidator:
    """
    Validate model inputs
    """
    
    @staticmethod
    def validate_tensor(
        tensor: Any,
        expected_shape: Optional[Tuple] = None,
        dtype: Optional[type] = None,
        device: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate tensor"""
        if not TORCH_AVAILABLE:
            return True, None  # Skip if PyTorch not available
        
        if not isinstance(tensor, torch.Tensor):
            return False, "Input must be torch.Tensor"
        
        if expected_shape and tensor.shape != expected_shape:
            return False, f"Shape mismatch: expected {expected_shape}, got {tensor.shape}"
        
        if dtype and tensor.dtype != dtype:
            return False, f"dtype mismatch: expected {dtype}, got {tensor.dtype}"
        
        if device and str(tensor.device) != device:
            return False, f"Device mismatch: expected {device}, got {tensor.device}"
        
        # Check for NaN/Inf
        if torch.isnan(tensor).any():
            return False, "Tensor contains NaN values"
        
        if torch.isinf(tensor).any():
            return False, "Tensor contains Inf values"
        
        return True, None
    
    @staticmethod
    def validate_batch(
        batch: Dict[str, Any],
        required_keys: List[str],
        expected_shapes: Optional[Dict[str, Tuple]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate batch"""
        # Check required keys
        for key in required_keys:
            if key not in batch:
                return False, f"Missing required key: {key}"
        
        # Check shapes
        if expected_shapes:
            for key, expected_shape in expected_shapes.items():
                if key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        if batch[key].shape != expected_shape:
                            return False, f"Shape mismatch for {key}: expected {expected_shape}, got {batch[key].shape}"
        
        return True, None

