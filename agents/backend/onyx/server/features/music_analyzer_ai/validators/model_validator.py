"""
Model Input Validators
"""

import torch
import numpy as np
from typing import Any, Dict

from .validator import BaseValidator, ValidationResult


class ModelInputValidator(BaseValidator):
    """
    Validator for model inputs
    """
    
    def __init__(self, expected_shape: tuple = None, expected_dtype: torch.dtype = None):
        super().__init__("ModelInputValidator")
        self.expected_shape = expected_shape
        self.expected_dtype = expected_dtype
    
    def validate(self, data: Any) -> ValidationResult:
        """Validate model input"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Check if tensor
        if not isinstance(data, torch.Tensor):
            self._add_error(result, "Input must be a torch.Tensor")
            return result
        
        # Check shape
        if self.expected_shape and data.shape != self.expected_shape:
            self._add_error(result, f"Shape mismatch: expected {self.expected_shape}, got {data.shape}")
        
        # Check dtype
        if self.expected_dtype and data.dtype != self.expected_dtype:
            self._add_warning(result, f"Dtype mismatch: expected {self.expected_dtype}, got {data.dtype}")
        
        # Check for NaN/Inf
        if torch.isnan(data).any():
            self._add_error(result, "Input contains NaN values")
        
        if torch.isinf(data).any():
            self._add_error(result, "Input contains Inf values")
        
        return result


class FeatureValidator(BaseValidator):
    """
    Validator for audio features
    """
    
    def __init__(self, expected_dim: int = None, min_value: float = None, max_value: float = None):
        super().__init__("FeatureValidator")
        self.expected_dim = expected_dim
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, data: Any) -> ValidationResult:
        """Validate features"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Convert to numpy if needed
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        elif not isinstance(data, np.ndarray):
            self._add_error(result, "Features must be numpy array or torch tensor")
            return result
        
        # Check dimension
        if self.expected_dim and data.shape[-1] != self.expected_dim:
            self._add_error(result, f"Feature dimension mismatch: expected {self.expected_dim}, got {data.shape[-1]}")
        
        # Check value range
        if self.min_value is not None and data.min() < self.min_value:
            self._add_warning(result, f"Values below minimum: {data.min()} < {self.min_value}")
        
        if self.max_value is not None and data.max() > self.max_value:
            self._add_warning(result, f"Values above maximum: {data.max()} > {self.max_value}")
        
        return result








