"""
Validation Utilities
Input validation and data quality checks
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class InputValidator:
    """
    Validate model inputs
    """
    
    @staticmethod
    def validate_tensor(
        tensor: torch.Tensor,
        expected_shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        check_nan: bool = True,
        check_inf: bool = True,
        check_finite: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate tensor
        
        Args:
            tensor: Input tensor
            expected_shape: Expected shape
            dtype: Expected dtype
            device: Expected device
            check_nan: Check for NaN
            check_inf: Check for Inf
            check_finite: Check for finite values
            
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(tensor, torch.Tensor):
            return False, "Input is not a tensor"
        
        if expected_shape and tensor.shape != expected_shape:
            return False, f"Shape mismatch: expected {expected_shape}, got {tensor.shape}"
        
        if dtype and tensor.dtype != dtype:
            return False, f"Dtype mismatch: expected {dtype}, got {tensor.dtype}"
        
        if device and tensor.device != device:
            return False, f"Device mismatch: expected {device}, got {tensor.device}"
        
        if check_nan and torch.isnan(tensor).any():
            return False, "Tensor contains NaN values"
        
        if check_inf and torch.isinf(tensor).any():
            return False, "Tensor contains Inf values"
        
        if check_finite and not torch.isfinite(tensor).all():
            return False, "Tensor contains non-finite values"
        
        return True, None
    
    @staticmethod
    def validate_features(
        features: List[float],
        expected_length: Optional[int] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate feature vector
        
        Args:
            features: Feature vector
            expected_length: Expected length
            min_value: Minimum value
            max_value: Maximum value
            
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(features, list):
            return False, "Features must be a list"
        
        if expected_length and len(features) != expected_length:
            return False, f"Length mismatch: expected {expected_length}, got {len(features)}"
        
        for i, val in enumerate(features):
            if not isinstance(val, (int, float)):
                return False, f"Feature {i} is not numeric: {val}"
            
            if np.isnan(val) or np.isinf(val):
                return False, f"Feature {i} is NaN or Inf"
            
            if min_value is not None and val < min_value:
                return False, f"Feature {i} < min_value: {val} < {min_value}"
            
            if max_value is not None and val > max_value:
                return False, f"Feature {i} > max_value: {val} > {max_value}"
        
        return True, None
    
    @staticmethod
    def validate_text(
        text: str,
        min_length: int = 1,
        max_length: int = 10000,
        allowed_chars: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate text input
        
        Args:
            text: Input text
            min_length: Minimum length
            max_length: Maximum length
            allowed_chars: Allowed characters (optional)
            
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(text, str):
            return False, "Text must be a string"
        
        if len(text) < min_length:
            return False, f"Text too short: {len(text)} < {min_length}"
        
        if len(text) > max_length:
            return False, f"Text too long: {len(text)} > {max_length}"
        
        if allowed_chars:
            invalid_chars = [c for c in text if c not in allowed_chars]
            if invalid_chars:
                return False, f"Invalid characters found: {invalid_chars[:10]}"
        
        return True, None


class ModelValidator:
    """
    Validate model state and outputs
    """
    
    @staticmethod
    def validate_model_state(model: torch.nn.Module) -> Dict[str, Any]:
        """
        Validate model state
        
        Args:
            model: PyTorch model
            
        Returns:
            Validation results
        """
        results = {
            "is_valid": True,
            "issues": [],
            "parameter_count": 0,
            "has_nan": False,
            "has_inf": False
        }
        
        # Check parameters
        for name, param in model.named_parameters():
            results["parameter_count"] += param.numel()
            
            if torch.isnan(param).any():
                results["has_nan"] = True
                results["issues"].append(f"NaN in parameter: {name}")
                results["is_valid"] = False
            
            if torch.isinf(param).any():
                results["has_inf"] = True
                results["issues"].append(f"Inf in parameter: {name}")
                results["is_valid"] = False
        
        return results
    
    @staticmethod
    def validate_output(
        output: torch.Tensor,
        expected_shape: Optional[Tuple[int, ...]] = None,
        check_nan: bool = True,
        check_inf: bool = True,
        check_range: Optional[Tuple[float, float]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate model output
        
        Args:
            output: Model output
            expected_shape: Expected shape
            check_nan: Check for NaN
            check_inf: Check for Inf
            check_range: Expected value range (min, max)
            
        Returns:
            (is_valid, error_message)
        """
        if expected_shape and output.shape != expected_shape:
            return False, f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
        
        if check_nan and torch.isnan(output).any():
            return False, "Output contains NaN values"
        
        if check_inf and torch.isinf(output).any():
            return False, "Output contains Inf values"
        
        if check_range:
            min_val, max_val = check_range
            if output.min() < min_val or output.max() > max_val:
                return False, f"Output out of range: [{output.min():.4f}, {output.max():.4f}] not in [{min_val}, {max_val}]"
        
        return True, None


def validate_input(tensor: torch.Tensor, **kwargs) -> Tuple[bool, Optional[str]]:
    """Validate input tensor"""
    return InputValidator.validate_tensor(tensor, **kwargs)


def validate_features(features: List[float], **kwargs) -> Tuple[bool, Optional[str]]:
    """Validate feature vector"""
    return InputValidator.validate_features(features, **kwargs)


def validate_text(text: str, **kwargs) -> Tuple[bool, Optional[str]]:
    """Validate text input"""
    return InputValidator.validate_text(text, **kwargs)













