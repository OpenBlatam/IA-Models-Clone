"""
Validators
Input validation utilities.
"""

from typing import Any, Callable, List, Optional
import torch
import numpy as np


class Validator:
    """Base validator class."""
    
    @staticmethod
    def validate(value: Any) -> bool:
        """Validate a value."""
        raise NotImplementedError


class TypeValidator(Validator):
    """Validate value type."""
    
    def __init__(self, expected_type: type):
        self.expected_type = expected_type
    
    def validate(self, value: Any) -> bool:
        return isinstance(value, self.expected_type)


class RangeValidator(Validator):
    """Validate numeric range."""
    
    def __init__(self, min_value: Optional[float] = None, max_value: Optional[float] = None):
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, value: Any) -> bool:
        if not isinstance(value, (int, float)):
            return False
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


class TensorShapeValidator(Validator):
    """Validate tensor shape."""
    
    def __init__(self, expected_shape: tuple):
        self.expected_shape = expected_shape
    
    def validate(self, value: Any) -> bool:
        if not isinstance(value, torch.Tensor):
            return False
        return value.shape == self.expected_shape


class TensorDtypeValidator(Validator):
    """Validate tensor dtype."""
    
    def __init__(self, expected_dtype: torch.dtype):
        self.expected_dtype = expected_dtype
    
    def validate(self, value: Any) -> bool:
        if not isinstance(value, torch.Tensor):
            return False
        return value.dtype == self.expected_dtype


class NotNoneValidator(Validator):
    """Validate value is not None."""
    
    @staticmethod
    def validate(value: Any) -> bool:
        return value is not None


class NotEmptyValidator(Validator):
    """Validate value is not empty."""
    
    @staticmethod
    def validate(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (str, list, tuple)):
            return len(value) > 0
        if isinstance(value, torch.Tensor):
            return value.numel() > 0
        return True


class InListValidator(Validator):
    """Validate value is in list."""
    
    def __init__(self, allowed_values: List[Any]):
        self.allowed_values = allowed_values
    
    def validate(self, value: Any) -> bool:
        return value in self.allowed_values


class CompositeValidator(Validator):
    """Combine multiple validators."""
    
    def __init__(self, validators: List[Validator], require_all: bool = True):
        self.validators = validators
        self.require_all = require_all
    
    def validate(self, value: Any) -> bool:
        if self.require_all:
            return all(v.validate(value) for v in self.validators)
        else:
            return any(v.validate(value) for v in self.validators)


def validate_model_input(
    input_tensor: torch.Tensor,
    expected_shape: Optional[tuple] = None,
    expected_dtype: Optional[torch.dtype] = None,
    check_finite: bool = True,
) -> bool:
    """
    Validate model input tensor.
    
    Args:
        input_tensor: Input tensor to validate
        expected_shape: Expected shape (optional)
        expected_dtype: Expected dtype (optional)
        check_finite: Check for NaN/Inf values
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise ValueError(f"Input must be a torch.Tensor, got {type(input_tensor)}")
    
    if expected_shape and input_tensor.shape != expected_shape:
        raise ValueError(
            f"Expected shape {expected_shape}, got {input_tensor.shape}"
        )
    
    if expected_dtype and input_tensor.dtype != expected_dtype:
        raise ValueError(
            f"Expected dtype {expected_dtype}, got {input_tensor.dtype}"
        )
    
    if check_finite:
        if torch.isnan(input_tensor).any():
            raise ValueError("Input contains NaN values")
        if torch.isinf(input_tensor).any():
            raise ValueError("Input contains Inf values")
    
    return True


def validate_generation_params(
    max_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> bool:
    """Validate text generation parameters."""
    if max_length < 1 or max_length > 2048:
        raise ValueError(f"max_length must be between 1 and 2048, got {max_length}")
    
    if temperature < 0.0 or temperature > 2.0:
        raise ValueError(f"temperature must be between 0.0 and 2.0, got {temperature}")
    
    if top_p < 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be between 0.0 and 1.0, got {top_p}")
    
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    
    return True



