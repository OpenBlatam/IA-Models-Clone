"""
Input Validator Module

Validates model inputs (tensors and arrays).
"""

from typing import Union
import logging

from .tensor_validator import TensorValidator
from .array_validator import ArrayValidator

logger = logging.getLogger(__name__)

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class InputValidator:
    """Validate model inputs."""
    
    @staticmethod
    def validate_features(
        features: Union[torch.Tensor, np.ndarray],
        expected_dim: int,
        name: str = "features"
    ) -> bool:
        """
        Validate feature input.
        
        Args:
            features: Input features (tensor or array).
            expected_dim: Expected feature dimension.
            name: Name for logging.
        
        Returns:
            True if valid, False otherwise.
        """
        if isinstance(features, torch.Tensor):
            validator = TensorValidator
            if features.dim() == 1:
                features = features.unsqueeze(0)
        elif isinstance(features, np.ndarray):
            validator = ArrayValidator
            if features.ndim == 1:
                features = features[np.newaxis, :]
        else:
            logger.error(f"Invalid input type for {name}: {type(features)}")
            return False
        
        # Check for NaN/Inf
        if validator.check_nan_inf(features, name):
            return False
        
        # Check dimension
        if features.shape[-1] != expected_dim:
            logger.error(
                f"Feature dimension mismatch in {name}: "
                f"expected {expected_dim}, got {features.shape[-1]}"
            )
            return False
        
        return True
    
    @staticmethod
    def sanitize_input(
        input_data: Union[torch.Tensor, np.ndarray],
        name: str = "input"
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Sanitize input data.
        
        Args:
            input_data: Input data (tensor or array).
            name: Name for logging.
        
        Returns:
            Sanitized input data.
        """
        if isinstance(input_data, torch.Tensor):
            if TensorValidator.check_nan_inf(input_data, name):
                return TensorValidator.fix_nan_inf(input_data)
        elif isinstance(input_data, np.ndarray):
            if ArrayValidator.check_nan_inf(input_data, name):
                return ArrayValidator.fix_nan_inf(input_data)
        
        return input_data



