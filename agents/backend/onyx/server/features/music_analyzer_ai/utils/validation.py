"""
Modular Validation Utilities
Input/output validation and sanitization
"""

from typing import Any, Optional, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class TensorValidator:
    """Validate and sanitize tensors"""
    
    @staticmethod
    def check_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> bool:
        """Check for NaN/Inf values"""
        if not TORCH_AVAILABLE:
            return False
        
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        
        if has_nan:
            logger.warning(f"NaN detected in {name}")
        if has_inf:
            logger.warning(f"Inf detected in {name}")
        
        return has_nan or has_inf
    
    @staticmethod
    def fix_nan_inf(
        tensor: torch.Tensor,
        nan_value: float = 0.0,
        posinf_value: float = 1.0,
        neginf_value: float = -1.0
    ) -> torch.Tensor:
        """Fix NaN/Inf values"""
        if not TORCH_AVAILABLE:
            return tensor
        
        return torch.nan_to_num(
            tensor,
            nan=nan_value,
            posinf=posinf_value,
            neginf=neginf_value
        )
    
    @staticmethod
    def validate_shape(
        tensor: torch.Tensor,
        expected_shape: tuple,
        name: str = "tensor"
    ) -> bool:
        """Validate tensor shape"""
        if not TORCH_AVAILABLE:
            return False
        
        if tensor.shape != expected_shape:
            logger.warning(
                f"Shape mismatch in {name}: "
                f"expected {expected_shape}, got {tensor.shape}"
            )
            return False
        return True
    
    @staticmethod
    def validate_range(
        tensor: torch.Tensor,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        name: str = "tensor"
    ) -> bool:
        """Validate tensor value range"""
        if not TORCH_AVAILABLE:
            return False
        
        if min_val is not None and (tensor < min_val).any():
            logger.warning(f"Values below {min_val} in {name}")
            return False
        
        if max_val is not None and (tensor > max_val).any():
            logger.warning(f"Values above {max_val} in {name}")
            return False
        
        return True


class ArrayValidator:
    """Validate and sanitize numpy arrays"""
    
    @staticmethod
    def check_nan_inf(array: np.ndarray, name: str = "array") -> bool:
        """Check for NaN/Inf values"""
        has_nan = np.isnan(array).any()
        has_inf = np.isinf(array).any()
        
        if has_nan:
            logger.warning(f"NaN detected in {name}")
        if has_inf:
            logger.warning(f"Inf detected in {name}")
        
        return has_nan or has_inf
    
    @staticmethod
    def fix_nan_inf(
        array: np.ndarray,
        nan_value: float = 0.0,
        posinf_value: float = 1.0,
        neginf_value: float = -1.0
    ) -> np.ndarray:
        """Fix NaN/Inf values"""
        return np.nan_to_num(
            array,
            nan=nan_value,
            posinf=posinf_value,
            neginf=neginf_value
        )
    
    @staticmethod
    def validate_shape(
        array: np.ndarray,
        expected_shape: tuple,
        name: str = "array"
    ) -> bool:
        """Validate array shape"""
        if array.shape != expected_shape:
            logger.warning(
                f"Shape mismatch in {name}: "
                f"expected {expected_shape}, got {array.shape}"
            )
            return False
        return True


class InputValidator:
    """Validate model inputs"""
    
    @staticmethod
    def validate_features(
        features: Union[torch.Tensor, np.ndarray],
        expected_dim: int,
        name: str = "features"
    ) -> bool:
        """Validate feature input"""
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
        """Sanitize input data"""
        if isinstance(input_data, torch.Tensor):
            if TensorValidator.check_nan_inf(input_data, name):
                return TensorValidator.fix_nan_inf(input_data)
        elif isinstance(input_data, np.ndarray):
            if ArrayValidator.check_nan_inf(input_data, name):
                return ArrayValidator.fix_nan_inf(input_data)
        
        return input_data



