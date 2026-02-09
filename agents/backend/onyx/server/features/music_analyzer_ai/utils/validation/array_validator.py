"""
Array Validator Module

Validates and sanitizes NumPy arrays.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class ArrayValidator:
    """Validate and sanitize numpy arrays."""
    
    @staticmethod
    def check_nan_inf(array: np.ndarray, name: str = "array") -> bool:
        """
        Check for NaN/Inf values in array.
        
        Args:
            array: Input array.
            name: Name for logging.
        
        Returns:
            True if NaN/Inf detected, False otherwise.
        """
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
        """
        Fix NaN/Inf values in array.
        
        Args:
            array: Input array.
            nan_value: Value to replace NaN.
            posinf_value: Value to replace positive infinity.
            neginf_value: Value to replace negative infinity.
        
        Returns:
            Fixed array.
        """
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
        """
        Validate array shape.
        
        Args:
            array: Input array.
            expected_shape: Expected shape tuple.
            name: Name for logging.
        
        Returns:
            True if shape matches, False otherwise.
        """
        if array.shape != expected_shape:
            logger.warning(
                f"Shape mismatch in {name}: "
                f"expected {expected_shape}, got {array.shape}"
            )
            return False
        return True



