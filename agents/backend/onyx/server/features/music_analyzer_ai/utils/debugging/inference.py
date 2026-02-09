"""
Inference Debugging Module

Debugging utilities for inference.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class InferenceDebugger:
    """
    Debugging tools for inference.
    """
    
    @staticmethod
    def validate_input(input_data: torch.Tensor, expected_shape: tuple):
        """
        Validate input shape and values.
        
        Args:
            input_data: Input tensor to validate
            expected_shape: Expected shape
        
        Raises:
            ValueError: If input is invalid
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        if input_data.shape != expected_shape:
            raise ValueError(
                f"Input shape mismatch: expected {expected_shape}, got {input_data.shape}"
            )
        
        if torch.isnan(input_data).any():
            raise ValueError("Input contains NaN values")
        
        if torch.isinf(input_data).any():
            raise ValueError("Input contains Inf values")
    
    @staticmethod
    def validate_output(output: torch.Tensor, expected_shape: Optional[tuple] = None) -> bool:
        """
        Validate output shape and values.
        
        Args:
            output: Output tensor to validate
            expected_shape: Expected shape (optional)
        
        Returns:
            True if output is valid, False otherwise
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        if expected_shape and output.shape != expected_shape:
            logger.warning(
                f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
            )
        
        if torch.isnan(output).any():
            logger.error("Output contains NaN values")
            return False
        
        if torch.isinf(output).any():
            logger.error("Output contains Inf values")
            return False
        
        return True



