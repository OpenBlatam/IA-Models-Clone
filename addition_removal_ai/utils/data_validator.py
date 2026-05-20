"""
Data Validation Utilities
"""

import torch
import numpy as np
from typing import Any, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Data validator for training and inference"""
    
    @staticmethod
    def validate_tensor(
        tensor: torch.Tensor,
        shape: Optional[tuple] = None,
        dtype: Optional[torch.dtype] = None,
        range_check: Optional[tuple] = None,
        name: str = "tensor"
    ) -> bool:
        """
        Validate tensor
        
        Args:
            tensor: Tensor to validate
            shape: Expected shape
            dtype: Expected dtype
            range_check: (min, max) range
            name: Tensor name for logging
            
        Returns:
            True if valid
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        
        if shape and tensor.shape != shape:
            raise ValueError(f"{name} shape mismatch: expected {shape}, got {tensor.shape}")
        
        if dtype and tensor.dtype != dtype:
            raise ValueError(f"{name} dtype mismatch: expected {dtype}, got {tensor.dtype}")
        
        if range_check:
            min_val, max_val = range_check
            if tensor.min() < min_val or tensor.max() > max_val:
                raise ValueError(f"{name} values out of range [{min_val}, {max_val}]")
        
        # Check for NaN/Inf
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} contains NaN values")
        
        if torch.isinf(tensor).any():
            raise ValueError(f"{name} contains Inf values")
        
        return True
    
    @staticmethod
    def validate_batch(
        batch: Any,
        expected_keys: Optional[List[str]] = None,
        min_batch_size: int = 1
    ) -> bool:
        """
        Validate data batch
        
        Args:
            batch: Batch to validate
            expected_keys: Expected dictionary keys
            min_batch_size: Minimum batch size
            
        Returns:
            True if valid
        """
        if isinstance(batch, dict):
            if expected_keys:
                for key in expected_keys:
                    if key not in batch:
                        raise KeyError(f"Missing key in batch: {key}")
            
            # Check batch size consistency
            sizes = [len(v) if isinstance(v, (list, torch.Tensor, np.ndarray)) else 1 
                    for v in batch.values()]
            if len(set(sizes)) > 1:
                raise ValueError("Inconsistent batch sizes in batch dictionary")
        
        elif isinstance(batch, (list, tuple)):
            if len(batch) < min_batch_size:
                raise ValueError(f"Batch size {len(batch)} < minimum {min_batch_size}")
        
        return True
    
    @staticmethod
    def sanitize_input(input_data: Any) -> Any:
        """
        Sanitize input data
        
        Args:
            input_data: Input to sanitize
            
        Returns:
            Sanitized input
        """
        if isinstance(input_data, np.ndarray):
            # Check for NaN/Inf
            if np.isnan(input_data).any() or np.isinf(input_data).any():
                logger.warning("NaN/Inf detected in input, replacing with zeros")
                input_data = np.nan_to_num(input_data)
        
        elif isinstance(input_data, torch.Tensor):
            # Check for NaN/Inf
            if torch.isnan(input_data).any() or torch.isinf(input_data).any():
                logger.warning("NaN/Inf detected in input, replacing with zeros")
                input_data = torch.nan_to_num(input_data)
        
        return input_data


def create_validator() -> DataValidator:
    """Factory function for data validator"""
    return DataValidator()

