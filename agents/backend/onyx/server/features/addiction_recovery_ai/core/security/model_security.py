"""
Model Security
Security utilities for model protection
"""

import torch
import torch.nn as nn
import hashlib
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelSecurity:
    """
    Security utilities for models
    """
    
    @staticmethod
    def compute_model_hash(model: nn.Module) -> str:
        """
        Compute hash of model weights
        
        Args:
            model: PyTorch model
            
        Returns:
            SHA256 hash string
        """
        # Get all parameters as bytes
        params_bytes = b""
        for param in model.parameters():
            params_bytes += param.data.cpu().numpy().tobytes()
        
        # Compute hash
        hash_obj = hashlib.sha256(params_bytes)
        return hash_obj.hexdigest()
    
    @staticmethod
    def verify_model_integrity(
        model: nn.Module,
        expected_hash: str
    ) -> bool:
        """
        Verify model integrity
        
        Args:
            model: Model to verify
            expected_hash: Expected hash
            
        Returns:
            True if integrity verified
        """
        actual_hash = ModelSecurity.compute_model_hash(model)
        return actual_hash == expected_hash
    
    @staticmethod
    def sanitize_input(
        tensor: torch.Tensor,
        max_value: Optional[float] = None,
        min_value: Optional[float] = None
    ) -> torch.Tensor:
        """
        Sanitize input tensor
        
        Args:
            tensor: Input tensor
            max_value: Maximum allowed value
            min_value: Minimum allowed value
            
        Returns:
            Sanitized tensor
        """
        # Clip values
        if max_value is not None or min_value is not None:
            tensor = torch.clamp(tensor, min=min_value, max=max_value)
        
        # Replace NaN and Inf
        tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
        
        return tensor


def compute_model_hash(model: nn.Module) -> str:
    """Compute model hash"""
    return ModelSecurity.compute_model_hash(model)


def verify_model_integrity(model: nn.Module, expected_hash: str) -> bool:
    """Verify model integrity"""
    return ModelSecurity.verify_model_integrity(model, expected_hash)


def sanitize_input(tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """Sanitize input"""
    return ModelSecurity.sanitize_input(tensor, **kwargs)








