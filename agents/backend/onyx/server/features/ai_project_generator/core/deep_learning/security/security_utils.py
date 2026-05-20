"""
Security Utilities
==================

Security and safety utilities for models.
"""

import logging
from typing import Optional, Dict, Any, List, Union
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


def validate_inputs(
    inputs: Union[torch.Tensor, np.ndarray, dict, list],
    expected_shape: Optional[tuple] = None,
    expected_dtype: Optional[torch.dtype] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    check_nan: bool = True,
    check_inf: bool = True
) -> tuple[bool, Optional[str]]:
    """
    Validate model inputs.
    
    Args:
        inputs: Input data
        expected_shape: Expected shape
        expected_dtype: Expected dtype
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        check_nan: Check for NaN values
        check_inf: Check for Inf values
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Convert to tensor if needed
    if isinstance(inputs, (dict, list)):
        # Handle complex inputs
        if isinstance(inputs, dict):
            tensor_inputs = [v for v in inputs.values() if isinstance(v, torch.Tensor)]
            if not tensor_inputs:
                return False, "No tensor inputs found in dictionary"
            inputs = tensor_inputs[0]  # Validate first tensor
        else:
            tensor_inputs = [v for v in inputs if isinstance(v, torch.Tensor)]
            if not tensor_inputs:
                return False, "No tensor inputs found in list"
            inputs = tensor_inputs[0]
    
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs)
    
    if not isinstance(inputs, torch.Tensor):
        return False, f"Input is not a tensor, got {type(inputs)}"
    
    # Check shape
    if expected_shape is not None:
        if inputs.shape != expected_shape:
            return False, f"Shape mismatch: {inputs.shape} != {expected_shape}"
    
    # Check dtype
    if expected_dtype is not None:
        if inputs.dtype != expected_dtype:
            return False, f"Dtype mismatch: {inputs.dtype} != {expected_dtype}"
    
    # Check value range
    if min_value is not None:
        if inputs.min().item() < min_value:
            return False, f"Value below minimum: {inputs.min().item()} < {min_value}"
    
    if max_value is not None:
        if inputs.max().item() > max_value:
            return False, f"Value above maximum: {inputs.max().item()} > {max_value}"
    
    # Check for NaN
    if check_nan:
        if torch.isnan(inputs).any():
            return False, "NaN values detected in inputs"
    
    # Check for Inf
    if check_inf:
        if torch.isinf(inputs).any():
            return False, "Inf values detected in inputs"
    
    return True, None


def sanitize_inputs(
    inputs: Union[torch.Tensor, np.ndarray],
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    replace_nan: Optional[float] = 0.0,
    replace_inf: Optional[float] = 0.0
) -> torch.Tensor:
    """
    Sanitize inputs (clip, replace NaN/Inf).
    
    Args:
        inputs: Input tensor
        clip_min: Minimum clipping value
        clip_max: Maximum clipping value
        replace_nan: Value to replace NaN with
        replace_inf: Value to replace Inf with
        
    Returns:
        Sanitized tensor
    """
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs)
    
    # Replace NaN
    if replace_nan is not None:
        inputs = torch.where(torch.isnan(inputs), torch.tensor(replace_nan), inputs)
    
    # Replace Inf
    if replace_inf is not None:
        inputs = torch.where(torch.isinf(inputs), torch.tensor(replace_inf), inputs)
    
    # Clip values
    if clip_min is not None or clip_max is not None:
        inputs = torch.clamp(inputs, min=clip_min, max=clip_max)
    
    return inputs


def detect_adversarial(
    inputs: torch.Tensor,
    model: nn.Module,
    threshold: float = 0.5,
    method: str = 'gradient'
) -> tuple[bool, float]:
    """
    Detect adversarial inputs.
    
    Args:
        inputs: Input tensor
        model: PyTorch model
        threshold: Detection threshold
        method: Detection method ('gradient', 'confidence')
        
    Returns:
        Tuple of (is_adversarial, score)
    """
    model.eval()
    
    if method == 'gradient':
        # Gradient-based detection
        inputs.requires_grad = True
        outputs = model(inputs)
        
        # Compute gradient
        loss = outputs.sum()
        loss.backward()
        
        gradient_norm = inputs.grad.norm().item()
        is_adversarial = gradient_norm > threshold
        
        return is_adversarial, gradient_norm
    
    elif method == 'confidence':
        # Confidence-based detection
        with torch.no_grad():
            outputs = model(inputs)
            if outputs.dim() > 1:
                probs = torch.softmax(outputs, dim=-1)
                max_prob = probs.max().item()
            else:
                max_prob = torch.sigmoid(outputs).item()
        
        is_adversarial = max_prob < threshold
        return is_adversarial, 1.0 - max_prob
    
    else:
        raise ValueError(f"Unknown detection method: {method}")


class ModelEncryption:
    """
    Simple model encryption/obfuscation.
    """
    
    @staticmethod
    def encrypt_state_dict(state_dict: Dict[str, torch.Tensor], key: int = 42) -> Dict[str, torch.Tensor]:
        """
        Encrypt model state dict (simple XOR encryption).
        
        Args:
            state_dict: Model state dict
            key: Encryption key
            
        Returns:
            Encrypted state dict
        """
        encrypted = {}
        for name, tensor in state_dict.items():
            # Simple XOR encryption (not secure, just obfuscation)
            encrypted_tensor = tensor.clone()
            encrypted_tensor = encrypted_tensor ^ (key % 256)
            encrypted[name] = encrypted_tensor
        
        return encrypted
    
    @staticmethod
    def decrypt_state_dict(encrypted_dict: Dict[str, torch.Tensor], key: int = 42) -> Dict[str, torch.Tensor]:
        """
        Decrypt model state dict.
        
        Args:
            encrypted_dict: Encrypted state dict
            key: Decryption key
            
        Returns:
            Decrypted state dict
        """
        return ModelEncryption.encrypt_state_dict(encrypted_dict, key)  # XOR is symmetric



