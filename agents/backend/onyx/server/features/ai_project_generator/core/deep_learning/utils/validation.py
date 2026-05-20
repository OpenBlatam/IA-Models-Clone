"""
Validation Utilities - Input Validation and Sanity Checks
==========================================================

Provides validation utilities for:
- Model inputs/outputs
- Data integrity
- Configuration validation
- Gradient checks
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


def validate_model_inputs(
    model: nn.Module,
    sample_input: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate model inputs before training/inference.
    
    Args:
        model: PyTorch model
        sample_input: Sample input dictionary
        device: Device to run on
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        model.eval()
        with torch.no_grad():
            output = model(**sample_input)
        
        # Check for NaN/Inf
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                return False, "Model output contains NaN"
            if torch.isinf(output).any():
                return False, "Model output contains Inf"
        elif isinstance(output, dict):
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any():
                        return False, f"Model output '{key}' contains NaN"
                    if torch.isinf(value).any():
                        return False, f"Model output '{key}' contains Inf"
        
        return True, None
        
    except Exception as e:
        return False, f"Model validation failed: {str(e)}"


def check_gradients(model: nn.Module, check_nan: bool = True, check_inf: bool = True) -> Dict[str, Any]:
    """
    Check model gradients for issues.
    
    Args:
        model: PyTorch model
        check_nan: Check for NaN gradients
        check_inf: Check for Inf gradients
        
    Returns:
        Dictionary with gradient statistics
    """
    stats = {
        'total_params': 0,
        'params_with_grad': 0,
        'params_with_nan_grad': 0,
        'params_with_inf_grad': 0,
        'max_grad_norm': 0.0,
        'mean_grad_norm': 0.0
    }
    
    grad_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            stats['params_with_grad'] += 1
            
            if check_nan and torch.isnan(param.grad).any():
                stats['params_with_nan_grad'] += 1
                logger.warning(f"NaN gradient detected in {name}")
            
            if check_inf and torch.isinf(param.grad).any():
                stats['params_with_inf_grad'] += 1
                logger.warning(f"Inf gradient detected in {name}")
            
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
        
        stats['total_params'] += 1
    
    if grad_norms:
        stats['max_grad_norm'] = max(grad_norms)
        stats['mean_grad_norm'] = np.mean(grad_norms)
    
    return stats


def validate_data_loader(
    dataloader: torch.utils.data.DataLoader,
    num_samples: int = 5
) -> Tuple[bool, Optional[str]]:
    """
    Validate data loader for common issues.
    
    Args:
        dataloader: DataLoader to validate
        num_samples: Number of samples to check
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            # Check batch structure
            if isinstance(batch, dict):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        if torch.isnan(value).any():
                            return False, f"NaN values in batch[{i}]['{key}']"
                        if torch.isinf(value).any():
                            return False, f"Inf values in batch[{i}]['{key}']"
            elif isinstance(batch, (tuple, list)):
                for j, item in enumerate(batch):
                    if isinstance(item, torch.Tensor):
                        if torch.isnan(item).any():
                            return False, f"NaN values in batch[{i}][{j}]"
                        if torch.isinf(item).any():
                            return False, f"Inf values in batch[{i}][{j}]"
            else:
                if isinstance(batch, torch.Tensor):
                    if torch.isnan(batch).any():
                        return False, f"NaN values in batch[{i}]"
                    if torch.isinf(batch).any():
                        return False, f"Inf values in batch[{i}]"
        
        return True, None
        
    except Exception as e:
        return False, f"DataLoader validation failed: {str(e)}"


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys (supports dot notation)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_keys = []
    
    for key in required_keys:
        keys = key.split('.')
        value = config
        
        for k in keys:
            if not isinstance(value, dict) or k not in value:
                missing_keys.append(key)
                break
            value = value[k]
    
    if missing_keys:
        return False, f"Missing required config keys: {', '.join(missing_keys)}"
    
    return True, None



