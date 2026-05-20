"""
API Utilities
============

Utility functions for the TruthGPT API.
"""

import logging
from datetime import datetime
import torch
import numpy as np
from typing import Any, Dict, List, Union, Optional


def serialize_tensor(tensor: Any) -> Union[List, float, int]:
    """
    Serialize PyTorch tensor or numpy array to Python native types.
    
    Args:
        tensor: PyTorch tensor or numpy array
        
    Returns:
        Serialized data as list, float, or int
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
        if tensor.numel() == 1:
            return float(tensor.item())
        return tensor.numpy().tolist()
    elif isinstance(tensor, np.ndarray):
        if tensor.size == 1:
            return float(tensor.item())
        return tensor.tolist()
    elif isinstance(tensor, (list, tuple)):
        return [serialize_tensor(item) for item in tensor]
    elif isinstance(tensor, dict):
        return {key: serialize_tensor(value) for key, value in tensor.items()}
    else:
        return tensor


def serialize_history(history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize training history to JSON-serializable format.
    
    Args:
        history: Training history dictionary
        
    Returns:
        Serialized history dictionary
    """
    serialized = {}
    for key, value in history.items():
        if isinstance(value, (list, tuple)):
            serialized[key] = [
                float(v) if isinstance(v, (int, float, torch.Tensor, np.number)) 
                else serialize_tensor(v)
                for v in value
            ]
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            serialized[key] = serialize_tensor(value)
        elif isinstance(value, (int, float, np.number)):
            serialized[key] = float(value)
        else:
            serialized[key] = value
    return serialized


def validate_array_shape(data: List[List[float]], expected_dim: Optional[int] = None) -> np.ndarray:
    """
    Validate and convert array data to numpy array.
    
    Args:
        data: Input data as list of lists
        expected_dim: Expected number of dimensions (optional)
        
    Returns:
        Validated numpy array
        
    Raises:
        ValueError: If data is invalid
    """
    if not data:
        raise ValueError("Data cannot be empty")
    
    try:
        array = np.array(data, dtype=np.float32)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid data format: {str(e)}")
    
    if expected_dim is not None and array.ndim != expected_dim:
        raise ValueError(
            f"Expected {expected_dim}D array, got {array.ndim}D array"
        )
    
    if np.any(np.isnan(array)) or np.any(np.isinf(array)):
        raise ValueError("Data contains NaN or Inf values")
    
    return array


def get_model_summary(model: Any) -> Dict[str, Any]:
    """
    Get summary information about a model.
    
    Args:
        model: Model instance
        
    Returns:
        Dictionary with model summary information
    """
    summary = {
        "type": type(model).__name__,
        "name": getattr(model, 'name', None),
    }
    
    if hasattr(model, 'layers_list'):
        summary["num_layers"] = len(model.layers_list)
        summary["layers"] = [
            {
                "type": type(layer).__name__,
                "name": getattr(layer, 'name', None)
            }
            for layer in model.layers_list
        ]
    
    if hasattr(model, '_compiled'):
        summary["compiled"] = model._compiled
    
    try:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        summary["trainable_parameters"] = total_params
    except (AttributeError, TypeError):
        pass
    
    return summary


def handle_model_operation_error(
    operation: str,
    model_id: Optional[str] = None,
    error: Optional[Exception] = None
) -> None:
    """
    Helper function to log and handle model operation errors consistently.
    
    Args:
        operation: Name of the operation that failed
        model_id: Optional model ID
        error: Optional exception that occurred
    """
    logger = logging.getLogger(__name__)
    
    model_context = f" for model {model_id}" if model_id else ""
    error_msg = f"Error {operation}{model_context}"
    if error:
        error_msg += f": {str(error)}"
    
    logger.error(error_msg, exc_info=error is not None)


def extract_layer_params(layer: Any) -> Dict[str, Any]:
    """
    Extract parameters from a layer object.
    
    Args:
        layer: Layer instance
        
    Returns:
        Dictionary of layer parameters
    """
    params = {}
    param_mapping = {
        'units': 'units',
        'out_features': 'units',
        'filters': 'filters',
        'kernel_size': 'kernel_size',
        'rate': 'rate',
        'activation': 'activation'
    }
    
    for attr, param_key in param_mapping.items():
        if hasattr(layer, attr):
            value = getattr(layer, attr)
            if attr == 'activation':
                params[param_key] = str(value) if value else None
            else:
                params[param_key] = value
    
    return params


def increment_counter(counter_dict: Dict[str, int], key: str, increment: int = 1) -> None:
    """
    Increment a counter in a dictionary.
    
    Args:
        counter_dict: Dictionary to update
        key: Key to increment
        increment: Amount to increment (default: 1)
    """
    counter_dict[key] = counter_dict.get(key, 0) + increment


def parse_iso_date(date_str: str) -> Optional[datetime]:
    """
    Parse ISO format date string, handling timezone variations.
    
    Args:
        date_str: ISO format date string
        
    Returns:
        Parsed datetime object or None if parsing fails
    """
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except (ValueError, AttributeError, TypeError):
        return None

