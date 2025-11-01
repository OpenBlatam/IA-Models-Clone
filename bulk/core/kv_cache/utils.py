"""
Utility functions for KV Cache.

Shared utilities following best practices.
"""
import logging
from typing import Dict, Any, Optional
import torch

logger = logging.getLogger(__name__)


def get_device_info(device: torch.device) -> Dict[str, Any]:
    """
    Get information about a device.
    
    Args:
        device: PyTorch device
        
    Returns:
        Dictionary with device information
    """
    info = {
        "type": device.type,
        "index": device.index if device.index is not None else 0,
    }
    
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            info["name"] = torch.cuda.get_device_name(device)
            info["capability"] = torch.cuda.get_device_capability(device)
            info["memory_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024**2)
            info["memory_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024**2)
            info["max_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2) if hasattr(torch.cuda, 'max_memory_allocated') else None
        except Exception as e:
            logger.warning(f"Error getting CUDA device info: {e}")
    
    return info


def validate_tensor_shapes(
    key: torch.Tensor,
    value: torch.Tensor,
    expected_dims: Optional[int] = None
) -> bool:
    """
    Validate that key and value tensors have compatible shapes.
    
    Args:
        key: Key tensor
        value: Value tensor
        expected_dims: Expected number of dimensions (optional)
        
    Returns:
        True if shapes are compatible, False otherwise
    """
    if key.shape != value.shape:
        logger.error(
            f"Shape mismatch: key={key.shape}, value={value.shape}"
        )
        return False
    
    if expected_dims is not None and len(key.shape) != expected_dims:
        logger.warning(
            f"Unexpected number of dimensions: got {len(key.shape)}, expected {expected_dims}"
        )
        return False
    
    return True


def format_memory_size(bytes_size: float) -> str:
    """
    Format memory size in human-readable format.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def safe_device_transfer(
    tensor: torch.Tensor,
    target_device: torch.device,
    non_blocking: bool = True
) -> torch.Tensor:
    """
    Safely transfer tensor to target device with error handling.
    
    Args:
        tensor: Tensor to transfer
        target_device: Target device
        non_blocking: Whether to use non-blocking transfer
        
    Returns:
        Tensor on target device
        
    Raises:
        RuntimeError: If transfer fails
    """
    try:
        if tensor.device == target_device:
            return tensor
        
        return tensor.to(target_device, non_blocking=non_blocking)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"OOM during device transfer to {target_device}")
            # Try to free cache
            if target_device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError(f"GPU out of memory during transfer: {e}") from e
        raise


def calculate_tensor_memory_mb(tensor: torch.Tensor) -> float:
    """
    Calculate memory usage of a tensor in MB.
    
    Args:
        tensor: Tensor to calculate memory for
        
    Returns:
        Memory size in MB
    """
    if tensor.numel() == 0:
        return 0.0
    
    # Get element size in bytes
    element_size = tensor.element_size()
    
    # Calculate total memory
    total_bytes = tensor.numel() * element_size
    
    return total_bytes / (1024**2)


def get_tensor_info(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Get information about a tensor.
    
    Args:
        tensor: Tensor to inspect
        
    Returns:
        Dictionary with tensor information
    """
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "memory_mb": calculate_tensor_memory_mb(tensor),
        "requires_grad": tensor.requires_grad,
    }

