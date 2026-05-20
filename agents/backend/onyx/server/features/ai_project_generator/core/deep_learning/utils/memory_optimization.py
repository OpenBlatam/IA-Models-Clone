"""
Memory Optimization Utilities
==============================

Utilities for optimizing memory usage in deep learning:
- Gradient checkpointing
- Memory-efficient attention
- Mixed precision memory management
- Cache clearing
"""

import logging
from typing import Optional
import torch
import torch.nn as nn
import gc

logger = logging.getLogger(__name__)


def clear_cache() -> None:
    """Clear PyTorch and Python caches."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.debug("Cache cleared")


def enable_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    Enable gradient checkpointing for memory efficiency.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model with gradient checkpointing enabled
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    else:
        logger.warning("Model does not support gradient checkpointing")
    
    return model


def optimize_model_for_inference(model: nn.Module) -> nn.Module:
    """
    Optimize model for inference (fuse operations, etc.).
    
    Args:
        model: PyTorch model
        
    Returns:
        Optimized model
    """
    model.eval()
    
    # Try to fuse operations if available
    try:
        if hasattr(torch.jit, 'optimize_for_inference'):
            model = torch.jit.optimize_for_inference(torch.jit.script(model))
            logger.info("Model optimized with TorchScript")
    except Exception as e:
        logger.debug(f"TorchScript optimization not available: {e}")
    
    # Set to eval mode
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def get_memory_stats(device: Optional[torch.device] = None) -> dict:
    """
    Get memory statistics.
    
    Args:
        device: Device to check (defaults to current CUDA device)
        
    Returns:
        Dictionary with memory statistics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    stats = {}
    
    if device.type == 'cuda':
        stats['allocated_mb'] = torch.cuda.memory_allocated(device) / 1024**2
        stats['reserved_mb'] = torch.cuda.memory_reserved(device) / 1024**2
        stats['max_allocated_mb'] = torch.cuda.max_memory_allocated(device) / 1024**2
        stats['max_reserved_mb'] = torch.cuda.max_memory_reserved(device) / 1024**2
        
        if torch.cuda.device_count() > 0:
            stats['total_memory_mb'] = torch.cuda.get_device_properties(device).total_memory / 1024**2
            stats['free_memory_mb'] = stats['total_memory_mb'] - stats['reserved_mb']
    else:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        stats['rss_mb'] = mem_info.rss / 1024**2
        stats['vms_mb'] = mem_info.vms / 1024**2
    
    return stats


def print_memory_stats(device: Optional[torch.device] = None) -> None:
    """Print memory statistics."""
    stats = get_memory_stats(device)
    logger.info(f"Memory stats: {stats}")


class MemoryMonitor:
    """Context manager for monitoring memory usage."""
    
    def __init__(self, device: Optional[torch.device] = None, log_interval: int = 10):
        """
        Initialize memory monitor.
        
        Args:
            device: Device to monitor
            log_interval: Log every N operations
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_interval = log_interval
        self.initial_stats = None
        self.operation_count = 0
    
    def __enter__(self):
        """Enter context."""
        self.initial_stats = get_memory_stats(self.device)
        logger.info(f"Memory monitor started. Initial: {self.initial_stats}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        final_stats = get_memory_stats(self.device)
        logger.info(f"Memory monitor ended. Final: {final_stats}")
        
        if self.initial_stats and self.device.type == 'cuda':
            delta = final_stats['allocated_mb'] - self.initial_stats['allocated_mb']
            logger.info(f"Memory delta: {delta:.2f} MB")
    
    def check(self, operation_name: str = "") -> None:
        """Check current memory usage."""
        self.operation_count += 1
        if self.operation_count % self.log_interval == 0:
            stats = get_memory_stats(self.device)
            logger.info(f"Memory check [{operation_name}]: {stats}")



