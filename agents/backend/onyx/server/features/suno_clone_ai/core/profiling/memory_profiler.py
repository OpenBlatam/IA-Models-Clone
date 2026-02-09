"""
Memory Profiling

Utilities for profiling memory usage.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Optional
import psutil
import os

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Profile memory usage of models and operations."""
    
    @staticmethod
    def get_memory_stats(
        device: Optional[torch.device] = None
    ) -> Dict[str, float]:
        """
        Get current memory statistics.
        
        Args:
            device: Device to check (if CUDA)
            
        Returns:
            Memory statistics
        """
        stats = {}
        
        # CPU memory
        process = psutil.Process(os.getpid())
        stats['cpu_memory_mb'] = process.memory_info().rss / (1024 ** 2)
        stats['cpu_memory_percent'] = process.memory_percent()
        
        # GPU memory
        if device and device.type == "cuda":
            stats['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated(device) / (1024 ** 3)
            stats['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved(device) / (1024 ** 3)
            stats['gpu_memory_max_allocated_gb'] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        
        return stats
    
    @staticmethod
    def profile_model_memory(
        model: nn.Module,
        input_size: tuple,
        device: Optional[torch.device] = None
    ) -> Dict[str, float]:
        """
        Profile model memory usage.
        
        Args:
            model: Model to profile
            input_size: Input tensor size
            device: Device to use
            
        Returns:
            Memory statistics
        """
        if device is None:
            device = next(model.parameters()).device
        
        # Clear cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        
        # Get baseline
        baseline_stats = MemoryProfiler.get_memory_stats(device)
        
        # Move model to device if needed
        model = model.to(device)
        
        # Create input
        input_tensor = torch.randn(input_size, device=device)
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Get peak memory
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        else:
            peak_memory = 0.0
        
        # Get current stats
        current_stats = MemoryProfiler.get_memory_stats(device)
        
        return {
            'baseline_memory_mb': baseline_stats.get('cpu_memory_mb', 0),
            'current_memory_mb': current_stats.get('cpu_memory_mb', 0),
            'peak_gpu_memory_gb': peak_memory,
            'memory_increase_mb': current_stats.get('cpu_memory_mb', 0) - baseline_stats.get('cpu_memory_mb', 0)
        }


def profile_memory_usage(
    model: nn.Module,
    input_size: tuple,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Convenience function to profile memory usage.
    
    Args:
        model: Model to profile
        input_size: Input size
        device: Device to use
        
    Returns:
        Memory statistics
    """
    return MemoryProfiler.profile_model_memory(model, input_size, device)


def get_memory_stats(
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Convenience function to get memory stats.
    
    Args:
        device: Device to check
        
    Returns:
        Memory statistics
    """
    return MemoryProfiler.get_memory_stats(device)



