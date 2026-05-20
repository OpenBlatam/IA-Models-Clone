"""
Memory Management Utilities
Memory optimization and management
"""

import torch
import gc
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Memory management utilities
    """
    
    @staticmethod
    def clear_cache() -> None:
        """Clear PyTorch and Python caches"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.debug("Cleared caches")
    
    @staticmethod
    def get_memory_usage() -> Dict[str, Any]:
        """
        Get current memory usage
        
        Returns:
            Dictionary with memory information
        """
        info = {}
        
        # CPU memory
        try:
            import psutil
            process = psutil.Process()
            info['cpu_memory_mb'] = process.memory_info().rss / (1024 ** 2)
            info['cpu_memory_percent'] = process.memory_info().rss / psutil.virtual_memory().total * 100
        except ImportError:
            logger.warning("psutil not available for CPU memory tracking")
        
        # GPU memory
        if torch.cuda.is_available():
            info['gpu_allocated_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
            info['gpu_reserved_mb'] = torch.cuda.memory_reserved() / (1024 ** 2)
            info['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated() / (1024 ** 2)
            info['gpu_max_reserved_mb'] = torch.cuda.max_memory_reserved() / (1024 ** 2)
        
        return info
    
    @staticmethod
    def optimize_memory(model: torch.nn.Module) -> None:
        """
        Optimize model memory usage
        
        Args:
            model: Model to optimize
        """
        # Set to eval mode
        model.eval()
        
        # Clear gradients
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None
        
        # Clear cache
        MemoryManager.clear_cache()
        
        logger.info("Optimized model memory")
    
    @staticmethod
    def print_memory_summary() -> None:
        """Print memory usage summary"""
        info = MemoryManager.get_memory_usage()
        
        print("\n" + "="*60)
        print("Memory Usage Summary")
        print("="*60)
        
        if 'cpu_memory_mb' in info:
            print(f"CPU Memory: {info['cpu_memory_mb']:.2f} MB ({info['cpu_memory_percent']:.2f}%)")
        
        if 'gpu_allocated_mb' in info:
            print(f"GPU Allocated: {info['gpu_allocated_mb']:.2f} MB")
            print(f"GPU Reserved: {info['gpu_reserved_mb']:.2f} MB")
            print(f"GPU Max Allocated: {info['gpu_max_allocated_mb']:.2f} MB")
        
        print("="*60 + "\n")



