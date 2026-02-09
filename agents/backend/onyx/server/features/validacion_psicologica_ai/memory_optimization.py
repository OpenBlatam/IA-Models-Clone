"""
Memory Optimization
===================
Memory optimization techniques for training and inference
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import structlog
import gc

logger = structlog.get_logger()


class MemoryOptimizer:
    """
    Memory optimization utilities
    """
    
    def __init__(self):
        """Initialize memory optimizer"""
        logger.info("MemoryOptimizer initialized")
    
    @staticmethod
    def clear_cache() -> None:
        """Clear PyTorch cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.debug("Cache cleared")
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """
        Get current memory usage
        
        Returns:
            Memory usage statistics in GB
        """
        stats = {}
        
        if torch.cuda.is_available():
            stats["cuda"] = {
                "allocated": torch.cuda.memory_allocated() / 1024**3,
                "reserved": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**3
            }
        
        return stats
    
    @staticmethod
    def optimize_model_memory(model: nn.Module) -> nn.Module:
        """
        Optimize model memory usage
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        # Set to eval mode
        model.eval()
        
        # Use half precision if possible
        if torch.cuda.is_available():
            try:
                model = model.half()
                logger.info("Model converted to half precision")
            except Exception as e:
                logger.warning("Could not convert to half precision", error=str(e))
        
        return model
    
    @staticmethod
    def gradient_checkpointing(
        model: nn.Module,
        enable: bool = True
    ) -> nn.Module:
        """
        Enable gradient checkpointing to save memory
        
        Args:
            model: Model
            enable: Enable gradient checkpointing
            
        Returns:
            Model with gradient checkpointing
        """
        if hasattr(model, "gradient_checkpointing_enable"):
            if enable:
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            else:
                model.gradient_checkpointing_disable()
                logger.info("Gradient checkpointing disabled")
        else:
            logger.warning("Model does not support gradient checkpointing")
        
        return model


class BatchMemoryManager:
    """Manage batch memory efficiently"""
    
    def __init__(self, max_batch_size: int = 32):
        """
        Initialize batch memory manager
        
        Args:
            max_batch_size: Maximum batch size
        """
        self.max_batch_size = max_batch_size
        logger.info("BatchMemoryManager initialized", max_batch_size=max_batch_size)
    
    def adaptive_batch_size(
        self,
        initial_batch_size: int,
        memory_threshold: float = 0.9
    ) -> int:
        """
        Calculate adaptive batch size based on memory
        
        Args:
            initial_batch_size: Initial batch size
            memory_threshold: Memory threshold (0-1)
            
        Returns:
            Adaptive batch size
        """
        if not torch.cuda.is_available():
            return initial_batch_size
        
        # Get current memory usage
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        
        usage_ratio = reserved / total
        
        if usage_ratio > memory_threshold:
            # Reduce batch size
            new_batch_size = max(1, int(initial_batch_size * (1 - usage_ratio)))
            logger.info(
                "Reducing batch size",
                old=initial_batch_size,
                new=new_batch_size,
                memory_usage=usage_ratio
            )
            return new_batch_size
        
        return initial_batch_size


# Global memory optimizer
memory_optimizer = MemoryOptimizer()




