"""
Memory Optimizer
================

Memory optimization utilities.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    @staticmethod
    def enable_gradient_checkpointing(model: nn.Module) -> nn.Module:
        """
        Enable gradient checkpointing to save memory.
        
        Args:
            model: Model to optimize
        
        Returns:
            Model with gradient checkpointing
        """
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            # For custom models, use torch.utils.checkpoint
            logger.warning("Gradient checkpointing not directly available")
        
        return model
    
    @staticmethod
    def clear_cache():
        """Clear CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")
    
    @staticmethod
    def optimize_memory_usage(model: nn.Module) -> nn.Module:
        """
        Optimize model memory usage.
        
        Args:
            model: Model to optimize
        
        Returns:
            Optimized model
        """
        # Set to eval mode to disable gradients
        model.eval()
        
        # Clear cache
        MemoryOptimizer.clear_cache()
        
        return model
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """
        Get current memory usage.
        
        Returns:
            Memory usage in MB
        """
        memory_info = {}
        
        if torch.cuda.is_available():
            memory_info["cuda_allocated"] = torch.cuda.memory_allocated() / 1024**2
            memory_info["cuda_reserved"] = torch.cuda.memory_reserved() / 1024**2
            memory_info["cuda_max_allocated"] = torch.cuda.max_memory_allocated() / 1024**2
        
        return memory_info




