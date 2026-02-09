"""
Memory Optimization
Aggressive memory optimizations for maximum efficiency
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging
import gc

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """
    Memory optimization utilities
    """
    
    @staticmethod
    def optimize_model_memory(model: nn.Module):
        """
        Optimize model memory usage
        
        Args:
            model: Model to optimize
        """
        # 1. Set to eval mode
        model.eval()
        
        # 2. Disable gradients
        for param in model.parameters():
            param.requires_grad = False
        
        # 3. Use channels_last memory format if applicable
        try:
            if hasattr(model, 'to'):
                model = model.to(memory_format=torch.channels_last)
        except:
            pass
        
        # 4. Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def clear_cache():
        """Clear all caches"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """
        Get memory statistics
        
        Returns:
            Dictionary with memory stats
        """
        stats = {}
        
        if torch.cuda.is_available():
            stats["allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            stats["reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            stats["max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            
            if torch.cuda.get_device_properties(0).total_memory:
                stats["total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
                stats["free_gb"] = stats["total_gb"] - stats["reserved_gb"]
                stats["usage_percent"] = (stats["reserved_gb"] / stats["total_gb"]) * 100
        
        return stats
    
    @staticmethod
    def optimize_batch_size(
        model: nn.Module,
        input_shape: tuple,
        device: torch.device,
        max_memory_gb: float = 8.0
    ) -> int:
        """
        Find optimal batch size for given memory constraint
        
        Args:
            model: Model
            input_shape: Input shape (batch_size, ...)
            device: Device
            max_memory_gb: Maximum memory in GB
            
        Returns:
            Optimal batch size
        """
        model.eval()
        
        # Start with batch size 1
        batch_size = 1
        
        while True:
            try:
                # Try with current batch size
                test_input = torch.randn((batch_size, *input_shape[1:])).to(device)
                
                with torch.inference_mode():
                    _ = model(test_input)
                
                # Check memory
                memory_gb = torch.cuda.memory_allocated() / 1024**3 if device.type == "cuda" else 0
                
                if memory_gb > max_memory_gb:
                    return max(1, batch_size - 1)
                
                batch_size *= 2
                
                # Clear cache
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    return max(1, batch_size - 1)
                raise
        
        return batch_size


class GradientCheckpointing:
    """
    Gradient checkpointing for memory efficiency during training
    """
    
    @staticmethod
    def apply_checkpointing(model: nn.Module, segments: int = 4):
        """
        Apply gradient checkpointing
        
        Args:
            model: Model to apply checkpointing to
            segments: Number of segments
        """
        try:
            from torch.utils.checkpoint import checkpoint
            
            # Wrap forward pass with checkpointing
            original_forward = model.forward
            
            def checkpointed_forward(*args, **kwargs):
                return checkpoint(original_forward, *args, **kwargs, use_reentrant=False)
            
            model.forward = checkpointed_forward
            logger.info(f"Gradient checkpointing applied with {segments} segments")
        except Exception as e:
            logger.warning(f"Gradient checkpointing failed: {e}")


def optimize_model_memory(model: nn.Module):
    """Optimize model memory"""
    MemoryOptimizer.optimize_model_memory(model)


def clear_memory_cache():
    """Clear memory cache"""
    MemoryOptimizer.clear_cache()


def get_memory_stats() -> Dict[str, float]:
    """Get memory statistics"""
    return MemoryOptimizer.get_memory_stats()













