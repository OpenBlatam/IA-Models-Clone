"""
Optimization Module

GPU and memory optimization utilities.
"""

from typing import Optional, Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class GPUOptimizer:
    """
    GPU optimization utilities.
    """
    
    @staticmethod
    def setup_optimal_gpu_settings():
        """Setup optimal GPU settings for music generation."""
        try:
            import torch
            
            if torch.cuda.is_available():
                # Enable cuDNN benchmarking for faster convolutions
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # Enable TensorFloat-32 for faster training on Ampere GPUs
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                logger.info("GPU optimizations enabled")
        except ImportError:
            logger.debug("PyTorch not available")
    
    @staticmethod
    def clear_gpu_cache():
        """Clear GPU cache."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass


class MemoryOptimizer:
    """
    Memory optimization utilities.
    """
    
    @staticmethod
    def quantize_model(model, bits: int = 8):
        """
        Quantize model to reduce memory usage.
        
        Args:
            model: PyTorch model
            bits: Number of bits (8 or 4)
            
        Returns:
            Quantized model
        """
        try:
            from bitsandbytes import quantize_model
            
            logger.info(f"Quantizing model to {bits}-bit")
            quantized = quantize_model(model, bits=bits)
            return quantized
        except ImportError:
            logger.warning("bitsandbytes not available, skipping quantization")
            return model
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return model
    
    @staticmethod
    def enable_gradient_checkpointing(model):
        """Enable gradient checkpointing to save memory."""
        try:
            if hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()
                logger.info("Gradient checkpointing enabled")
        except Exception as e:
            logger.warning(f"Failed to enable gradient checkpointing: {e}")


class BatchOptimizer:
    """
    Batch processing optimization.
    """
    
    def __init__(self, max_batch_size: int = 4):
        self.max_batch_size = max_batch_size
    
    def optimize_batch_size(
        self,
        available_memory_gb: float,
        model_size_gb: float
    ) -> int:
        """
        Calculate optimal batch size based on available memory.
        
        Args:
            available_memory_gb: Available GPU memory in GB
            model_size_gb: Model size in GB
            
        Returns:
            Optimal batch size
        """
        # Simple heuristic: leave 2GB for overhead
        usable_memory = available_memory_gb - 2.0
        
        if usable_memory < model_size_gb:
            return 1
        
        # Estimate batch size (rough calculation)
        estimated_batch_size = int((usable_memory - model_size_gb) / 2.0)
        
        return min(max(estimated_batch_size, 1), self.max_batch_size)


class ModelCache:
    """
    Cache for models to avoid reloading.
    """
    
    def __init__(self, max_size: int = 2):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
    
    def get(self, key: str):
        """Get model from cache."""
        return self.cache.get(key)
    
    def set(self, key: str, model: Any) -> None:
        """Store model in cache."""
        if len(self.cache) >= self.max_size:
            # Remove oldest (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = model
    
    def clear(self) -> None:
        """Clear cache."""
        for model in self.cache.values():
            try:
                del model
            except:
                pass
        self.cache.clear()
        
        # Clear GPU cache
        GPUOptimizer.clear_gpu_cache()


# Global model cache
_model_cache = ModelCache()


def get_model_cache() -> ModelCache:
    """Get global model cache."""
    return _model_cache















