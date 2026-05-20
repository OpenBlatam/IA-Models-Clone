"""
Performance Optimization Utilities
==================================

Advanced performance optimizations for deep learning models.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
import logging
from functools import lru_cache

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Model optimization utilities.
    
    Provides:
    - Model compilation (torch.compile)
    - Memory optimization
    - Inference optimization
    - GPU optimizations
    """
    
    @staticmethod
    def compile_model(
        model: nn.Module,
        mode: str = "default",
        fullgraph: bool = False,
        dynamic: bool = False
    ) -> nn.Module:
        """
        Compile model with torch.compile for faster execution.
        
        Args:
            model: Model to compile
            mode: Compilation mode (default, reduce-overhead, max-autotune)
            fullgraph: Whether to compile the full graph
            dynamic: Whether to use dynamic shapes
        
        Returns:
            Compiled model
        """
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile not available (requires PyTorch 2.0+)")
            return model
        
        try:
            compiled_model = torch.compile(
                model,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic
            )
            logger.info(f"✅ Model compiled with mode: {mode}")
            return compiled_model
        except Exception as e:
            logger.warning(f"Compilation failed: {e}, returning original model")
            return model
    
    @staticmethod
    def optimize_for_inference(model: nn.Module) -> nn.Module:
        """
        Optimize model for inference.
        
        Args:
            model: Model to optimize
        
        Returns:
            Optimized model
        """
        model.eval()
        
        # Fuse operations if available
        try:
            if hasattr(torch.ao.quantization, 'fuse_modules'):
                # Example: fuse Conv+BN+ReLU
                torch.ao.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])
        except Exception:
            pass
        
        # Enable inference optimizations
        with torch.no_grad():
            # Set to eval mode
            model.eval()
            
            # Disable gradient computation
            for param in model.parameters():
                param.requires_grad = False
        
        logger.info("✅ Model optimized for inference")
        return model
    
    @staticmethod
    def enable_tf32(model: nn.Module) -> None:
        """Enable TF32 for faster computation on Ampere+ GPUs."""
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("✅ TF32 enabled")
    
    @staticmethod
    def enable_flash_attention(model: nn.Module) -> bool:
        """
        Enable flash attention if available.
        
        Args:
            model: Model to enable flash attention for
        
        Returns:
            True if enabled, False otherwise
        """
        try:
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
                logger.info("✅ Flash attention enabled")
                return True
        except Exception:
            pass
        
        logger.debug("Flash attention not available")
        return False


class MemoryOptimizer:
    """
    Memory optimization utilities.
    """
    
    @staticmethod
    def clear_cache() -> None:
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """
        Get current GPU memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        if not torch.cuda.is_available():
            return {"available": False}
        
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
            "available": True
        }
    
    @staticmethod
    def set_memory_fraction(fraction: float) -> None:
        """
        Set GPU memory fraction.
        
        Args:
            fraction: Memory fraction (0.0-1.0)
        """
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(fraction)
            logger.info(f"✅ GPU memory fraction set to {fraction}")


class InferenceOptimizer:
    """
    Inference optimization utilities.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize inference optimizer.
        
        Args:
            model: Model to optimize
            device: Target device
        """
        self.model = model
        self.device = device
        self.model.eval()
        self._cache = {}
    
    @torch.no_grad()
    def batch_inference(
        self,
        inputs: torch.Tensor,
        batch_size: int = 32,
        use_amp: bool = True
    ) -> torch.Tensor:
        """
        Perform batched inference for large inputs.
        
        Args:
            inputs: Input tensor
            batch_size: Batch size for inference
            use_amp: Whether to use mixed precision
        
        Returns:
            Output tensor
        """
        outputs = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size].to(self.device)
            
            if use_amp and self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    batch_output = self.model(batch)
            else:
                batch_output = self.model(batch)
            
            outputs.append(batch_output.cpu())
        
        return torch.cat(outputs, dim=0)
    
    @lru_cache(maxsize=128)
    def cached_inference(self, input_hash: int) -> torch.Tensor:
        """
        Cached inference for repeated inputs.
        
        Args:
            input_hash: Hash of input
        
        Returns:
            Cached output
        """
        # This is a placeholder - actual implementation would hash inputs
        pass


class DataLoaderOptimizer:
    """
    DataLoader optimization utilities.
    """
    
    @staticmethod
    def optimize_dataloader(
        dataloader,
        prefetch_factor: int = 4,
        persistent_workers: bool = True,
        pin_memory: bool = True
    ):
        """
        Optimize DataLoader settings.
        
        Args:
            dataloader: DataLoader to optimize
            prefetch_factor: Prefetch factor
            persistent_workers: Keep workers alive
            pin_memory: Pin memory for faster GPU transfer
        
        Returns:
            Optimized DataLoader
        """
        # DataLoader settings are set at creation time
        # This function provides recommendations
        recommendations = {
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
            "pin_memory": pin_memory and torch.cuda.is_available(),
            "num_workers": min(4, torch.cuda.device_count() * 2) if torch.cuda.is_available() else 2
        }
        
        logger.info(f"DataLoader optimization recommendations: {recommendations}")
        return recommendations


def optimize_model_for_production(
    model: nn.Module,
    device: torch.device,
    compile_model: bool = True,
    enable_tf32: bool = True,
    enable_flash_attention: bool = True
) -> nn.Module:
    """
    Comprehensive model optimization for production.
    
    Args:
        model: Model to optimize
        device: Target device
        compile_model: Whether to compile model
        enable_tf32: Whether to enable TF32
        enable_flash_attention: Whether to enable flash attention
    
    Returns:
        Optimized model
    """
    # Optimize for inference
    model = ModelOptimizer.optimize_for_inference(model)
    
    # Enable TF32
    if enable_tf32:
        ModelOptimizer.enable_tf32(model)
    
    # Enable flash attention
    if enable_flash_attention:
        ModelOptimizer.enable_flash_attention(model)
    
    # Compile model
    if compile_model and hasattr(torch, 'compile'):
        model = ModelOptimizer.compile_model(model, mode="reduce-overhead")
    
    # Move to device
    model = model.to(device)
    
    logger.info("✅ Model optimized for production")
    
    return model



