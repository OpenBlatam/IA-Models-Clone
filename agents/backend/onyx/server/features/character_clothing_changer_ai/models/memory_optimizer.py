"""
Memory Optimizer for Flux2 Clothing Changer
===========================================

Advanced memory optimization techniques for large models and images.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
import logging
import gc

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """Advanced memory optimization utilities."""
    
    def __init__(self, device: torch.device):
        """
        Initialize memory optimizer.
        
        Args:
            device: Device to optimize for
        """
        self.device = device
        self.original_forward = {}
    
    @staticmethod
    def enable_gradient_checkpointing(model: nn.Module) -> None:
        """
        Enable gradient checkpointing to save memory during training.
        
        Args:
            model: Model to enable checkpointing for
        """
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            logger.warning("Model does not support gradient checkpointing")
    
    @staticmethod
    def disable_gradient_checkpointing(model: nn.Module) -> None:
        """
        Disable gradient checkpointing.
        
        Args:
            model: Model to disable checkpointing for
        """
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
            logger.info("Gradient checkpointing disabled")
    
    @staticmethod
    def enable_cpu_offload(model: nn.Module) -> None:
        """
        Enable CPU offloading for model components.
        
        Args:
            model: Model to enable offloading for
        """
        if hasattr(model, "enable_model_cpu_offload"):
            model.enable_model_cpu_offload()
            logger.info("CPU offloading enabled")
        else:
            logger.warning("Model does not support CPU offloading")
    
    @staticmethod
    def enable_sequential_cpu_offload(model: nn.Module) -> None:
        """
        Enable sequential CPU offloading.
        
        Args:
            model: Model to enable sequential offloading for
        """
        if hasattr(model, "enable_sequential_cpu_offload"):
            model.enable_sequential_cpu_offload()
            logger.info("Sequential CPU offloading enabled")
        else:
            logger.warning("Model does not support sequential CPU offloading")
    
    @staticmethod
    def clear_cache() -> None:
        """Clear PyTorch and CUDA caches."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.debug("Memory cache cleared")
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """
        Get current memory usage.
        
        Returns:
            Dict with memory usage in MB
        """
        memory_info = {}
        
        # CPU memory
        import psutil
        process = psutil.Process()
        memory_info["cpu_mb"] = process.memory_info().rss / (1024 * 1024)
        
        # GPU memory
        if torch.cuda.is_available():
            memory_info["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            memory_info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            memory_info["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        return memory_info
    
    @staticmethod
    def optimize_model_for_inference(model: nn.Module) -> None:
        """
        Optimize model for inference (reduce memory usage).
        
        Args:
            model: Model to optimize
        """
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        # Enable inference mode
        if hasattr(torch, "inference_mode"):
            # Use inference_mode context manager in actual usage
            pass
        
        logger.info("Model optimized for inference")
    
    @staticmethod
    def enable_attention_slicing(
        model: nn.Module,
        slice_size: Optional[Union[str, int]] = "max",
    ) -> None:
        """
        Enable attention slicing to reduce memory usage.
        
        Args:
            model: Model to enable slicing for
            slice_size: Slice size ("max", "auto", or int)
        """
        if hasattr(model, "enable_attention_slicing"):
            model.enable_attention_slicing(slice_size)
            logger.info(f"Attention slicing enabled with slice_size={slice_size}")
        else:
            logger.warning("Model does not support attention slicing")
    
    @staticmethod
    def enable_vae_slicing(model: nn.Module) -> None:
        """
        Enable VAE slicing to reduce memory usage.
        
        Args:
            model: Model to enable VAE slicing for
        """
        if hasattr(model, "enable_vae_slicing"):
            model.enable_vae_slicing()
            logger.info("VAE slicing enabled")
        else:
            logger.warning("Model does not support VAE slicing")
    
    @staticmethod
    def enable_vae_tiling(model: nn.Module) -> None:
        """
        Enable VAE tiling for large images.
        
        Args:
            model: Model to enable VAE tiling for
        """
        if hasattr(model, "enable_vae_tiling"):
            model.enable_vae_tiling()
            logger.info("VAE tiling enabled")
        else:
            logger.warning("Model does not support VAE tiling")
    
    @staticmethod
    def set_memory_fraction(fraction: float) -> None:
        """
        Set GPU memory fraction.
        
        Args:
            fraction: Memory fraction (0.0 to 1.0)
        """
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(fraction)
            logger.info(f"GPU memory fraction set to {fraction}")
    
    @staticmethod
    def enable_torch_compile(
        model: nn.Module,
        mode: str = "reduce-overhead",
    ) -> Optional[nn.Module]:
        """
        Enable torch.compile for faster inference.
        
        Args:
            model: Model to compile
            mode: Compilation mode
            
        Returns:
            Compiled model or original if compilation fails
        """
        if hasattr(torch, "compile"):
            try:
                compiled = torch.compile(model, mode=mode)
                logger.info(f"Model compiled with torch.compile (mode={mode})")
                return compiled
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
                return model
        else:
            logger.warning("torch.compile not available")
            return model
    
    @staticmethod
    def enable_xformers(model: nn.Module) -> None:
        """
        Enable xformers memory efficient attention.
        
        Args:
            model: Model to enable xformers for
        """
        if hasattr(model, "enable_xformers_memory_efficient_attention"):
            try:
                model.enable_xformers_memory_efficient_attention()
                logger.info("XFormers memory efficient attention enabled")
            except Exception as e:
                logger.warning(f"XFormers not available: {e}")
        else:
            logger.warning("Model does not support xformers")
    
    def apply_all_optimizations(
        self,
        model: nn.Module,
        enable_checkpointing: bool = False,
        enable_cpu_offload: bool = False,
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
        enable_xformers: bool = True,
    ) -> None:
        """
        Apply all memory optimizations.
        
        Args:
            model: Model to optimize
            enable_checkpointing: Enable gradient checkpointing
            enable_cpu_offload: Enable CPU offloading
            enable_attention_slicing: Enable attention slicing
            enable_vae_slicing: Enable VAE slicing
            enable_xformers: Enable xformers
        """
        logger.info("Applying memory optimizations...")
        
        if enable_checkpointing:
            self.enable_gradient_checkpointing(model)
        
        if enable_cpu_offload:
            self.enable_cpu_offload(model)
        
        if enable_attention_slicing:
            self.enable_attention_slicing(model)
        
        if enable_vae_slicing:
            self.enable_vae_slicing(model)
        
        if enable_xformers:
            self.enable_xformers(model)
        
        self.optimize_model_for_inference(model)
        logger.info("Memory optimizations applied")


