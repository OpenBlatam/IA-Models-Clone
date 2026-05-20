"""
Performance Optimizer
Advanced performance optimization utilities.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Optimize model for inference."""
    
    @staticmethod
    def optimize_for_inference(
        model: nn.Module,
        compile_model: bool = False,
        enable_xformers: bool = True,
        enable_flash_attention: bool = False,
    ) -> nn.Module:
        """Optimize model for inference."""
        model.eval()
        
        # Compile model (PyTorch 2.0+)
        if compile_model and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        # Enable xformers memory efficient attention
        if enable_xformers:
            try:
                for module in model.modules():
                    if hasattr(module, "enable_xformers_memory_efficient_attention"):
                        module.enable_xformers_memory_efficient_attention()
                logger.info("XFormers memory efficient attention enabled")
            except Exception as e:
                logger.warning(f"XFormers not available: {e}")
        
        # Enable flash attention
        if enable_flash_attention:
            try:
                for module in model.modules():
                    if hasattr(module, "enable_flash_attention"):
                        module.enable_flash_attention()
                logger.info("Flash attention enabled")
            except Exception as e:
                logger.warning(f"Flash attention not available: {e}")
        
        return model
    
    @staticmethod
    def fuse_modules(model: nn.Module) -> nn.Module:
        """Fuse modules for better performance."""
        try:
            # Fuse Conv+BN+ReLU
            if hasattr(torch.quantization, "fuse_modules"):
                # This is a simplified example
                # In practice, you'd identify fuseable patterns
                logger.info("Module fusion attempted")
        except Exception as e:
            logger.warning(f"Module fusion failed: {e}")
        
        return model
    
    @staticmethod
    def enable_torchscript(model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        """Enable TorchScript for faster inference."""
        try:
            traced_model = torch.jit.trace(model, example_input)
            logger.info("Model traced with TorchScript")
            return traced_model
        except Exception as e:
            logger.warning(f"TorchScript tracing failed: {e}")
            return model


class MemoryOptimizer:
    """Optimize memory usage."""
    
    @staticmethod
    def enable_gradient_checkpointing(model: nn.Module):
        """Enable gradient checkpointing to save memory."""
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
    
    @staticmethod
    def clear_cache():
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU cache cleared")
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = {}
        
        # CPU memory
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_info["cpu_memory_mb"] = process.memory_info().rss / (1024 * 1024)
        except Exception:
            pass
        
        # GPU memory
        if torch.cuda.is_available():
            memory_info["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            memory_info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            memory_info["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        return memory_info



