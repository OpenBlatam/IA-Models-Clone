"""
Model Optimizer Utilities
==========================

Utilities for applying optimizations to PyTorch models.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Handles model optimizations for memory and speed."""
    
    @staticmethod
    def apply_optimizations(model: nn.Module, device: torch.device) -> None:
        """
        Apply memory and speed optimizations to a model.
        
        Args:
            model: PyTorch model to optimize
            device: Device the model is on
        """
        if device.type != "cuda":
            logger.debug("Skipping CUDA optimizations for non-CUDA device")
            return
        
        try:
            ModelOptimizer._enable_attention_slicing(model)
            ModelOptimizer._enable_xformers(model)
            ModelOptimizer._compile_model(model)
        except Exception as e:
            logger.warning(f"Some optimizations failed: {e}")
    
    @staticmethod
    def _enable_attention_slicing(model: nn.Module) -> None:
        """Enable attention slicing for memory efficiency."""
        if hasattr(model, "enable_attention_slicing"):
            model.enable_attention_slicing(1)
            logger.debug("Attention slicing enabled")
    
    @staticmethod
    def _enable_xformers(model: nn.Module) -> None:
        """Enable xformers memory efficient attention."""
        try:
            if hasattr(model, "enable_xformers_memory_efficient_attention"):
                model.enable_xformers_memory_efficient_attention()
                logger.info("XFormers memory efficient attention enabled")
        except Exception as e:
            logger.warning(f"XFormers not available: {e}")
    
    @staticmethod
    def _compile_model(model: nn.Module) -> None:
        """Compile model with torch.compile (PyTorch 2.0+)."""
        try:
            if hasattr(torch, "compile"):
                torch.compile(model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile")
        except Exception as e:
            logger.warning(f"torch.compile not available or failed: {e}")
    
    @staticmethod
    def get_optimization_info(model: nn.Module) -> dict:
        """
        Get information about applied optimizations.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with optimization information
        """
        info = {
            "attention_slicing": hasattr(model, "enable_attention_slicing"),
            "xformers_available": hasattr(model, "enable_xformers_memory_efficient_attention"),
            "torch_compile_available": hasattr(torch, "compile"),
        }
        return info


