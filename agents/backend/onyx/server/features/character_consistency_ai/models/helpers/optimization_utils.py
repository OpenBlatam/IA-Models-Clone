"""
Model Optimization Utilities
=============================

Utilities for model optimizations (memory, speed, etc.).
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Handles model optimizations for memory and speed."""
    
    @staticmethod
    def apply_optimizations(model: nn.Module, device: torch.device) -> None:
        """
        Apply memory and speed optimizations.
        
        Args:
            model: PyTorch model to optimize
            device: Device the model is on
        """
        if device.type != "cuda":
            logger.debug("Skipping optimizations for non-CUDA device")
            return
        
        try:
            # Enable attention slicing
            if hasattr(model, "enable_attention_slicing"):
                model.enable_attention_slicing(1)
                logger.debug("Attention slicing enabled")
            
            # Enable xformers
            try:
                if hasattr(model, "enable_xformers_memory_efficient_attention"):
                    model.enable_xformers_memory_efficient_attention()
                    logger.info("XFormers memory efficient attention enabled")
            except Exception as e:
                logger.warning(f"XFormers not available: {e}")
            
            # Compile model (PyTorch 2.0+)
            try:
                if hasattr(torch, "compile"):
                    torch.compile(model, mode="reduce-overhead")
                    logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile not available or failed: {e}")
        
        except Exception as e:
            logger.warning(f"Some optimizations failed: {e}")


