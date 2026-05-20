"""
Performance Optimization Utilities
Advanced performance optimization tools
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    Performance optimization utilities
    """
    
    @staticmethod
    def optimize_model_for_inference(
        model: nn.Module,
        device: torch.device,
    ) -> nn.Module:
        """
        Optimize model for inference
        
        Args:
            model: Model to optimize
            device: Target device
            
        Returns:
            Optimized model
        """
        model.eval()
        model = model.to(device)
        
        # Fuse operations if possible
        if hasattr(torch.jit, 'optimize_for_inference'):
            try:
                model = torch.jit.optimize_for_inference(model)
                logger.info("Optimized model with JIT")
            except Exception as e:
                logger.warning(f"JIT optimization failed: {e}")
        
        # Enable inference mode
        with torch.inference_mode():
            # Warmup
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            _ = model(dummy_input)
        
        return model
    
    @staticmethod
    def enable_cudnn_benchmark() -> None:
        """Enable cuDNN benchmark for faster convolutions"""
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled cuDNN benchmark")
    
    @staticmethod
    def disable_cudnn_benchmark() -> None:
        """Disable cuDNN benchmark"""
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            logger.info("Disabled cuDNN benchmark")
    
    @staticmethod
    def set_deterministic(seed: int = 42) -> None:
        """
        Set deterministic behavior
        
        Args:
            seed: Random seed
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Set deterministic mode with seed {seed}")
    
    @staticmethod
    def compile_model(
        model: nn.Module,
        mode: str = "default",
    ) -> nn.Module:
        """
        Compile model with torch.compile (PyTorch 2.0+)
        
        Args:
            model: Model to compile
            mode: Compilation mode
            
        Returns:
            Compiled model
        """
        try:
            compiled_model = torch.compile(model, mode=mode)
            logger.info(f"Compiled model with mode: {mode}")
            return compiled_model
        except AttributeError:
            logger.warning("torch.compile not available (requires PyTorch 2.0+)")
            return model
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")
            return model



