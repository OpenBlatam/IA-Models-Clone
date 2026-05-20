"""
TorchScript Optimizer
=====================

TorchScript optimizations for maximum speed.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class TorchScriptOptimizer:
    """
    TorchScript optimizer for faster inference.
    
    Features:
    - JIT trace optimization
    - JIT script optimization
    - Optimize for inference
    - Freeze model
    """
    
    @staticmethod
    def trace_and_optimize(
        model: nn.Module,
        example_input: torch.Tensor,
        optimize: bool = True
    ) -> torch.jit.ScriptModule:
        """
        Trace and optimize model.
        
        Args:
            model: PyTorch model
            example_input: Example input
            optimize: Whether to optimize
        
        Returns:
            Traced and optimized model
        """
        model.eval()
        
        try:
            with torch.no_grad():
                # Trace
                traced = torch.jit.trace(model, example_input)
                
                # Optimize for inference
                if optimize:
                    traced = torch.jit.optimize_for_inference(traced)
                    traced = torch.jit.freeze(traced)
                
                logger.info("Model traced and optimized")
                return traced
        except Exception as e:
            logger.warning(f"TorchScript trace failed: {str(e)}")
            raise
    
    @staticmethod
    def script_and_optimize(
        model: nn.Module,
        optimize: bool = True
    ) -> torch.jit.ScriptModule:
        """
        Script and optimize model.
        
        Args:
            model: PyTorch model
            optimize: Whether to optimize
        
        Returns:
            Scripted and optimized model
        """
        try:
            # Script
            scripted = torch.jit.script(model)
            
            # Optimize for inference
            if optimize:
                scripted = torch.jit.optimize_for_inference(scripted)
                scripted = torch.jit.freeze(scripted)
            
            logger.info("Model scripted and optimized")
            return scripted
        except Exception as e:
            logger.warning(f"TorchScript script failed: {str(e)}")
            raise
    
    @staticmethod
    def save_optimized(
        model: torch.jit.ScriptModule,
        path: str
    ):
        """
        Save optimized model.
        
        Args:
            model: TorchScript model
            path: Save path
        """
        try:
            model.save(path)
            logger.info(f"Optimized model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    @staticmethod
    def load_optimized(path: str) -> torch.jit.ScriptModule:
        """
        Load optimized model.
        
        Args:
            path: Model path
        
        Returns:
            Loaded model
        """
        try:
            model = torch.jit.load(path)
            logger.info(f"Optimized model loaded from {path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise




