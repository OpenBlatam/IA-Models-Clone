"""
TorchScript Exporter
Export models to TorchScript
"""

from typing import Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class TorchScriptExporter:
    """Export models to TorchScript"""
    
    @staticmethod
    def export_trace(
        model: nn.Module,
        path: str,
        input_shape: Tuple[int, ...],
        optimize: bool = True
    ):
        """
        Export model using tracing
        
        Args:
            model: Model to export
            path: Output path
            input_shape: Input tensor shape
            optimize: Whether to optimize
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        traced_model = torch.jit.trace(model, dummy_input)
        
        if optimize:
            traced_model = torch.jit.optimize_for_inference(traced_model)
        
        traced_model.save(str(path))
        logger.info(f"Model exported to TorchScript (trace): {path}")
    
    @staticmethod
    def export_script(
        model: nn.Module,
        path: str
    ):
        """
        Export model using scripting
        
        Args:
            model: Model to export
            path: Output path
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        scripted_model = torch.jit.script(model)
        scripted_model.save(str(path))
        logger.info(f"Model exported to TorchScript (script): {path}")



