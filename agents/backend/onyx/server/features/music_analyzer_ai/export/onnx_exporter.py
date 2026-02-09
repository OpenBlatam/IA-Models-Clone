"""
ONNX Exporter
Export models to ONNX format
"""

from typing import Optional, Tuple, List
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


class ONNXExporter:
    """Export models to ONNX"""
    
    @staticmethod
    def export(
        model: nn.Module,
        path: str,
        input_shape: Tuple[int, ...],
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Any]] = None,
        opset_version: int = 11
    ):
        """
        Export model to ONNX
        
        Args:
            model: Model to export
            path: Output path
            input_shape: Input tensor shape
            input_names: Optional input names
            output_names: Optional output names
            dynamic_axes: Optional dynamic axes
            opset_version: ONNX opset version
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        try:
            import torch.onnx
        except ImportError:
            raise ImportError("ONNX export not available")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            str(path),
            input_names=input_names or ["input"],
            output_names=output_names or ["output"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            verbose=False
        )
        
        logger.info(f"Model exported to ONNX: {path}")



