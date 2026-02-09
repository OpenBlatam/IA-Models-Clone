"""
Model Export

Utilities for exporting models to different formats.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Try to import ONNX
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX not available for export")


class ModelExporter:
    """Export models to different formats."""
    
    @staticmethod
    def export_onnx(
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, ...],
        input_names: Optional[list] = None,
        output_names: Optional[list] = None,
        opset_version: int = 11
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            model: Model to export
            output_path: Output path
            input_shape: Input tensor shape
            input_names: Input names
            output_names: Output names
            opset_version: ONNX opset version
            
        Returns:
            Output path
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX required for ONNX export")
        
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        input_names = input_names or ["input"]
        output_names = output_names or ["output"]
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=opset_version,
                dynamic_axes=None
            )
            logger.info(f"Exported model to ONNX: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
    
    @staticmethod
    def export_torchscript(
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, ...],
        method: str = "trace"
    ) -> str:
        """
        Export model to TorchScript.
        
        Args:
            model: Model to export
            output_path: Output path
            input_shape: Input tensor shape
            method: Export method ('trace' or 'script')
            
        Returns:
            Output path
        """
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        try:
            if method == "trace":
                traced_model = torch.jit.trace(model, dummy_input)
            elif method == "script":
                traced_model = torch.jit.script(model)
            else:
                raise ValueError(f"Unknown export method: {method}")
            
            traced_model.save(output_path)
            logger.info(f"Exported model to TorchScript: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            raise
    
    @staticmethod
    def export_tensorrt(
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, ...],
        precision: str = "fp32"
    ) -> str:
        """
        Export model to TensorRT (via ONNX).
        
        Args:
            model: Model to export
            output_path: Output path
            input_shape: Input tensor shape
            precision: Precision ('fp32', 'fp16', 'int8')
            
        Returns:
            Output path
        """
        # First export to ONNX
        onnx_path = output_path.replace('.trt', '.onnx')
        ModelExporter.export_onnx(model, onnx_path, input_shape)
        
        logger.info(f"Exported to ONNX for TensorRT conversion: {onnx_path}")
        logger.warning("TensorRT conversion requires TensorRT SDK. Use trtexec or TensorRT Python API.")
        
        return onnx_path


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...],
    **kwargs
) -> str:
    """Export model to ONNX."""
    return ModelExporter.export_onnx(model, output_path, input_shape, **kwargs)


def export_to_torchscript(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...],
    **kwargs
) -> str:
    """Export model to TorchScript."""
    return ModelExporter.export_torchscript(model, output_path, input_shape, **kwargs)


def export_to_tensorrt(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...],
    **kwargs
) -> str:
    """Export model to TensorRT."""
    return ModelExporter.export_tensorrt(model, output_path, input_shape, **kwargs)



