"""
Model Export
Export models to various formats
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelExporter:
    """
    Export models to various formats
    """
    
    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str,
        opset_version: int = 14,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Export model to ONNX
        
        Args:
            model: Model to export
            input_shape: Input shape
            output_path: Output path
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes for variable batch/sequence length
            device: Device to use
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        
        dummy_input = torch.randn(input_shape).to(device)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if dynamic_axes is None:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True
        )
        
        logger.info(f"Model exported to ONNX: {output_path}")
    
    @staticmethod
    def export_to_torchscript(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str,
        method: str = "trace",
        device: Optional[torch.device] = None
    ):
        """
        Export model to TorchScript
        
        Args:
            model: Model to export
            input_shape: Input shape
            output_path: Output path
            method: Export method (trace or script)
            device: Device to use
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        
        dummy_input = torch.randn(input_shape).to(device)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if method == "trace":
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(output_path)
        elif method == "script":
            scripted_model = torch.jit.script(model)
            scripted_model.save(output_path)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Model exported to TorchScript: {output_path}")
    
    @staticmethod
    def export_to_tensorrt(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str,
        precision: str = "fp16"
    ):
        """
        Export model to TensorRT (requires tensorrt package)
        
        Args:
            model: Model to export
            input_shape: Input shape
            output_path: Output path
            precision: Precision (fp32, fp16, int8)
        """
        try:
            import tensorrt as trt
            logger.info("TensorRT export requires additional setup")
            logger.warning("TensorRT export not fully implemented - use ONNX conversion first")
        except ImportError:
            logger.warning("TensorRT not available - install tensorrt package")
    
    @staticmethod
    def export_summary(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: Optional[str] = None
    ) -> str:
        """
        Export model summary
        
        Args:
            model: Model to summarize
            input_shape: Input shape
            output_path: Optional path to save summary
            
        Returns:
            Summary string
        """
        from torchsummary import summary
        
        try:
            summary_str = str(summary(model, input_shape, device='cpu'))
        except:
            # Fallback manual summary
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            summary_str = f"""
Model Summary:
- Total Parameters: {total_params:,}
- Trainable Parameters: {trainable_params:,}
- Non-trainable Parameters: {total_params - trainable_params:,}
"""
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(summary_str)
            logger.info(f"Model summary saved to {output_path}")
        
        return summary_str


def export_to_onnx(model: nn.Module, input_shape: Tuple[int, ...], output_path: str, **kwargs):
    """Export to ONNX"""
    return ModelExporter.export_to_onnx(model, input_shape, output_path, **kwargs)


def export_to_torchscript(model: nn.Module, input_shape: Tuple[int, ...], output_path: str, **kwargs):
    """Export to TorchScript"""
    return ModelExporter.export_to_torchscript(model, input_shape, output_path, **kwargs)








