"""
Model Compiler
==============

Advanced model compilation utilities.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelCompiler:
    """
    Advanced model compiler with multiple backends.
    
    Features:
    - torch.compile with different modes
    - TorchScript optimization
    - ONNX export
    - TensorRT (if available)
    """
    
    @staticmethod
    def compile_for_inference(
        model: nn.Module,
        example_input: torch.Tensor,
        backend: str = "torch_compile",
        mode: str = "reduce-overhead"
    ) -> nn.Module:
        """
        Compile model for inference.
        
        Args:
            model: PyTorch model
            example_input: Example input
            backend: Compilation backend
            mode: Compilation mode
        
        Returns:
            Compiled model
        """
        model.eval()
        
        if backend == "torch_compile":
            if hasattr(torch, 'compile'):
                return torch.compile(model, mode=mode, fullgraph=True)
            else:
                logger.warning("torch.compile not available")
                return model
        
        elif backend == "torchscript":
            try:
                with torch.no_grad():
                    traced = torch.jit.trace(model, example_input)
                    optimized = torch.jit.optimize_for_inference(traced)
                    return optimized
            except Exception as e:
                logger.warning(f"TorchScript failed: {str(e)}")
                return model
        
        elif backend == "onnx":
            # ONNX export would be handled separately
            logger.info("ONNX export should be done separately")
            return model
        
        else:
            logger.warning(f"Unknown backend: {backend}")
            return model
    
    @staticmethod
    def export_to_onnx_runtime(
        model: nn.Module,
        example_input: torch.Tensor,
        output_path: str,
        opset_version: int = 11,
        dynamic_axes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Export to ONNX for ONNX Runtime.
        
        Args:
            model: PyTorch model
            example_input: Example input
            output_path: Output path
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes
        """
        try:
            import torch.onnx
            
            model.eval()
            
            dynamic_axes = dynamic_axes or {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
            
            torch.onnx.export(
                model,
                example_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes
            )
            
            logger.info(f"Model exported to ONNX: {output_path}")
        except ImportError:
            logger.error("ONNX export requires torch.onnx")
        except Exception as e:
            logger.error(f"ONNX export failed: {str(e)}")




