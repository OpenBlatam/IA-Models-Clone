"""
Model Export Utilities
Export models to ONNX, TorchScript, and other formats
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import onnx
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ModelExporter:
    """
    Export PyTorch models to various formats
    """
    
    @staticmethod
    def export_onnx(
        model: nn.Module,
        output_path: Path,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        input_names: Optional[list] = None,
        output_names: Optional[list] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 11,
        device: torch.device = None,
    ) -> Path:
        """
        Export model to ONNX format
        
        Args:
            model: Model to export
            output_path: Output file path
            input_shape: Input tensor shape
            input_names: Input tensor names
            output_names: Output tensor names
            dynamic_axes: Dynamic axes for variable input sizes
            opset_version: ONNX opset version
            device: Device to export on
            
        Returns:
            Path to exported model
        """
        if device is None:
            device = torch.device('cpu')
        
        model.eval()
        model = model.to(device)
        
        if input_names is None:
            input_names = ['input']
        if output_names is None:
            output_names = ['output']
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(device)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=False,
            )
            logger.info(f"Exported ONNX model to {output_path}")
            
            # Verify ONNX model
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model verification passed")
            
            return output_path
        except Exception as e:
            logger.error(f"Error exporting ONNX model: {e}")
            raise
    
    @staticmethod
    def export_torchscript(
        model: nn.Module,
        output_path: Path,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        device: torch.device = None,
        method: str = "trace",
    ) -> Path:
        """
        Export model to TorchScript format
        
        Args:
            model: Model to export
            output_path: Output file path
            input_shape: Input tensor shape
            device: Device to export on
            method: 'trace' or 'script'
            
        Returns:
            Path to exported model
        """
        if device is None:
            device = torch.device('cpu')
        
        model.eval()
        model = model.to(device)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if method == "trace":
                dummy_input = torch.randn(input_shape).to(device)
                traced_model = torch.jit.trace(model, dummy_input)
            elif method == "script":
                traced_model = torch.jit.script(model)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            traced_model.save(str(output_path))
            logger.info(f"Exported TorchScript model to {output_path}")
            
            return output_path
        except Exception as e:
            logger.error(f"Error exporting TorchScript model: {e}")
            raise
    
    @staticmethod
    def test_onnx_model(
        onnx_path: Path,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> Dict[str, Any]:
        """
        Test exported ONNX model
        
        Args:
            onnx_path: Path to ONNX model
            input_shape: Input tensor shape
            
        Returns:
            Dictionary with test results
        """
        try:
            import numpy as np
            
            # Create dummy input
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Create ONNX Runtime session
            session = ort.InferenceSession(str(onnx_path))
            
            # Get input/output names
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            # Run inference
            outputs = session.run([output_name], {input_name: dummy_input})
            
            return {
                'success': True,
                'output_shape': outputs[0].shape,
                'input_name': input_name,
                'output_name': output_name,
            }
        except Exception as e:
            logger.error(f"Error testing ONNX model: {e}")
            return {
                'success': False,
                'error': str(e),
            }



