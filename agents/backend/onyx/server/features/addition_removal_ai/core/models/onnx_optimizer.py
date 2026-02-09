"""
ONNX Export and Optimization for Ultra-Fast Inference
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX not available. Install: pip install onnx onnxruntime")


class ONNXOptimizer:
    """ONNX model optimizer for fast inference"""
    
    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        example_input: torch.Tensor,
        output_path: str,
        opset_version: int = 14,
        dynamic_axes: Optional[dict] = None,
        optimize: bool = True
    ) -> str:
        """
        Export PyTorch model to ONNX
        
        Args:
            model: PyTorch model
            example_input: Example input tensor
            output_path: Output ONNX file path
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes configuration
            optimize: Optimize ONNX model
            
        Returns:
            Path to exported ONNX model
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available")
        
        model.eval()
        
        # Export
        torch.onnx.export(
            model,
            example_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes or {}
        )
        
        # Optimize
        if optimize:
            try:
                onnx_model = onnx.load(output_path)
                optimized_model = onnx.optimizer.optimize_model(onnx_model)
                optimized_model.save_model_to_file(output_path)
                logger.info(f"ONNX model optimized and saved to {output_path}")
            except Exception as e:
                logger.warning(f"ONNX optimization failed: {e}")
        
        logger.info(f"Model exported to ONNX: {output_path}")
        return output_path
    
    @staticmethod
    def create_onnx_session(
        model_path: str,
        providers: Optional[list] = None,
        session_options: Optional[dict] = None
    ):
        """
        Create ONNXRuntime session
        
        Args:
            model_path: Path to ONNX model
            providers: Execution providers
            session_options: Session options
            
        Returns:
            ONNXRuntime session
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available")
        
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        opts = ort.SessionOptions()
        if session_options:
            for key, value in session_options.items():
                setattr(opts, key, value)
        
        # Enable optimizations
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True
        
        session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=providers
        )
        
        logger.info(f"ONNX session created with providers: {providers}")
        return session
    
    @staticmethod
    def quantize_onnx(
        model_path: str,
        output_path: str,
        quantization_type: str = "dynamic"
    ) -> str:
        """
        Quantize ONNX model
        
        Args:
            model_path: Input ONNX model path
            output_path: Output quantized model path
            quantization_type: "dynamic" or "static"
            
        Returns:
            Path to quantized model
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available")
        
        try:
            from onnxruntime.quantization import quantize_dynamic, quantize_static
            
            if quantization_type == "dynamic":
                quantize_dynamic(
                    model_path,
                    output_path,
                    weight_type=onnx.TensorProto.INT8
                )
            else:
                # Static quantization requires calibration data
                logger.warning("Static quantization requires calibration data")
                quantize_dynamic(
                    model_path,
                    output_path,
                    weight_type=onnx.TensorProto.INT8
                )
            
            logger.info(f"ONNX model quantized: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"ONNX quantization failed: {e}")
            return model_path


class ONNXInference:
    """ONNX inference wrapper"""
    
    def __init__(self, model_path: str, use_gpu: bool = True):
        """
        Initialize ONNX inference
        
        Args:
            model_path: Path to ONNX model
            use_gpu: Use GPU execution provider
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        
        self.session = ONNXOptimizer.create_onnx_session(
            model_path,
            providers=providers
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def __call__(self, input_data):
        """Run inference"""
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu().numpy()
        
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        return outputs[0]


def export_model_to_onnx(
    model: nn.Module,
    example_input: torch.Tensor,
    output_path: str,
    optimize: bool = True
) -> str:
    """Export model to ONNX"""
    return ONNXOptimizer.export_to_onnx(model, example_input, output_path, optimize=optimize)

