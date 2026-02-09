"""
ONNX Export and Optimization for Recovery Models
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
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


def export_to_onnx(
    model: nn.Module,
    input_shape: tuple,
    output_path: str,
    device: Optional[torch.device] = None,
    opset_version: int = 14,
    dynamic_axes: Optional[Dict] = None
) -> bool:
    """
    Export PyTorch model to ONNX
    
    Args:
        model: PyTorch model
        input_shape: Input shape (batch_size, ...)
        output_path: Output ONNX file path
        device: Device to use
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes for variable batch size
    
    Returns:
        True if successful
    """
    if not ONNX_AVAILABLE:
        logger.error("ONNX not available")
        return False
    
    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes
        )
        
        logger.info(f"Model exported to ONNX: {output_path}")
        return True
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return False


def optimize_onnx(
    input_path: str,
    output_path: str,
    optimization_level: str = "all"
) -> bool:
    """
    Optimize ONNX model
    
    Args:
        input_path: Input ONNX file
        output_path: Output ONNX file
        optimization_level: Optimization level (basic, extended, all)
    
    Returns:
        True if successful
    """
    if not ONNX_AVAILABLE:
        logger.error("ONNX not available")
        return False
    
    try:
        # Load model
        model = onnx.load(input_path)
        
        # Optimize
        if optimization_level == "all":
            optimized_model = onnx.optimizer.optimize(model, [
                "eliminate_nop_transpose",
                "eliminate_nop_pad",
                "fuse_matmul_add_bias_into_gemm",
                "fuse_transpose_into_gemm",
                "fuse_bn_into_conv",
                "fuse_consecutive_concats",
                "fuse_consecutive_reduce_unsqueeze",
                "fuse_consecutive_squeezes",
                "fuse_consecutive_transposes",
                "fuse_transpose_into_gemm"
            ])
        else:
            optimized_model = model
        
        # Save
        onnx.save(optimized_model, output_path)
        logger.info(f"ONNX model optimized: {output_path}")
        return True
    except Exception as e:
        logger.error(f"ONNX optimization failed: {e}")
        return False


class ONNXRuntimeInference:
    """ONNX Runtime inference wrapper"""
    
    def __init__(self, onnx_path: str, providers: Optional[list] = None):
        """
        Initialize ONNX Runtime inference
        
        Args:
            onnx_path: Path to ONNX model
            providers: Execution providers (e.g., ["CUDAExecutionProvider", "CPUExecutionProvider"])
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available")
        
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        
        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        logger.info(f"ONNX Runtime session created: {onnx_path}")
    
    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Run inference
        
        Args:
            input_data: Input tensor
        
        Returns:
            Output tensor
        """
        # Convert to numpy
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu().numpy()
        
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        
        return torch.tensor(outputs[0])
    
    def benchmark(self, input_shape: tuple, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark ONNX inference"""
        import time
        
        dummy_input = torch.randn(input_shape).cpu().numpy()
        
        # Warmup
        for _ in range(10):
            _ = self.predict(dummy_input)
        
        # Benchmark
        start = time.time()
        for _ in range(num_runs):
            _ = self.predict(dummy_input)
        elapsed = time.time() - start
        
        return {
            "avg_time_ms": (elapsed / num_runs) * 1000,
            "throughput": num_runs / elapsed
        }

