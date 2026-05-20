"""
Model Serving Optimizations

Optimizations for production model serving.
"""

import logging
import torch
from typing import Optional, Dict, Any, List
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class ModelServer:
    """
    Optimized model server for production.
    
    Features:
    - Model caching
    - Batch processing
    - Async inference
    - Health checks
    - Metrics tracking
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        max_batch_size: int = 32,
        warmup_steps: int = 10
    ):
        """
        Initialize model server.
        
        Args:
            model: PyTorch model
            device: Device to run on
            max_batch_size: Maximum batch size
            warmup_steps: Number of warmup steps
        """
        self.model = model
        self.device = device
        self.max_batch_size = max_batch_size
        self.warmup_steps = warmup_steps
        
        # Move model to device and set to eval
        self.model = self.model.to(device)
        self.model.eval()
        
        # Warmup
        self._warmup()
        
        # Metrics
        self.request_count = 0
        self.total_latency = 0.0
        self.batch_count = 0
        
        logger.info(f"Model server initialized on {device}")
    
    def _warmup(self) -> None:
        """Warmup model for faster inference."""
        logger.info(f"Warming up model with {self.warmup_steps} steps...")
        
        # Create dummy input
        dummy_input = torch.randn(1, 10).to(self.device)
        
        with torch.no_grad():
            for _ in range(self.warmup_steps):
                _ = self.model(dummy_input)
        
        # Clear cache
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        logger.info("Model warmup complete")
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Single prediction.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Output tensor
        """
        start_time = time.time()
        
        inputs = inputs.to(self.device, non_blocking=True)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                outputs = self.model(inputs)
        
        latency = time.time() - start_time
        
        self.request_count += 1
        self.total_latency += latency
        
        return outputs
    
    def predict_batch(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Batch prediction.
        
        Args:
            inputs: List of input tensors
            
        Returns:
            List of output tensors
        """
        start_time = time.time()
        
        # Pad to same length if needed
        max_len = max(inp.shape[0] for inp in inputs)
        padded = []
        for inp in inputs:
            if inp.shape[0] < max_len:
                pad = torch.zeros(max_len - inp.shape[0], *inp.shape[1:])
                padded.append(torch.cat([inp, pad]))
            else:
                padded.append(inp)
        
        # Stack and move to device
        batch = torch.stack(padded).to(self.device, non_blocking=True)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                outputs = self.model(batch)
        
        # Convert back to list
        results = [out for out in outputs]
        
        latency = time.time() - start_time
        
        self.request_count += len(inputs)
        self.batch_count += 1
        self.total_latency += latency
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics."""
        avg_latency = (
            self.total_latency / self.request_count 
            if self.request_count > 0 else 0.0
        )
        
        return {
            "request_count": self.request_count,
            "batch_count": self.batch_count,
            "total_latency": self.total_latency,
            "avg_latency": avg_latency,
            "throughput": self.request_count / self.total_latency if self.total_latency > 0 else 0.0
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Test inference
            dummy_input = torch.randn(1, 10).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            return {
                "status": "healthy",
                "device": str(self.device),
                "model_loaded": True
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


def export_to_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    opset_version: int = 14,
    dynamic_axes: Optional[Dict[str, List[int]]] = None
) -> None:
    """
    Export model to ONNX for faster inference.
    
    Args:
        model: PyTorch model
        dummy_input: Example input
        output_path: Output path for ONNX model
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes for variable input sizes
    """
    model.eval()
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes
    )
    
    logger.info(f"Model exported to ONNX: {output_path}")


def export_to_torchscript(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    method: str = "script"
) -> None:
    """
    Export model to TorchScript for faster inference.
    
    Args:
        model: PyTorch model
        dummy_input: Example input
        output_path: Output path for TorchScript model
        method: Export method ("script" or "trace")
    """
    model.eval()
    
    if method == "script":
        scripted = torch.jit.script(model)
    else:
        scripted = torch.jit.trace(model, dummy_input)
    
    scripted.save(str(output_path))
    logger.info(f"Model exported to TorchScript: {output_path}")













