"""
Model Testing
Automated testing for models
"""

from typing import Dict, Any, Optional, List
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelTester:
    """
    Automated testing for models
    """
    
    def __init__(self):
        self.test_results: List[Dict[str, Any]] = []
    
    def test_model_forward(
        self,
        model: nn.Module,
        input_shape: tuple,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Test model forward pass"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        try:
            model.eval()
            dummy_input = torch.randn(*input_shape).to(device)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            return {
                "test": "forward_pass",
                "status": "passed",
                "input_shape": input_shape,
                "output_shape": list(output.shape),
                "output_dtype": str(output.dtype)
            }
        except Exception as e:
            return {
                "test": "forward_pass",
                "status": "failed",
                "error": str(e)
            }
    
    def test_model_gradient(
        self,
        model: nn.Module,
        input_shape: tuple,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Test model gradient flow"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        try:
            model.train()
            dummy_input = torch.randn(*input_shape).to(device)
            dummy_target = torch.randint(0, 10, (input_shape[0],)).to(device)
            
            output = model(dummy_input)
            if isinstance(output, dict):
                output = output.get("genre", output.get(list(output.keys())[0]))
            
            loss = nn.CrossEntropyLoss()(output, dummy_target)
            loss.backward()
            
            # Check gradients
            has_gradients = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.parameters()
            )
            
            return {
                "test": "gradient_flow",
                "status": "passed" if has_gradients else "failed",
                "has_gradients": has_gradients
            }
        except Exception as e:
            return {
                "test": "gradient_flow",
                "status": "failed",
                "error": str(e)
            }
    
    def test_model_inference_speed(
        self,
        model: nn.Module,
        input_shape: tuple,
        num_runs: int = 100,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Test model inference speed"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        import time
        
        model.eval()
        dummy_input = torch.randn(*input_shape).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(dummy_input)
                times.append(time.time() - start)
        
        return {
            "test": "inference_speed",
            "status": "passed",
            "mean_latency_ms": float(np.mean(times) * 1000),
            "std_latency_ms": float(np.std(times) * 1000),
            "min_latency_ms": float(np.min(times) * 1000),
            "max_latency_ms": float(np.max(times) * 1000),
            "throughput": float(1.0 / np.mean(times))
        }
    
    def run_all_tests(
        self,
        model: nn.Module,
        input_shape: tuple,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Run all model tests"""
        results = {
            "forward_pass": self.test_model_forward(model, input_shape, device),
            "gradient_flow": self.test_model_gradient(model, input_shape, device),
            "inference_speed": self.test_model_inference_speed(model, input_shape, device=device)
        }
        
        all_passed = all(r.get("status") == "passed" for r in results.values())
        
        return {
            "all_tests_passed": all_passed,
            "results": results
        }

