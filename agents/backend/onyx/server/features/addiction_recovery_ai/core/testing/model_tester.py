"""
Model Testing Utilities
Comprehensive testing for models
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ModelTester:
    """
    Comprehensive model testing
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize model tester
        
        Args:
            device: Device to use
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_forward_pass(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_tests: int = 10
    ) -> Dict[str, Any]:
        """
        Test forward pass
        
        Args:
            model: Model to test
            input_shape: Input shape
            num_tests: Number of test runs
            
        Returns:
            Test results
        """
        model = model.to(self.device).eval()
        results = {
            "passed": True,
            "errors": [],
            "times": []
        }
        
        for i in range(num_tests):
            try:
                dummy_input = torch.randn(input_shape).to(self.device)
                
                import time
                start = time.perf_counter()
                with torch.inference_mode():
                    output = model(dummy_input)
                end = time.perf_counter()
                
                results["times"].append((end - start) * 1000)  # ms
                
                # Check output
                if torch.isnan(output).any():
                    results["passed"] = False
                    results["errors"].append(f"Test {i}: NaN in output")
                
                if torch.isinf(output).any():
                    results["passed"] = False
                    results["errors"].append(f"Test {i}: Inf in output")
                
            except Exception as e:
                results["passed"] = False
                results["errors"].append(f"Test {i}: {str(e)}")
        
        if results["times"]:
            results["mean_time_ms"] = np.mean(results["times"])
            results["std_time_ms"] = np.std(results["times"])
        
        return results
    
    def test_gradient_flow(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        target_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """
        Test gradient flow
        
        Args:
            model: Model to test
            input_shape: Input shape
            target_shape: Target shape
            
        Returns:
            Test results
        """
        model = model.to(self.device).train()
        results = {
            "passed": True,
            "errors": [],
            "gradient_norms": []
        }
        
        try:
            dummy_input = torch.randn(input_shape).to(self.device)
            dummy_target = torch.randn(target_shape).to(self.device)
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()
            
            # Check gradients
            total_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    results["gradient_norms"].append((name, param_norm.item()))
                    
                    if torch.isnan(param.grad).any():
                        results["passed"] = False
                        results["errors"].append(f"NaN gradient in {name}")
                    
                    if torch.isinf(param.grad).any():
                        results["passed"] = False
                        results["errors"].append(f"Inf gradient in {name}")
                else:
                    results["errors"].append(f"No gradient for {name}")
            
            results["total_gradient_norm"] = total_norm ** 0.5
            
        except Exception as e:
            results["passed"] = False
            results["errors"].append(f"Gradient test failed: {str(e)}")
        
        return results
    
    def test_batch_processing(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        batch_sizes: List[int] = [1, 4, 8, 16, 32]
    ) -> Dict[str, Any]:
        """
        Test batch processing
        
        Args:
            model: Model to test
            input_shape: Input shape (without batch dimension)
            batch_sizes: List of batch sizes to test
            
        Returns:
            Test results
        """
        model = model.to(self.device).eval()
        results = {
            "passed": True,
            "errors": [],
            "batch_results": {}
        }
        
        for batch_size in batch_sizes:
            try:
                batch_shape = (batch_size, *input_shape)
                dummy_input = torch.randn(batch_shape).to(self.device)
                
                with torch.inference_mode():
                    output = model(dummy_input)
                
                # Check output batch size
                if output.shape[0] != batch_size:
                    results["passed"] = False
                    results["errors"].append(
                        f"Batch size {batch_size}: output batch size mismatch"
                    )
                
                results["batch_results"][batch_size] = {
                    "output_shape": list(output.shape),
                    "success": True
                }
                
            except Exception as e:
                results["passed"] = False
                results["errors"].append(f"Batch size {batch_size}: {str(e)}")
                results["batch_results"][batch_size] = {"success": False, "error": str(e)}
        
        return results
    
    def test_memory_usage(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Test memory usage
        
        Args:
            model: Model to test
            input_shape: Input shape
            num_iterations: Number of iterations
            
        Returns:
            Test results
        """
        model = model.to(self.device).eval()
        results = {
            "passed": True,
            "errors": [],
            "memory_usage": []
        }
        
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            
            for i in range(num_iterations):
                dummy_input = torch.randn(input_shape).to(self.device)
                
                with torch.inference_mode():
                    _ = model(dummy_input)
                
                memory_mb = torch.cuda.memory_allocated() / 1024**2
                results["memory_usage"].append(memory_mb)
            
            results["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2
            results["mean_memory_mb"] = np.mean(results["memory_usage"])
        else:
            results["errors"].append("Memory testing only available on CUDA")
        
        return results


def create_model_tester(device: Optional[torch.device] = None) -> ModelTester:
    """Factory for model tester"""
    return ModelTester(device)













