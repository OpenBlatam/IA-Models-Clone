"""
Model Testing

Utilities for testing models.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ModelTester:
    """Test models for correctness and performance."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize model tester.
        
        Args:
            device: Device to test on
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_forward(
        self,
        model: nn.Module,
        input_shape: tuple,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Test model forward pass.
        
        Args:
            model: Model to test
            input_shape: Input tensor shape
            **kwargs: Additional arguments
            
        Returns:
            Test results
        """
        model = model.to(self.device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=self.device)
        
        try:
            with torch.no_grad():
                output = model(dummy_input, **kwargs)
            
            return {
                'success': True,
                'output_shape': output.shape if isinstance(output, torch.Tensor) else None,
                'output_type': type(output).__name__
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_gradients(
        self,
        model: nn.Module,
        input_shape: tuple,
        loss_fn: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Test model gradients.
        
        Args:
            model: Model to test
            input_shape: Input tensor shape
            loss_fn: Loss function (uses dummy if None)
            
        Returns:
            Test results
        """
        model = model.to(self.device)
        model.train()
        
        # Create dummy input and target
        dummy_input = torch.randn(input_shape, device=self.device)
        dummy_target = torch.randn(input_shape, device=self.device)
        
        try:
            output = model(dummy_input)
            
            if loss_fn is None:
                loss = ((output - dummy_target) ** 2).mean()
            else:
                loss = loss_fn(output, dummy_target)
            
            loss.backward()
            
            # Check gradients
            has_gradients = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.parameters()
            )
            
            return {
                'success': True,
                'has_gradients': has_gradients,
                'loss_value': loss.item()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_inference(
        self,
        model: nn.Module,
        input_shape: tuple,
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Test model inference performance.
        
        Args:
            model: Model to test
            input_shape: Input tensor shape
            num_iterations: Number of iterations
            
        Returns:
            Performance metrics
        """
        import time
        
        model = model.to(self.device)
        model.eval()
        
        dummy_input = torch.randn(input_shape, device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if self.device.type == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    _ = model(dummy_input)
                    end.record()
                    torch.cuda.synchronize()
                    elapsed = start.elapsed_time(end) / 1000
                else:
                    start = time.time()
                    _ = model(dummy_input)
                    elapsed = time.time() - start
                
                times.append(elapsed)
        
        return {
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'throughput': input_shape[0] / (sum(times) / len(times))
        }


def test_model_forward(
    model: nn.Module,
    input_shape: tuple,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to test forward pass."""
    tester = ModelTester()
    return tester.test_forward(model, input_shape, **kwargs)


def test_model_gradients(
    model: nn.Module,
    input_shape: tuple,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to test gradients."""
    tester = ModelTester()
    return tester.test_gradients(model, input_shape, **kwargs)


def test_model_inference(
    model: nn.Module,
    input_shape: tuple,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to test inference."""
    tester = ModelTester()
    return tester.test_inference(model, input_shape, **kwargs)



