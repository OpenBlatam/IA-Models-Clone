"""
Test Utilities
Utilities for testing ML models and services.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a test."""
    name: str
    passed: bool
    message: str
    metrics: Optional[Dict[str, float]] = None
    duration: float = 0.0


class ModelTester:
    """Test ML models."""
    
    @staticmethod
    def test_forward_pass(
        model: torch.nn.Module,
        input_shape: tuple,
        device: str = "cpu",
    ) -> TestResult:
        """Test model forward pass."""
        import time
        start_time = time.time()
        
        try:
            model.eval()
            dummy_input = torch.randn(input_shape).to(device)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            duration = time.time() - start_time
            
            return TestResult(
                name="forward_pass",
                passed=True,
                message="Forward pass successful",
                metrics={
                    "input_shape": input_shape,
                    "output_shape": tuple(output.shape),
                    "duration": duration,
                },
                duration=duration,
            )
        except Exception as e:
            return TestResult(
                name="forward_pass",
                passed=False,
                message=f"Forward pass failed: {str(e)}",
                duration=time.time() - start_time,
            )
    
    @staticmethod
    def test_gradient_flow(
        model: torch.nn.Module,
        input_shape: tuple,
        device: str = "cpu",
    ) -> TestResult:
        """Test gradient flow."""
        import time
        start_time = time.time()
        
        try:
            model.train()
            dummy_input = torch.randn(input_shape).to(device)
            dummy_target = torch.randn(input_shape[0], 1).to(device)
            
            output = model(dummy_input)
            loss = torch.nn.functional.mse_loss(output, dummy_target)
            loss.backward()
            
            # Check gradients
            has_gradients = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.parameters()
            )
            
            duration = time.time() - start_time
            
            return TestResult(
                name="gradient_flow",
                passed=has_gradients,
                message="Gradient flow OK" if has_gradients else "No gradients detected",
                duration=duration,
            )
        except Exception as e:
            return TestResult(
                name="gradient_flow",
                passed=False,
                message=f"Gradient flow test failed: {str(e)}",
                duration=time.time() - start_time,
            )
    
    @staticmethod
    def test_model_size(
        model: torch.nn.Module,
        max_size_mb: Optional[float] = None,
    ) -> TestResult:
        """Test model size."""
        import time
        start_time = time.time()
        
        try:
            param_count = sum(p.numel() for p in model.parameters())
            trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Estimate size in MB (assuming float32)
            size_mb = param_count * 4 / (1024 * 1024)
            
            passed = True
            message = "Model size OK"
            
            if max_size_mb and size_mb > max_size_mb:
                passed = False
                message = f"Model size {size_mb:.2f}MB exceeds limit {max_size_mb}MB"
            
            return TestResult(
                name="model_size",
                passed=passed,
                message=message,
                metrics={
                    "param_count": param_count,
                    "trainable_count": trainable_count,
                    "size_mb": size_mb,
                },
                duration=time.time() - start_time,
            )
        except Exception as e:
            return TestResult(
                name="model_size",
                passed=False,
                message=f"Model size test failed: {str(e)}",
                duration=time.time() - start_time,
            )


class DataTester:
    """Test data and datasets."""
    
    @staticmethod
    def test_dataset_loading(
        dataset: Any,
        expected_size: Optional[int] = None,
    ) -> TestResult:
        """Test dataset loading."""
        import time
        start_time = time.time()
        
        try:
            size = len(dataset)
            
            passed = True
            message = f"Dataset loaded successfully (size: {size})"
            
            if expected_size and size != expected_size:
                passed = False
                message = f"Dataset size {size} != expected {expected_size}"
            
            # Try to get a sample
            sample = dataset[0]
            
            return TestResult(
                name="dataset_loading",
                passed=passed,
                message=message,
                metrics={"size": size},
                duration=time.time() - start_time,
            )
        except Exception as e:
            return TestResult(
                name="dataset_loading",
                passed=False,
                message=f"Dataset loading failed: {str(e)}",
                duration=time.time() - start_time,
            )
    
    @staticmethod
    def test_data_consistency(
        data: List[Any],
        validator: Optional[Callable[[Any], bool]] = None,
    ) -> TestResult:
        """Test data consistency."""
        import time
        start_time = time.time()
        
        try:
            if not data:
                return TestResult(
                    name="data_consistency",
                    passed=False,
                    message="Data is empty",
                    duration=time.time() - start_time,
                )
            
            if validator:
                all_valid = all(validator(item) for item in data)
                if not all_valid:
                    return TestResult(
                        name="data_consistency",
                        passed=False,
                        message="Some items failed validation",
                        duration=time.time() - start_time,
                    )
            
            return TestResult(
                name="data_consistency",
                passed=True,
                message=f"All {len(data)} items are consistent",
                metrics={"count": len(data)},
                duration=time.time() - start_time,
            )
        except Exception as e:
            return TestResult(
                name="data_consistency",
                passed=False,
                message=f"Data consistency test failed: {str(e)}",
                duration=time.time() - start_time,
            )


class ServiceTester:
    """Test ML services."""
    
    @staticmethod
    def test_service_health(
        health_check: Callable[[], Dict[str, Any]],
    ) -> TestResult:
        """Test service health."""
        import time
        start_time = time.time()
        
        try:
            health = health_check()
            status = health.get("status", "unknown")
            
            passed = status == "healthy"
            message = f"Service status: {status}"
            
            return TestResult(
                name="service_health",
                passed=passed,
                message=message,
                metrics=health,
                duration=time.time() - start_time,
            )
        except Exception as e:
            return TestResult(
                name="service_health",
                passed=False,
                message=f"Health check failed: {str(e)}",
                duration=time.time() - start_time,
            )
    
    @staticmethod
    def test_endpoint(
        endpoint: Callable,
        test_input: Any,
        expected_output_type: Optional[type] = None,
    ) -> TestResult:
        """Test service endpoint."""
        import time
        start_time = time.time()
        
        try:
            output = endpoint(test_input)
            
            passed = True
            message = "Endpoint test passed"
            
            if expected_output_type and not isinstance(output, expected_output_type):
                passed = False
                message = f"Output type {type(output)} != expected {expected_output_type}"
            
            return TestResult(
                name="endpoint_test",
                passed=passed,
                message=message,
                duration=time.time() - start_time,
            )
        except Exception as e:
            return TestResult(
                name="endpoint_test",
                passed=False,
                message=f"Endpoint test failed: {str(e)}",
                duration=time.time() - start_time,
            )


class TestSuite:
    """Run a suite of tests."""
    
    def __init__(self, name: str = "Test Suite"):
        self.name = name
        self.tests: List[Callable[[], TestResult]] = []
    
    def add_test(self, test_func: Callable[[], TestResult]):
        """Add a test to the suite."""
        self.tests.append(test_func)
    
    def run(self) -> List[TestResult]:
        """Run all tests."""
        results = []
        for test_func in self.tests:
            try:
                result = test_func()
                results.append(result)
            except Exception as e:
                results.append(TestResult(
                    name="test_execution",
                    passed=False,
                    message=f"Test execution failed: {str(e)}",
                ))
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        results = self.run()
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        
        return {
            "name": self.name,
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "results": [r.__dict__ for r in results],
        }



