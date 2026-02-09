"""
TruthGPT Advanced Testing Module
Comprehensive testing utilities for TruthGPT models
"""

import torch
import torch.nn as nn
import pytest
import unittest
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import time
import warnings

logger = logging.getLogger(__name__)

@dataclass
class TruthGPTTestingConfig:
    """Configuration for TruthGPT testing."""
    # Test settings
    enable_unit_tests: bool = True
    enable_integration_tests: bool = True
    enable_performance_tests: bool = True
    enable_property_tests: bool = True
    
    # Test data settings
    test_batch_size: int = 32
    num_test_samples: int = 1000
    test_data_dir: str = "./test_data"
    
    # Performance test settings
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'max_latency_ms': 100.0,
        'min_throughput_samples_per_sec': 10.0,
        'max_memory_mb': 2048.0
    })
    
    # Property test settings
    property_test_iterations: int = 100
    property_test_seed: int = 42
    
    # Reporting settings
    enable_coverage_reporting: bool = True
    enable_performance_reporting: bool = True
    report_path: str = "./test_reports"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enable_unit_tests': self.enable_unit_tests,
            'enable_integration_tests': self.enable_integration_tests,
            'enable_performance_tests': self.enable_performance_tests,
            'enable_property_tests': self.enable_property_tests,
            'test_batch_size': self.test_batch_size,
            'num_test_samples': self.num_test_samples,
            'test_data_dir': self.test_data_dir,
            'performance_thresholds': self.performance_thresholds,
            'property_test_iterations': self.property_test_iterations,
            'property_test_seed': self.property_test_seed,
            'enable_coverage_reporting': self.enable_coverage_reporting,
            'enable_performance_reporting': self.enable_performance_reporting,
            'report_path': self.report_path
        }

class TruthGPTUnitTester:
    """Advanced unit tester for TruthGPT."""
    
    def __init__(self, config: TruthGPTTestingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Test state
        self.test_results = []
        self.test_stats = {}
    
    def test_model_forward(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Test model forward pass."""
        self.logger.info(f"Testing model forward pass with input shape: {input_shape}")
        
        try:
            # Create dummy input
            dummy_input = torch.randn(self.config.test_batch_size, *input_shape)
            
            # Forward pass
            start_time = time.time()
            output = model(dummy_input)
            latency_ms = (time.time() - start_time) * 1000
            
            # Validate output
            assert output is not None, "Output is None"
            assert output.shape[0] == self.config.test_batch_size, "Batch size mismatch"
            
            result = {
                'test_name': 'model_forward',
                'status': 'passed',
                'latency_ms': latency_ms,
                'output_shape': output.shape
            }
            
            self.logger.info("Model forward pass test passed")
            return result
            
        except Exception as e:
            result = {
                'test_name': 'model_forward',
                'status': 'failed',
                'error': str(e)
            }
            self.logger.error(f"Model forward pass test failed: {e}")
            return result
    
    def test_model_gradients(self, model: nn.Module, loss_fn: Callable) -> Dict[str, Any]:
        """Test model gradient computation."""
        self.logger.info("Testing model gradient computation")
        
        try:
            # Create dummy input and target
            dummy_input = torch.randn(self.config.test_batch_size, 768, requires_grad=True)
            dummy_target = torch.randn(self.config.test_batch_size, 1000)
            
            # Forward pass
            output = model(dummy_input)
            loss = loss_fn(output, dummy_target)
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            has_gradients = any(p.grad is not None for p in model.parameters())
            
            result = {
                'test_name': 'model_gradients',
                'status': 'passed' if has_gradients else 'failed',
                'has_gradients': has_gradients
            }
            
            self.logger.info("Model gradient computation test passed")
            return result
            
        except Exception as e:
            result = {
                'test_name': 'model_gradients',
                'status': 'failed',
                'error': str(e)
            }
            self.logger.error(f"Model gradient computation test failed: {e}")
            return result
    
    def run_all_unit_tests(self, model: nn.Module, input_shape: Tuple[int, ...], 
                          loss_fn: Callable) -> List[Dict[str, Any]]:
        """Run all unit tests."""
        results = []
        
        # Test forward pass
        forward_result = self.test_model_forward(model, input_shape)
        results.append(forward_result)
        
        # Test gradients
        gradient_result = self.test_model_gradients(model, loss_fn)
        results.append(gradient_result)
        
        # Store results
        self.test_results.extend(results)
        
        return results
    
    def get_test_stats(self) -> Dict[str, Any]:
        """Get test statistics."""
        if not self.test_results:
            return {}
        
        passed = sum(1 for r in self.test_results if r['status'] == 'passed')
        failed = sum(1 for r in self.test_results if r['status'] == 'failed')
        
        return {
            'total_tests': len(self.test_results),
            'passed': passed,
            'failed': failed,
            'success_rate': passed / len(self.test_results) if self.test_results else 0
        }

class TruthGPTPerformanceTester:
    """Advanced performance tester for TruthGPT."""
    
    def __init__(self, config: TruthGPTTestingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance test state
        self.performance_results = []
    
    def test_latency(self, model: nn.Module, input_shape: Tuple[int, ...], 
                     num_iterations: int = 100) -> Dict[str, Any]:
        """Test model latency."""
        self.logger.info(f"Testing model latency with {num_iterations} iterations")
        
        latencies = []
        model.eval()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                dummy_input = torch.randn(1, *input_shape)
                
                start_time = time.perf_counter()
                _ = model(dummy_input)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
        
        avg_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # Check threshold
        max_latency = self.config.performance_thresholds['max_latency_ms']
        passed = avg_latency <= max_latency
        
        result = {
            'test_name': 'latency',
            'status': 'passed' if passed else 'failed',
            'avg_latency_ms': avg_latency,
            'median_latency_ms': median_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'threshold_ms': max_latency
        }
        
        self.logger.info(f"Latency test {'passed' if passed else 'failed'}: {avg_latency:.2f} ms")
        return result
    
    def test_throughput(self, model: nn.Module, input_shape: Tuple[int, ...],
                       duration_seconds: int = 10) -> Dict[str, Any]:
        """Test model throughput."""
        self.logger.info(f"Testing model throughput for {duration_seconds} seconds")
        
        samples_processed = 0
        model.eval()
        
        start_time = time.time()
        with torch.no_grad():
            while time.time() - start_time < duration_seconds:
                dummy_input = torch.randn(1, *input_shape)
                _ = model(dummy_input)
                samples_processed += 1
        
        elapsed_time = time.time() - start_time
        throughput = samples_processed / elapsed_time
        
        # Check threshold
        min_throughput = self.config.performance_thresholds['min_throughput_samples_per_sec']
        passed = throughput >= min_throughput
        
        result = {
            'test_name': 'throughput',
            'status': 'passed' if passed else 'failed',
            'throughput_samples_per_sec': throughput,
            'samples_processed': samples_processed,
            'duration_seconds': elapsed_time,
            'threshold_samples_per_sec': min_throughput
        }
        
        self.logger.info(f"Throughput test {'passed' if passed else 'failed'}: {throughput:.2f} samples/sec")
        return result
    
    def run_all_performance_tests(self, model: nn.Module, input_shape: Tuple[int, ...]) -> List[Dict[str, Any]]:
        """Run all performance tests."""
        results = []
        
        # Test latency
        latency_result = self.test_latency(model, input_shape)
        results.append(latency_result)
        
        # Test throughput
        throughput_result = self.test_throughput(model, input_shape)
        results.append(throughput_result)
        
        # Store results
        self.performance_results.extend(results)
        
        return results

class TruthGPTPropertyTester:
    """Advanced property tester for TruthGPT."""
    
    def __init__(self, config: TruthGPTTestingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Property test state
        self.property_results = []
    
    def test_output_consistency(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Test output consistency."""
        self.logger.info("Testing model output consistency")
        
        try:
            # Test multiple forward passes
            model.eval()
            outputs = []
            
            with torch.no_grad():
                for _ in range(10):
                    dummy_input = torch.randn(1, *input_shape)
                    output = model(dummy_input)
                    outputs.append(output)
            
            # Check consistency
            output_variance = np.var([o.numpy().flatten() for o in outputs])
            is_consistent = output_variance < 1e-6  # Very small variance
            
            result = {
                'test_name': 'output_consistency',
                'status': 'passed' if is_consistent else 'failed',
                'output_variance': output_variance
            }
            
            self.logger.info(f"Output consistency test {'passed' if is_consistent else 'failed'}")
            return result
            
        except Exception as e:
            result = {
                'test_name': 'output_consistency',
                'status': 'failed',
                'error': str(e)
            }
            self.logger.error(f"Output consistency test failed: {e}")
            return result
    
    def test_gradient_properties(self, model: nn.Module, loss_fn: Callable) -> Dict[str, Any]:
        """Test gradient properties."""
        self.logger.info("Testing model gradient properties")
        
        try:
            # Create dummy input
            dummy_input = torch.randn(2, 768, requires_grad=True)
            dummy_target = torch.randn(2, 1000)
            
            # Forward pass
            output = model(dummy_input)
            loss = loss_fn(output, dummy_target)
            
            # Backward pass
            loss.backward()
            
            # Check gradient properties
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            
            if not gradients:
                result = {
                    'test_name': 'gradient_properties',
                    'status': 'failed',
                    'error': 'No gradients found'
                }
                return result
            
            # Check for NaN/Inf
            has_nan = any(torch.isnan(g).any() for g in gradients)
            has_inf = any(torch.isinf(g).any() for g in gradients)
            
            # Check gradient norms
            gradient_norms = [g.norm().item() for g in gradients]
            avg_gradient_norm = np.mean(gradient_norms)
            
            result = {
                'test_name': 'gradient_properties',
                'status': 'passed' if not (has_nan or has_inf) else 'failed',
                'has_nan': has_nan,
                'has_inf': has_inf,
                'avg_gradient_norm': avg_gradient_norm
            }
            
            self.logger.info(f"Gradient properties test {'passed' if result['status'] == 'passed' else 'failed'}")
            return result
            
        except Exception as e:
            result = {
                'test_name': 'gradient_properties',
                'status': 'failed',
                'error': str(e)
            }
            self.logger.error(f"Gradient properties test failed: {e}")
            return result

class TruthGPTTestingManager:
    """Advanced testing manager for TruthGPT."""
    
    def __init__(self, config: TruthGPTTestingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Testing components
        self.unit_tester = TruthGPTUnitTester(config) if config.enable_unit_tests else None
        self.performance_tester = TruthGPTPerformanceTester(config) if config.enable_performance_tests else None
        self.property_tester = TruthGPTPropertyTester(config) if config.enable_property_tests else None
        
        # Testing state
        self.all_test_results = []
    
    def run_all_tests(self, model: nn.Module, input_shape: Tuple[int, ...],
                     loss_fn: Callable) -> Dict[str, Any]:
        """Run all tests."""
        self.logger.info("🚀 Running comprehensive TruthGPT tests")
        
        results = {
            'timestamp': time.time(),
            'unit_tests': [],
            'performance_tests': [],
            'property_tests': []
        }
        
        # Run unit tests
        if self.unit_tester:
            unit_results = self.unit_tester.run_all_unit_tests(model, input_shape, loss_fn)
            results['unit_tests'] = unit_results
        
        # Run performance tests
        if self.performance_tester:
            performance_results = self.performance_tester.run_all_performance_tests(model, input_shape)
            results['performance_tests'] = performance_results
        
        # Run property tests
        if self.property_tester:
            consistency_result = self.property_tester.test_output_consistency(model, input_shape)
            gradient_result = self.property_tester.test_gradient_properties(model, loss_fn)
            results['property_tests'] = [consistency_result, gradient_result]
        
        self.all_test_results.append(results)
        
        # Generate summary
        summary = self._generate_test_summary(results)
        results['summary'] = summary
        
        self.logger.info(f"✅ Testing completed - Success rate: {summary['success_rate']:.2%}")
        return results
    
    def _generate_test_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test summary."""
        total_tests = 0
        passed_tests = 0
        
        for test_category, test_results in results.items():
            if isinstance(test_results, list):
                for test_result in test_results:
                    total_tests += 1
                    if test_result.get('status') == 'passed':
                        passed_tests += 1
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': success_rate
        }

# Factory functions
def create_truthgpt_testing_manager(config: TruthGPTTestingConfig) -> TruthGPTTestingManager:
    """Create TruthGPT testing manager."""
    return TruthGPTTestingManager(config)

def test_truthgpt_model(model: nn.Module, input_shape: Tuple[int, ...], 
                       loss_fn: Callable, config: TruthGPTTestingConfig) -> Dict[str, Any]:
    """Quick test TruthGPT model."""
    manager = create_truthgpt_testing_manager(config)
    return manager.run_all_tests(model, input_shape, loss_fn)

# Example usage
if __name__ == "__main__":
    # Example TruthGPT testing
    print("🚀 TruthGPT Advanced Testing Demo")
    print("=" * 50)
    
    # Create testing configuration
    config = TruthGPTTestingConfig(
        enable_unit_tests=True,
        enable_performance_tests=True,
        enable_property_tests=True
    )
    
    # Create testing manager
    manager = create_truthgpt_testing_manager(config)
    
    # Create sample model
    class TruthGPTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(768, 1000)
        
        def forward(self, x):
            return self.linear(x)
    
    model = TruthGPTModel()
    loss_fn = nn.MSELoss()
    
    # Run tests
    results = manager.run_all_tests(model, (768,), loss_fn)
    print(f"Test results: {results['summary']}")
    
    print("✅ TruthGPT testing demo completed!")
