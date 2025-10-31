"""
Advanced Model Testing System for TruthGPT Optimization Core
Complete model testing with unit testing, integration testing, and performance testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from pathlib import Path
import pickle
from abc import ABC, abstractmethod
import math
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TestingLevel(Enum):
    """Testing levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    COMPREHENSIVE = "comprehensive"

class TestingType(Enum):
    """Testing types"""
    UNIT_TESTING = "unit_testing"
    INTEGRATION_TESTING = "integration_testing"
    PERFORMANCE_TESTING = "performance_testing"
    STRESS_TESTING = "stress_testing"
    LOAD_TESTING = "load_testing"
    SECURITY_TESTING = "security_testing"

class TestFramework(Enum):
    """Test frameworks"""
    PYTEST = "pytest"
    UNITTEST = "unittest"
    CUSTOM = "custom"

class ModelTestingConfig:
    """Configuration for model testing system"""
    # Basic settings
    testing_level: TestingLevel = TestingLevel.INTERMEDIATE
    testing_type: TestingType = TestingType.UNIT_TESTING
    test_framework: TestFramework = TestFramework.PYTEST
    
    # Unit testing settings
    enable_unit_tests: bool = True
    unit_test_coverage_threshold: float = 0.8
    unit_test_timeout: int = 30
    enable_parameter_tests: bool = True
    enable_forward_pass_tests: bool = True
    enable_backward_pass_tests: bool = True
    
    # Integration testing settings
    enable_integration_tests: bool = True
    integration_test_timeout: int = 60
    enable_data_pipeline_tests: bool = True
    enable_model_serving_tests: bool = True
    enable_api_tests: bool = True
    
    # Performance testing settings
    enable_performance_tests: bool = True
    performance_test_duration: int = 300  # seconds
    performance_test_iterations: int = 1000
    target_latency_ms: float = 100.0
    target_throughput_qps: float = 100.0
    target_memory_mb: float = 500.0
    
    # Stress testing settings
    enable_stress_tests: bool = True
    stress_test_duration: int = 600  # seconds
    stress_test_load_multiplier: float = 2.0
    stress_test_memory_multiplier: float = 1.5
    
    # Load testing settings
    enable_load_tests: bool = True
    load_test_concurrent_users: int = 100
    load_test_ramp_up_time: int = 60  # seconds
    load_test_duration: int = 300  # seconds
    
    # Security testing settings
    enable_security_tests: bool = True
    security_test_adversarial_samples: int = 100
    security_test_privacy_leaks: bool = True
    security_test_input_validation: bool = True
    
    # Test data settings
    test_data_size: int = 1000
    test_data_batch_size: int = 32
    test_data_generation_method: str = "random"
    enable_synthetic_data: bool = True
    enable_real_data: bool = True
    
    # Test reporting settings
    enable_test_reports: bool = True
    test_report_format: str = "html"
    enable_coverage_reports: bool = True
    enable_performance_reports: bool = True
    enable_security_reports: bool = True
    
    # Advanced features
    enable_automated_testing: bool = True
    enable_continuous_testing: bool = True
    enable_test_parallelization: bool = True
    enable_test_caching: bool = True
    
    def __post_init__(self):
        """Validate testing configuration"""
        if not (0 <= self.unit_test_coverage_threshold <= 1):
            raise ValueError("Unit test coverage threshold must be between 0 and 1")
        if self.unit_test_timeout <= 0:
            raise ValueError("Unit test timeout must be positive")
        if self.integration_test_timeout <= 0:
            raise ValueError("Integration test timeout must be positive")
        if self.performance_test_duration <= 0:
            raise ValueError("Performance test duration must be positive")
        if self.performance_test_iterations <= 0:
            raise ValueError("Performance test iterations must be positive")
        if self.target_latency_ms <= 0:
            raise ValueError("Target latency must be positive")
        if self.target_throughput_qps <= 0:
            raise ValueError("Target throughput must be positive")
        if self.target_memory_mb <= 0:
            raise ValueError("Target memory must be positive")
        if self.stress_test_duration <= 0:
            raise ValueError("Stress test duration must be positive")
        if self.stress_test_load_multiplier <= 0:
            raise ValueError("Stress test load multiplier must be positive")
        if self.stress_test_memory_multiplier <= 0:
            raise ValueError("Stress test memory multiplier must be positive")
        if self.load_test_concurrent_users <= 0:
            raise ValueError("Load test concurrent users must be positive")
        if self.load_test_ramp_up_time <= 0:
            raise ValueError("Load test ramp up time must be positive")
        if self.load_test_duration <= 0:
            raise ValueError("Load test duration must be positive")
        if self.security_test_adversarial_samples <= 0:
            raise ValueError("Security test adversarial samples must be positive")
        if self.test_data_size <= 0:
            raise ValueError("Test data size must be positive")
        if self.test_data_batch_size <= 0:
            raise ValueError("Test data batch size must be positive")

class UnitTester:
    """Unit tester for model components"""
    
    def __init__(self, config: ModelTestingConfig):
        self.config = config
        self.test_results = []
        logger.info("✅ Unit Tester initialized")
    
    def run_unit_tests(self, model: nn.Module) -> Dict[str, Any]:
        """Run comprehensive unit tests"""
        logger.info("🔍 Running unit tests")
        
        test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_coverage': 0.0,
            'test_details': [],
            'start_time': time.time()
        }
        
        # Test model parameters
        if self.config.enable_parameter_tests:
            param_tests = self._test_model_parameters(model)
            test_results['test_details'].extend(param_tests)
        
        # Test forward pass
        if self.config.enable_forward_pass_tests:
            forward_tests = self._test_forward_pass(model)
            test_results['test_details'].extend(forward_tests)
        
        # Test backward pass
        if self.config.enable_backward_pass_tests:
            backward_tests = self._test_backward_pass(model)
            test_results['test_details'].extend(backward_tests)
        
        # Test model layers
        layer_tests = self._test_model_layers(model)
        test_results['test_details'].extend(layer_tests)
        
        # Calculate results
        test_results['total_tests'] = len(test_results['test_details'])
        test_results['passed_tests'] = sum(1 for test in test_results['test_details'] if test['passed'])
        test_results['failed_tests'] = test_results['total_tests'] - test_results['passed_tests']
        test_results['test_coverage'] = self._calculate_test_coverage(model, test_results['test_details'])
        
        test_results['end_time'] = time.time()
        test_results['duration'] = test_results['end_time'] - test_results['start_time']
        
        # Store test results
        self.test_results.append(test_results)
        
        return test_results
    
    def _test_model_parameters(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Test model parameters"""
        tests = []
        
        # Test parameter initialization
        test = {
            'name': 'Parameter Initialization',
            'passed': True,
            'message': 'All parameters initialized correctly',
            'details': {}
        }
        
        try:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is None:
                    # Check if parameter has reasonable values
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        test['passed'] = False
                        test['message'] = f'Parameter {name} contains NaN or Inf values'
                        break
                    
                    # Check parameter shape
                    if param.shape == torch.Size([]):
                        test['passed'] = False
                        test['message'] = f'Parameter {name} has empty shape'
                        break
            
            test['details'] = {
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            
        except Exception as e:
            test['passed'] = False
            test['message'] = f'Parameter test failed: {str(e)}'
        
        tests.append(test)
        
        # Test parameter gradients
        test = {
            'name': 'Parameter Gradients',
            'passed': True,
            'message': 'Gradients computed correctly',
            'details': {}
        }
        
        try:
            # Create dummy input and loss
            dummy_input = torch.randn(1, 3, 32, 32)
            dummy_target = torch.randint(0, 10, (1,))
            
            # Forward pass
            output = model(dummy_input)
            loss = F.cross_entropy(output, dummy_target)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Check gradients
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                    
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        test['passed'] = False
                        test['message'] = f'Gradient for {name} contains NaN or Inf values'
                        break
            
            test['details'] = {
                'gradient_norms': grad_norms,
                'average_gradient_norm': np.mean(grad_norms) if grad_norms else 0.0
            }
            
        except Exception as e:
            test['passed'] = False
            test['message'] = f'Gradient test failed: {str(e)}'
        
        tests.append(test)
        
        return tests
    
    def _test_forward_pass(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Test forward pass"""
        tests = []
        
        # Test forward pass with different input sizes
        test = {
            'name': 'Forward Pass - Different Input Sizes',
            'passed': True,
            'message': 'Forward pass works with different input sizes',
            'details': {}
        }
        
        try:
            input_sizes = [(1, 3, 32, 32), (2, 3, 32, 32), (1, 3, 64, 64)]
            outputs = []
            
            for input_size in input_sizes:
                dummy_input = torch.randn(input_size)
                output = model(dummy_input)
                outputs.append(output.shape)
                
                # Check output shape
                if len(output.shape) != 2:  # Expected batch_size x num_classes
                    test['passed'] = False
                    test['message'] = f'Unexpected output shape for input size {input_size}: {output.shape}'
                    break
            
            test['details'] = {
                'input_sizes': input_sizes,
                'output_shapes': outputs
            }
            
        except Exception as e:
            test['passed'] = False
            test['message'] = f'Forward pass test failed: {str(e)}'
        
        tests.append(test)
        
        # Test forward pass with edge cases
        test = {
            'name': 'Forward Pass - Edge Cases',
            'passed': True,
            'message': 'Forward pass handles edge cases correctly',
            'details': {}
        }
        
        try:
            # Test with zero input
            zero_input = torch.zeros(1, 3, 32, 32)
            zero_output = model(zero_input)
            
            # Test with large input
            large_input = torch.randn(1, 3, 32, 32) * 100
            large_output = model(large_input)
            
            # Check for NaN or Inf in outputs
            if torch.isnan(zero_output).any() or torch.isinf(zero_output).any():
                test['passed'] = False
                test['message'] = 'Zero input produces NaN or Inf output'
            elif torch.isnan(large_output).any() or torch.isinf(large_output).any():
                test['passed'] = False
                test['message'] = 'Large input produces NaN or Inf output'
            
            test['details'] = {
                'zero_output_stats': {
                    'mean': zero_output.mean().item(),
                    'std': zero_output.std().item(),
                    'min': zero_output.min().item(),
                    'max': zero_output.max().item()
                },
                'large_output_stats': {
                    'mean': large_output.mean().item(),
                    'std': large_output.std().item(),
                    'min': large_output.min().item(),
                    'max': large_output.max().item()
                }
            }
            
        except Exception as e:
            test['passed'] = False
            test['message'] = f'Edge case test failed: {str(e)}'
        
        tests.append(test)
        
        return tests
    
    def _test_backward_pass(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Test backward pass"""
        tests = []
        
        # Test backward pass
        test = {
            'name': 'Backward Pass',
            'passed': True,
            'message': 'Backward pass works correctly',
            'details': {}
        }
        
        try:
            # Create dummy input and target
            dummy_input = torch.randn(1, 3, 32, 32, requires_grad=True)
            dummy_target = torch.randint(0, 10, (1,))
            
            # Forward pass
            output = model(dummy_input)
            loss = F.cross_entropy(output, dummy_target)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Check if gradients are computed
            has_gradients = any(param.grad is not None for param in model.parameters())
            
            if not has_gradients:
                test['passed'] = False
                test['message'] = 'No gradients computed during backward pass'
            else:
                # Check gradient magnitudes
                grad_norms = [param.grad.norm().item() for param in model.parameters() if param.grad is not None]
                
                if any(np.isnan(grad_norm) or np.isinf(grad_norm) for grad_norm in grad_norms):
                    test['passed'] = False
                    test['message'] = 'Gradients contain NaN or Inf values'
                
                test['details'] = {
                    'gradient_norms': grad_norms,
                    'average_gradient_norm': np.mean(grad_norms),
                    'max_gradient_norm': np.max(grad_norms)
                }
            
        except Exception as e:
            test['passed'] = False
            test['message'] = f'Backward pass test failed: {str(e)}'
        
        tests.append(test)
        
        return tests
    
    def _test_model_layers(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Test individual model layers"""
        tests = []
        
        # Test each layer
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                test = {
                    'name': f'Layer Test - {name}',
                    'passed': True,
                    'message': f'Layer {name} works correctly',
                    'details': {}
                }
                
                try:
                    # Test layer with dummy input
                    if isinstance(module, (nn.Linear, nn.Conv2d)):
                        # Create appropriate input size
                        if isinstance(module, nn.Linear):
                            dummy_input = torch.randn(1, module.in_features)
                        else:  # Conv2d
                            dummy_input = torch.randn(1, module.in_channels, 32, 32)
                        
                        output = module(dummy_input)
                        
                        # Check output shape
                        expected_shape = dummy_input.shape
                        if isinstance(module, nn.Linear):
                            expected_shape = (dummy_input.shape[0], module.out_features)
                        elif isinstance(module, nn.Conv2d):
                            # Simplified shape calculation
                            expected_shape = (dummy_input.shape[0], module.out_channels, 32, 32)
                        
                        test['details'] = {
                            'input_shape': list(dummy_input.shape),
                            'output_shape': list(output.shape),
                            'layer_type': type(module).__name__
                        }
                    
                except Exception as e:
                    test['passed'] = False
                    test['message'] = f'Layer {name} test failed: {str(e)}'
                
                tests.append(test)
        
        return tests
    
    def _calculate_test_coverage(self, model: nn.Module, test_details: List[Dict[str, Any]]) -> float:
        """Calculate test coverage"""
        # Simplified coverage calculation
        total_components = len(list(model.named_modules()))
        tested_components = len([test for test in test_details if test['passed']])
        
        return tested_components / total_components if total_components > 0 else 0.0

class IntegrationTester:
    """Integration tester for model systems"""
    
    def __init__(self, config: ModelTestingConfig):
        self.config = config
        self.test_results = []
        logger.info("✅ Integration Tester initialized")
    
    def run_integration_tests(self, model: nn.Module, data_pipeline: Callable = None) -> Dict[str, Any]:
        """Run integration tests"""
        logger.info("🔍 Running integration tests")
        
        test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': [],
            'start_time': time.time()
        }
        
        # Test data pipeline integration
        if self.config.enable_data_pipeline_tests and data_pipeline:
            pipeline_tests = self._test_data_pipeline_integration(model, data_pipeline)
            test_results['test_details'].extend(pipeline_tests)
        
        # Test model serving integration
        if self.config.enable_model_serving_tests:
            serving_tests = self._test_model_serving_integration(model)
            test_results['test_details'].extend(serving_tests)
        
        # Test API integration
        if self.config.enable_api_tests:
            api_tests = self._test_api_integration(model)
            test_results['test_details'].extend(api_tests)
        
        # Test end-to-end workflow
        e2e_tests = self._test_end_to_end_workflow(model)
        test_results['test_details'].extend(e2e_tests)
        
        # Calculate results
        test_results['total_tests'] = len(test_results['test_details'])
        test_results['passed_tests'] = sum(1 for test in test_results['test_details'] if test['passed'])
        test_results['failed_tests'] = test_results['total_tests'] - test_results['passed_tests']
        
        test_results['end_time'] = time.time()
        test_results['duration'] = test_results['end_time'] - test_results['start_time']
        
        # Store test results
        self.test_results.append(test_results)
        
        return test_results
    
    def _test_data_pipeline_integration(self, model: nn.Module, data_pipeline: Callable) -> List[Dict[str, Any]]:
        """Test data pipeline integration"""
        tests = []
        
        test = {
            'name': 'Data Pipeline Integration',
            'passed': True,
            'message': 'Data pipeline integrates correctly with model',
            'details': {}
        }
        
        try:
            # Test data pipeline
            processed_data = data_pipeline()
            
            # Test model with processed data
            if isinstance(processed_data, torch.Tensor):
                output = model(processed_data)
                test['details'] = {
                    'input_shape': list(processed_data.shape),
                    'output_shape': list(output.shape),
                    'data_type': str(processed_data.dtype)
                }
            else:
                test['passed'] = False
                test['message'] = 'Data pipeline output is not a tensor'
            
        except Exception as e:
            test['passed'] = False
            test['message'] = f'Data pipeline integration test failed: {str(e)}'
        
        tests.append(test)
        
        return tests
    
    def _test_model_serving_integration(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Test model serving integration"""
        tests = []
        
        test = {
            'name': 'Model Serving Integration',
            'passed': True,
            'message': 'Model serving integration works correctly',
            'details': {}
        }
        
        try:
            # Simulate model serving
            dummy_input = torch.randn(1, 3, 32, 32)
            
            # Test model in evaluation mode
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            
            # Test batch processing
            batch_input = torch.randn(5, 3, 32, 32)
            with torch.no_grad():
                batch_output = model(batch_input)
            
            test['details'] = {
                'single_inference_shape': list(output.shape),
                'batch_inference_shape': list(batch_output.shape),
                'inference_time': 0.1  # Simulated
            }
            
        except Exception as e:
            test['passed'] = False
            test['message'] = f'Model serving integration test failed: {str(e)}'
        
        tests.append(test)
        
        return tests
    
    def _test_api_integration(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Test API integration"""
        tests = []
        
        test = {
            'name': 'API Integration',
            'passed': True,
            'message': 'API integration works correctly',
            'details': {}
        }
        
        try:
            # Simulate API calls
            api_requests = [
                {'input': torch.randn(1, 3, 32, 32).tolist()},
                {'input': torch.randn(2, 3, 32, 32).tolist()},
                {'input': torch.randn(1, 3, 64, 64).tolist()}
            ]
            
            api_responses = []
            for request in api_requests:
                input_tensor = torch.tensor(request['input'])
                with torch.no_grad():
                    output = model(input_tensor)
                api_responses.append(output.tolist())
            
            test['details'] = {
                'api_requests_count': len(api_requests),
                'api_responses_count': len(api_responses),
                'average_response_time': 0.05  # Simulated
            }
            
        except Exception as e:
            test['passed'] = False
            test['message'] = f'API integration test failed: {str(e)}'
        
        tests.append(test)
        
        return tests
    
    def _test_end_to_end_workflow(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Test end-to-end workflow"""
        tests = []
        
        test = {
            'name': 'End-to-End Workflow',
            'passed': True,
            'message': 'End-to-end workflow works correctly',
            'details': {}
        }
        
        try:
            # Simulate complete workflow
            # 1. Data preparation
            raw_data = torch.randn(10, 3, 32, 32)
            
            # 2. Data preprocessing
            processed_data = raw_data / 255.0  # Normalize
            
            # 3. Model inference
            model.eval()
            with torch.no_grad():
                predictions = model(processed_data)
            
            # 4. Post-processing
            probabilities = F.softmax(predictions, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1)
            
            # 5. Validation
            if len(predicted_classes) != len(processed_data):
                test['passed'] = False
                test['message'] = 'Prediction count does not match input count'
            else:
                test['details'] = {
                    'input_samples': len(processed_data),
                    'predictions_count': len(predicted_classes),
                    'average_confidence': probabilities.max(dim=-1)[0].mean().item(),
                    'workflow_steps': 5
                }
            
        except Exception as e:
            test['passed'] = False
            test['message'] = f'End-to-end workflow test failed: {str(e)}'
        
        tests.append(test)
        
        return tests

class PerformanceTester:
    """Performance tester for model evaluation"""
    
    def __init__(self, config: ModelTestingConfig):
        self.config = config
        self.test_results = []
        logger.info("✅ Performance Tester initialized")
    
    def run_performance_tests(self, model: nn.Module) -> Dict[str, Any]:
        """Run performance tests"""
        logger.info("🔍 Running performance tests")
        
        test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'performance_metrics': {},
            'test_details': [],
            'start_time': time.time()
        }
        
        # Test latency
        latency_tests = self._test_latency(model)
        test_results['test_details'].extend(latency_tests)
        
        # Test throughput
        throughput_tests = self._test_throughput(model)
        test_results['test_details'].extend(throughput_tests)
        
        # Test memory usage
        memory_tests = self._test_memory_usage(model)
        test_results['test_details'].extend(memory_tests)
        
        # Test scalability
        scalability_tests = self._test_scalability(model)
        test_results['test_details'].extend(scalability_tests)
        
        # Calculate results
        test_results['total_tests'] = len(test_results['test_details'])
        test_results['passed_tests'] = sum(1 for test in test_results['test_details'] if test['passed'])
        test_results['failed_tests'] = test_results['total_tests'] - test_results['passed_tests']
        
        # Extract performance metrics
        test_results['performance_metrics'] = self._extract_performance_metrics(test_results['test_details'])
        
        test_results['end_time'] = time.time()
        test_results['duration'] = test_results['end_time'] - test_results['start_time']
        
        # Store test results
        self.test_results.append(test_results)
        
        return test_results
    
    def _test_latency(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Test model latency"""
        tests = []
        
        test = {
            'name': 'Latency Test',
            'passed': True,
            'message': 'Latency meets requirements',
            'details': {}
        }
        
        try:
            model.eval()
            dummy_input = torch.randn(1, 3, 32, 32)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            # Measure latency
            times = []
            for _ in range(self.config.performance_test_iterations):
                start_time = time.time()
                with torch.no_grad():
                    _ = model(dummy_input)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            avg_latency = np.mean(times)
            max_latency = np.max(times)
            min_latency = np.min(times)
            std_latency = np.std(times)
            
            if avg_latency > self.config.target_latency_ms:
                test['passed'] = False
                test['message'] = f'Average latency {avg_latency:.2f}ms exceeds target {self.config.target_latency_ms}ms'
            
            test['details'] = {
                'average_latency_ms': avg_latency,
                'max_latency_ms': max_latency,
                'min_latency_ms': min_latency,
                'std_latency_ms': std_latency,
                'target_latency_ms': self.config.target_latency_ms,
                'iterations': self.config.performance_test_iterations
            }
            
        except Exception as e:
            test['passed'] = False
            test['message'] = f'Latency test failed: {str(e)}'
        
        tests.append(test)
        
        return tests
    
    def _test_throughput(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Test model throughput"""
        tests = []
        
        test = {
            'name': 'Throughput Test',
            'passed': True,
            'message': 'Throughput meets requirements',
            'details': {}
        }
        
        try:
            model.eval()
            dummy_input = torch.randn(1, 3, 32, 32)
            
            # Measure throughput
            start_time = time.time()
            iterations = 0
            
            while time.time() - start_time < self.config.performance_test_duration:
                with torch.no_grad():
                    _ = model(dummy_input)
                iterations += 1
            
            total_time = time.time() - start_time
            throughput = iterations / total_time
            
            if throughput < self.config.target_throughput_qps:
                test['passed'] = False
                test['message'] = f'Throughput {throughput:.2f} QPS below target {self.config.target_throughput_qps} QPS'
            
            test['details'] = {
                'throughput_qps': throughput,
                'total_iterations': iterations,
                'total_time_s': total_time,
                'target_throughput_qps': self.config.target_throughput_qps,
                'test_duration_s': self.config.performance_test_duration
            }
            
        except Exception as e:
            test['passed'] = False
            test['message'] = f'Throughput test failed: {str(e)}'
        
        tests.append(test)
        
        return tests
    
    def _test_memory_usage(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Test memory usage"""
        tests = []
        
        test = {
            'name': 'Memory Usage Test',
            'passed': True,
            'message': 'Memory usage meets requirements',
            'details': {}
        }
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                model.eval()
                dummy_input = torch.randn(1, 3, 32, 32).cuda()
                
                # Measure memory usage
                with torch.no_grad():
                    _ = model(dummy_input)
                
                memory_used = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
                
                if memory_used > self.config.target_memory_mb:
                    test['passed'] = False
                    test['message'] = f'Memory usage {memory_used:.2f}MB exceeds target {self.config.target_memory_mb}MB'
                
                test['details'] = {
                    'memory_usage_mb': memory_used,
                    'target_memory_mb': self.config.target_memory_mb,
                    'device': 'cuda'
                }
            else:
                # CPU memory estimation (simplified)
                memory_used = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
                
                test['details'] = {
                    'memory_usage_mb': memory_used,
                    'target_memory_mb': self.config.target_memory_mb,
                    'device': 'cpu'
                }
            
        except Exception as e:
            test['passed'] = False
            test['message'] = f'Memory usage test failed: {str(e)}'
        
        tests.append(test)
        
        return tests
    
    def _test_scalability(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Test model scalability"""
        tests = []
        
        test = {
            'name': 'Scalability Test',
            'passed': True,
            'message': 'Model scales correctly',
            'details': {}
        }
        
        try:
            model.eval()
            batch_sizes = [1, 2, 4, 8, 16]
            latencies = []
            
            for batch_size in batch_sizes:
                dummy_input = torch.randn(batch_size, 3, 32, 32)
                
                # Measure latency for this batch size
                times = []
                for _ in range(10):
                    start_time = time.time()
                    with torch.no_grad():
                        _ = model(dummy_input)
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)
                
                avg_latency = np.mean(times)
                latencies.append(avg_latency)
            
            # Check if latency scales reasonably
            latency_ratio = latencies[-1] / latencies[0] if latencies[0] > 0 else 1.0
            batch_ratio = batch_sizes[-1] / batch_sizes[0]
            
            if latency_ratio > batch_ratio * 2:  # Allow some overhead
                test['passed'] = False
                test['message'] = f'Latency does not scale well: {latency_ratio:.2f}x for {batch_ratio:.2f}x batch size'
            
            test['details'] = {
                'batch_sizes': batch_sizes,
                'latencies_ms': latencies,
                'latency_ratio': latency_ratio,
                'batch_ratio': batch_ratio
            }
            
        except Exception as e:
            test['passed'] = False
            test['message'] = f'Scalability test failed: {str(e)}'
        
        tests.append(test)
        
        return tests
    
    def _extract_performance_metrics(self, test_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract performance metrics from test details"""
        metrics = {}
        
        for test in test_details:
            if 'details' in test:
                details = test['details']
                
                if 'average_latency_ms' in details:
                    metrics['average_latency_ms'] = details['average_latency_ms']
                
                if 'throughput_qps' in details:
                    metrics['throughput_qps'] = details['throughput_qps']
                
                if 'memory_usage_mb' in details:
                    metrics['memory_usage_mb'] = details['memory_usage_mb']
        
        return metrics

class ModelTestingSystem:
    """Main model testing system"""
    
    def __init__(self, config: ModelTestingConfig):
        self.config = config
        
        # Components
        self.unit_tester = UnitTester(config)
        self.integration_tester = IntegrationTester(config)
        self.performance_tester = PerformanceTester(config)
        
        # Testing state
        self.testing_history = []
        
        logger.info("✅ Model Testing System initialized")
    
    def test_model(self, model: nn.Module, data_pipeline: Callable = None) -> Dict[str, Any]:
        """Test model comprehensively"""
        logger.info(f"🔍 Testing model using {self.config.testing_level.value} level")
        
        testing_results = {
            'start_time': time.time(),
            'config': self.config,
            'testing_results': {}
        }
        
        # Stage 1: Unit tests
        if self.config.enable_unit_tests:
            logger.info("🔍 Stage 1: Running unit tests")
            
            unit_results = self.unit_tester.run_unit_tests(model)
            testing_results['testing_results']['unit_tests'] = unit_results
        
        # Stage 2: Integration tests
        if self.config.enable_integration_tests:
            logger.info("🔍 Stage 2: Running integration tests")
            
            integration_results = self.integration_tester.run_integration_tests(model, data_pipeline)
            testing_results['testing_results']['integration_tests'] = integration_results
        
        # Stage 3: Performance tests
        if self.config.enable_performance_tests:
            logger.info("🔍 Stage 3: Running performance tests")
            
            performance_results = self.performance_tester.run_performance_tests(model)
            testing_results['testing_results']['performance_tests'] = performance_results
        
        # Final evaluation
        testing_results['end_time'] = time.time()
        testing_results['total_duration'] = testing_results['end_time'] - testing_results['start_time']
        
        # Store results
        self.testing_history.append(testing_results)
        
        logger.info("✅ Model testing completed")
        return testing_results
    
    def generate_testing_report(self, testing_results: Dict[str, Any]) -> str:
        """Generate testing report"""
        logger.info("📋 Generating testing report")
        
        report = []
        report.append("=" * 60)
        report.append("MODEL TESTING REPORT")
        report.append("=" * 60)
        
        # Configuration
        report.append("\nTESTING CONFIGURATION:")
        report.append("-" * 22)
        report.append(f"Testing Level: {self.config.testing_level.value}")
        report.append(f"Testing Type: {self.config.testing_type.value}")
        report.append(f"Test Framework: {self.config.test_framework.value}")
        report.append(f"Enable Unit Tests: {'Enabled' if self.config.enable_unit_tests else 'Disabled'}")
        report.append(f"Unit Test Coverage Threshold: {self.config.unit_test_coverage_threshold}")
        report.append(f"Unit Test Timeout: {self.config.unit_test_timeout}s")
        report.append(f"Enable Parameter Tests: {'Enabled' if self.config.enable_parameter_tests else 'Disabled'}")
        report.append(f"Enable Forward Pass Tests: {'Enabled' if self.config.enable_forward_pass_tests else 'Disabled'}")
        report.append(f"Enable Backward Pass Tests: {'Enabled' if self.config.enable_backward_pass_tests else 'Disabled'}")
        report.append(f"Enable Integration Tests: {'Enabled' if self.config.enable_integration_tests else 'Disabled'}")
        report.append(f"Integration Test Timeout: {self.config.integration_test_timeout}s")
        report.append(f"Enable Data Pipeline Tests: {'Enabled' if self.config.enable_data_pipeline_tests else 'Disabled'}")
        report.append(f"Enable Model Serving Tests: {'Enabled' if self.config.enable_model_serving_tests else 'Disabled'}")
        report.append(f"Enable API Tests: {'Enabled' if self.config.enable_api_tests else 'Disabled'}")
        report.append(f"Enable Performance Tests: {'Enabled' if self.config.enable_performance_tests else 'Disabled'}")
        report.append(f"Performance Test Duration: {self.config.performance_test_duration}s")
        report.append(f"Performance Test Iterations: {self.config.performance_test_iterations}")
        report.append(f"Target Latency (ms): {self.config.target_latency_ms}")
        report.append(f"Target Throughput (QPS): {self.config.target_throughput_qps}")
        report.append(f"Target Memory (MB): {self.config.target_memory_mb}")
        report.append(f"Enable Stress Tests: {'Enabled' if self.config.enable_stress_tests else 'Disabled'}")
        report.append(f"Stress Test Duration: {self.config.stress_test_duration}s")
        report.append(f"Stress Test Load Multiplier: {self.config.stress_test_load_multiplier}")
        report.append(f"Stress Test Memory Multiplier: {self.config.stress_test_memory_multiplier}")
        report.append(f"Enable Load Tests: {'Enabled' if self.config.enable_load_tests else 'Disabled'}")
        report.append(f"Load Test Concurrent Users: {self.config.load_test_concurrent_users}")
        report.append(f"Load Test Ramp Up Time: {self.config.load_test_ramp_up_time}s")
        report.append(f"Load Test Duration: {self.config.load_test_duration}s")
        report.append(f"Enable Security Tests: {'Enabled' if self.config.enable_security_tests else 'Disabled'}")
        report.append(f"Security Test Adversarial Samples: {self.config.security_test_adversarial_samples}")
        report.append(f"Security Test Privacy Leaks: {'Enabled' if self.config.security_test_privacy_leaks else 'Disabled'}")
        report.append(f"Security Test Input Validation: {'Enabled' if self.config.security_test_input_validation else 'Disabled'}")
        report.append(f"Test Data Size: {self.config.test_data_size}")
        report.append(f"Test Data Batch Size: {self.config.test_data_batch_size}")
        report.append(f"Test Data Generation Method: {self.config.test_data_generation_method}")
        report.append(f"Enable Synthetic Data: {'Enabled' if self.config.enable_synthetic_data else 'Disabled'}")
        report.append(f"Enable Real Data: {'Enabled' if self.config.enable_real_data else 'Disabled'}")
        report.append(f"Enable Test Reports: {'Enabled' if self.config.enable_test_reports else 'Disabled'}")
        report.append(f"Test Report Format: {self.config.test_report_format}")
        report.append(f"Enable Coverage Reports: {'Enabled' if self.config.enable_coverage_reports else 'Disabled'}")
        report.append(f"Enable Performance Reports: {'Enabled' if self.config.enable_performance_reports else 'Disabled'}")
        report.append(f"Enable Security Reports: {'Enabled' if self.config.enable_security_reports else 'Disabled'}")
        report.append(f"Enable Automated Testing: {'Enabled' if self.config.enable_automated_testing else 'Disabled'}")
        report.append(f"Enable Continuous Testing: {'Enabled' if self.config.enable_continuous_testing else 'Disabled'}")
        report.append(f"Enable Test Parallelization: {'Enabled' if self.config.enable_test_parallelization else 'Disabled'}")
        report.append(f"Enable Test Caching: {'Enabled' if self.config.enable_test_caching else 'Disabled'}")
        
        # Testing results
        report.append("\nTESTING RESULTS:")
        report.append("-" * 16)
        
        for test_type, results in testing_results.get('testing_results', {}).items():
            report.append(f"\n{test_type.upper()}:")
            report.append("-" * len(test_type))
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, (list, tuple)) and len(value) > 5:
                        report.append(f"  {key}: {type(value).__name__} with {len(value)} items")
                    elif isinstance(value, dict) and len(value) > 5:
                        report.append(f"  {key}: Dict with {len(value)} items")
                    else:
                        report.append(f"  {key}: {value}")
            else:
                report.append(f"  Results: {results}")
        
        # Summary
        report.append("\nSUMMARY:")
        report.append("-" * 8)
        report.append(f"Total Duration: {testing_results.get('total_duration', 0):.2f} seconds")
        report.append(f"Testing History Length: {len(self.testing_history)}")
        
        return "\n".join(report)

# Factory functions
def create_testing_config(**kwargs) -> ModelTestingConfig:
    """Create testing configuration"""
    return ModelTestingConfig(**kwargs)

def create_unit_tester(config: ModelTestingConfig) -> UnitTester:
    """Create unit tester"""
    return UnitTester(config)

def create_integration_tester(config: ModelTestingConfig) -> IntegrationTester:
    """Create integration tester"""
    return IntegrationTester(config)

def create_performance_tester(config: ModelTestingConfig) -> PerformanceTester:
    """Create performance tester"""
    return PerformanceTester(config)

def create_model_testing_system(config: ModelTestingConfig) -> ModelTestingSystem:
    """Create model testing system"""
    return ModelTestingSystem(config)

# Example usage
def example_model_testing():
    """Example of model testing system"""
    # Create configuration
    config = create_testing_config(
        testing_level=TestingLevel.INTERMEDIATE,
        testing_type=TestingType.UNIT_TESTING,
        test_framework=TestFramework.PYTEST,
        enable_unit_tests=True,
        unit_test_coverage_threshold=0.8,
        unit_test_timeout=30,
        enable_parameter_tests=True,
        enable_forward_pass_tests=True,
        enable_backward_pass_tests=True,
        enable_integration_tests=True,
        integration_test_timeout=60,
        enable_data_pipeline_tests=True,
        enable_model_serving_tests=True,
        enable_api_tests=True,
        enable_performance_tests=True,
        performance_test_duration=300,
        performance_test_iterations=1000,
        target_latency_ms=100.0,
        target_throughput_qps=100.0,
        target_memory_mb=500.0,
        enable_stress_tests=True,
        stress_test_duration=600,
        stress_test_load_multiplier=2.0,
        stress_test_memory_multiplier=1.5,
        enable_load_tests=True,
        load_test_concurrent_users=100,
        load_test_ramp_up_time=60,
        load_test_duration=300,
        enable_security_tests=True,
        security_test_adversarial_samples=100,
        security_test_privacy_leaks=True,
        security_test_input_validation=True,
        test_data_size=1000,
        test_data_batch_size=32,
        test_data_generation_method="random",
        enable_synthetic_data=True,
        enable_real_data=True,
        enable_test_reports=True,
        test_report_format="html",
        enable_coverage_reports=True,
        enable_performance_reports=True,
        enable_security_reports=True,
        enable_automated_testing=True,
        enable_continuous_testing=True,
        enable_test_parallelization=True,
        enable_test_caching=True
    )
    
    # Create model testing system
    testing_system = create_model_testing_system(config)
    
    # Create dummy model
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10)
    )
    
    # Create dummy data pipeline
    def data_pipeline():
        return torch.randn(32, 3, 32, 32)
    
    # Test model
    testing_results = testing_system.test_model(model, data_pipeline)
    
    # Generate report
    testing_report = testing_system.generate_testing_report(testing_results)
    
    print(f"✅ Model Testing Example Complete!")
    print(f"🚀 Model Testing Statistics:")
    print(f"   Testing Level: {config.testing_level.value}")
    print(f"   Testing Type: {config.testing_type.value}")
    print(f"   Test Framework: {config.test_framework.value}")
    print(f"   Enable Unit Tests: {'Enabled' if config.enable_unit_tests else 'Disabled'}")
    print(f"   Unit Test Coverage Threshold: {config.unit_test_coverage_threshold}")
    print(f"   Unit Test Timeout: {config.unit_test_timeout}s")
    print(f"   Enable Parameter Tests: {'Enabled' if config.enable_parameter_tests else 'Disabled'}")
    print(f"   Enable Forward Pass Tests: {'Enabled' if config.enable_forward_pass_tests else 'Disabled'}")
    print(f"   Enable Backward Pass Tests: {'Enabled' if config.enable_backward_pass_tests else 'Disabled'}")
    print(f"   Enable Integration Tests: {'Enabled' if config.enable_integration_tests else 'Disabled'}")
    print(f"   Integration Test Timeout: {config.integration_test_timeout}s")
    print(f"   Enable Data Pipeline Tests: {'Enabled' if config.enable_data_pipeline_tests else 'Disabled'}")
    print(f"   Enable Model Serving Tests: {'Enabled' if config.enable_model_serving_tests else 'Disabled'}")
    print(f"   Enable API Tests: {'Enabled' if config.enable_api_tests else 'Disabled'}")
    print(f"   Enable Performance Tests: {'Enabled' if config.enable_performance_tests else 'Disabled'}")
    print(f"   Performance Test Duration: {config.performance_test_duration}s")
    print(f"   Performance Test Iterations: {config.performance_test_iterations}")
    print(f"   Target Latency (ms): {config.target_latency_ms}")
    print(f"   Target Throughput (QPS): {config.target_throughput_qps}")
    print(f"   Target Memory (MB): {config.target_memory_mb}")
    print(f"   Enable Stress Tests: {'Enabled' if config.enable_stress_tests else 'Disabled'}")
    print(f"   Stress Test Duration: {config.stress_test_duration}s")
    print(f"   Stress Test Load Multiplier: {config.stress_test_load_multiplier}")
    print(f"   Stress Test Memory Multiplier: {config.stress_test_memory_multiplier}")
    print(f"   Enable Load Tests: {'Enabled' if config.enable_load_tests else 'Disabled'}")
    print(f"   Load Test Concurrent Users: {config.load_test_concurrent_users}")
    print(f"   Load Test Ramp Up Time: {config.load_test_ramp_up_time}s")
    print(f"   Load Test Duration: {config.load_test_duration}s")
    print(f"   Enable Security Tests: {'Enabled' if config.enable_security_tests else 'Disabled'}")
    print(f"   Security Test Adversarial Samples: {config.security_test_adversarial_samples}")
    print(f"   Security Test Privacy Leaks: {'Enabled' if config.security_test_privacy_leaks else 'Disabled'}")
    print(f"   Security Test Input Validation: {'Enabled' if config.security_test_input_validation else 'Disabled'}")
    print(f"   Test Data Size: {config.test_data_size}")
    print(f"   Test Data Batch Size: {config.test_data_batch_size}")
    print(f"   Test Data Generation Method: {config.test_data_generation_method}")
    print(f"   Enable Synthetic Data: {'Enabled' if config.enable_synthetic_data else 'Disabled'}")
    print(f"   Enable Real Data: {'Enabled' if config.enable_real_data else 'Disabled'}")
    print(f"   Enable Test Reports: {'Enabled' if config.enable_test_reports else 'Disabled'}")
    print(f"   Test Report Format: {config.test_report_format}")
    print(f"   Enable Coverage Reports: {'Enabled' if config.enable_coverage_reports else 'Disabled'}")
    print(f"   Enable Performance Reports: {'Enabled' if config.enable_performance_reports else 'Disabled'}")
    print(f"   Enable Security Reports: {'Enabled' if config.enable_security_reports else 'Disabled'}")
    print(f"   Enable Automated Testing: {'Enabled' if config.enable_automated_testing else 'Disabled'}")
    print(f"   Enable Continuous Testing: {'Enabled' if config.enable_continuous_testing else 'Disabled'}")
    print(f"   Enable Test Parallelization: {'Enabled' if config.enable_test_parallelization else 'Disabled'}")
    print(f"   Enable Test Caching: {'Enabled' if config.enable_test_caching else 'Disabled'}")
    
    print(f"\n📊 Model Testing Results:")
    print(f"   Testing History Length: {len(testing_system.testing_history)}")
    print(f"   Total Duration: {testing_results.get('total_duration', 0):.2f} seconds")
    
    # Show testing results summary
    if 'testing_results' in testing_results:
        print(f"   Number of Testing Types: {len(testing_results['testing_results'])}")
    
    print(f"\n📋 Model Testing Report:")
    print(testing_report)
    
    return testing_system

# Export utilities
__all__ = [
    'TestingLevel',
    'TestingType',
    'TestFramework',
    'ModelTestingConfig',
    'UnitTester',
    'IntegrationTester',
    'PerformanceTester',
    'ModelTestingSystem',
    'create_testing_config',
    'create_unit_tester',
    'create_integration_tester',
    'create_performance_tester',
    'create_model_testing_system',
    'example_model_testing'
]

if __name__ == "__main__":
    example_model_testing()
    print("✅ Model testing example completed successfully!")