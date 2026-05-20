"""
Model Testing Utilities
========================

Utilities for testing and validating models.
"""

import logging
from typing import Optional, Dict, Any, List, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

logger = logging.getLogger(__name__)


class ModelTester:
    """
    Comprehensive model testing utility.
    """
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Initialize model tester.
        
        Args:
            model: PyTorch model
            device: Device to test on
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def test_forward_pass(self, input_shape: tuple) -> Dict[str, Any]:
        """
        Test forward pass.
        
        Args:
            input_shape: Input tensor shape
            
        Returns:
            Test results
        """
        dummy_input = torch.randn(input_shape).to(self.device)
        
        try:
            with torch.no_grad():
                output = self.model(dummy_input)
            
            return {
                'success': True,
                'output_shape': tuple(output.shape) if isinstance(output, torch.Tensor) else None,
                'output_type': type(output).__name__
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_batch_processing(self, batch_sizes: List[int], input_shape: tuple) -> Dict[str, Any]:
        """
        Test batch processing.
        
        Args:
            batch_sizes: List of batch sizes to test
            input_shape: Input tensor shape (without batch dimension)
            
        Returns:
            Test results
        """
        results = {}
        
        for batch_size in batch_sizes:
            try:
                dummy_input = torch.randn((batch_size, *input_shape)).to(self.device)
                
                start_time = time.time()
                with torch.no_grad():
                    _ = self.model(dummy_input)
                elapsed = time.time() - start_time
                
                results[batch_size] = {
                    'success': True,
                    'time': elapsed,
                    'throughput': batch_size / elapsed
                }
            except Exception as e:
                results[batch_size] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def test_memory_usage(self, input_shape: tuple) -> Dict[str, Any]:
        """
        Test memory usage.
        
        Args:
            input_shape: Input tensor shape
            
        Returns:
            Memory usage statistics
        """
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        dummy_input = torch.randn(input_shape).to(self.device)
        
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        return {
            'memory_allocated_mb': memory_allocated,
            'memory_reserved_mb': memory_reserved,
            'peak_memory_mb': peak_memory
        }
    
    def validate_outputs(
        self,
        dataloader: DataLoader,
        expected_output_shape: Optional[tuple] = None,
        max_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Validate model outputs on dataset.
        
        Args:
            dataloader: DataLoader with test data
            expected_output_shape: Expected output shape
            max_samples: Maximum samples to test
            
        Returns:
            Validation results
        """
        results = {
            'total_samples': 0,
            'valid_outputs': 0,
            'invalid_outputs': 0,
            'errors': []
        }
        
        for idx, batch in enumerate(dataloader):
            if idx >= max_samples:
                break
            
            try:
                if isinstance(batch, dict):
                    inputs = batch.get('input_ids', batch.get('inputs', None))
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                
                if inputs is None:
                    continue
                
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.tensor(inputs)
                
                inputs = inputs.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(inputs)
                
                # Validate shape
                if expected_output_shape:
                    if isinstance(outputs, torch.Tensor):
                        if outputs.shape != expected_output_shape:
                            results['invalid_outputs'] += 1
                            results['errors'].append(f"Shape mismatch: {outputs.shape} != {expected_output_shape}")
                            continue
                
                # Check for NaN/Inf
                if isinstance(outputs, torch.Tensor):
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        results['invalid_outputs'] += 1
                        results['errors'].append("NaN or Inf in outputs")
                        continue
                
                results['valid_outputs'] += 1
                results['total_samples'] += 1
                
            except Exception as e:
                results['invalid_outputs'] += 1
                results['errors'].append(str(e))
                results['total_samples'] += 1
        
        results['success_rate'] = results['valid_outputs'] / max(results['total_samples'], 1)
        return results


def benchmark_model(
    model: nn.Module,
    dataloader: DataLoader,
    num_warmup: int = 5,
    num_runs: int = 20,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Benchmark model performance.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with test data
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
        device: Device to run on
        
    Returns:
        Benchmark results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    # Warmup
    for _ in range(num_warmup):
        batch = next(iter(dataloader))
        if isinstance(batch, (list, tuple)):
            inputs = batch[0]
        elif isinstance(batch, dict):
            inputs = batch.get('input_ids', batch.get('inputs', None))
        else:
            inputs = batch
        
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs)
        
        inputs = inputs.to(device)
        
        with torch.no_grad():
            _ = model(inputs)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        batch = next(iter(dataloader))
        if isinstance(batch, (list, tuple)):
            inputs = batch[0]
        elif isinstance(batch, dict):
            inputs = batch.get('input_ids', batch.get('inputs', None))
        else:
            inputs = batch
        
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs)
        
        inputs = inputs.to(device)
        batch_size = inputs.shape[0]
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        with torch.no_grad():
            _ = model(inputs)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        'avg_time_ms': avg_time * 1000,
        'min_time_ms': min_time * 1000,
        'max_time_ms': max_time * 1000,
        'throughput_samples_per_sec': batch_size / avg_time
    }


def create_test_suite(model: nn.Module) -> ModelTester:
    """
    Create test suite for model.
    
    Args:
        model: PyTorch model
        
    Returns:
        ModelTester instance
    """
    return ModelTester(model)



