"""
Profiling Utilities - Performance Profiling and Optimization
===========================================================

Provides utilities for profiling and optimizing deep learning code:
- PyTorch profiler integration
- Memory profiling
- Performance bottlenecks identification
"""

import logging
from typing import Optional, Dict, Any, Callable
import torch
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)


@contextmanager
def profile_operation(operation_name: str, use_torch_profiler: bool = False):
    """
    Context manager for profiling operations.
    
    Args:
        operation_name: Name of the operation
        use_torch_profiler: Use PyTorch profiler
    """
    start_time = time.time()
    
    if use_torch_profiler and torch.cuda.is_available():
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True,
            profile_memory=True
        ) as prof:
            yield prof
            prof.export_chrome_trace(f"{operation_name}_trace.json")
            logger.info(f"Profiler trace saved: {operation_name}_trace.json")
    else:
        yield None
    
    elapsed_time = time.time() - start_time
    logger.info(f"{operation_name} took {elapsed_time:.4f} seconds")


def profile_model(
    model: torch.nn.Module,
    input_shape: tuple,
    device: Optional[torch.device] = None,
    num_runs: int = 10,
    warmup_runs: int = 3
) -> Dict[str, float]:
    """
    Profile model inference performance.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch_size, ...)
        device: Device to run on
        num_runs: Number of profiling runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Dictionary with profiling results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    # Synchronize if CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Profile
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_runs
    
    # Memory usage
    if device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**2  # MB
    else:
        memory_allocated = memory_reserved = 0.0
    
    results = {
        'avg_inference_time': avg_time,
        'total_time': elapsed_time,
        'num_runs': num_runs,
        'memory_allocated_mb': memory_allocated,
        'memory_reserved_mb': memory_reserved,
        'throughput': input_shape[0] / avg_time  # samples per second
    }
    
    logger.info(f"Model profiling results: {results}")
    return results


def check_for_bottlenecks(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
    num_batches: int = 5
) -> Dict[str, float]:
    """
    Check for bottlenecks in data loading vs model inference.
    
    Args:
        dataloader: DataLoader to profile
        model: Model to profile
        device: Device to run on
        num_batches: Number of batches to profile
        
    Returns:
        Dictionary with timing breakdown
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    data_times = []
    inference_times = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            # Time data loading
            data_start = time.time()
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            elif isinstance(batch, (tuple, list)):
                batch = tuple(v.to(device) if isinstance(v, torch.Tensor) else v
                             for v in batch)
            else:
                batch = batch.to(device)
            data_time = time.time() - data_start
            data_times.append(data_time)
            
            # Time inference
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inf_start = time.time()
            if isinstance(batch, dict):
                _ = model(**batch)
            elif isinstance(batch, (tuple, list)):
                _ = model(*batch)
            else:
                _ = model(batch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inf_time = time.time() - inf_start
            inference_times.append(inf_time)
    
    avg_data_time = sum(data_times) / len(data_times)
    avg_inf_time = sum(inference_times) / len(inference_times)
    
    results = {
        'avg_data_loading_time': avg_data_time,
        'avg_inference_time': avg_inf_time,
        'data_loading_ratio': avg_data_time / (avg_data_time + avg_inf_time),
        'inference_ratio': avg_inf_time / (avg_data_time + avg_inf_time),
        'bottleneck': 'data_loading' if avg_data_time > avg_inf_time else 'inference'
    }
    
    logger.info(f"Bottleneck analysis: {results}")
    if results['bottleneck'] == 'data_loading':
        logger.warning("Data loading is the bottleneck. Consider increasing num_workers or using prefetch_factor.")
    
    return results



