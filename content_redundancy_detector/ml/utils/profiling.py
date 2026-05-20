"""
Profiling Utilities
Performance profiling and bottleneck identification
"""

import torch
import time
import logging
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class Profiler:
    """
    Performance profiler for training and inference
    """
    
    def __init__(self):
        """Initialize profiler"""
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.counts: Dict[str, int] = defaultdict(int)
        self.enabled = True
    
    @contextmanager
    def profile(self, name: str):
        """
        Context manager for profiling code blocks
        
        Args:
            name: Name of the operation to profile
        """
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.timings[name].append(elapsed)
            self.counts[name] += 1
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get profiling statistics
        
        Returns:
            Dictionary with statistics for each operation
        """
        stats = {}
        for name, timings in self.timings.items():
            if timings:
                stats[name] = {
                    'mean': np.mean(timings),
                    'std': np.std(timings),
                    'min': np.min(timings),
                    'max': np.max(timings),
                    'total': np.sum(timings),
                    'count': self.counts[name],
                }
        return stats
    
    def print_stats(self) -> None:
        """Print profiling statistics"""
        stats = self.get_stats()
        logger.info("Profiling Statistics:")
        for name, stat in sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True):
            logger.info(
                f"  {name}: "
                f"mean={stat['mean']:.4f}s, "
                f"std={stat['std']:.4f}s, "
                f"min={stat['min']:.4f}s, "
                f"max={stat['max']:.4f}s, "
                f"total={stat['total']:.4f}s, "
                f"count={stat['count']}"
            )
    
    def reset(self) -> None:
        """Reset profiling data"""
        self.timings.clear()
        self.counts.clear()


def profile_model(
    model: torch.nn.Module,
    input_shape: tuple,
    device: torch.device,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, Any]:
    """
    Profile model inference
    
    Args:
        model: Model to profile
        input_shape: Input tensor shape (B, C, H, W)
        device: Device to run on
        num_runs: Number of profiling runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Dictionary with profiling results
    """
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    # Synchronize GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Profile
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            timings.append(time.time() - start)
    
    return {
        'mean_time': np.mean(timings),
        'std_time': np.std(timings),
        'min_time': np.min(timings),
        'max_time': np.max(timings),
        'fps': 1.0 / np.mean(timings),
        'throughput': input_shape[0] / np.mean(timings),
    }


def profile_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int = 10,
) -> Dict[str, Any]:
    """
    Profile training step
    
    Args:
        model: Model to profile
        optimizer: Optimizer
        criterion: Loss function
        data_loader: Data loader
        device: Device
        num_batches: Number of batches to profile
        
    Returns:
        Dictionary with profiling results
    """
    model.train()
    profiler = Profiler()
    
    batch_count = 0
    for inputs, targets in data_loader:
        if batch_count >= num_batches:
            break
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        with profiler.profile('forward'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        with profiler.profile('backward'):
            loss.backward()
        
        with profiler.profile('optimizer'):
            optimizer.step()
            optimizer.zero_grad()
        
        batch_count += 1
    
    return profiler.get_stats()



