"""
Benchmarking Utilities
=======================

Comprehensive benchmarking utilities.
"""

import logging
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import statistics

logger = logging.getLogger(__name__)


def benchmark_inference(
    model: nn.Module,
    dataloader: DataLoader,
    num_warmup: int = 5,
    num_runs: int = 50,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Benchmark inference performance.
    
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
    batch_sizes = []
    
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
        batch_sizes.append(batch_size)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        with torch.no_grad():
            _ = model(inputs)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        
        times.append(elapsed)
    
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    avg_batch_size = statistics.mean(batch_sizes)
    
    return {
        'avg_time_ms': avg_time * 1000,
        'min_time_ms': min_time * 1000,
        'max_time_ms': max_time * 1000,
        'std_time_ms': std_time * 1000,
        'throughput_samples_per_sec': avg_batch_size / avg_time,
        'p50_time_ms': statistics.median(times) * 1000,
        'p95_time_ms': sorted(times)[int(len(times) * 0.95)] * 1000,
        'p99_time_ms': sorted(times)[int(len(times) * 0.99)] * 1000
    }


def benchmark_training(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    num_batches: int = 10,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Benchmark training performance.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        optimizer: Optimizer
        loss_fn: Loss function
        num_batches: Number of batches to benchmark
        device: Device to run on
        
    Returns:
        Benchmark results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.train()
    
    times = []
    batch_sizes = []
    
    for idx, batch in enumerate(train_loader):
        if idx >= num_batches:
            break
        
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch[0], batch[1]
        elif isinstance(batch, dict):
            inputs = batch.get('input_ids', batch.get('inputs', None))
            targets = batch.get('labels', None)
        else:
            inputs = batch
            targets = None
        
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs)
        if targets is not None and not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)
        
        inputs = inputs.to(device)
        if targets is not None:
            targets = targets.to(device)
        
        batch_size = inputs.shape[0]
        batch_sizes.append(batch_size)
        
        optimizer.zero_grad()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        outputs = model(inputs)
        if targets is not None:
            loss = loss_fn(outputs, targets)
            loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        
        times.append(elapsed)
    
    avg_time = statistics.mean(times)
    avg_batch_size = statistics.mean(batch_sizes)
    
    return {
        'avg_time_per_batch_ms': avg_time * 1000,
        'throughput_samples_per_sec': avg_batch_size / avg_time,
        'total_batches': len(times)
    }


def compare_models(
    models: Dict[str, nn.Module],
    dataloader: DataLoader,
    num_runs: int = 20,
    device: Optional[torch.device] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models.
    
    Args:
        models: Dictionary of model name -> model
        dataloader: DataLoader with test data
        num_runs: Number of benchmark runs
        device: Device to run on
        
    Returns:
        Comparison results
    """
    results = {}
    
    for name, model in models.items():
        logger.info(f"Benchmarking {name}...")
        results[name] = benchmark_inference(model, dataloader, num_runs=num_runs, device=device)
    
    return results


class BenchmarkSuite:
    """
    Comprehensive benchmark suite.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize benchmark suite.
        
        Args:
            device: Device to run on
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
    
    def add_model(self, name: str, model: nn.Module) -> None:
        """
        Add model to suite.
        
        Args:
            name: Model name
            model: PyTorch model
        """
        self.results[name] = {'model': model}
    
    def run_inference_benchmark(
        self,
        dataloader: DataLoader,
        num_runs: int = 20
    ) -> Dict[str, Dict[str, float]]:
        """
        Run inference benchmark on all models.
        
        Args:
            dataloader: DataLoader with test data
            num_runs: Number of runs
            
        Returns:
            Benchmark results
        """
        results = {}
        
        for name, data in self.results.items():
            model = data['model']
            results[name] = benchmark_inference(model, dataloader, num_runs=num_runs, device=self.device)
        
        return results
    
    def run_training_benchmark(
        self,
        train_loader: DataLoader,
        optimizer_fn,
        loss_fn: nn.Module,
        num_batches: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Run training benchmark on all models.
        
        Args:
            train_loader: Training DataLoader
            optimizer_fn: Function to create optimizer
            loss_fn: Loss function
            num_batches: Number of batches
            
        Returns:
            Benchmark results
        """
        results = {}
        
        for name, data in self.results.items():
            model = data['model']
            optimizer = optimizer_fn(model.parameters())
            results[name] = benchmark_training(
                model, train_loader, optimizer, loss_fn,
                num_batches=num_batches, device=self.device
            )
        
        return results



