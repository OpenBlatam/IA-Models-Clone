"""
Performance Benchmarking for Recovery AI
"""

import time
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import logging
from statistics import mean, median, stdev
from collections import defaultdict

logger = logging.getLogger(__name__)


class Benchmark:
    """Performance benchmark"""
    
    def __init__(self):
        """Initialize benchmark"""
        self.results = defaultdict(list)
        logger.info("Benchmark initialized")
    
    def measure(
        self,
        name: str,
        func: Callable,
        *args,
        iterations: int = 10,
        warmup: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Measure function performance
        
        Args:
            name: Benchmark name
            func: Function to benchmark
            *args: Function arguments
            iterations: Number of iterations
            warmup: Warmup iterations
            **kwargs: Function keyword arguments
        
        Returns:
            Benchmark results
        """
        # Warmup
        for _ in range(warmup):
            try:
                func(*args, **kwargs)
            except Exception:
                pass
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Benchmark error: {e}")
                continue
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        if not times:
            return {"error": "No successful iterations"}
        
        stats = {
            "name": name,
            "iterations": len(times),
            "mean_ms": mean(times) * 1000,
            "median_ms": median(times) * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
            "std_ms": stdev(times) * 1000 if len(times) > 1 else 0,
            "throughput": 1.0 / mean(times) if mean(times) > 0 else 0
        }
        
        self.results[name].append(stats)
        return stats
    
    def compare(self, names: List[str]) -> Dict[str, Any]:
        """
        Compare benchmarks
        
        Args:
            names: Benchmark names to compare
        
        Returns:
            Comparison results
        """
        comparison = {}
        
        for name in names:
            if name in self.results:
                latest = self.results[name][-1]
                comparison[name] = {
                    "mean_ms": latest["mean_ms"],
                    "throughput": latest["throughput"]
                }
        
        return comparison
    
    def get_all_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all benchmark results"""
        return dict(self.results)


class ModelBenchmark:
    """Benchmark for PyTorch models"""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize model benchmark
        
        Args:
            device: Device to use
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ModelBenchmark initialized on {self.device}")
    
    def benchmark_inference(
        self,
        model: torch.nn.Module,
        input_shape: tuple,
        iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark model inference
        
        Args:
            model: PyTorch model
            input_shape: Input shape (batch_size, ...)
            iterations: Number of iterations
            warmup: Warmup iterations
        
        Returns:
            Benchmark results
        """
        model = model.to(self.device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
        
        # Synchronize if CUDA
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = model(dummy_input)
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                elapsed = time.perf_counter() - start
                times.append(elapsed)
        
        stats = {
            "device": str(self.device),
            "input_shape": input_shape,
            "iterations": len(times),
            "mean_ms": mean(times) * 1000,
            "median_ms": median(times) * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
            "std_ms": stdev(times) * 1000 if len(times) > 1 else 0,
            "throughput": input_shape[0] / mean(times) if mean(times) > 0 else 0
        }
        
        return stats
    
    def benchmark_training_step(
        self,
        model: torch.nn.Module,
        input_shape: tuple,
        iterations: int = 50,
        warmup: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmark training step
        
        Args:
            model: PyTorch model
            input_shape: Input shape
            iterations: Number of iterations
            warmup: Warmup iterations
        
        Returns:
            Benchmark results
        """
        model = model.to(self.device)
        model.train()
        
        dummy_input = torch.randn(*input_shape).to(self.device)
        dummy_target = torch.randn(input_shape[0], 1).to(self.device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Warmup
        for _ in range(warmup):
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()
            optimizer.step()
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(iterations):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()
            optimizer.step()
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        stats = {
            "device": str(self.device),
            "input_shape": input_shape,
            "iterations": len(times),
            "mean_ms": mean(times) * 1000,
            "median_ms": median(times) * 1000,
            "throughput": input_shape[0] / mean(times) if mean(times) > 0 else 0
        }
        
        return stats

