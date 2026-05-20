"""
Benchmarking Utilities
Performance benchmarking for models
"""

import torch
import torch.nn as nn
import time
import statistics
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelBenchmark:
    """
    Benchmark model performance
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize benchmark
        
        Args:
            device: Device to use
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def benchmark_inference(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_warmup: int = 10,
        num_runs: int = 100,
        batch_sizes: List[int] = [1, 4, 8, 16, 32]
    ) -> Dict[str, Any]:
        """
        Benchmark inference performance
        
        Args:
            model: Model to benchmark
            input_shape: Input shape (without batch dimension)
            num_warmup: Number of warmup runs
            num_runs: Number of benchmark runs
            batch_sizes: Batch sizes to test
            
        Returns:
            Benchmark results
        """
        model = model.to(self.device).eval()
        results = {
            "device": str(self.device),
            "batch_results": {}
        }
        
        # Warmup
        dummy_input = torch.randn((1, *input_shape)).to(self.device)
        with torch.inference_mode():
            for _ in range(num_warmup):
                _ = model(dummy_input)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark each batch size
        for batch_size in batch_sizes:
            batch_shape = (batch_size, *input_shape)
            dummy_input = torch.randn(batch_shape).to(self.device)
            
            times = []
            
            for _ in range(num_runs):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                
                with torch.inference_mode():
                    _ = model(dummy_input)
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms
            
            results["batch_results"][batch_size] = {
                "mean_ms": statistics.mean(times),
                "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
                "min_ms": min(times),
                "max_ms": max(times),
                "median_ms": statistics.median(times),
                "throughput": batch_size / (statistics.mean(times) / 1000)  # samples/sec
            }
        
        return results
    
    def benchmark_memory(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """
        Benchmark memory usage
        
        Args:
            model: Model to benchmark
            input_shape: Input shape
            batch_size: Batch size
            
        Returns:
            Memory benchmark results
        """
        if self.device.type != "cuda":
            return {"error": "Memory benchmarking only available on CUDA"}
        
        model = model.to(self.device).eval()
        torch.cuda.reset_peak_memory_stats()
        
        dummy_input = torch.randn((batch_size, *input_shape)).to(self.device)
        
        with torch.inference_mode():
            _ = model(dummy_input)
        
        torch.cuda.synchronize()
        
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        }
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        input_shape: Tuple[int, ...],
        num_runs: int = 50
    ) -> Dict[str, Any]:
        """
        Compare multiple models
        
        Args:
            models: Dictionary of model_name -> model
            input_shape: Input shape
            num_runs: Number of runs
            
        Returns:
            Comparison results
        """
        results = {
            "models": {},
            "comparison": {}
        }
        
        for name, model in models.items():
            model_results = self.benchmark_inference(
                model,
                input_shape,
                num_runs=num_runs,
                batch_sizes=[1]
            )
            results["models"][name] = model_results["batch_results"][1]
        
        # Find fastest
        if results["models"]:
            fastest = min(
                results["models"].items(),
                key=lambda x: x[1]["mean_ms"]
            )
            results["comparison"]["fastest"] = fastest[0]
            results["comparison"]["fastest_time_ms"] = fastest[1]["mean_ms"]
        
        return results


def create_benchmark(device: Optional[torch.device] = None) -> ModelBenchmark:
    """Factory for benchmark"""
    return ModelBenchmark(device)













