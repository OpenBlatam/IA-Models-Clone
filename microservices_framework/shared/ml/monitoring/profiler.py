"""
Model Profiler
Performance profiling and bottleneck identification.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import time
import logging
from contextlib import contextmanager
from collections import defaultdict

logger = logging.getLogger(__name__)


class ModelProfiler:
    """
    Profiler for model performance analysis.
    """
    
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.profile_data: Dict[str, List[float]] = defaultdict(list)
        self.hooks = []
    
    @contextmanager
    def profile_forward(self):
        """Context manager for profiling forward pass."""
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        yield
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        self.profile_data["forward_time"].append(duration)
        self.profile_data["memory_delta"].append(memory_delta)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
    
    def profile_layer(self, layer_name: str, layer: nn.Module):
        """
        Add profiling hook to a specific layer.
        
        Args:
            layer_name: Name of the layer
            layer: Layer module
        """
        def hook_fn(module, input, output):
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            start = time.time()
            
            # Measure output size
            if isinstance(output, torch.Tensor):
                output_size = output.numel() * output.element_size() / (1024 * 1024)
            elif isinstance(output, tuple):
                output_size = sum(
                    t.numel() * t.element_size() / (1024 * 1024)
                    for t in output if isinstance(t, torch.Tensor)
                )
            else:
                output_size = 0
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            duration = time.time() - start
            
            self.profile_data[f"layer_{layer_name}_time"].append(duration)
            self.profile_data[f"layer_{layer_name}_output_size"].append(output_size)
        
        hook = layer.register_forward_hook(hook_fn)
        self.hooks.append(hook)
    
    def profile_model(self, input_shape: tuple, num_runs: int = 10, warmup: int = 3):
        """
        Profile entire model.
        
        Args:
            input_shape: Input tensor shape
            num_runs: Number of profiling runs
            warmup: Number of warmup runs
            
        Returns:
            Profiling results
        """
        # Warmup
        dummy_input = torch.randn(input_shape).to(self.device)
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Profile
        times = []
        memory_usage = []
        
        for _ in range(num_runs):
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            
            start_memory = self._get_memory_usage()
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
            
            if self.device == "cuda":
                peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
                memory_usage[-1] = peak_memory
        
        results = {
            "mean_time": sum(times) / len(times),
            "std_time": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
            "min_time": min(times),
            "max_time": max(times),
            "mean_memory_mb": sum(memory_usage) / len(memory_usage),
            "max_memory_mb": max(memory_usage),
            "throughput": input_shape[0] / (sum(times) / len(times)),  # samples per second
        }
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        summary = {}
        
        for key, values in self.profile_data.items():
            if values:
                summary[key] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
        
        return summary
    
    def cleanup(self):
        """Remove profiling hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class BottleneckAnalyzer:
    """
    Analyze bottlenecks in model inference.
    """
    
    @staticmethod
    def analyze_model(model: nn.Module, input_shape: tuple, device: str = "cuda") -> Dict[str, Any]:
        """
        Analyze model for bottlenecks.
        
        Args:
            model: Model to analyze
            input_shape: Input shape
            device: Device to run on
            
        Returns:
            Bottleneck analysis
        """
        profiler = ModelProfiler(model, device)
        
        # Profile model
        results = profiler.profile_model(input_shape)
        
        # Identify bottlenecks
        bottlenecks = []
        
        if results["mean_time"] > 0.1:  # Threshold for slow inference
            bottlenecks.append({
                "type": "slow_inference",
                "severity": "high" if results["mean_time"] > 1.0 else "medium",
                "message": f"Mean inference time: {results['mean_time']:.3f}s",
            })
        
        if results["max_memory_mb"] > 1000:  # Threshold for high memory
            bottlenecks.append({
                "type": "high_memory",
                "severity": "high" if results["max_memory_mb"] > 5000 else "medium",
                "message": f"Peak memory: {results['max_memory_mb']:.2f} MB",
            })
        
        if results["throughput"] < 10:  # Threshold for low throughput
            bottlenecks.append({
                "type": "low_throughput",
                "severity": "medium",
                "message": f"Throughput: {results['throughput']:.2f} samples/s",
            })
        
        return {
            "profile_results": results,
            "bottlenecks": bottlenecks,
            "recommendations": BottleneckAnalyzer._get_recommendations(bottlenecks),
        }
    
    @staticmethod
    def _get_recommendations(bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Get recommendations based on bottlenecks."""
        recommendations = []
        
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "slow_inference":
                recommendations.append("Consider model quantization (INT8/INT4)")
                recommendations.append("Use model compilation (torch.compile)")
                recommendations.append("Enable mixed precision inference")
            elif bottleneck["type"] == "high_memory":
                recommendations.append("Use gradient checkpointing")
                recommendations.append("Consider model quantization")
                recommendations.append("Use smaller batch sizes")
            elif bottleneck["type"] == "low_throughput":
                recommendations.append("Increase batch size if memory allows")
                recommendations.append("Use batching for inference")
                recommendations.append("Consider model optimization")
        
        return list(set(recommendations))  # Remove duplicates



