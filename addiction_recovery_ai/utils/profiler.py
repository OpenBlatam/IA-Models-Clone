"""
Performance Profiling and Monitoring
"""

import torch
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict
import logging
import json
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Performance profiler for models and operations"""
    
    def __init__(self):
        """Initialize profiler"""
        self.metrics = defaultdict(list)
        self.timings = {}
        self.enabled = True
    
    @contextmanager
    def profile(self, operation_name: str):
        """Context manager for profiling operations"""
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            elapsed = (end_time - start_time) * 1000  # ms
            memory_used = (end_memory - start_memory) / 1024**2  # MB
            
            self.metrics[operation_name].append({
                "time_ms": elapsed,
                "memory_mb": memory_used,
                "timestamp": time.time()
            })
            
            logger.debug(f"{operation_name}: {elapsed:.2f}ms, {memory_used:.2f}MB")
    
    def get_stats(self, operation_name: str) -> Dict[str, float]:
        """
        Get statistics for operation
        
        Args:
            operation_name: Operation name
        
        Returns:
            Statistics dictionary
        """
        if operation_name not in self.metrics:
            return {}
        
        times = [m["time_ms"] for m in self.metrics[operation_name]]
        memories = [m["memory_mb"] for m in self.metrics[operation_name]]
        
        return {
            "count": len(times),
            "avg_time_ms": sum(times) / len(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "std_time_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
            "avg_memory_mb": sum(memories) / len(memories) if memories else 0,
            "max_memory_mb": max(memories) if memories else 0
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations"""
        return {op: self.get_stats(op) for op in self.metrics.keys()}
    
    def reset(self):
        """Reset profiler"""
        self.metrics.clear()
        self.timings.clear()
        logger.info("Profiler reset")
    
    def export_report(self, filepath: str):
        """Export profiling report to JSON"""
        report = {
            "operations": self.get_all_stats(),
            "summary": {
                "total_operations": len(self.metrics),
                "total_calls": sum(len(v) for v in self.metrics.values())
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Profiling report exported to {filepath}")


class ModelProfiler:
    """Profiler specifically for PyTorch models"""
    
    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        """
        Initialize model profiler
        
        Args:
            model: Model to profile
            device: Device to use
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.profiler = PerformanceProfiler()
    
    def profile_forward(
        self,
        inputs: torch.Tensor,
        num_runs: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """
        Profile forward pass
        
        Args:
            inputs: Input tensor
            num_runs: Number of runs
            warmup: Warmup runs
        
        Returns:
            Performance metrics
        """
        self.model.eval()
        inputs = inputs.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(inputs)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Profile
        with self.profiler.profile("forward_pass"):
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = self.model(inputs)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        stats = self.profiler.get_stats("forward_pass")
        stats["throughput"] = num_runs / (stats["avg_time_ms"] / 1000)
        
        return stats
    
    def profile_memory(self) -> Dict[str, float]:
        """Profile memory usage"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        torch.cuda.reset_peak_memory_stats()
        
        # Get current memory
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "max_allocated_mb": max_allocated
        }
    
    def get_model_size(self) -> Dict[str, float]:
        """Get model size information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate size (assuming float32)
        size_mb = total_params * 4 / 1024**2
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "size_mb": size_mb,
            "trainable_percent": (trainable_params / total_params * 100) if total_params > 0 else 0
        }


class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self):
        """Initialize system monitor"""
        self.metrics = []
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        if not torch.cuda.is_available():
            return {"available": False}
        
        info = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "memory": {}
        }
        
        for i in range(torch.cuda.device_count()):
            info["memory"][f"gpu_{i}"] = {
                "allocated_mb": torch.cuda.memory_allocated(i) / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved(i) / 1024**2,
                "max_allocated_mb": torch.cuda.max_memory_allocated(i) / 1024**2
            }
        
        return info
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        import platform
        import psutil
        
        return {
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "memory_available_gb": psutil.virtual_memory().available / 1024**3,
            "gpu": self.get_gpu_info()
        }

