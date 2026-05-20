"""
Memory management utilities for Blaze AI.

This module provides comprehensive memory management capabilities including:
- Memory profiling and monitoring
- Memory optimization strategies
- GPU memory management
- Memory pooling and allocation
"""

from __future__ import annotations

import gc
import psutil
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available, memory management limited")

try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False
    warnings.warn("GPUtil not available, GPU memory monitoring limited")

class MemoryProfiler:
    """Memory profiling and monitoring utilities."""
    
    def __init__(self):
        self.memory_history: List[Dict[str, float]] = []
        self.max_history_size = 1000
    
    def get_system_memory(self) -> Dict[str, float]:
        """Get current system memory usage."""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent_used": memory.percent,
            "free_gb": memory.free / (1024**3)
        }
    
    def get_gpu_memory(self) -> Dict[str, Any]:
        """Get current GPU memory usage."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        gpu_info = {}
        for i in range(torch.cuda.device_count()):
            memory_stats = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            total = memory_stats.total_memory / (1024**3)
            
            gpu_info[f"gpu_{i}"] = {
                "name": memory_stats.name,
                "total_gb": total,
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "free_gb": total - reserved,
                "utilization_percent": (allocated / total) * 100
            }
        
        return gpu_info
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory summary."""
        summary = {
            "timestamp": time.time(),
            "system": self.get_system_memory(),
            "gpu": self.get_gpu_memory()
        }
        
        # Store in history
        self.memory_history.append(summary)
        if len(self.memory_history) > self.max_history_size:
            self.memory_history.pop(0)
        
        return summary
    
    def get_memory_trends(self, minutes: int = 60) -> Dict[str, List[float]]:
        """Get memory usage trends over time."""
        cutoff_time = time.time() - (minutes * 60)
        recent_data = [entry for entry in self.memory_history if entry["timestamp"] > cutoff_time]
        
        if not recent_data:
            return {}
        
        timestamps = [entry["timestamp"] for entry in recent_data]
        system_used = [entry["system"]["used_gb"] for entry in recent_data]
        
        trends = {
            "timestamps": timestamps,
            "system_used_gb": system_used
        }
        
        # Add GPU trends if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_key = f"gpu_{i}"
                if gpu_key in recent_data[0]["gpu"]:
                    trends[f"gpu_{i}_allocated_gb"] = [
                        entry["gpu"][gpu_key]["allocated_gb"] for entry in recent_data
                    ]
        
        return trends

class MemoryOptimizer:
    """Memory optimization strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.optimization_history: List[Dict[str, Any]] = []
    
    def optimize_pytorch_memory(self, model: Any) -> Dict[str, Any]:
        """Apply PyTorch memory optimizations."""
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
        
        optimizations = {}
        
        # Enable memory efficient attention if available
        if hasattr(model, 'config'):
            if hasattr(model.config, 'attention_mode'):
                model.config.attention_mode = "flash_attention_2"
                optimizations["attention_mode"] = "flash_attention_2"
            
            if hasattr(model.config, 'use_memory_efficient_attention'):
                model.config.use_memory_efficient_attention = True
                optimizations["memory_efficient_attention"] = True
        
        # Enable gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            optimizations["gradient_checkpointing"] = True
        
        # Set memory fraction if specified
        if self.config.get('memory_fraction'):
            fraction = self.config['memory_fraction']
            torch.cuda.set_per_process_memory_fraction(fraction)
            optimizations["memory_fraction"] = fraction
        
        return optimizations
    
    def clear_cache(self, clear_gpu: bool = True, clear_cpu: bool = True) -> Dict[str, Any]:
        """Clear memory caches."""
        results = {}
        
        if clear_cpu:
            gc.collect()
            results["cpu_cache_cleared"] = True
        
        if clear_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            results["gpu_cache_cleared"] = True
        
        return results
    
    def optimize_batch_size(self, model: Any, max_memory_gb: float) -> int:
        """Calculate optimal batch size based on available memory."""
        if not TORCH_AVAILABLE:
            return 1
        
        # Simple heuristic - can be improved with actual memory profiling
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            available_memory = min(gpu_memory * 0.8, max_memory_gb)
            
            # Rough estimate: 1GB per batch item for large models
            estimated_batch_size = max(1, int(available_memory))
            return min(estimated_batch_size, 32)  # Cap at 32
        
        return 1

class MemoryManager:
    """Comprehensive memory management system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.profiler = MemoryProfiler()
        self.optimizer = MemoryOptimizer(config)
        self.memory_thresholds = self.config.get('memory_thresholds', {
            'system_warning': 80.0,  # 80% system memory usage
            'gpu_warning': 85.0,     # 85% GPU memory usage
            'system_critical': 95.0, # 95% system memory usage
            'gpu_critical': 95.0     # 95% GPU memory usage
        })
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status with warnings."""
        summary = self.profiler.get_memory_summary()
        warnings = []
        critical = []
        
        # Check system memory
        system_usage = summary["system"]["percent_used"]
        if system_usage > self.memory_thresholds["system_critical"]:
            critical.append(f"System memory critical: {system_usage:.1f}%")
        elif system_usage > self.memory_thresholds["system_warning"]:
            warnings.append(f"System memory high: {system_usage:.1f}%")
        
        # Check GPU memory
        if "error" not in summary["gpu"]:
            for gpu_id, gpu_info in summary["gpu"].items():
                gpu_usage = gpu_info["utilization_percent"]
                if gpu_usage > self.memory_thresholds["gpu_critical"]:
                    critical.append(f"{gpu_id} memory critical: {gpu_usage:.1f}%")
                elif gpu_usage > self.memory_thresholds["gpu_warning"]:
                    warnings.append(f"{gpu_id} memory high: {gpu_usage:.1f}%")
        
        summary["warnings"] = warnings
        summary["critical"] = critical
        summary["status"] = "critical" if critical else "warning" if warnings else "healthy"
        
        return summary
    
    def optimize_memory(self, model: Optional[Any] = None) -> Dict[str, Any]:
        """Apply comprehensive memory optimizations."""
        optimizations = {}
        
        # Clear caches
        cache_results = self.optimizer.clear_cache()
        optimizations.update(cache_results)
        
        # Apply PyTorch optimizations if model provided
        if model is not None:
            pytorch_results = self.optimizer.optimize_pytorch_memory(model)
            optimizations.update(pytorch_results)
        
        # Get memory status after optimization
        optimizations["memory_status"] = self.get_memory_status()
        
        return optimizations
    
    def monitor_memory(self, interval_seconds: int = 30, duration_minutes: int = 60):
        """Monitor memory usage over time."""
        import threading
        
        def monitor_loop():
            start_time = time.time()
            while time.time() - start_time < (duration_minutes * 60):
                try:
                    status = self.get_memory_status()
                    if status["status"] != "healthy":
                        print(f"Memory warning: {status['warnings']}")
                        if status["critical"]:
                            print(f"CRITICAL: {status['critical']}")
                    
                    time.sleep(interval_seconds)
                except Exception as e:
                    print(f"Memory monitoring error: {e}")
                    time.sleep(interval_seconds)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        return monitor_thread
    
    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        recommendations = []
        status = self.get_memory_status()
        
        if status["status"] == "critical":
            recommendations.append("IMMEDIATE ACTION REQUIRED: Memory usage is critical")
            recommendations.append("Consider reducing batch size or model size")
            recommendations.append("Clear GPU cache and restart if necessary")
        
        elif status["status"] == "warning":
            recommendations.append("Memory usage is high - consider optimizations")
            recommendations.append("Enable gradient checkpointing if not already enabled")
            recommendations.append("Consider using mixed precision training")
        
        # General recommendations
        if TORCH_AVAILABLE and torch.cuda.is_available():
            recommendations.append("Use torch.cuda.empty_cache() periodically")
            recommendations.append("Consider using torch.compile() for memory optimization")
        
        return recommendations

# Utility functions
def get_memory_info() -> Dict[str, Any]:
    """Get basic memory information."""
    profiler = MemoryProfiler()
    return profiler.get_memory_summary()

def optimize_model_memory(model: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Quick memory optimization for a model."""
    optimizer = MemoryOptimizer(config)
    return optimizer.optimize_pytorch_memory(model)

def clear_memory_caches(clear_gpu: bool = True, clear_cpu: bool = True) -> Dict[str, Any]:
    """Quick cache clearing."""
    optimizer = MemoryOptimizer()
    return optimizer.clear_cache(clear_gpu, clear_cpu)

# Export main classes
__all__ = [
    "MemoryProfiler",
    "MemoryOptimizer", 
    "MemoryManager",
    "get_memory_info",
    "optimize_model_memory",
    "clear_memory_caches"
]
