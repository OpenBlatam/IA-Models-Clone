from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import cProfile
import pstats
import io
import time
import asyncio
import threading
import psutil
import gc
import os
import sys
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
import logging
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.profiler import profile, record_function, ProfilerActivity
import torch.cuda.amp as amp
    import line_profiler
    import memory_profiler
    import pyinstrument
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.performance_optimizer import (
from onyx.server.features.ads.multi_gpu_training import (
        import concurrent.futures
from typing import Any, List, Dict, Optional
"""
Profiling and Optimization System for Onyx Ads Backend

This module provides comprehensive profiling capabilities including:
- Performance profiling with cProfile and line_profiler
- Memory profiling with memory_profiler
- GPU profiling with torch.profiler
- Data loading and preprocessing optimization
- Bottleneck identification and resolution
- Automatic optimization recommendations
- Real-time monitoring and alerting
"""


# Third-party profiling tools
try:
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

try:
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

try:
    PYINSTRUMENT_AVAILABLE = True
except ImportError:
    PYINSTRUMENT_AVAILABLE = False

    performance_monitor,
    cache_result,
    performance_context,
    memory_context,
    optimizer
)
    GPUConfig,
    GPUMonitor,
    gpu_monitoring_context
)

logger = setup_logger()

@dataclass
class ProfilingConfig:
    """Configuration for profiling and optimization."""
    
    # Profiling settings
    enabled: bool = True
    profile_cpu: bool = True
    profile_memory: bool = True
    profile_gpu: bool = True
    profile_data_loading: bool = True
    profile_preprocessing: bool = True
    
    # Profiling depth
    profile_depth: int = 10
    min_time_threshold: float = 0.001  # 1ms
    min_memory_threshold: float = 1024 * 1024  # 1MB
    
    # Output settings
    save_profiles: bool = True
    profile_dir: str = "profiles"
    export_formats: List[str] = field(default_factory=lambda: ["json", "html", "txt"])
    
    # Optimization settings
    auto_optimize: bool = True
    optimization_threshold: float = 0.1  # 10% improvement required
    max_optimization_iterations: int = 5
    
    # Monitoring settings
    real_time_monitoring: bool = True
    alert_threshold: float = 5.0  # 5 seconds
    monitoring_interval: float = 1.0  # 1 second
    
    # Data loading optimization
    optimize_data_loading: bool = True
    prefetch_factor: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Preprocessing optimization
    optimize_preprocessing: bool = True
    batch_preprocessing: bool = True
    cache_preprocessing: bool = True
    parallel_preprocessing: bool = True

@dataclass
class ProfilingResult:
    """Results from profiling analysis."""
    
    # Timing information
    total_time: float = 0.0
    function_times: Dict[str, float] = field(default_factory=dict)
    bottleneck_functions: List[str] = field(default_factory=list)
    
    # Memory information
    peak_memory: float = 0.0
    memory_usage: Dict[str, float] = field(default_factory=dict)
    memory_leaks: List[str] = field(default_factory=list)
    
    # GPU information
    gpu_utilization: float = 0.0
    gpu_memory_usage: float = 0.0
    gpu_bottlenecks: List[str] = field(default_factory=list)
    
    # Data loading information
    data_loading_time: float = 0.0
    preprocessing_time: float = 0.0
    io_bottlenecks: List[str] = field(default_factory=list)
    
    # Optimization recommendations
    recommendations: List[str] = field(default_factory=list)
    estimated_improvements: Dict[str, float] = field(default_factory=dict)

class ProfilingOptimizer:
    """Comprehensive profiling and optimization system."""
    
    def __init__(self, config: ProfilingConfig):
        
    """__init__ function."""
self.config = config
        self.profiler = None
        self.line_profiler = None
        self.memory_profiler = None
        self.gpu_monitor = GPUMonitor(GPUConfig())
        self.profiling_results = []
        self.optimization_history = []
        
        # Create profile directory
        if self.config.save_profiles:
            Path(self.config.profile_dir).mkdir(exist_ok=True)
        
        # Initialize profilers
        self._initialize_profilers()
    
    def _initialize_profilers(self) -> Any:
        """Initialize available profilers."""
        if LINE_PROFILER_AVAILABLE:
            self.line_profiler = line_profiler.LineProfiler()
        
        if MEMORY_PROFILER_AVAILABLE:
            self.memory_profiler = memory_profiler.profile
    
    @contextmanager
    def profile_function(self, function_name: str = None):
        """Context manager for profiling functions."""
        if not self.config.enabled:
            yield
            return
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        # Start GPU profiling if available
        gpu_start_memory = 0
        if torch.cuda.is_available() and self.config.profile_gpu:
            gpu_start_memory = torch.cuda.memory_allocated()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            gpu_memory_used = 0
            if torch.cuda.is_available() and self.config.profile_gpu:
                gpu_end_memory = torch.cuda.memory_allocated()
                gpu_memory_used = gpu_end_memory - gpu_start_memory
            
            # Log if thresholds exceeded
            if execution_time > self.config.min_time_threshold:
                logger.info(f"Function {function_name or 'unknown'}: "
                           f"Time: {execution_time:.4f}s, "
                           f"Memory: {memory_used / 1024 / 1024:.2f}MB, "
                           f"GPU Memory: {gpu_memory_used / 1024 / 1024:.2f}MB")
    
    @performance_monitor("profiling_analysis")
    def profile_code(self, func: Callable, *args, **kwargs) -> ProfilingResult:
        """Profile a function and return detailed analysis."""
        result = ProfilingResult()
        
        # CPU profiling
        if self.config.profile_cpu:
            cpu_result = self._profile_cpu(func, *args, **kwargs)
            result.function_times.update(cpu_result)
            result.total_time = sum(cpu_result.values())
        
        # Memory profiling
        if self.config.profile_memory and MEMORY_PROFILER_AVAILABLE:
            memory_result = self._profile_memory(func, *args, **kwargs)
            result.memory_usage.update(memory_result)
            result.peak_memory = max(memory_result.values()) if memory_result else 0
        
        # GPU profiling
        if self.config.profile_gpu and torch.cuda.is_available():
            gpu_result = self._profile_gpu(func, *args, **kwargs)
            result.gpu_utilization = gpu_result.get("utilization", 0)
            result.gpu_memory_usage = gpu_result.get("memory_usage", 0)
            result.gpu_bottlenecks = gpu_result.get("bottlenecks", [])
        
        # Identify bottlenecks
        result.bottleneck_functions = self._identify_bottlenecks(result.function_times)
        result.memory_leaks = self._identify_memory_leaks(result.memory_usage)
        
        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)
        result.estimated_improvements = self._estimate_improvements(result)
        
        # Save results
        self.profiling_results.append(result)
        
        return result
    
    def _profile_cpu(self, func: Callable, *args, **kwargs) -> Dict[str, float]:
        """Profile CPU usage of a function."""
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            func(*args, **kwargs)
        finally:
            profiler.disable()
        
        # Analyze results
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        stats.print_stats(self.config.profile_depth)
        
        # Parse results
        function_times = {}
        for line in s.getvalue().split('\n'):
            if line.strip() and 'function' in line:
                parts = line.split()
                if len(parts) >= 6:
                    function_name = parts[5]
                    cumulative_time = float(parts[3])
                    if cumulative_time > self.config.min_time_threshold:
                        function_times[function_name] = cumulative_time
        
        return function_times
    
    def _profile_memory(self, func: Callable, *args, **kwargs) -> Dict[str, float]:
        """Profile memory usage of a function."""
        if not MEMORY_PROFILER_AVAILABLE:
            return {}
        
        memory_usage = {}
        
        @self.memory_profiler
        def wrapped_func():
            
    """wrapped_func function."""
return func(*args, **kwargs)
        
        # Run memory profiling
        result = wrapped_func()
        
        # Parse memory profiler output (simplified)
        # In practice, you'd parse the detailed output
        memory_usage["peak"] = psutil.Process().memory_info().rss / 1024 / 1024
        
        return memory_usage
    
    def _profile_gpu(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile GPU usage of a function."""
        if not torch.cuda.is_available():
            return {}
        
        gpu_info = {}
        
        # Use torch.profiler for detailed GPU analysis
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        ) as prof:
            with record_function("gpu_profiling"):
                func(*args, **kwargs)
        
        # Analyze profiler results
        key_averages = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        
        # Extract GPU metrics
        gpu_info["utilization"] = self._calculate_gpu_utilization(prof)
        gpu_info["memory_usage"] = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_info["bottlenecks"] = self._identify_gpu_bottlenecks(prof)
        
        return gpu_info
    
    def _calculate_gpu_utilization(self, prof) -> float:
        """Calculate GPU utilization from profiler results."""
        total_cuda_time = 0
        total_cpu_time = 0
        
        for event in prof.function_events:
            if event.device_type == torch.profiler.DeviceType.CUDA:
                total_cuda_time += event.cuda_time_total
            else:
                total_cpu_time += event.cpu_time_total
        
        if total_cpu_time > 0:
            return (total_cuda_time / total_cpu_time) * 100
        return 0.0
    
    def _identify_gpu_bottlenecks(self, prof) -> List[str]:
        """Identify GPU bottlenecks from profiler results."""
        bottlenecks = []
        
        for event in prof.function_events:
            if event.device_type == torch.profiler.DeviceType.CUDA:
                if event.cuda_time_total > 1000:  # 1ms threshold
                    bottlenecks.append(f"{event.name}: {event.cuda_time_total:.2f}ms")
        
        return bottlenecks[:5]  # Top 5 bottlenecks
    
    def _identify_bottlenecks(self, function_times: Dict[str, float]) -> List[str]:
        """Identify performance bottlenecks."""
        if not function_times:
            return []
        
        total_time = sum(function_times.values())
        threshold = total_time * 0.1  # 10% threshold
        
        bottlenecks = []
        for func_name, time_taken in function_times.items():
            if time_taken > threshold:
                bottlenecks.append(f"{func_name}: {time_taken:.4f}s ({time_taken/total_time*100:.1f}%)")
        
        return sorted(bottlenecks, key=lambda x: float(x.split(':')[1].split('s')[0]), reverse=True)
    
    def _identify_memory_leaks(self, memory_usage: Dict[str, float]) -> List[str]:
        """Identify potential memory leaks."""
        leaks = []
        
        # Simple heuristic: check if memory usage is unusually high
        if memory_usage.get("peak", 0) > 1024:  # 1GB threshold
            leaks.append(f"High memory usage: {memory_usage['peak']:.2f}MB")
        
        return leaks
    
    def _generate_recommendations(self, result: ProfilingResult) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # CPU optimization recommendations
        if result.bottleneck_functions:
            recommendations.append("Consider optimizing bottleneck functions")
            for bottleneck in result.bottleneck_functions[:3]:
                recommendations.append(f"  - {bottleneck}")
        
        # Memory optimization recommendations
        if result.memory_leaks:
            recommendations.append("Address potential memory leaks")
            for leak in result.memory_leaks:
                recommendations.append(f"  - {leak}")
        
        # GPU optimization recommendations
        if result.gpu_utilization < 50:
            recommendations.append("Low GPU utilization - consider batch size optimization")
        
        if result.gpu_memory_usage > 0.8:  # 80% threshold
            recommendations.append("High GPU memory usage - consider model optimization")
        
        # Data loading optimization
        if result.data_loading_time > result.total_time * 0.3:
            recommendations.append("Data loading is a bottleneck - optimize I/O operations")
        
        return recommendations
    
    def _estimate_improvements(self, result: ProfilingResult) -> Dict[str, float]:
        """Estimate potential improvements."""
        improvements = {}
        
        # Estimate based on bottlenecks
        if result.bottleneck_functions:
            improvements["bottleneck_optimization"] = 0.2  # 20% improvement
        
        # Estimate based on memory optimization
        if result.memory_leaks:
            improvements["memory_optimization"] = 0.15  # 15% improvement
        
        # Estimate based on GPU optimization
        if result.gpu_utilization < 50:
            improvements["gpu_optimization"] = 0.25  # 25% improvement
        
        return improvements

class DataLoadingOptimizer:
    """Specialized optimizer for data loading and preprocessing."""
    
    def __init__(self, config: ProfilingConfig):
        
    """__init__ function."""
self.config = config
        self.cache = {}
        self.preprocessing_cache = {}
        self.optimization_stats = {}
    
    @performance_monitor("optimize_dataloader")
    def optimize_dataloader(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        **kwargs
    ) -> DataLoader:
        """Optimize DataLoader configuration for better performance."""
        
        # Determine optimal number of workers
        optimal_workers = self._calculate_optimal_workers()
        
        # Determine optimal prefetch factor
        optimal_prefetch = self._calculate_optimal_prefetch()
        
        # Create optimized DataLoader
        optimized_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=optimal_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=optimal_prefetch,
            **kwargs
        )
        
        # Profile the optimized loader
        self._profile_dataloader(optimized_loader)
        
        return optimized_loader
    
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers for data loading."""
        cpu_count = os.cpu_count()
        
        # Base calculation
        optimal_workers = min(cpu_count, 8)  # Cap at 8 workers
        
        # Adjust based on system memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 8:
            optimal_workers = min(optimal_workers, 2)
        elif memory_gb < 16:
            optimal_workers = min(optimal_workers, 4)
        
        return max(1, optimal_workers)
    
    async def _calculate_optimal_prefetch(self) -> int:
        """Calculate optimal prefetch factor."""
        # Base prefetch factor
        prefetch = self.config.prefetch_factor
        
        # Adjust based on available memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 8:
            prefetch = 1
        elif memory_gb > 32:
            prefetch = 4
        
        return prefetch
    
    def _profile_dataloader(self, dataloader: DataLoader):
        """Profile DataLoader performance."""
        start_time = time.time()
        
        # Measure data loading time
        for i, batch in enumerate(dataloader):
            if i >= 10:  # Profile first 10 batches
                break
        
        loading_time = time.time() - start_time
        avg_loading_time = loading_time / 10
        
        self.optimization_stats["avg_loading_time"] = avg_loading_time
        self.optimization_stats["total_loading_time"] = loading_time
        
        logger.info(f"DataLoader profiling: "
                   f"Avg loading time: {avg_loading_time:.4f}s, "
                   f"Total time: {loading_time:.4f}s")
    
    @performance_monitor("optimize_preprocessing")
    def optimize_preprocessing(
        self,
        preprocessing_func: Callable,
        data: Any,
        cache_key: str = None
    ) -> Callable:
        """Optimize preprocessing function with caching and batching."""
        
        if self.config.cache_preprocessing and cache_key:
            # Use cached preprocessing
            if cache_key in self.preprocessing_cache:
                return self.preprocessing_cache[cache_key]
        
        # Create optimized preprocessing function
        optimized_func = self._create_optimized_preprocessing(preprocessing_func)
        
        # Cache if enabled
        if self.config.cache_preprocessing and cache_key:
            self.preprocessing_cache[cache_key] = optimized_func
        
        return optimized_func
    
    def _create_optimized_preprocessing(self, func: Callable) -> Callable:
        """Create an optimized version of preprocessing function."""
        
        @wraps(func)
        def optimized_preprocessing(*args, **kwargs) -> Any:
            # Add performance monitoring
            start_time = time.time()
            
            # Execute preprocessing
            result = func(*args, **kwargs)
            
            # Record performance
            processing_time = time.time() - start_time
            self.optimization_stats[f"{func.__name__}_time"] = processing_time
            
            return result
        
        return optimized_preprocessing

class PreprocessingOptimizer:
    """Specialized optimizer for data preprocessing."""
    
    def __init__(self, config: ProfilingConfig):
        
    """__init__ function."""
self.config = config
        self.batch_cache = {}
        self.parallel_pool = None
    
    @performance_monitor("batch_preprocessing")
    def batch_preprocess(
        self,
        data: List[Any],
        preprocessing_func: Callable,
        batch_size: int = 32
    ) -> List[Any]:
        """Apply preprocessing in batches for better performance."""
        
        if not self.config.batch_preprocessing:
            return [preprocessing_func(item) for item in data]
        
        results = []
        
        # Process in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            # Apply preprocessing to batch
            batch_results = self._process_batch(batch, preprocessing_func)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, batch: List[Any], func: Callable) -> List[Any]:
        """Process a batch of data."""
        return [func(item) for item in batch]
    
    @performance_monitor("parallel_preprocessing")
    def parallel_preprocess(
        self,
        data: List[Any],
        preprocessing_func: Callable,
        num_workers: int = None
    ) -> List[Any]:
        """Apply preprocessing in parallel for better performance."""
        
        if not self.config.parallel_preprocessing:
            return [preprocessing_func(item) for item in data]
        
        if num_workers is None:
            num_workers = min(os.cpu_count(), 4)
        
        # Use ThreadPoolExecutor for I/O bound tasks
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(preprocessing_func, data))
        
        return results

class RealTimeProfiler:
    """Real-time profiling and monitoring system."""
    
    def __init__(self, config: ProfilingConfig):
        
    """__init__ function."""
self.config = config
        self.monitoring_active = False
        self.monitoring_thread = None
        self.performance_metrics = []
        self.alert_callbacks = []
    
    def start_monitoring(self) -> Any:
        """Start real-time monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Real-time profiling started")
    
    def stop_monitoring(self) -> Any:
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        logger.info("Real-time profiling stopped")
    
    def _monitoring_loop(self) -> Any:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.performance_metrics.append(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Sleep for monitoring interval
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics."""
        metrics = {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used": psutil.virtual_memory().used / (1024**3),  # GB
        }
        
        # GPU metrics if available
        if torch.cuda.is_available():
            metrics["gpu_memory_used"] = torch.cuda.memory_allocated() / (1024**3)  # GB
            metrics["gpu_memory_percent"] = (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100
        
        return metrics
    
    def _check_alerts(self, metrics: Dict[str, float]):
        """Check for performance alerts."""
        alerts = []
        
        # CPU alert
        if metrics["cpu_percent"] > 90:
            alerts.append(f"High CPU usage: {metrics['cpu_percent']:.1f}%")
        
        # Memory alert
        if metrics["memory_percent"] > 90:
            alerts.append(f"High memory usage: {metrics['memory_percent']:.1f}%")
        
        # GPU alert
        if "gpu_memory_percent" in metrics and metrics["gpu_memory_percent"] > 90:
            alerts.append(f"High GPU memory usage: {metrics['gpu_memory_percent']:.1f}%")
        
        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert, metrics)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add an alert callback function."""
        self.alert_callbacks.append(callback)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from monitoring data."""
        if not self.performance_metrics:
            return {}
        
        # Calculate statistics
        cpu_percentages = [m["cpu_percent"] for m in self.performance_metrics]
        memory_percentages = [m["memory_percent"] for m in self.performance_metrics]
        
        summary = {
            "avg_cpu_percent": np.mean(cpu_percentages),
            "max_cpu_percent": np.max(cpu_percentages),
            "avg_memory_percent": np.mean(memory_percentages),
            "max_memory_percent": np.max(memory_percentages),
            "monitoring_duration": len(self.performance_metrics) * self.config.monitoring_interval
        }
        
        # GPU statistics if available
        gpu_memory_percentages = [m.get("gpu_memory_percent", 0) for m in self.performance_metrics]
        if any(gpu_memory_percentages):
            summary["avg_gpu_memory_percent"] = np.mean(gpu_memory_percentages)
            summary["max_gpu_memory_percent"] = np.max(gpu_memory_percentages)
        
        return summary

# Utility functions
def profile_function(func: Callable = None, config: ProfilingConfig = None):
    """Decorator for profiling functions."""
    def decorator(f) -> Any:
        @wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            if config and config.enabled:
                profiler = ProfilingOptimizer(config)
                result = profiler.profile_code(f, *args, **kwargs)
                return f(*args, **kwargs), result
            else:
                return f(*args, **kwargs)
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)

def optimize_dataloader(
    dataset: Dataset,
    config: ProfilingConfig = None,
    **kwargs
) -> DataLoader:
    """Optimize DataLoader for better performance."""
    if config is None:
        config = ProfilingConfig()
    
    optimizer = DataLoadingOptimizer(config)
    return optimizer.optimize_dataloader(dataset, **kwargs)

def optimize_preprocessing(
    preprocessing_func: Callable,
    config: ProfilingConfig = None
) -> Callable:
    """Optimize preprocessing function."""
    if config is None:
        config = ProfilingConfig()
    
    optimizer = PreprocessingOptimizer(config)
    return optimizer.optimize_preprocessing(preprocessing_func)

@contextmanager
def profiling_context(config: ProfilingConfig = None):
    """Context manager for profiling code blocks."""
    if config is None:
        config = ProfilingConfig()
    
    profiler = ProfilingOptimizer(config)
    real_time_profiler = RealTimeProfiler(config)
    
    try:
        real_time_profiler.start_monitoring()
        yield profiler
    finally:
        real_time_profiler.stop_monitoring()
        
        # Generate summary
        summary = real_time_profiler.get_performance_summary()
        if summary:
            logger.info(f"Profiling summary: {summary}")

# Example usage
async def example_profiling():
    """Example of profiling usage."""
    
    # Configuration
    config = ProfilingConfig(
        enabled=True,
        profile_cpu=True,
        profile_memory=True,
        profile_gpu=True,
        real_time_monitoring=True
    )
    
    # Create profiler
    profiler = ProfilingOptimizer(config)
    
    # Example function to profile
    def sample_function():
        
    """sample_function function."""
time.sleep(0.1)  # Simulate work
        return "result"
    
    # Profile the function
    result = profiler.profile_code(sample_function)
    
    print(f"Profiling result: {result}")
    print(f"Bottlenecks: {result.bottleneck_functions}")
    print(f"Recommendations: {result.recommendations}")
    
    return result

match __name__:
    case "__main__":
    asyncio.run(example_profiling()) 