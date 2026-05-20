from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import time
import cProfile
import pstats
import io
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, ContextManager
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import contextmanager
import functools
import threading
import multiprocessing
from collections import defaultdict, deque
import gc
import tracemalloc
import line_profiler
import memory_profiler
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from core.training_logger import TrainingLogger, TrainingEventType, LogLevel
from core.error_handling import ErrorHandler, ModelError
    import torch
    from torch.utils.data import DataLoader, TensorDataset
from typing import Any, List, Dict, Optional
import logging
"""
Code Profiler System

Comprehensive code profiling system for identifying and optimizing bottlenecks
in data loading, preprocessing, and training pipelines.
"""




@dataclass
class ProfilerConfig:
    """Configuration for code profiling"""
    
    # General profiling settings
    enable_profiling: bool = True
    profile_level: str = "detailed"  # basic, detailed, comprehensive
    save_profiles: bool = True
    profile_dir: str = "profiles"
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    monitor_interval: float = 1.0  # seconds
    track_memory: bool = True
    track_cpu: bool = True
    track_gpu: bool = True
    track_io: bool = True
    
    # Data loading profiling
    profile_data_loading: bool = True
    profile_preprocessing: bool = True
    profile_augmentation: bool = True
    profile_collate: bool = True
    
    # Training profiling
    profile_forward_pass: bool = True
    profile_backward_pass: bool = True
    profile_optimizer_step: bool = True
    profile_gradient_computation: bool = True
    
    # Memory profiling
    enable_memory_profiling: bool = True
    track_allocations: bool = True
    track_deallocations: bool = True
    memory_snapshots: bool = True
    
    # GPU profiling
    enable_gpu_profiling: bool = True
    track_gpu_memory: bool = True
    track_gpu_utilization: bool = True
    track_cuda_events: bool = True
    
    # I/O profiling
    enable_io_profiling: bool = True
    track_file_operations: bool = True
    track_network_operations: bool = True
    
    # Reporting
    generate_reports: bool = True
    create_visualizations: bool = True
    export_to_json: bool = True
    export_to_csv: bool = True


@dataclass
class ProfilerMetrics:
    """Profiler metrics container"""
    
    # Timing metrics
    execution_times: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    total_times: Dict[str, float] = field(default_factory=dict)
    average_times: Dict[str, float] = field(default_factory=dict)
    min_times: Dict[str, float] = field(default_factory=dict)
    max_times: Dict[str, float] = field(default_factory=dict)
    
    # Memory metrics
    memory_usage: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    peak_memory: Dict[str, float] = field(default_factory=dict)
    memory_leaks: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    # GPU metrics
    gpu_memory_usage: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    gpu_utilization: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    cuda_events: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    # I/O metrics
    io_operations: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    file_operations: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    network_operations: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    # Bottleneck analysis
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Call statistics
    call_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    call_frequencies: Dict[str, float] = field(default_factory=dict)


class CodeProfiler:
    """Comprehensive code profiler for identifying bottlenecks"""
    
    def __init__(
        self,
        config: ProfilerConfig,
        logger: Optional[TrainingLogger] = None
    ):
        
    """__init__ function."""
self.config = config
        self.logger = logger
        self.error_handler = ErrorHandler(debug_mode=True)
        
        # Initialize metrics
        self.metrics = ProfilerMetrics()
        
        # Profiling state
        self.active_profilers = {}
        self.profiling_stack = []
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Performance monitoring
        self.performance_data = defaultdict(list)
        self.start_time = time.time()
        
        # Memory profiling
        if self.config.enable_memory_profiling:
            tracemalloc.start()
        
        # GPU profiling
        self.gpu_events = {}
        if self.config.enable_gpu_profiling and torch.cuda.is_available():
            self._setup_gpu_profiling()
        
        # Create profile directory
        if self.config.save_profiles:
            Path(self.config.profile_dir).mkdir(parents=True, exist_ok=True)
        
        if self.logger:
            self.logger.log_info("Code profiler initialized successfully")
    
    def _setup_gpu_profiling(self) -> Any:
        """Setup GPU profiling with CUDA events"""
        
        try:
            if torch.cuda.is_available():
                # Create CUDA events for timing
                self.gpu_events = {
                    "start": torch.cuda.Event(enable_timing=True),
                    "end": torch.cuda.Event(enable_timing=True)
                }
                
                if self.logger:
                    self.logger.log_info("GPU profiling setup completed")
                    
        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"GPU profiling setup failed: {e}")
    
    @contextmanager
    def profile_section(self, section_name: str, category: str = "general"):
        """Context manager for profiling code sections"""
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_gpu_memory = self._get_gpu_memory_usage()
        
        # Record GPU event start
        if self.config.enable_gpu_profiling and torch.cuda.is_available():
            self.gpu_events["start"].record()
        
        try:
            yield
        finally:
            # Record GPU event end
            if self.config.enable_gpu_profiling and torch.cuda.is_available():
                self.gpu_events["end"].record()
                torch.cuda.synchronize()
                gpu_time = self.gpu_events["start"].elapsed_time(self.gpu_events["end"])
                self.metrics.cuda_events[section_name].append(gpu_time)
            
            # Calculate metrics
            execution_time = time.time() - start_time
            end_memory = self._get_memory_usage()
            end_gpu_memory = self._get_gpu_memory_usage()
            
            # Store metrics
            self.metrics.execution_times[section_name].append(execution_time)
            self.metrics.memory_usage[section_name].append(end_memory - start_memory)
            self.metrics.gpu_memory_usage[section_name].append(end_gpu_memory - start_gpu_memory)
            self.metrics.call_counts[section_name] += 1
            
            # Update statistics
            self._update_statistics(section_name)
            
            # Log if significant
            if execution_time > 1.0:  # Log sections taking more than 1 second
                if self.logger:
                    self.logger.log_info(
                        f"Profiled section '{section_name}': {execution_time:.4f}s, "
                        f"Memory: {end_memory - start_memory:.3f}MB, "
                        f"GPU Memory: {end_gpu_memory - start_gpu_memory:.3f}MB"
                    )
    
    def profile_function(self, func: Callable, category: str = "function"):
        """Decorator for profiling functions"""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            function_name = f"{category}.{func.__name__}"
            
            with self.profile_section(function_name, category):
                return func(*args, **kwargs)
        
        return wrapper
    
    def profile_data_loading(self, dataloader: DataLoader, num_batches: int = 10):
        """Profile data loading performance"""
        
        if not self.config.profile_data_loading:
            return {}
        
        print(f"\n{'='*60}")
        print("PROFILING DATA LOADING")
        print(f"{'='*60}")
        
        metrics = {}
        
        # Profile DataLoader initialization
        with self.profile_section("dataloader_init", "data_loading"):
            # Measure initialization time
            pass
        
        # Profile batch loading
        batch_times = []
        batch_sizes = []
        memory_usage = []
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            batch_start = time.time()
            batch_memory_start = self._get_memory_usage()
            
            # Process batch
            if isinstance(batch, (list, tuple)):
                batch_size = len(batch[0]) if batch[0] is not None else 0
            else:
                batch_size = len(batch) if batch is not None else 0
            
            batch_memory_end = self._get_memory_usage()
            batch_time = time.time() - batch_start
            
            batch_times.append(batch_time)
            batch_sizes.append(batch_size)
            memory_usage.append(batch_memory_end - batch_memory_start)
            
            print(f"Batch {i+1}: Time={batch_time:.4f}s, Size={batch_size}, "
                  f"Memory={batch_memory_end - batch_memory_start:.3f}MB")
        
        # Calculate statistics
        metrics.update({
            "avg_batch_time": np.mean(batch_times),
            "min_batch_time": np.min(batch_times),
            "max_batch_time": np.max(batch_times),
            "std_batch_time": np.std(batch_times),
            "avg_batch_size": np.mean(batch_sizes),
            "avg_memory_per_batch": np.mean(memory_usage),
            "throughput": np.mean(batch_sizes) / np.mean(batch_times),
            "total_batches_profiled": len(batch_times)
        })
        
        # Store in metrics
        self.metrics.execution_times["data_loading_batch"] = batch_times
        self.metrics.memory_usage["data_loading_batch"] = memory_usage
        
        # Identify bottlenecks
        bottlenecks = self._identify_data_loading_bottlenecks(metrics)
        self.metrics.bottlenecks.extend(bottlenecks)
        
        # Print summary
        print(f"\nData Loading Summary:")
        print(f"  Average batch time: {metrics['avg_batch_time']:.4f}s")
        print(f"  Average batch size: {metrics['avg_batch_size']:.1f}")
        print(f"  Throughput: {metrics['throughput']:.1f} samples/second")
        print(f"  Average memory per batch: {metrics['avg_memory_per_batch']:.3f}MB")
        
        if bottlenecks:
            print(f"\nIdentified Bottlenecks:")
            for bottleneck in bottlenecks:
                print(f"  - {bottleneck['description']}")
        
        return metrics
    
    def profile_preprocessing(self, preprocessing_func: Callable, sample_data: Any, num_samples: int = 100):
        """Profile preprocessing performance"""
        
        if not self.config.profile_preprocessing:
            return {}
        
        print(f"\n{'='*60}")
        print("PROFILING PREPROCESSING")
        print(f"{'='*60}")
        
        metrics = {}
        processing_times = []
        memory_usage = []
        
        for i in range(num_samples):
            with self.profile_section(f"preprocessing_sample_{i}", "preprocessing"):
                start_memory = self._get_memory_usage()
                start_time = time.time()
                
                # Apply preprocessing
                processed_data = preprocessing_func(sample_data)
                
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                processing_times.append(end_time - start_time)
                memory_usage.append(end_memory - start_memory)
        
        # Calculate statistics
        metrics.update({
            "avg_processing_time": np.mean(processing_times),
            "min_processing_time": np.min(processing_times),
            "max_processing_time": np.max(processing_times),
            "std_processing_time": np.std(processing_times),
            "avg_memory_usage": np.mean(memory_usage),
            "throughput": num_samples / np.sum(processing_times),
            "total_samples_profiled": num_samples
        })
        
        # Store in metrics
        self.metrics.execution_times["preprocessing"] = processing_times
        self.metrics.memory_usage["preprocessing"] = memory_usage
        
        # Print summary
        print(f"\nPreprocessing Summary:")
        print(f"  Average processing time: {metrics['avg_processing_time']:.6f}s")
        print(f"  Throughput: {metrics['throughput']:.1f} samples/second")
        print(f"  Average memory usage: {metrics['avg_memory_usage']:.3f}MB")
        
        return metrics
    
    def profile_model_training(self, model: nn.Module, dataloader: DataLoader, num_batches: int = 10):
        """Profile model training performance"""
        
        print(f"\n{'='*60}")
        print("PROFILING MODEL TRAINING")
        print(f"{'='*60}")
        
        device = next(model.parameters()).device
        model.train()
        
        metrics = {
            "forward_times": [],
            "backward_times": [],
            "optimizer_times": [],
            "total_times": [],
            "memory_usage": [],
            "gpu_memory_usage": []
        }
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            batch_start = time.time()
            batch_memory_start = self._get_memory_usage()
            batch_gpu_memory_start = self._get_gpu_memory_usage()
            
            # Forward pass
            if self.config.profile_forward_pass:
                with self.profile_section(f"forward_pass_batch_{i}", "training"):
                    forward_start = time.time()
                    outputs = model(inputs)
                    forward_time = time.time() - forward_start
                    metrics["forward_times"].append(forward_time)
            
            # Loss computation
            loss = loss_fn(outputs, targets)
            
            # Backward pass
            if self.config.profile_backward_pass:
                with self.profile_section(f"backward_pass_batch_{i}", "training"):
                    backward_start = time.time()
                    loss.backward()
                    backward_time = time.time() - backward_start
                    metrics["backward_times"].append(backward_time)
            
            # Optimizer step
            if self.config.profile_optimizer_step:
                with self.profile_section(f"optimizer_step_batch_{i}", "training"):
                    optimizer_start = time.time()
                    optimizer.step()
                    optimizer.zero_grad()
                    optimizer_time = time.time() - optimizer_start
                    metrics["optimizer_times"].append(optimizer_time)
            
            batch_time = time.time() - batch_start
            batch_memory_end = self._get_memory_usage()
            batch_gpu_memory_end = self._get_gpu_memory_usage()
            
            metrics["total_times"].append(batch_time)
            metrics["memory_usage"].append(batch_memory_end - batch_memory_start)
            metrics["gpu_memory_usage"].append(batch_gpu_memory_end - batch_gpu_memory_start)
            
            print(f"Batch {i+1}: Total={batch_time:.4f}s, "
                  f"Forward={metrics['forward_times'][-1]:.4f}s, "
                  f"Backward={metrics['backward_times'][-1]:.4f}s, "
                  f"Optimizer={metrics['optimizer_times'][-1]:.4f}s")
        
        # Calculate statistics
        summary = {
            "avg_total_time": np.mean(metrics["total_times"]),
            "avg_forward_time": np.mean(metrics["forward_times"]),
            "avg_backward_time": np.mean(metrics["backward_times"]),
            "avg_optimizer_time": np.mean(metrics["optimizer_times"]),
            "avg_memory_usage": np.mean(metrics["memory_usage"]),
            "avg_gpu_memory_usage": np.mean(metrics["gpu_memory_usage"]),
            "throughput": len(metrics["total_times"]) / np.sum(metrics["total_times"])
        }
        
        # Store in metrics
        self.metrics.execution_times["training_total"] = metrics["total_times"]
        self.metrics.execution_times["training_forward"] = metrics["forward_times"]
        self.metrics.execution_times["training_backward"] = metrics["backward_times"]
        self.metrics.execution_times["training_optimizer"] = metrics["optimizer_times"]
        self.metrics.memory_usage["training"] = metrics["memory_usage"]
        self.metrics.gpu_memory_usage["training"] = metrics["gpu_memory_usage"]
        
        # Print summary
        print(f"\nTraining Summary:")
        print(f"  Average total time: {summary['avg_total_time']:.4f}s")
        print(f"  Average forward time: {summary['avg_forward_time']:.4f}s")
        print(f"  Average backward time: {summary['avg_backward_time']:.4f}s")
        print(f"  Average optimizer time: {summary['avg_optimizer_time']:.4f}s")
        print(f"  Throughput: {summary['throughput']:.1f} batches/second")
        print(f"  Average memory usage: {summary['avg_memory_usage']:.3f}MB")
        print(f"  Average GPU memory usage: {summary['avg_gpu_memory_usage']:.3f}MB")
        
        return summary
    
    def _identify_data_loading_bottlenecks(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify data loading bottlenecks"""
        
        bottlenecks = []
        
        # Check for slow batch loading
        if metrics.get("avg_batch_time", 0) > 0.1:  # More than 100ms per batch
            bottlenecks.append({
                "type": "slow_batch_loading",
                "severity": "high" if metrics["avg_batch_time"] > 0.5 else "medium",
                "description": f"Slow batch loading: {metrics['avg_batch_time']:.4f}s average",
                "recommendation": "Consider increasing num_workers, using pin_memory, or optimizing data preprocessing"
            })
        
        # Check for memory issues
        if metrics.get("avg_memory_per_batch", 0) > 100:  # More than 100MB per batch
            bottlenecks.append({
                "type": "high_memory_usage",
                "severity": "high" if metrics["avg_memory_per_batch"] > 500 else "medium",
                "description": f"High memory usage per batch: {metrics['avg_memory_per_batch']:.3f}MB",
                "recommendation": "Consider reducing batch size or optimizing memory usage in data loading"
            })
        
        # Check for low throughput
        if metrics.get("throughput", 0) < 100:  # Less than 100 samples/second
            bottlenecks.append({
                "type": "low_throughput",
                "severity": "high" if metrics["throughput"] < 50 else "medium",
                "description": f"Low throughput: {metrics['throughput']:.1f} samples/second",
                "recommendation": "Optimize data loading pipeline, increase num_workers, or use prefetching"
            })
        
        return bottlenecks
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
            return 0.0
        except Exception:
            return 0.0
    
    def _update_statistics(self, section_name: str):
        """Update statistics for a profiled section"""
        
        times = self.metrics.execution_times[section_name]
        if times:
            self.metrics.total_times[section_name] = sum(times)
            self.metrics.average_times[section_name] = np.mean(times)
            self.metrics.min_times[section_name] = np.min(times)
            self.metrics.max_times[section_name] = np.max(times)
            
            # Calculate call frequency
            total_time = time.time() - self.start_time
            if total_time > 0:
                self.metrics.call_frequencies[section_name] = len(times) / total_time
    
    def start_performance_monitoring(self) -> Any:
        """Start continuous performance monitoring"""
        
        if not self.config.enable_performance_monitoring:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        if self.logger:
            self.logger.log_info("Performance monitoring started")
    
    def stop_performance_monitoring(self) -> Any:
        """Stop continuous performance monitoring"""
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        if self.logger:
            self.logger.log_info("Performance monitoring stopped")
    
    def _monitoring_loop(self) -> Any:
        """Performance monitoring loop"""
        
        while self.monitoring_active:
            try:
                # CPU usage
                if self.config.track_cpu:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.performance_data["cpu_usage"].append(cpu_percent)
                
                # Memory usage
                if self.config.track_memory:
                    memory_usage = self._get_memory_usage()
                    self.performance_data["memory_usage"].append(memory_usage)
                
                # GPU usage
                if self.config.track_gpu and torch.cuda.is_available():
                    gpu_memory = self._get_gpu_memory_usage()
                    self.performance_data["gpu_memory_usage"].append(gpu_memory)
                
                time.sleep(self.config.monitor_interval)
                
            except Exception as e:
                if self.logger:
                    self.logger.log_warning(f"Performance monitoring error: {e}")
    
    def generate_profiling_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive profiling report"""
        
        if not self.config.generate_reports:
            return {}
        
        print(f"\n{'='*60}")
        print("GENERATING PROFILING REPORT")
        print(f"{'='*60}")
        
        report = {
            "summary": self._generate_summary(),
            "bottlenecks": self.metrics.bottlenecks,
            "recommendations": self._generate_recommendations(),
            "detailed_metrics": self._generate_detailed_metrics(),
            "performance_data": dict(self.performance_data)
        }
        
        # Save report
        if save_path is None:
            save_path = Path(self.config.profile_dir) / "profiling_report.json"
        
        with open(save_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(report, f, indent=2)
        
        # Print summary
        self._print_report_summary(report)
        
        # Create visualizations
        if self.config.create_visualizations:
            self._create_visualizations(report)
        
        if self.logger:
            self.logger.log_info(f"Profiling report saved to {save_path}")
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate profiling summary"""
        
        total_sections = len(self.metrics.execution_times)
        total_calls = sum(self.metrics.call_counts.values())
        total_time = sum(self.metrics.total_times.values())
        
        # Find slowest sections
        slowest_sections = sorted(
            self.metrics.average_times.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Find most called sections
        most_called = sorted(
            self.metrics.call_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "total_sections_profiled": total_sections,
            "total_function_calls": total_calls,
            "total_profiling_time": total_time,
            "slowest_sections": slowest_sections,
            "most_called_sections": most_called,
            "bottlenecks_found": len(self.metrics.bottlenecks)
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        
        recommendations = []
        
        # Analyze timing data
        for section, avg_time in self.metrics.average_times.items():
            if avg_time > 1.0:  # More than 1 second
                recommendations.append(f"Optimize '{section}': {avg_time:.4f}s average execution time")
            
            if self.metrics.call_counts[section] > 1000:  # Called more than 1000 times
                recommendations.append(f"Consider caching results for '{section}': called {self.metrics.call_counts[section]} times")
        
        # Analyze memory usage
        for section, memory_usage in self.metrics.memory_usage.items():
            if memory_usage and np.mean(memory_usage) > 100:  # More than 100MB average
                recommendations.append(f"Optimize memory usage in '{section}': {np.mean(memory_usage):.3f}MB average")
        
        # Add bottleneck recommendations
        for bottleneck in self.metrics.bottlenecks:
            recommendations.append(bottleneck["recommendation"])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_detailed_metrics(self) -> Dict[str, Any]:
        """Generate detailed metrics"""
        
        return {
            "execution_times": dict(self.metrics.execution_times),
            "memory_usage": dict(self.metrics.memory_usage),
            "gpu_memory_usage": dict(self.metrics.gpu_memory_usage),
            "call_counts": dict(self.metrics.call_counts),
            "call_frequencies": dict(self.metrics.call_frequencies)
        }
    
    def _print_report_summary(self, report: Dict[str, Any]):
        """Print report summary"""
        
        summary = report["summary"]
        
        print(f"\nProfiling Report Summary:")
        print(f"  Sections profiled: {summary['total_sections_profiled']}")
        print(f"  Total function calls: {summary['total_function_calls']}")
        print(f"  Total profiling time: {summary['total_profiling_time']:.4f}s")
        print(f"  Bottlenecks found: {summary['bottlenecks_found']}")
        
        if summary["slowest_sections"]:
            print(f"\nSlowest sections:")
            for section, time in summary["slowest_sections"]:
                print(f"  {section}: {time:.4f}s average")
        
        if report["recommendations"]:
            print(f"\nTop recommendations:")
            for i, rec in enumerate(report["recommendations"][:5], 1):
                print(f"  {i}. {rec}")
    
    def _create_visualizations(self, report: Dict[str, Any]):
        """Create profiling visualizations"""
        
        viz_dir = Path(self.config.profile_dir) / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Execution time distribution
        if self.metrics.execution_times:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            sections = list(self.metrics.average_times.keys())
            times = list(self.metrics.average_times.values())
            
            bars = ax.bar(range(len(sections)), times)
            ax.set_title("Average Execution Times by Section")
            ax.set_xlabel("Section")
            ax.set_ylabel("Time (seconds)")
            ax.set_xticks(range(len(sections)))
            ax.set_xticklabels(sections, rotation=45, ha='right')
            
            # Add value labels
            for bar, time in zip(bars, times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{time:.4f}s', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(viz_dir / "execution_times.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Memory usage over time
        if self.performance_data.get("memory_usage"):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            memory_data = self.performance_data["memory_usage"]
            time_points = range(len(memory_data))
            
            ax.plot(time_points, memory_data, 'b-', linewidth=2)
            ax.set_title("Memory Usage Over Time")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Memory Usage (MB)")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(viz_dir / "memory_usage.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Call frequency analysis
        if self.metrics.call_frequencies:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            sections = list(self.metrics.call_frequencies.keys())
            frequencies = list(self.metrics.call_frequencies.values())
            
            bars = ax.bar(range(len(sections)), frequencies)
            ax.set_title("Function Call Frequencies")
            ax.set_xlabel("Section")
            ax.set_ylabel("Calls per Second")
            ax.set_xticks(range(len(sections)))
            ax.set_xticklabels(sections, rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(viz_dir / "call_frequencies.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {viz_dir}")
    
    def cleanup(self) -> Any:
        """Cleanup profiler resources"""
        
        try:
            # Stop monitoring
            self.stop_performance_monitoring()
            
            # Stop memory profiling
            if self.config.enable_memory_profiling:
                tracemalloc.stop()
            
            # Generate final report
            if self.config.generate_reports:
                self.generate_profiling_report()
            
            if self.logger:
                self.logger.log_info("Code profiler cleanup completed")
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Profiler cleanup", "cleanup")


# Utility functions
def create_code_profiler(
    enable_profiling: bool = True,
    profile_level: str = "detailed",
    save_profiles: bool = True,
    logger: Optional[TrainingLogger] = None,
    **kwargs
) -> CodeProfiler:
    """Create a code profiler with default settings"""
    
    config = ProfilerConfig(
        enable_profiling=enable_profiling,
        profile_level=profile_level,
        save_profiles=save_profiles,
        **kwargs
    )
    
    return CodeProfiler(config, logger)


def profile_function(func: Callable, category: str = "function"):
    """Decorator for profiling functions"""
    
    def decorator(*args, **kwargs) -> Any:
        # This would need to be used with a global profiler instance
        # For now, just add timing
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        print(f"Function '{func.__name__}' took {execution_time:.6f}s")
        return result
    
    return decorator


if __name__ == "__main__":
    # Example usage
    
    # Create sample data
    data = torch.randn(1000, 10)
    labels = torch.randint(0, 2, (1000,))
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create profiler
    profiler = create_code_profiler()
    
    # Profile data loading
    profiler.profile_data_loading(dataloader)
    
    # Profile model training
    model = torch.nn.Linear(10, 2)
    profiler.profile_model_training(model, dataloader)
    
    # Generate report
    profiler.generate_profiling_report()
    
    # Cleanup
    profiler.cleanup() 