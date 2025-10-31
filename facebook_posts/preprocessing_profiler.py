#!/usr/bin/env python3
"""
Preprocessing Profiler
Specialized profiler for identifying and optimizing bottlenecks in data preprocessing operations.
"""

import time
import psutil
import threading
import multiprocessing
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, List, Dict, Optional, Callable, Tuple, Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from logging_config import get_logger, log_performance_metrics


class PreprocessingBottleneck(Enum):
    """Types of preprocessing bottlenecks."""
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    GPU_UNDERUTILIZED = "gpu_underutilized"
    I_O_BOUND = "i_o_bound"
    VECTORIZATION_INEFFICIENT = "vectorization_inefficient"
    CACHING_MISSED = "caching_missed"
    BATCH_SIZE_INEFFICIENT = "batch_size_inefficient"
    ALGORITHM_INEFFICIENT = "algorithm_inefficient"


@dataclass
class PreprocessingProfile:
    """Profile of preprocessing performance."""
    operation_name: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    execution_time: float
    memory_usage_mb: float
    gpu_memory_usage_mb: float
    cpu_usage_percent: float
    throughput_samples_per_second: float
    bottlenecks: List[PreprocessingBottleneck] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessingProfilerConfig:
    """Configuration for preprocessing profiling."""
    enable_memory_tracking: bool = True
    enable_gpu_tracking: bool = True
    enable_cpu_tracking: bool = True
    enable_io_tracking: bool = True
    profiling_duration: int = 60  # seconds
    batch_size_range: List[int] = field(default_factory=lambda: [1, 8, 16, 32, 64])
    enable_caching: bool = True
    cache_size: int = 1000
    save_profiles: bool = True
    profile_output_dir: str = "preprocessing_profiles"
    auto_optimize: bool = False


class PreprocessingProfiler:
    """Specialized profiler for preprocessing bottlenecks."""
    
    def __init__(self, config: PreprocessingProfilerConfig):
        self.config = config
        self.logger = get_logger("preprocessing_profiler")
        self.profiles: List[PreprocessingProfile] = []
        self.current_profile: Optional[PreprocessingProfile] = None
        
        # Performance tracking
        self.performance_metrics = defaultdict(float)
        self.bottleneck_history = []
        
        # Caching
        self.preprocessing_cache = {}
        
        # Initialize tracking
        self._initialize_tracking()
    
    def _initialize_tracking(self):
        """Initialize performance tracking."""
        if self.config.enable_memory_tracking:
            self.memory_tracker = MemoryTracker()
        if self.config.enable_gpu_tracking:
            self.gpu_tracker = GPUTracker()
        if self.config.enable_cpu_tracking:
            self.cpu_tracker = CPUTracker()
        if self.config.enable_io_tracking:
            self.io_tracker = IOTracker()
    
    def profile_preprocessing_function(self, preprocessing_func: Callable, 
                                    sample_data: torch.Tensor,
                                    num_iterations: int = 10,
                                    operation_name: str = None) -> PreprocessingProfile:
        """Profile a specific preprocessing function."""
        operation_name = operation_name or preprocessing_func.__name__
        self.logger.info(f"Profiling preprocessing function: {operation_name}")
        
        # Start tracking
        self._start_tracking()
        
        # Profile preprocessing
        start_time = time.time()
        execution_times = []
        memory_usage = []
        gpu_memory_usage = []
        cpu_usage = []
        
        for i in range(num_iterations):
            iter_start = time.time()
            
            # Apply preprocessing
            if self.config.enable_caching and sample_data.shape in self.preprocessing_cache:
                result = self.preprocessing_cache[sample_data.shape]
            else:
                result = preprocessing_func(sample_data)
                if self.config.enable_caching and len(self.preprocessing_cache) < self.config.cache_size:
                    self.preprocessing_cache[sample_data.shape] = result
            
            iter_end = time.time()
            iter_time = iter_end - iter_start
            
            # Record metrics
            execution_times.append(iter_time)
            memory_usage.append(self._get_memory_usage())
            gpu_memory_usage.append(self._get_gpu_memory_usage())
            cpu_usage.append(self._get_cpu_usage())
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Stop tracking
        self._stop_tracking()
        
        # Calculate metrics
        avg_execution_time = np.mean(execution_times)
        throughput = num_iterations / total_time
        avg_memory = np.mean(memory_usage)
        avg_gpu_memory = np.mean(gpu_memory_usage)
        avg_cpu = np.mean(cpu_usage)
        
        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(
            avg_execution_time, avg_memory, avg_gpu_memory, avg_cpu,
            sample_data, throughput, preprocessing_func
        )
        
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(bottlenecks, preprocessing_func, sample_data)
        
        # Create profile
        profile = PreprocessingProfile(
            operation_name=operation_name,
            input_shape=sample_data.shape,
            output_shape=result.shape if hasattr(result, 'shape') else None,
            execution_time=avg_execution_time,
            memory_usage_mb=avg_memory,
            gpu_memory_usage_mb=avg_gpu_memory,
            cpu_usage_percent=avg_cpu,
            throughput_samples_per_second=throughput,
            bottlenecks=bottlenecks,
            optimization_suggestions=suggestions,
            preprocessing_config=self._extract_preprocessing_config(preprocessing_func)
        )
        
        self.profiles.append(profile)
        self.current_profile = profile
        
        # Log results
        self.logger.info(f"Preprocessing profile completed:")
        self.logger.info(f"  Operation: {operation_name}")
        self.logger.info(f"  Input shape: {sample_data.shape}")
        self.logger.info(f"  Avg execution time: {avg_execution_time:.4f}s")
        self.logger.info(f"  Throughput: {throughput:.2f} samples/s")
        self.logger.info(f"  Bottlenecks detected: {len(bottlenecks)}")
        
        return profile
    
    def profile_batch_preprocessing(self, preprocessing_func: Callable,
                                  sample_data: torch.Tensor,
                                  batch_sizes: List[int] = None) -> List[PreprocessingProfile]:
        """Profile preprocessing with different batch sizes."""
        batch_sizes = batch_sizes or self.config.batch_size_range
        
        self.logger.info(f"Profiling batch preprocessing with {len(batch_sizes)} batch sizes")
        
        profiles = []
        
        for batch_size in batch_sizes:
            try:
                # Create batch data
                if batch_size == 1:
                    batch_data = sample_data.unsqueeze(0)
                else:
                    batch_data = sample_data.repeat(batch_size, *([1] * (len(sample_data.shape) - 1)))
                
                # Profile this batch size
                profile = self.profile_preprocessing_function(
                    preprocessing_func, batch_data, num_iterations=5,
                    operation_name=f"{preprocessing_func.__name__}_batch_{batch_size}"
                )
                profiles.append(profile)
                
            except Exception as e:
                self.logger.warning(f"Failed to profile batch_size={batch_size}: {e}")
                continue
        
        return profiles
    
    def _detect_bottlenecks(self, avg_execution_time: float, avg_memory: float,
                           avg_gpu_memory: float, avg_cpu: float,
                           sample_data: torch.Tensor, throughput: float,
                           preprocessing_func: Callable) -> List[PreprocessingBottleneck]:
        """Detect bottlenecks in preprocessing."""
        bottlenecks = []
        
        # Check for CPU-intensive operations
        if avg_cpu > 80:  # More than 80% CPU usage
            bottlenecks.append(PreprocessingBottleneck.CPU_INTENSIVE)
        
        # Check for memory-intensive operations
        if avg_memory > 500:  # More than 500MB
            bottlenecks.append(PreprocessingBottleneck.MEMORY_INTENSIVE)
        
        # Check for GPU underutilization
        if torch.cuda.is_available() and avg_gpu_memory < 100:  # Less than 100MB
            bottlenecks.append(PreprocessingBottleneck.GPU_UNDERUTILIZED)
        
        # Check for I/O bound operations
        if avg_execution_time > 0.1:  # More than 100ms
            bottlenecks.append(PreprocessingBottleneck.I_O_BOUND)
        
        # Check for vectorization inefficiency
        if sample_data.numel() > 1000 and avg_execution_time > 0.01:  # Large data, slow processing
            bottlenecks.append(PreprocessingBottleneck.VECTORIZATION_INEFFICIENT)
        
        # Check for missed caching opportunities
        if not self.config.enable_caching and avg_execution_time > 0.01:
            bottlenecks.append(PreprocessingBottleneck.CACHING_MISSED)
        
        # Check for batch size inefficiency
        if sample_data.shape[0] < 8:  # Small batch size
            bottlenecks.append(PreprocessingBottleneck.BATCH_SIZE_INEFFICIENT)
        
        # Check for algorithm inefficiency
        if hasattr(preprocessing_func, '__name__'):
            func_name = preprocessing_func.__name__.lower()
            if any(keyword in func_name for keyword in ['loop', 'iter', 'for']):
                bottlenecks.append(PreprocessingBottleneck.ALGORITHM_INEFFICIENT)
        
        return bottlenecks
    
    def _generate_optimization_suggestions(self, bottlenecks: List[PreprocessingBottleneck],
                                         preprocessing_func: Callable,
                                         sample_data: torch.Tensor) -> List[str]:
        """Generate optimization suggestions based on detected bottlenecks."""
        suggestions = []
        
        for bottleneck in bottlenecks:
            if bottleneck == PreprocessingBottleneck.CPU_INTENSIVE:
                suggestions.extend([
                    "Move preprocessing to GPU using torch.cuda.amp",
                    "Use vectorized operations instead of loops",
                    "Implement parallel processing with multiprocessing",
                    "Consider using numba or Cython for hot paths"
                ])
            
            elif bottleneck == PreprocessingBottleneck.MEMORY_INTENSIVE:
                suggestions.extend([
                    "Reduce batch size to decrease memory usage",
                    "Use in-place operations when possible",
                    "Implement memory pooling",
                    "Consider using torch.utils.checkpoint"
                ])
            
            elif bottleneck == PreprocessingBottleneck.GPU_UNDERUTILIZED:
                suggestions.extend([
                    "Move preprocessing operations to GPU",
                    "Use torch.cuda.amp for mixed precision",
                    "Batch GPU operations for better utilization",
                    "Consider using GPU-optimized libraries"
                ])
            
            elif bottleneck == PreprocessingBottleneck.I_O_BOUND:
                suggestions.extend([
                    "Use async I/O operations",
                    "Implement data prefetching",
                    "Use memory-mapped files",
                    "Consider using databases for structured data"
                ])
            
            elif bottleneck == PreprocessingBottleneck.VECTORIZATION_INEFFICIENT:
                suggestions.extend([
                    "Replace loops with vectorized operations",
                    "Use torch.vectorize for custom functions",
                    "Implement batch processing",
                    "Consider using torch.jit.script"
                ])
            
            elif bottleneck == PreprocessingBottleneck.CACHING_MISSED:
                suggestions.extend([
                    "Enable preprocessing result caching",
                    "Cache intermediate results",
                    "Use LRU cache for frequently accessed data",
                    "Implement smart cache invalidation"
                ])
            
            elif bottleneck == PreprocessingBottleneck.BATCH_SIZE_INEFFICIENT:
                current_batch_size = sample_data.shape[0]
                suggestions.extend([
                    f"Increase batch size from {current_batch_size} to at least 8",
                    "Use gradient accumulation for larger effective batch sizes",
                    "Implement dynamic batch sizing"
                ])
            
            elif bottleneck == PreprocessingBottleneck.ALGORITHM_INEFFICIENT:
                suggestions.extend([
                    "Replace loops with vectorized operations",
                    "Use torch.vectorize for custom functions",
                    "Implement parallel processing",
                    "Consider using specialized libraries"
                ])
        
        return list(set(suggestions))  # Remove duplicates
    
    def _extract_preprocessing_config(self, preprocessing_func: Callable) -> Dict[str, Any]:
        """Extract configuration information from preprocessing function."""
        config = {}
        
        # Check if function has config attributes
        if hasattr(preprocessing_func, 'config'):
            config.update(preprocessing_func.config)
        
        # Check for common preprocessing parameters
        if hasattr(preprocessing_func, 'mean'):
            config['mean'] = preprocessing_func.mean
        if hasattr(preprocessing_func, 'std'):
            config['std'] = preprocessing_func.std
        if hasattr(preprocessing_func, 'size'):
            config['size'] = preprocessing_func.size
        
        return config
    
    def _start_tracking(self):
        """Start performance tracking."""
        if hasattr(self, 'memory_tracker'):
            self.memory_tracker.start_session()
        if hasattr(self, 'gpu_tracker'):
            self.gpu_tracker.start_session()
        if hasattr(self, 'cpu_tracker'):
            self.cpu_tracker.start_session()
        if hasattr(self, 'io_tracker'):
            self.io_tracker.start_session()
    
    def _stop_tracking(self):
        """Stop performance tracking."""
        if hasattr(self, 'memory_tracker'):
            self.memory_tracker.stop_session()
        if hasattr(self, 'gpu_tracker'):
            self.gpu_tracker.stop_session()
        if hasattr(self, 'cpu_tracker'):
            self.cpu_tracker.stop_session()
        if hasattr(self, 'io_tracker'):
            self.io_tracker.stop_session()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if hasattr(self, 'memory_tracker'):
            return self.memory_tracker.get_current_memory()
        return 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if hasattr(self, 'gpu_tracker'):
            return self.gpu_tracker.get_current_memory()
        return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if hasattr(self, 'cpu_tracker'):
            return self.cpu_tracker.get_current_usage()
        return 0.0
    
    def get_optimal_preprocessing_config(self) -> Optional[PreprocessingProfile]:
        """Get the optimal preprocessing configuration based on profiles."""
        if not self.profiles:
            return None
        
        # Score profiles based on multiple criteria
        scored_profiles = []
        for profile in self.profiles:
            score = self._calculate_profile_score(profile)
            scored_profiles.append((score, profile))
        
        # Return the profile with the highest score
        scored_profiles.sort(key=lambda x: x[0], reverse=True)
        return scored_profiles[0][1]
    
    def _calculate_profile_score(self, profile: PreprocessingProfile) -> float:
        """Calculate a score for a profile based on multiple criteria."""
        score = 0.0
        
        # Higher throughput is better
        score += profile.throughput_samples_per_second * 10
        
        # Lower execution time is better
        score -= profile.execution_time * 100
        
        # Lower memory usage is better
        score -= profile.memory_usage_mb / 100
        
        # Lower GPU memory usage is better (but some usage is good)
        if profile.gpu_memory_usage_mb > 0:
            score += 1  # Bonus for using GPU
            score -= profile.gpu_memory_usage_mb / 200  # Penalty for excessive usage
        
        # Lower CPU usage is better
        score -= profile.cpu_usage_percent / 10
        
        # Fewer bottlenecks is better
        score -= len(profile.bottlenecks) * 5
        
        # Prefer larger batch sizes (efficiency)
        if profile.input_shape and len(profile.input_shape) > 0:
            batch_size = profile.input_shape[0]
            if batch_size >= 16:
                score += 2
            elif batch_size >= 8:
                score += 1
        
        return score
    
    def generate_optimization_report(self, output_dir: str = None) -> str:
        """Generate comprehensive optimization report."""
        output_dir = output_dir or self.config.profile_output_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not self.profiles:
            return "No profiles available for report generation"
        
        # Find optimal configuration
        optimal_profile = self.get_optimal_preprocessing_config()
        
        # Generate report
        report = f"""# Preprocessing Optimization Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Summary
Total preprocessing operations profiled: {len(self.profiles)}

## Optimal Configuration
- Operation: {optimal_profile.operation_name}
- Input Shape: {optimal_profile.input_shape}
- Output Shape: {optimal_profile.output_shape}
- Execution Time: {optimal_profile.execution_time:.4f} seconds
- Throughput: {optimal_profile.throughput_samples_per_second:.2f} samples/second
- Memory Usage: {optimal_profile.memory_usage_mb:.1f} MB
- GPU Memory Usage: {optimal_profile.gpu_memory_usage_mb:.1f} MB
- CPU Usage: {optimal_profile.cpu_usage_percent:.1f}%
- Bottlenecks: {len(optimal_profile.bottlenecks)}

## All Operations
"""
        
        for i, profile in enumerate(self.profiles, 1):
            report += f"""
### Operation {i}: {profile.operation_name}
- Input Shape: {profile.input_shape}
- Execution Time: {profile.execution_time:.4f}s
- Throughput: {profile.throughput_samples_per_second:.2f} samples/s
- Memory: {profile.memory_usage_mb:.1f} MB
- GPU Memory: {profile.gpu_memory_usage_mb:.1f} MB
- CPU: {profile.cpu_usage_percent:.1f}%
- Bottlenecks: {len(profile.bottlenecks)}
- Suggestions: {len(profile.optimization_suggestions)}
"""
        
        # Save report
        report_file = output_path / "preprocessing_optimization_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Generate visualizations
        self._generate_optimization_visualizations(output_path)
        
        return f"Report generated: {report_file}"
    
    def _generate_optimization_visualizations(self, output_path: Path):
        """Generate visualization charts for optimization analysis."""
        if not self.profiles:
            return
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extract data
        operation_names = [p.operation_name for p in self.profiles]
        execution_times = [p.execution_time for p in self.profiles]
        throughputs = [p.throughput_samples_per_second for p in self.profiles]
        memory_usage = [p.memory_usage_mb for p in self.profiles]
        gpu_memory_usage = [p.gpu_memory_usage_mb for p in self.profiles]
        cpu_usage = [p.cpu_usage_percent for p in self.profiles]
        bottleneck_counts = [len(p.bottlenecks) for p in self.profiles]
        
        # Plot 1: Execution Time by Operation
        axes[0, 0].bar(range(len(execution_times)), execution_times, alpha=0.7)
        axes[0, 0].set_xlabel('Operation')
        axes[0, 0].set_ylabel('Execution Time (s)')
        axes[0, 0].set_title('Execution Time by Operation')
        axes[0, 0].set_xticks(range(len(operation_names)))
        axes[0, 0].set_xticklabels(operation_names, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Throughput by Operation
        axes[0, 1].bar(range(len(throughputs)), throughputs, alpha=0.7)
        axes[0, 1].set_xlabel('Operation')
        axes[0, 1].set_ylabel('Throughput (samples/s)')
        axes[0, 1].set_title('Throughput by Operation')
        axes[0, 1].set_xticks(range(len(operation_names)))
        axes[0, 1].set_xticklabels(operation_names, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Memory Usage by Operation
        axes[0, 2].bar(range(len(memory_usage)), memory_usage, alpha=0.7)
        axes[0, 2].set_xlabel('Operation')
        axes[0, 2].set_ylabel('Memory Usage (MB)')
        axes[0, 2].set_title('Memory Usage by Operation')
        axes[0, 2].set_xticks(range(len(operation_names)))
        axes[0, 2].set_xticklabels(operation_names, rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: GPU Memory Usage by Operation
        axes[1, 0].bar(range(len(gpu_memory_usage)), gpu_memory_usage, alpha=0.7)
        axes[1, 0].set_xlabel('Operation')
        axes[1, 0].set_ylabel('GPU Memory Usage (MB)')
        axes[1, 0].set_title('GPU Memory Usage by Operation')
        axes[1, 0].set_xticks(range(len(operation_names)))
        axes[1, 0].set_xticklabels(operation_names, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: CPU Usage by Operation
        axes[1, 1].bar(range(len(cpu_usage)), cpu_usage, alpha=0.7)
        axes[1, 1].set_xlabel('Operation')
        axes[1, 1].set_ylabel('CPU Usage (%)')
        axes[1, 1].set_title('CPU Usage by Operation')
        axes[1, 1].set_xticks(range(len(operation_names)))
        axes[1, 1].set_xticklabels(operation_names, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Bottlenecks by Operation
        axes[1, 2].bar(range(len(bottleneck_counts)), bottleneck_counts, alpha=0.7)
        axes[1, 2].set_xlabel('Operation')
        axes[1, 2].set_ylabel('Number of Bottlenecks')
        axes[1, 2].set_title('Bottlenecks by Operation')
        axes[1, 2].set_xticks(range(len(operation_names)))
        axes[1, 2].set_xticklabels(operation_names, rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "preprocessing_optimization_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()


class MemoryTracker:
    """Track memory usage during profiling."""
    
    def __init__(self):
        self.session_stats = []
        self.start_time = None
    
    def start_session(self):
        """Start memory tracking session."""
        self.start_time = time.time()
        self.session_stats = []
    
    def stop_session(self):
        """Stop memory tracking session."""
        self.start_time = None
    
    def get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024


class GPUTracker:
    """Track GPU usage during profiling."""
    
    def __init__(self):
        self.session_stats = []
        self.start_time = None
    
    def start_session(self):
        """Start GPU tracking session."""
        self.start_time = time.time()
        self.session_stats = []
    
    def stop_session(self):
        """Stop GPU tracking session."""
        self.start_time = None
    
    def get_current_memory(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0


class CPUTracker:
    """Track CPU usage during profiling."""
    
    def __init__(self):
        self.session_stats = []
        self.start_time = None
    
    def start_session(self):
        """Start CPU tracking session."""
        self.start_time = time.time()
        self.session_stats = []
    
    def stop_session(self):
        """Stop CPU tracking session."""
        self.start_time = None
    
    def get_current_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)


class IOTracker:
    """Track I/O operations during profiling."""
    
    def __init__(self):
        self.session_stats = []
        self.start_time = None
    
    def start_session(self):
        """Start I/O tracking session."""
        self.start_time = time.time()
        self.session_stats = []
    
    def stop_session(self):
        """Stop I/O tracking session."""
        self.start_time = None


def demonstrate_preprocessing_profiling():
    """Demonstrate preprocessing profiling capabilities."""
    print("Preprocessing Profiler Demonstration")
    print("=" * 50)
    
    # Create configuration
    config = PreprocessingProfilerConfig(
        enable_memory_tracking=True,
        enable_gpu_tracking=True,
        enable_cpu_tracking=True,
        enable_io_tracking=True,
        profiling_duration=30,
        batch_size_range=[1, 8, 16, 32, 64],
        enable_caching=True,
        cache_size=1000,
        save_profiles=True,
        auto_optimize=False
    )
    
    # Create profiler
    profiler = PreprocessingProfiler(config)
    
    # Define sample preprocessing functions
    def normalize_data(data):
        """Normalize data to [0, 1] range."""
        return (data - data.min()) / (data.max() - data.min())
    
    def standardize_data(data):
        """Standardize data to zero mean and unit variance."""
        return (data - data.mean()) / data.std()
    
    def augment_data(data):
        """Simple data augmentation."""
        # Simulate augmentation
        time.sleep(0.001)  # Simulate work
        return data + torch.randn_like(data) * 0.1
    
    def resize_data(data):
        """Resize data using interpolation."""
        if len(data.shape) == 3:  # [C, H, W]
            return F.interpolate(data.unsqueeze(0), size=(224, 224), mode='bilinear').squeeze(0)
        return data
    
    # Create sample data
    sample_data = torch.randn(3, 64, 64)
    
    # Profile different preprocessing functions
    print("Profiling different preprocessing functions...")
    
    functions_to_profile = [
        (normalize_data, "normalize_data"),
        (standardize_data, "standardize_data"),
        (augment_data, "augment_data"),
        (resize_data, "resize_data")
    ]
    
    for func, name in functions_to_profile:
        try:
            profile = profiler.profile_preprocessing_function(
                func, sample_data, num_iterations=5, operation_name=name
            )
        except Exception as e:
            print(f"Failed to profile {name}: {e}")
            continue
    
    # Profile batch preprocessing
    print("\nProfiling batch preprocessing...")
    batch_profiles = profiler.profile_batch_preprocessing(
        normalize_data, sample_data, batch_sizes=[1, 8, 16, 32]
    )
    
    # Get optimal configuration
    optimal_profile = profiler.get_optimal_preprocessing_config()
    
    if optimal_profile:
        print(f"\nOptimal preprocessing configuration found:")
        print(f"  Operation: {optimal_profile.operation_name}")
        print(f"  Input shape: {optimal_profile.input_shape}")
        print(f"  Execution time: {optimal_profile.execution_time:.4f}s")
        print(f"  Throughput: {optimal_profile.throughput_samples_per_second:.2f} samples/s")
        print(f"  Bottlenecks: {len(optimal_profile.bottlenecks)}")
    
    # Generate optimization report
    print("\nGenerating optimization report...")
    report_path = profiler.generate_optimization_report()
    print(f"Report generated: {report_path}")
    
    print("\nPreprocessing profiling demonstration completed!")


if __name__ == "__main__":
    # Demonstrate preprocessing profiling
    demonstrate_preprocessing_profiling()






