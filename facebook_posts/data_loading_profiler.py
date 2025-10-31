#!/usr/bin/env python3
"""
Data Loading Profiler
Specialized profiler for identifying and optimizing bottlenecks in data loading and preprocessing.
"""

import time
import psutil
import threading
import multiprocessing
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, List, Dict, Optional, Callable, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

from logging_config import get_logger, log_performance_metrics


class DataLoadingBottleneck(Enum):
    """Types of data loading bottlenecks."""
    SLOW_DISK_IO = "slow_disk_io"
    INSUFFICIENT_WORKERS = "insufficient_workers"
    MEMORY_PRESSURE = "memory_pressure"
    GPU_TRANSFER_OVERHEAD = "gpu_transfer_overhead"
    PREPROCESSING_BOTTLENECK = "preprocessing_bottleneck"
    BATCH_SIZE_INEFFICIENCY = "batch_size_inefficiency"
    SAMPLER_OVERHEAD = "sampler_overhead"
    COLLATE_FUNCTION_SLOW = "collate_function_slow"


@dataclass
class DataLoadingProfile:
    """Profile of data loading performance."""
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    loading_time_per_batch: float
    memory_usage_mb: float
    gpu_memory_usage_mb: float
    cpu_usage_percent: float
    throughput_batches_per_second: float
    bottlenecks: List[DataLoadingBottleneck] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)


@dataclass
class DataLoadingProfilerConfig:
    """Configuration for data loading profiling."""
    enable_memory_tracking: bool = True
    enable_gpu_tracking: bool = True
    enable_cpu_tracking: bool = True
    enable_io_tracking: bool = True
    profiling_duration: int = 60  # seconds
    batch_size_range: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    worker_range: List[int] = field(default_factory=lambda: [0, 2, 4, 8])
    save_profiles: bool = True
    profile_output_dir: str = "data_loading_profiles"
    auto_optimize: bool = False


class DataLoadingProfiler:
    """Specialized profiler for data loading bottlenecks."""
    
    def __init__(self, config: DataLoadingProfilerConfig):
        self.config = config
        self.logger = get_logger("data_loading_profiler")
        self.profiles: List[DataLoadingProfile] = []
        self.current_profile: Optional[DataLoadingProfile] = None
        
        # Performance tracking
        self.performance_metrics = defaultdict(float)
        self.bottleneck_history = []
        
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
    
    def profile_data_loader(self, data_loader: data.DataLoader, 
                           num_batches: int = 10) -> DataLoadingProfile:
        """Profile a specific data loader configuration."""
        self.logger.info(f"Profiling data loader: {num_batches} batches")
        
        # Start tracking
        self._start_tracking()
        
        # Profile data loading
        start_time = time.time()
        batch_times = []
        memory_usage = []
        gpu_memory_usage = []
        cpu_usage = []
        
        for i, (data_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            
            batch_start = time.time()
            
            # Move to device if available
            if torch.cuda.is_available():
                data_batch = data_batch.cuda(non_blocking=True)
                target_batch = target_batch.cuda(non_blocking=True)
            
            batch_end = time.time()
            batch_time = batch_end - batch_start
            
            # Record metrics
            batch_times.append(batch_time)
            memory_usage.append(self._get_memory_usage())
            gpu_memory_usage.append(self._get_gpu_memory_usage())
            cpu_usage.append(self._get_cpu_usage())
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Stop tracking
        self._stop_tracking()
        
        # Calculate metrics
        avg_batch_time = np.mean(batch_times)
        throughput = num_batches / total_time
        avg_memory = np.mean(memory_usage)
        avg_gpu_memory = np.mean(gpu_memory_usage)
        avg_cpu = np.mean(cpu_usage)
        
        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(
            avg_batch_time, avg_memory, avg_gpu_memory, avg_cpu, 
            data_loader, throughput
        )
        
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(bottlenecks, data_loader)
        
        # Create profile
        profile = DataLoadingProfile(
            batch_size=data_loader.batch_size,
            num_workers=data_loader.num_workers,
            pin_memory=data_loader.pin_memory,
            persistent_workers=getattr(data_loader, 'persistent_workers', False),
            prefetch_factor=getattr(data_loader, 'prefetch_factor', 2),
            loading_time_per_batch=avg_batch_time,
            memory_usage_mb=avg_memory,
            gpu_memory_usage_mb=avg_gpu_memory,
            cpu_usage_percent=avg_cpu,
            throughput_batches_per_second=throughput,
            bottlenecks=bottlenecks,
            optimization_suggestions=suggestions
        )
        
        self.profiles.append(profile)
        self.current_profile = profile
        
        # Log results
        self.logger.info(f"Data loading profile completed:")
        self.logger.info(f"  Batch size: {profile.batch_size}")
        self.logger.info(f"  Workers: {profile.num_workers}")
        self.logger.info(f"  Avg batch time: {avg_batch_time:.4f}s")
        self.logger.info(f"  Throughput: {throughput:.2f} batches/s")
        self.logger.info(f"  Bottlenecks detected: {len(bottlenecks)}")
        
        return profile
    
    def profile_configuration_range(self, dataset: data.Dataset, 
                                  batch_sizes: List[int] = None,
                                  worker_counts: List[int] = None) -> List[DataLoadingProfile]:
        """Profile multiple data loader configurations."""
        batch_sizes = batch_sizes or self.config.batch_size_range
        worker_counts = worker_counts or self.config.worker_range
        
        self.logger.info(f"Profiling {len(batch_sizes)} batch sizes × {len(worker_counts)} worker counts")
        
        profiles = []
        
        for batch_size in batch_sizes:
            for num_workers in worker_counts:
                try:
                    # Create data loader
                    data_loader = data.DataLoader(
                        dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=num_workers > 0,
                        prefetch_factor=2 if num_workers > 0 else 2
                    )
                    
                    # Profile this configuration
                    profile = self.profile_data_loader(data_loader, num_batches=5)
                    profiles.append(profile)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to profile batch_size={batch_size}, workers={num_workers}: {e}")
                    continue
        
        return profiles
    
    def _detect_bottlenecks(self, avg_batch_time: float, avg_memory: float,
                           avg_gpu_memory: float, avg_cpu: float,
                           data_loader: data.DataLoader, throughput: float) -> List[DataLoadingBottleneck]:
        """Detect bottlenecks in data loading."""
        bottlenecks = []
        
        # Check for slow disk I/O
        if avg_batch_time > 0.1:  # More than 100ms per batch
            bottlenecks.append(DataLoadingBottleneck.SLOW_DISK_IO)
        
        # Check for insufficient workers
        if data_loader.num_workers == 0 and throughput < 10:  # Less than 10 batches/s
            bottlenecks.append(DataLoadingBottleneck.INSUFFICIENT_WORKERS)
        
        # Check for memory pressure
        if avg_memory > 1000:  # More than 1GB
            bottlenecks.append(DataLoadingBottleneck.MEMORY_PRESSURE)
        
        # Check for GPU transfer overhead
        if torch.cuda.is_available() and avg_gpu_memory > 500:  # More than 500MB
            bottlenecks.append(DataLoadingBottleneck.GPU_TRANSFER_OVERHEAD)
        
        # Check for preprocessing bottleneck
        if avg_cpu > 80:  # More than 80% CPU usage
            bottlenecks.append(DataLoadingBottleneck.PREPROCESSING_BOTTLENECK)
        
        # Check for batch size inefficiency
        if data_loader.batch_size < 16 or data_loader.batch_size > 256:
            bottlenecks.append(DataLoadingBottleneck.BATCH_SIZE_INEFFICIENCY)
        
        # Check for sampler overhead
        if hasattr(data_loader, 'sampler') and data_loader.sampler is not None:
            if avg_batch_time > 0.05:  # More than 50ms per batch
                bottlenecks.append(DataLoadingBottleneck.SAMPLER_OVERHEAD)
        
        return bottlenecks
    
    def _generate_optimization_suggestions(self, bottlenecks: List[DataLoadingBottleneck],
                                         data_loader: data.DataLoader) -> List[str]:
        """Generate optimization suggestions based on detected bottlenecks."""
        suggestions = []
        
        for bottleneck in bottlenecks:
            if bottleneck == DataLoadingBottleneck.SLOW_DISK_IO:
                suggestions.extend([
                    "Use SSD storage for faster I/O",
                    "Increase num_workers for parallel loading",
                    "Enable persistent_workers to avoid worker initialization",
                    "Use memory-mapped files for large datasets"
                ])
            
            elif bottleneck == DataLoadingBottleneck.INSUFFICIENT_WORKERS:
                suggestions.extend([
                    f"Increase num_workers from {data_loader.num_workers} to {min(8, multiprocessing.cpu_count())}",
                    "Enable persistent_workers=True",
                    "Use prefetch_factor=2 for data prefetching"
                ])
            
            elif bottleneck == DataLoadingBottleneck.MEMORY_PRESSURE:
                suggestions.extend([
                    "Reduce batch size to decrease memory usage",
                    "Enable gradient checkpointing",
                    "Use mixed precision training",
                    "Implement memory pooling"
                ])
            
            elif bottleneck == DataLoadingBottleneck.GPU_TRANSFER_OVERHEAD:
                suggestions.extend([
                    "Use pin_memory=True for faster GPU transfer",
                    "Enable non_blocking transfers",
                    "Batch GPU operations",
                    "Use gradient accumulation to reduce memory pressure"
                ])
            
            elif bottleneck == DataLoadingBottleneck.PREPROCESSING_BOTTLENECK:
                suggestions.extend([
                    "Move preprocessing to GPU using torch.cuda.amp",
                    "Cache preprocessed data",
                    "Use torch.jit.script for preprocessing functions",
                    "Implement batch preprocessing"
                ])
            
            elif bottleneck == DataLoadingBottleneck.BATCH_SIZE_INEFFICIENCY:
                current_batch_size = data_loader.batch_size
                if current_batch_size < 16:
                    suggestions.append(f"Increase batch size from {current_batch_size} to at least 16")
                elif current_batch_size > 256:
                    suggestions.append(f"Reduce batch size from {current_batch_size} to 128 or less")
                suggestions.append("Use gradient accumulation for larger effective batch sizes")
            
            elif bottleneck == DataLoadingBottleneck.SAMPLER_OVERHEAD:
                suggestions.extend([
                    "Optimize custom sampler implementation",
                    "Use simpler sampling strategies",
                    "Cache sampler results when possible"
                ])
            
            elif bottleneck == DataLoadingBottleneck.COLLATE_FUNCTION_SLOW:
                suggestions.extend([
                    "Optimize custom collate function",
                    "Use vectorized operations in collate function",
                    "Consider using torch.jit.script for collate function"
                ])
        
        return list(set(suggestions))  # Remove duplicates
    
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
    
    def get_optimal_configuration(self) -> Optional[DataLoadingProfile]:
        """Get the optimal data loading configuration based on profiles."""
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
    
    def _calculate_profile_score(self, profile: DataLoadingProfile) -> float:
        """Calculate a score for a profile based on multiple criteria."""
        score = 0.0
        
        # Higher throughput is better
        score += profile.throughput_batches_per_second * 10
        
        # Lower memory usage is better
        score -= profile.memory_usage_mb / 100
        
        # Lower GPU memory usage is better
        score -= profile.gpu_memory_usage_mb / 100
        
        # Lower CPU usage is better
        score -= profile.cpu_usage_percent / 10
        
        # Fewer bottlenecks is better
        score -= len(profile.bottlenecks) * 5
        
        # Prefer configurations with workers
        if profile.num_workers > 0:
            score += 2
        
        # Prefer configurations with pin_memory
        if profile.pin_memory:
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
        optimal_profile = self.get_optimal_configuration()
        
        # Generate report
        report = f"""# Data Loading Optimization Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Summary
Total configurations profiled: {len(self.profiles)}

## Optimal Configuration
- Batch Size: {optimal_profile.batch_size}
- Number of Workers: {optimal_profile.num_workers}
- Pin Memory: {optimal_profile.pin_memory}
- Persistent Workers: {optimal_profile.persistent_workers}
- Prefetch Factor: {optimal_profile.prefetch_factor}
- Throughput: {optimal_profile.throughput_batches_per_second:.2f} batches/second
- Memory Usage: {optimal_profile.memory_usage_mb:.1f} MB
- GPU Memory Usage: {optimal_profile.gpu_memory_usage_mb:.1f} MB
- CPU Usage: {optimal_profile.cpu_usage_percent:.1f}%

## All Configurations
"""
        
        for i, profile in enumerate(self.profiles, 1):
            report += f"""
### Configuration {i}
- Batch Size: {profile.batch_size}
- Workers: {profile.num_workers}
- Pin Memory: {profile.pin_memory}
- Throughput: {profile.throughput_batches_per_second:.2f} batches/s
- Memory: {profile.memory_usage_mb:.1f} MB
- GPU Memory: {profile.gpu_memory_usage_mb:.1f} MB
- CPU: {profile.cpu_usage_percent:.1f}%
- Bottlenecks: {len(profile.bottlenecks)}
- Suggestions: {len(profile.optimization_suggestions)}
"""
        
        # Save report
        report_file = output_path / "data_loading_optimization_report.md"
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
        batch_sizes = [p.batch_size for p in self.profiles]
        worker_counts = [p.num_workers for p in self.profiles]
        throughputs = [p.throughput_batches_per_second for p in self.profiles]
        memory_usage = [p.memory_usage_mb for p in self.profiles]
        gpu_memory_usage = [p.gpu_memory_usage_mb for p in self.profiles]
        cpu_usage = [p.cpu_usage_percent for p in self.profiles]
        bottleneck_counts = [len(p.bottlenecks) for p in self.profiles]
        
        # Plot 1: Throughput vs Batch Size
        axes[0, 0].scatter(batch_sizes, throughputs, alpha=0.7)
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Throughput (batches/s)')
        axes[0, 0].set_title('Throughput vs Batch Size')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Throughput vs Workers
        axes[0, 1].scatter(worker_counts, throughputs, alpha=0.7)
        axes[0, 1].set_xlabel('Number of Workers')
        axes[0, 1].set_ylabel('Throughput (batches/s)')
        axes[0, 1].set_title('Throughput vs Workers')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Memory Usage vs Batch Size
        axes[0, 2].scatter(batch_sizes, memory_usage, alpha=0.7)
        axes[0, 2].set_xlabel('Batch Size')
        axes[0, 2].set_ylabel('Memory Usage (MB)')
        axes[0, 2].set_title('Memory Usage vs Batch Size')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: GPU Memory vs Batch Size
        axes[1, 0].scatter(batch_sizes, gpu_memory_usage, alpha=0.7)
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('GPU Memory Usage (MB)')
        axes[1, 0].set_title('GPU Memory vs Batch Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: CPU Usage vs Workers
        axes[1, 1].scatter(worker_counts, cpu_usage, alpha=0.7)
        axes[1, 1].set_xlabel('Number of Workers')
        axes[1, 1].set_ylabel('CPU Usage (%)')
        axes[1, 1].set_title('CPU Usage vs Workers')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Bottlenecks vs Configuration
        config_labels = [f"BS{p.batch_size}_W{p.num_workers}" for p in self.profiles]
        axes[1, 2].bar(range(len(bottleneck_counts)), bottleneck_counts)
        axes[1, 2].set_xlabel('Configuration')
        axes[1, 2].set_ylabel('Number of Bottlenecks')
        axes[1, 2].set_title('Bottlenecks by Configuration')
        axes[1, 2].set_xticks(range(len(config_labels)))
        axes[1, 2].set_xticklabels(config_labels, rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "data_loading_optimization_analysis.png", 
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


def demonstrate_data_loading_profiling():
    """Demonstrate data loading profiling capabilities."""
    print("Data Loading Profiler Demonstration")
    print("=" * 50)
    
    # Create configuration
    config = DataLoadingProfilerConfig(
        enable_memory_tracking=True,
        enable_gpu_tracking=True,
        enable_cpu_tracking=True,
        enable_io_tracking=True,
        profiling_duration=30,
        batch_size_range=[16, 32, 64, 128],
        worker_range=[0, 2, 4, 8],
        save_profiles=True,
        auto_optimize=False
    )
    
    # Create profiler
    profiler = DataLoadingProfiler(config)
    
    # Create dummy dataset
    class DummyDataset(data.Dataset):
        def __init__(self, num_samples=1000):
            self.data = torch.randn(num_samples, 784)
            self.targets = torch.randint(0, 10, (num_samples,))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    
    # Create dataset
    dataset = DummyDataset(1000)
    
    # Profile multiple configurations
    print("Profiling multiple data loader configurations...")
    profiles = profiler.profile_configuration_range(dataset)
    
    # Get optimal configuration
    optimal_profile = profiler.get_optimal_configuration()
    
    if optimal_profile:
        print(f"\nOptimal configuration found:")
        print(f"  Batch size: {optimal_profile.batch_size}")
        print(f"  Workers: {optimal_profile.num_workers}")
        print(f"  Throughput: {optimal_profile.throughput_batches_per_second:.2f} batches/s")
        print(f"  Memory usage: {optimal_profile.memory_usage_mb:.1f} MB")
        print(f"  Bottlenecks: {len(optimal_profile.bottlenecks)}")
    
    # Generate optimization report
    print("\nGenerating optimization report...")
    report_path = profiler.generate_optimization_report()
    print(f"Report generated: {report_path}")
    
    print("\nData loading profiling demonstration completed!")


if __name__ == "__main__":
    # Demonstrate data loading profiling
    demonstrate_data_loading_profiling()






