#!/usr/bin/env python3
"""
Advanced Bottleneck Profiler
Comprehensive profiling system to identify and optimize bottlenecks in data loading and preprocessing.
"""

import io
import os
import gc
import time
import psutil
import queue
import tracemalloc
import warnings
import logging
import multiprocessing
import cProfile
import pstats
import threading
import asyncio
from datetime import datetime, timedelta
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
from typing import Any, List, Dict, Optional, Union, Tuple, Callable, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import line_profiler
    import memory_profiler
    import pyinstrument
    import scalene
    PROFILING_AVAILABLE = True
except ImportError:
    line_profiler = None
    memory_profiler = None
    pyinstrument = None
    scalene = None
    PROFILING_AVAILABLE = False

from logging_config import get_logger, log_performance_metrics


class BottleneckType(Enum):
    """Types of bottlenecks."""
    DATA_LOADING = "data_loading"
    PREPROCESSING = "preprocessing"
    MEMORY_ALLOCATION = "memory_allocation"
    GPU_TRANSFER = "gpu_transfer"
    CPU_COMPUTATION = "cpu_computation"
    I_O_OPERATION = "i_o_operation"
    NETWORK_LATENCY = "network_latency"
    SYNCHRONIZATION = "synchronization"


class ProfilingLevel(Enum):
    """Profiling levels."""
    BASIC = "basic"           # Basic timing and memory
    DETAILED = "detailed"     # Detailed with bottlenecks
    COMPREHENSIVE = "comprehensive"  # Full profiling with optimizations
    PRODUCTION = "production" # Production-ready profiling


@dataclass
class BottleneckProfile:
    """Profile of a specific bottleneck."""
    bottleneck_type: BottleneckType
    operation_name: str
    execution_time: float
    memory_usage: float
    gpu_memory_usage: float
    cpu_usage: float
    frequency: int
    severity: float  # 0.0 to 1.0
    optimization_potential: float  # 0.0 to 1.0
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ProfilingSession:
    """Complete profiling session."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    bottlenecks: List[BottleneckProfile] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    optimization_applied: List[str] = field(default_factory=list)
    memory_profile: Dict[str, float] = field(default_factory=dict)
    gpu_profile: Dict[str, float] = field(default_factory=dict)
    cpu_profile: Dict[str, float] = field(default_factory=dict)


@dataclass
class BottleneckProfilerConfig:
    """Configuration for bottleneck profiling."""
    profiling_level: ProfilingLevel = ProfilingLevel.DETAILED
    enable_real_time_monitoring: bool = True
    enable_memory_tracking: bool = True
    enable_gpu_tracking: bool = True
    enable_cpu_tracking: bool = True
    enable_io_tracking: bool = True
    sampling_interval: float = 0.1  # seconds
    bottleneck_threshold: float = 0.05  # 5% of total time
    memory_threshold: float = 0.8  # 80% of available memory
    gpu_threshold: float = 0.9  # 90% of GPU memory
    auto_optimize: bool = False
    save_profiles: bool = True
    profile_output_dir: str = "bottleneck_profiles"
    enable_async_profiling: bool = True
    max_concurrent_profiles: int = 4


class BottleneckProfiler:
    """Advanced bottleneck profiling system."""
    
    def __init__(self, config: BottleneckProfilerConfig):
        self.config = config
        self.logger = get_logger("bottleneck_profiler")
        self.current_session: Optional[ProfilingSession] = None
        self.bottlenecks: List[BottleneckProfile] = []
        self.performance_metrics = defaultdict(float)
        self.memory_tracker = MemoryTracker()
        self.gpu_tracker = GPUTracker()
        self.cpu_tracker = CPUTracker()
        self.io_tracker = IOTracker()
        
        # Profiling state
        self.is_profiling = False
        self.profiling_thread = None
        self.stop_event = threading.Event()
        
        # Initialize profiling
        self._initialize_profiling()
    
    def _initialize_profiling(self):
        """Initialize profiling components."""
        if self.config.enable_memory_tracking:
            tracemalloc.start()
        
        if self.config.enable_gpu_tracking and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Bottleneck profiler initialized")
    
    def start_profiling_session(self, session_name: str = None) -> str:
        """Start a new profiling session."""
        if self.is_profiling:
            self.logger.warning("Profiling session already active")
            return self.current_session.session_id
        
        session_id = session_name or f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = ProfilingSession(
            session_id=session_id,
            start_time=datetime.now()
        )
        
        self.is_profiling = True
        self.stop_event.clear()
        
        # Start real-time monitoring if enabled
        if self.config.enable_real_time_monitoring:
            self._start_real_time_monitoring()
        
        self.logger.info(f"Started profiling session: {session_id}")
        return session_id
    
    def stop_profiling_session(self) -> ProfilingSession:
        """Stop the current profiling session."""
        if not self.is_profiling:
            self.logger.warning("No active profiling session")
            return None
        
        self.is_profiling = False
        self.stop_event.set()
        
        if self.profiling_thread and self.profiling_thread.is_alive():
            self.profiling_thread.join()
        
        if self.current_session:
            self.current_session.end_time = datetime.now()
            
            # Generate final report
            self._generate_session_summary()
            
            # Save profile if enabled
            if self.config.save_profiles:
                self._save_session_profile()
        
        self.logger.info(f"Stopped profiling session: {self.current_session.session_id}")
        return self.current_session
    
    def _start_real_time_monitoring(self):
        """Start real-time monitoring thread."""
        self.profiling_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.profiling_thread.start()
    
    def _monitoring_loop(self):
        """Real-time monitoring loop."""
        while not self.stop_event.is_set():
            try:
                # Collect metrics
                self._collect_real_time_metrics()
                
                # Check for bottlenecks
                self._detect_real_time_bottlenecks()
                
                # Sleep for sampling interval
                time.sleep(self.config.sampling_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                break
    
    def _collect_real_time_metrics(self):
        """Collect real-time performance metrics."""
        if self.config.enable_memory_tracking:
            memory_stats = self.memory_tracker.get_current_stats()
            self.performance_metrics['memory_usage'] = memory_stats['used_percent']
        
        if self.config.enable_gpu_tracking:
            gpu_stats = self.gpu_tracker.get_current_stats()
            self.performance_metrics['gpu_memory_usage'] = gpu_stats['memory_used_percent']
        
        if self.config.enable_cpu_tracking:
            cpu_stats = self.cpu_tracker.get_current_stats()
            self.performance_metrics['cpu_usage'] = cpu_stats['usage_percent']
    
    def _detect_real_time_bottlenecks(self):
        """Detect bottlenecks in real-time."""
        # Check memory threshold
        if (self.config.enable_memory_tracking and 
            self.performance_metrics.get('memory_usage', 0) > self.config.memory_threshold * 100):
            self._record_bottleneck(
                BottleneckType.MEMORY_ALLOCATION,
                "high_memory_usage",
                severity=0.8,
                suggestions=["Reduce batch size", "Enable gradient checkpointing", "Use mixed precision"]
            )
        
        # Check GPU threshold
        if (self.config.enable_gpu_tracking and 
            self.performance_metrics.get('gpu_memory_usage', 0) > self.config.gpu_threshold * 100):
            self._record_bottleneck(
                BottleneckType.GPU_TRANSFER,
                "high_gpu_memory_usage",
                severity=0.9,
                suggestions=["Reduce batch size", "Enable gradient checkpointing", "Use mixed precision"]
            )
    
    @contextmanager
    def profile_operation(self, operation_name: str, operation_type: BottleneckType):
        """Context manager for profiling operations."""
        start_time = time.time()
        start_memory = self.memory_tracker.get_current_memory()
        start_gpu_memory = self.gpu_tracker.get_current_memory()
        start_cpu_usage = self.cpu_tracker.get_current_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.memory_tracker.get_current_memory()
            end_gpu_memory = self.gpu_tracker.get_current_memory()
            end_cpu_usage = self.cpu_tracker.get_current_usage()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            gpu_memory_delta = end_gpu_memory - start_gpu_memory
            cpu_usage_delta = end_cpu_usage - start_cpu_usage
            
            # Record operation metrics
            self._record_operation_metrics(
                operation_name, operation_type, execution_time,
                memory_delta, gpu_memory_delta, cpu_usage_delta
            )
    
    def _record_operation_metrics(self, operation_name: str, operation_type: BottleneckType,
                                 execution_time: float, memory_delta: float,
                                 gpu_memory_delta: float, cpu_usage_delta: float):
        """Record metrics for an operation."""
        # Update performance metrics
        self.performance_metrics[f'{operation_name}_time'] = execution_time
        self.performance_metrics[f'{operation_name}_memory'] = memory_delta
        self.performance_metrics[f'{operation_name}_gpu_memory'] = gpu_memory_delta
        self.performance_metrics[f'{operation_name}_cpu'] = cpu_usage_delta
        
        # Check if this operation is a bottleneck
        if execution_time > self.config.bottleneck_threshold * self.performance_metrics.get('total_time', 1):
            self._record_bottleneck(
                operation_type,
                operation_name,
                execution_time,
                memory_delta,
                gpu_memory_delta,
                cpu_usage_delta,
                severity=min(execution_time / self.performance_metrics.get('total_time', 1), 1.0)
            )
    
    def _record_bottleneck(self, bottleneck_type: BottleneckType, operation_name: str,
                          execution_time: float = 0, memory_usage: float = 0,
                          gpu_memory_usage: float = 0, cpu_usage: float = 0,
                          severity: float = 0.5, suggestions: List[str] = None):
        """Record a detected bottleneck."""
        # Check if bottleneck already exists
        existing_bottleneck = next(
            (b for b in self.bottlenecks 
             if b.bottleneck_type == bottleneck_type and b.operation_name == operation_name),
            None
        )
        
        if existing_bottleneck:
            # Update existing bottleneck
            existing_bottleneck.frequency += 1
            existing_bottleneck.execution_time = max(existing_bottleneck.execution_time, execution_time)
            existing_bottleneck.severity = max(existing_bottleneck.severity, severity)
        else:
            # Create new bottleneck
            bottleneck = BottleneckProfile(
                bottleneck_type=bottleneck_type,
                operation_name=operation_name,
                execution_time=execution_time,
                memory_usage=memory_usage,
                gpu_memory_usage=gpu_memory_usage,
                cpu_usage=cpu_usage,
                frequency=1,
                severity=severity,
                optimization_potential=self._calculate_optimization_potential(bottleneck_type),
                suggestions=suggestions or self._get_default_suggestions(bottleneck_type)
            )
            self.bottlenecks.append(bottleneck)
        
        # Log bottleneck detection
        self.logger.warning(f"Bottleneck detected: {bottleneck_type.value} - {operation_name} "
                           f"(severity: {severity:.2f})")
    
    def _calculate_optimization_potential(self, bottleneck_type: BottleneckType) -> float:
        """Calculate optimization potential for a bottleneck type."""
        optimization_potentials = {
            BottleneckType.DATA_LOADING: 0.8,
            BottleneckType.PREPROCESSING: 0.7,
            BottleneckType.MEMORY_ALLOCATION: 0.6,
            BottleneckType.GPU_TRANSFER: 0.7,
            BottleneckType.CPU_COMPUTATION: 0.5,
            BottleneckType.I_O_OPERATION: 0.8,
            BottleneckType.NETWORK_LATENCY: 0.3,
            BottleneckType.SYNCHRONIZATION: 0.4
        }
        return optimization_potentials.get(bottleneck_type, 0.5)
    
    def _get_default_suggestions(self, bottleneck_type: BottleneckType) -> List[str]:
        """Get default optimization suggestions for a bottleneck type."""
        suggestions = {
            BottleneckType.DATA_LOADING: [
                "Increase num_workers for parallel loading",
                "Enable pin_memory for faster GPU transfer",
                "Use persistent_workers to avoid worker initialization overhead",
                "Implement data prefetching",
                "Consider using memory-mapped files for large datasets"
            ],
            BottleneckType.PREPROCESSING: [
                "Move preprocessing to GPU using torch.cuda.amp",
                "Cache preprocessed data",
                "Use torch.jit.script for preprocessing functions",
                "Implement batch preprocessing",
                "Consider using specialized preprocessing libraries"
            ],
            BottleneckType.MEMORY_ALLOCATION: [
                "Reduce batch size",
                "Enable gradient checkpointing",
                "Use mixed precision training",
                "Implement memory pooling",
                "Consider using torch.utils.checkpoint"
            ],
            BottleneckType.GPU_TRANSFER: [
                "Use pin_memory=True",
                "Enable non_blocking transfers",
                "Batch GPU operations",
                "Consider using torch.cuda.Stream for overlapping",
                "Use gradient accumulation to reduce memory pressure"
            ],
            BottleneckType.CPU_COMPUTATION: [
                "Move computations to GPU",
                "Use vectorized operations",
                "Implement parallel processing",
                "Consider using numba or Cython",
                "Profile and optimize hot paths"
            ],
            BottleneckType.I_O_OPERATION: [
                "Use async I/O operations",
                "Implement data prefetching",
                "Use memory-mapped files",
                "Consider using databases for structured data",
                "Implement data compression"
            ],
            BottleneckType.NETWORK_LATENCY: [
                "Use local data storage",
                "Implement data caching",
                "Use compression for network transfers",
                "Consider using CDN for distributed training",
                "Implement data prefetching"
            ],
            BottleneckType.SYNCHRONIZATION: [
                "Reduce synchronization frequency",
                "Use asynchronous operations",
                "Implement gradient accumulation",
                "Consider using torch.distributed",
                "Profile synchronization overhead"
            ]
        }
        return suggestions.get(bottleneck_type, ["Profile and analyze the specific operation"])
    
    def profile_data_loading(self, data_loader: data.DataLoader, num_batches: int = 10) -> Dict[str, Any]:
        """Profile data loading operations."""
        self.logger.info(f"Profiling data loading for {num_batches} batches")
        
        loading_times = []
        memory_usage = []
        gpu_memory_usage = []
        cpu_usage = []
        
        for i, (data_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            
            with self.profile_operation(f"data_loading_batch_{i}", BottleneckType.DATA_LOADING):
                # Move data to device
                if torch.cuda.is_available():
                    data_batch = data_batch.cuda(non_blocking=True)
                    target_batch = target_batch.cuda(non_blocking=True)
            
            # Record metrics
            loading_times.append(self.performance_metrics.get(f'data_loading_batch_{i}_time', 0))
            memory_usage.append(self.memory_tracker.get_current_memory())
            gpu_memory_usage.append(self.gpu_tracker.get_current_memory())
            cpu_usage.append(self.cpu_tracker.get_current_usage())
        
        return {
            'loading_times': loading_times,
            'memory_usage': memory_usage,
            'gpu_memory_usage': gpu_memory_usage,
            'cpu_usage': cpu_usage,
            'average_loading_time': np.mean(loading_times),
            'total_batches': num_batches,
            'bottlenecks': [b for b in self.bottlenecks if b.bottleneck_type == BottleneckType.DATA_LOADING]
        }
    
    def profile_preprocessing(self, preprocessing_func: Callable, data: Any) -> Dict[str, Any]:
        """Profile preprocessing operations."""
        self.logger.info("Profiling preprocessing operations")
        
        with self.profile_operation("preprocessing", BottleneckType.PREPROCESSING):
            result = preprocessing_func(data)
        
        return {
            'preprocessing_time': self.performance_metrics.get('preprocessing_time', 0),
            'memory_usage': self.memory_tracker.get_current_memory(),
            'gpu_memory_usage': self.gpu_tracker.get_current_memory(),
            'cpu_usage': self.cpu_tracker.get_current_usage(),
            'result_shape': getattr(result, 'shape', None),
            'bottlenecks': [b for b in self.bottlenecks if b.bottleneck_type == BottleneckType.PREPROCESSING]
        }
    
    def get_bottleneck_summary(self) -> Dict[str, Any]:
        """Get summary of all detected bottlenecks."""
        if not self.bottlenecks:
            return {"message": "No bottlenecks detected"}
        
        # Group bottlenecks by type
        bottlenecks_by_type = defaultdict(list)
        for bottleneck in self.bottlenecks:
            bottlenecks_by_type[bottleneck.bottleneck_type.value].append(bottleneck)
        
        # Calculate overall statistics
        total_bottlenecks = len(self.bottlenecks)
        avg_severity = np.mean([b.severity for b in self.bottlenecks])
        avg_optimization_potential = np.mean([b.optimization_potential for b in self.bottlenecks])
        
        return {
            'total_bottlenecks': total_bottlenecks,
            'bottlenecks_by_type': dict(bottlenecks_by_type),
            'average_severity': avg_severity,
            'average_optimization_potential': avg_optimization_potential,
            'most_critical_bottlenecks': sorted(
                self.bottlenecks, 
                key=lambda x: x.severity * x.frequency, 
                reverse=True
            )[:5],
            'optimization_priority': sorted(
                self.bottlenecks,
                key=lambda x: x.optimization_potential * x.severity,
                reverse=True
            )[:5]
        }
    
    def _generate_session_summary(self):
        """Generate summary for the current session."""
        if not self.current_session:
            return
        
        # Calculate total time
        if self.current_session.end_time:
            total_time = (self.current_session.end_time - self.current_session.start_time).total_seconds()
        else:
            total_time = (datetime.now() - self.current_session.start_time).total_seconds()
        
        # Update session metrics
        self.current_session.performance_metrics.update(self.performance_metrics)
        self.current_session.bottlenecks = self.bottlenecks.copy()
        self.current_session.memory_profile = self.memory_tracker.get_session_stats()
        self.current_session.gpu_profile = self.gpu_tracker.get_session_stats()
        self.current_session.cpu_profile = self.cpu_tracker.get_session_stats()
        
        # Log session summary
        self.logger.info(f"Profiling session {self.current_session.session_id} completed:")
        self.logger.info(f"  Duration: {total_time:.2f} seconds")
        self.logger.info(f"  Bottlenecks detected: {len(self.bottlenecks)}")
        self.logger.info(f"  Memory usage: {self.performance_metrics.get('memory_usage', 0):.1f}%")
        self.logger.info(f"  GPU memory usage: {self.performance_metrics.get('gpu_memory_usage', 0):.1f}%")
        self.logger.info(f"  CPU usage: {self.performance_metrics.get('cpu_usage', 0):.1f}%")
    
    def _save_session_profile(self):
        """Save the current session profile."""
        if not self.current_session:
            return
        
        output_dir = Path(self.config.profile_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save profile data
        profile_file = output_dir / f"{self.current_session.session_id}_profile.pt"
        torch.save({
            'session': self.current_session,
            'bottlenecks': self.bottlenecks,
            'performance_metrics': self.performance_metrics
        }, profile_file)
        
        # Generate and save report
        self._generate_profile_report(output_dir)
        
        self.logger.info(f"Session profile saved to {profile_file}")
    
    def _generate_profile_report(self, output_dir: Path):
        """Generate comprehensive profile report."""
        report_file = output_dir / f"{self.current_session.session_id}_report.md"
        
        with open(report_file, 'w') as f:
            f.write(self._format_profile_report())
        
        # Generate visualizations
        self._generate_profile_visualizations(output_dir)
    
    def _format_profile_report(self) -> str:
        """Format the profile report."""
        if not self.current_session:
            return "No active session"
        
        report = f"""# Bottleneck Profiling Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session ID: {self.current_session.session_id}

## Session Summary
- Start Time: {self.current_session.start_time}
- End Time: {self.current_session.end_time}
- Duration: {(self.current_session.end_time - self.current_session.start_time).total_seconds():.2f} seconds

## Performance Metrics
"""
        
        for metric, value in self.current_session.performance_metrics.items():
            if isinstance(value, float):
                report += f"- {metric}: {value:.4f}\n"
            else:
                report += f"- {metric}: {value}\n"
        
        report += f"""
## Bottleneck Analysis
Total Bottlenecks Detected: {len(self.current_session.bottlenecks)}

### Bottlenecks by Type
"""
        
        bottlenecks_by_type = defaultdict(list)
        for bottleneck in self.current_session.bottlenecks:
            bottlenecks_by_type[bottleneck.bottleneck_type.value].append(bottleneck)
        
        for bottleneck_type, bottlenecks in bottlenecks_by_type.items():
            report += f"\n#### {bottleneck_type.title()}\n"
            for bottleneck in bottlenecks:
                report += f"- **{bottleneck.operation_name}**: "
                report += f"Severity: {bottleneck.severity:.2f}, "
                report += f"Frequency: {bottleneck.frequency}, "
                report += f"Time: {bottleneck.execution_time:.4f}s\n"
                
                if bottleneck.suggestions:
                    report += "  - Suggestions:\n"
                    for suggestion in bottleneck.suggestions:
                        report += f"    - {suggestion}\n"
        
        return report
    
    def _generate_profile_visualizations(self, output_dir: Path):
        """Generate visualization charts for the profile."""
        if not self.current_session or not self.current_session.bottlenecks:
            return
        
        # Create bottleneck severity chart
        plt.figure(figsize=(12, 8))
        
        bottleneck_names = [b.operation_name for b in self.current_session.bottlenecks]
        severities = [b.severity for b in self.current_session.bottlenecks]
        frequencies = [b.frequency for b in self.current_session.bottlenecks]
        
        # Subplot 1: Severity by bottleneck
        plt.subplot(2, 2, 1)
        plt.barh(bottleneck_names, severities)
        plt.title('Bottleneck Severity')
        plt.xlabel('Severity (0-1)')
        
        # Subplot 2: Frequency by bottleneck
        plt.subplot(2, 2, 2)
        plt.barh(bottleneck_names, frequencies)
        plt.title('Bottleneck Frequency')
        plt.xlabel('Frequency')
        
        # Subplot 3: Optimization potential
        plt.subplot(2, 2, 3)
        optimization_potentials = [b.optimization_potential for b in self.current_session.bottlenecks]
        plt.barh(bottleneck_names, optimization_potentials)
        plt.title('Optimization Potential')
        plt.xlabel('Potential (0-1)')
        
        # Subplot 4: Performance metrics over time
        plt.subplot(2, 2, 4)
        if self.current_session.memory_profile:
            memory_values = list(self.current_session.memory_profile.values())
            plt.plot(memory_values, label='Memory Usage')
        if self.current_session.gpu_profile:
            gpu_values = list(self.current_session.gpu_profile.values())
            plt.plot(gpu_values, label='GPU Memory Usage')
        plt.title('Resource Usage Over Time')
        plt.xlabel('Time')
        plt.ylabel('Usage (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{self.current_session.session_id}_visualizations.png", 
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
    
    def get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def get_current_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        memory = psutil.virtual_memory()
        current_memory = self.get_current_memory()
        
        stats = {
            'used_mb': current_memory,
            'used_percent': memory.percent,
            'available_gb': memory.available / 1024**3,
            'total_gb': memory.total / 1024**3
        }
        
        # Record for session
        if self.start_time:
            self.session_stats.append({
                'timestamp': time.time() - self.start_time,
                'memory_mb': current_memory,
                'memory_percent': memory.percent
            })
        
        return stats
    
    def get_session_stats(self) -> Dict[str, float]:
        """Get memory statistics for the session."""
        if not self.session_stats:
            return {}
        
        memory_values = [s['memory_mb'] for s in self.session_stats]
        percent_values = [s['memory_percent'] for s in self.session_stats]
        
        return {
            'min_memory_mb': min(memory_values),
            'max_memory_mb': max(memory_values),
            'avg_memory_mb': np.mean(memory_values),
            'min_percent': min(percent_values),
            'max_percent': max(percent_values),
            'avg_percent': np.mean(percent_values)
        }


class GPUTracker:
    """Track GPU usage during profiling."""
    
    def __init__(self):
        self.session_stats = []
        self.start_time = None
    
    def start_session(self):
        """Start GPU tracking session."""
        self.start_time = time.time()
        self.session_stats = []
    
    def get_current_memory(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def get_current_stats(self) -> Dict[str, float]:
        """Get current GPU statistics."""
        if not torch.cuda.is_available():
            return {'memory_mb': 0.0, 'memory_used_percent': 0.0}
        
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        
        stats = {
            'memory_mb': allocated / 1024 / 1024,
            'memory_used_percent': (allocated / total) * 100,
            'reserved_mb': reserved / 1024 / 1024,
            'total_gb': total / 1024**3
        }
        
        # Record for session
        if self.start_time:
            self.session_stats.append({
                'timestamp': time.time() - self.start_time,
                'memory_mb': stats['memory_mb'],
                'memory_percent': stats['memory_used_percent']
            })
        
        return stats
    
    def get_session_stats(self) -> Dict[str, float]:
        """Get GPU statistics for the session."""
        if not self.session_stats:
            return {}
        
        memory_values = [s['memory_mb'] for s in self.session_stats]
        percent_values = [s['memory_percent'] for s in self.session_stats]
        
        return {
            'min_memory_mb': min(memory_values),
            'max_memory_mb': max(memory_values),
            'avg_memory_mb': np.mean(memory_values),
            'min_percent': min(percent_values),
            'max_percent': max(percent_values),
            'avg_percent': np.mean(percent_values)
        }


class CPUTracker:
    """Track CPU usage during profiling."""
    
    def __init__(self):
        self.session_stats = []
        self.start_time = None
    
    def start_session(self):
        """Start CPU tracking session."""
        self.start_time = time.time()
        self.session_stats = []
    
    def get_current_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)
    
    def get_current_stats(self) -> Dict[str, float]:
        """Get current CPU statistics."""
        cpu_percent = self.get_current_usage()
        
        stats = {
            'usage_percent': cpu_percent,
            'count': psutil.cpu_count(),
            'frequency_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0
        }
        
        # Record for session
        if self.start_time:
            self.session_stats.append({
                'timestamp': time.time() - self.start_time,
                'cpu_percent': cpu_percent
            })
        
        return stats
    
    def get_session_stats(self) -> Dict[str, float]:
        """Get CPU statistics for the session."""
        if not self.session_stats:
            return {}
        
        cpu_values = [s['cpu_percent'] for s in self.session_stats]
        
        return {
            'min_percent': min(cpu_values),
            'max_percent': max(cpu_values),
            'avg_percent': np.mean(cpu_values)
        }


class IOTracker:
    """Track I/O operations during profiling."""
    
    def __init__(self):
        self.session_stats = []
        self.start_time = None
    
    def start_session(self):
        """Start I/O tracking session."""
        self.start_time = time.time()
        self.session_stats = []
    
    def get_current_stats(self) -> Dict[str, float]:
        """Get current I/O statistics."""
        try:
            process = psutil.Process(os.getpid())
            io_counters = process.io_counters()
            
            stats = {
                'read_bytes': io_counters.read_bytes,
                'write_bytes': io_counters.write_bytes,
                'read_count': io_counters.read_count,
                'write_count': io_counters.write_count
            }
            
            # Record for session
            if self.start_time:
                self.session_stats.append({
                    'timestamp': time.time() - self.start_time,
                    'read_bytes': stats['read_bytes'],
                    'write_bytes': stats['write_bytes']
                })
            
            return stats
        except Exception:
            return {}
    
    def get_session_stats(self) -> Dict[str, float]:
        """Get I/O statistics for the session."""
        if not self.session_stats:
            return {}
        
        read_values = [s['read_bytes'] for s in self.session_stats]
        write_values = [s['write_bytes'] for s in self.session_stats]
        
        return {
            'total_read_mb': sum(read_values) / 1024 / 1024,
            'total_write_mb': sum(write_values) / 1024 / 1024,
            'avg_read_mb': np.mean(read_values) / 1024 / 1024,
            'avg_write_mb': np.mean(write_values) / 1024 / 1024
        }


def demonstrate_bottleneck_profiling():
    """Demonstrate bottleneck profiling capabilities."""
    print("Advanced Bottleneck Profiling Demonstration")
    print("=" * 60)
    
    # Create configuration
    config = BottleneckProfilerConfig(
        profiling_level=ProfilingLevel.DETAILED,
        enable_real_time_monitoring=True,
        enable_memory_tracking=True,
        enable_gpu_tracking=True,
        enable_cpu_tracking=True,
        enable_io_tracking=True,
        sampling_interval=0.1,
        bottleneck_threshold=0.05,
        auto_optimize=False,
        save_profiles=True
    )
    
    # Create profiler
    profiler = BottleneckProfiler(config)
    
    # Start profiling session
    session_id = profiler.start_profiling_session("demo_session")
    print(f"Started profiling session: {session_id}")
    
    # Create dummy dataset and data loader
    class DummyDataset(data.Dataset):
        def __init__(self, num_samples=1000):
            self.data = torch.randn(num_samples, 784)
            self.targets = torch.randint(0, 10, (num_samples,))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    
    # Create data loader
    dataset = DummyDataset(1000)
    data_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    
    # Profile data loading
    print("\nProfiling data loading...")
    data_loading_profile = profiler.profile_data_loading(data_loader, num_batches=5)
    
    # Profile preprocessing
    print("\nProfiling preprocessing...")
    def preprocessing_function(data_batch):
        # Simulate preprocessing
        time.sleep(0.01)  # Simulate work
        return data_batch.float() / 255.0
    
    preprocessing_profile = profiler.profile_preprocessing(preprocessing_function, torch.randn(32, 784))
    
    # Profile model inference
    print("\nProfiling model inference...")
    with profiler.profile_operation("model_inference", BottleneckType.CPU_COMPUTATION):
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(32, 784)
            output = model(dummy_input)
    
    # Stop profiling session
    print("\nStopping profiling session...")
    session = profiler.stop_profiling_session()
    
    # Get bottleneck summary
    print("\nBottleneck Summary:")
    summary = profiler.get_bottleneck_summary()
    
    if 'message' in summary:
        print(summary['message'])
    else:
        print(f"Total bottlenecks detected: {summary['total_bottlenecks']}")
        print(f"Average severity: {summary['average_severity']:.2f}")
        print(f"Average optimization potential: {summary['average_optimization_potential']:.2f}")
        
        print("\nMost critical bottlenecks:")
        for bottleneck in summary['most_critical_bottlenecks'][:3]:
            print(f"  - {bottleneck.operation_name}: severity {bottleneck.severity:.2f}, "
                  f"frequency {bottleneck.frequency}")
        
        print("\nOptimization priority:")
        for bottleneck in summary['optimization_priority'][:3]:
            print(f"  - {bottleneck.operation_name}: potential {bottleneck.optimization_potential:.2f}")
    
    print(f"\nProfiling session completed! Check {config.profile_output_dir} for detailed reports.")


if __name__ == "__main__":
    # Demonstrate bottleneck profiling
    demonstrate_bottleneck_profiling()






