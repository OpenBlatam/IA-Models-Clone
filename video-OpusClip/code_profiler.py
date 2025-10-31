"""
Code Profiler for Video-OpusClip

Comprehensive profiling system to identify and optimize bottlenecks,
especially in data loading and preprocessing operations.
"""

import time
import cProfile
import pstats
import io
import line_profiler
import memory_profiler
import psutil
import torch
import torch.utils.data as data
import numpy as np
import structlog
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
import multiprocessing
from pathlib import Path
import json
import pickle
import functools
import tracemalloc
from collections import defaultdict, deque
import gc
import os

logger = structlog.get_logger()

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ProfilerConfig:
    """Configuration for code profiling."""
    # General profiling
    enabled: bool = True
    profile_level: str = "detailed"  # basic, detailed, comprehensive
    
    # Performance profiling
    enable_cprofile: bool = True
    enable_line_profiler: bool = True
    enable_memory_profiler: bool = True
    enable_gpu_profiler: bool = True
    
    # Data loading profiling
    profile_data_loading: bool = True
    profile_preprocessing: bool = True
    profile_augmentation: bool = True
    
    # Memory profiling
    enable_memory_tracking: bool = True
    enable_gc_profiling: bool = True
    memory_snapshots: bool = True
    
    # GPU profiling
    enable_cuda_profiling: bool = True
    profile_cuda_memory: bool = True
    profile_cuda_operations: bool = True
    
    # Output settings
    save_profiles: bool = True
    output_dir: str = "profiles"
    detailed_reports: bool = True
    
    # Sampling settings
    sample_interval: float = 0.1  # seconds
    max_samples: int = 10000
    
    # Threading settings
    enable_thread_profiling: bool = True
    enable_process_profiling: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.profile_level not in ["basic", "detailed", "comprehensive"]:
            raise ValueError("profile_level must be one of: basic, detailed, comprehensive")
        
        if self.sample_interval <= 0:
            raise ValueError("sample_interval must be positive")
        
        if self.max_samples <= 0:
            raise ValueError("max_samples must be positive")

# =============================================================================
# PERFORMANCE PROFILER
# =============================================================================

class PerformanceProfiler:
    """Comprehensive performance profiler for Video-OpusClip."""
    
    def __init__(self, config: ProfilerConfig):
        self.config = config
        self.profiles = {}
        self.stats = {}
        self.line_profiler = None
        self.memory_profiler = None
        
        if config.enable_line_profiler:
            self.line_profiler = line_profiler.LineProfiler()
        
        if config.enable_memory_profiler:
            self.memory_profiler = memory_profiler.profile
        
        # Create output directory
        if config.save_profiles:
            Path(config.output_dir).mkdir(exist_ok=True)
        
        logger.info(f"Performance profiler initialized: level={config.profile_level}")
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile a function."""
        if not self.config.enabled:
            return func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            # Profile with cProfile
            if self.config.enable_cprofile:
                profiler = cProfile.Profile()
                profiler.enable()
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Stop cProfile
                if self.config.enable_cprofile:
                    profiler.disable()
                
                # Record metrics
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                # Store profile data
                profile_data = {
                    'function': func.__name__,
                    'execution_time': execution_time,
                    'memory_delta': memory_delta,
                    'start_memory': start_memory,
                    'end_memory': end_memory,
                    'timestamp': time.time()
                }
                
                if self.config.enable_cprofile:
                    s = io.StringIO()
                    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                    stats.print_stats(20)
                    profile_data['cprofile_stats'] = s.getvalue()
                
                self.profiles[func.__name__] = profile_data
                
                # Log if significant
                if execution_time > 1.0 or abs(memory_delta) > 100 * 1024 * 1024:  # 100MB
                    logger.warning(f"Slow function detected: {func.__name__} took {execution_time:.2f}s, "
                                 f"memory delta: {memory_delta / 1024 / 1024:.1f}MB")
            
            return result
        
        return wrapper
    
    def profile_class(self, cls: type) -> type:
        """Decorator to profile all methods in a class."""
        if not self.config.enabled:
            return cls
        
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if callable(attr) and not attr_name.startswith('_'):
                setattr(cls, attr_name, self.profile_function(attr))
        
        return cls
    
    def profile_data_loader(self, data_loader: data.DataLoader) -> data.DataLoader:
        """Profile a data loader with custom iterator."""
        if not self.config.enabled or not self.config.profile_data_loading:
            return data_loader
        
        class ProfiledDataLoader:
            def __init__(self, loader, profiler):
                self.loader = loader
                self.profiler = profiler
                self.batch_times = []
                self.batch_sizes = []
                self.memory_usage = []
            
            def __iter__(self):
                self.batch_times = []
                self.batch_sizes = []
                self.memory_usage = []
                
                for batch_idx, batch in enumerate(self.loader):
                    start_time = time.time()
                    start_memory = self.profiler._get_memory_usage()
                    
                    yield batch
                    
                    end_time = time.time()
                    end_memory = self.profiler._get_memory_usage()
                    
                    batch_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    self.batch_times.append(batch_time)
                    self.batch_sizes.append(len(batch) if hasattr(batch, '__len__') else 1)
                    self.memory_usage.append(memory_delta)
                    
                    # Log slow batches
                    if batch_time > 1.0:
                        logger.warning(f"Slow batch {batch_idx}: {batch_time:.2f}s, "
                                     f"memory delta: {memory_delta / 1024 / 1024:.1f}MB")
            
            def __len__(self):
                return len(self.loader)
            
            def get_stats(self):
                if not self.batch_times:
                    return {}
                
                return {
                    'avg_batch_time': np.mean(self.batch_times),
                    'max_batch_time': np.max(self.batch_times),
                    'min_batch_time': np.min(self.batch_times),
                    'std_batch_time': np.std(self.batch_times),
                    'total_batches': len(self.batch_times),
                    'avg_memory_delta': np.mean(self.memory_usage),
                    'max_memory_delta': np.max(self.memory_usage),
                    'min_memory_delta': np.min(self.memory_usage)
                }
        
        return ProfiledDataLoader(data_loader, self)
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        process = psutil.Process()
        return process.memory_info().rss
    
    def get_profile_stats(self) -> Dict[str, Any]:
        """Get comprehensive profile statistics."""
        stats = {
            'total_functions_profiled': len(self.profiles),
            'slow_functions': [],
            'memory_intensive_functions': [],
            'profiles': self.profiles
        }
        
        # Identify slow functions
        for func_name, profile_data in self.profiles.items():
            if profile_data['execution_time'] > 1.0:
                stats['slow_functions'].append({
                    'function': func_name,
                    'execution_time': profile_data['execution_time'],
                    'memory_delta': profile_data['memory_delta']
                })
            
            if abs(profile_data['memory_delta']) > 100 * 1024 * 1024:  # 100MB
                stats['memory_intensive_functions'].append({
                    'function': func_name,
                    'execution_time': profile_data['execution_time'],
                    'memory_delta': profile_data['memory_delta']
                })
        
        return stats
    
    def save_profile_report(self, filename: str = None):
        """Save profile report to file."""
        if not self.config.save_profiles:
            return
        
        if filename is None:
            filename = f"profile_report_{int(time.time())}.json"
        
        filepath = Path(self.config.output_dir) / filename
        
        report = {
            'config': self.config.__dict__,
            'stats': self.get_profile_stats(),
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Profile report saved to {filepath}")

# =============================================================================
# MEMORY PROFILER
# =============================================================================

class MemoryProfiler:
    """Advanced memory profiling for Video-OpusClip."""
    
    def __init__(self, config: ProfilerConfig):
        self.config = config
        self.memory_snapshots = []
        self.memory_traces = []
        self.gc_stats = []
        
        if config.enable_memory_tracking:
            tracemalloc.start()
        
        logger.info("Memory profiler initialized")
    
    @contextmanager
    def memory_context(self, context_name: str):
        """Context manager for memory profiling."""
        if not self.config.enabled:
            yield
            return
        
        # Take snapshot before
        if self.config.memory_snapshots:
            snapshot1 = tracemalloc.take_snapshot()
        
        start_memory = self._get_memory_usage()
        start_gc_stats = self._get_gc_stats()
        
        try:
            yield
        finally:
            # Take snapshot after
            if self.config.memory_snapshots:
                snapshot2 = tracemalloc.take_snapshot()
            
            end_memory = self._get_memory_usage()
            end_gc_stats = self._get_gc_stats()
            
            # Calculate differences
            memory_delta = end_memory - start_memory
            gc_delta = {
                'collections': end_gc_stats['collections'] - start_gc_stats['collections'],
                'collected': end_gc_stats['collected'] - start_gc_stats['collected'],
                'uncollectable': end_gc_stats['uncollectable'] - start_gc_stats['uncollectable']
            }
            
            # Store memory trace
            memory_trace = {
                'context': context_name,
                'start_memory': start_memory,
                'end_memory': end_memory,
                'memory_delta': memory_delta,
                'gc_delta': gc_delta,
                'timestamp': time.time()
            }
            
            self.memory_traces.append(memory_trace)
            
            # Analyze snapshots if available
            if self.config.memory_snapshots and 'snapshot1' in locals() and 'snapshot2' in locals():
                top_stats = snapshot2.compare_to(snapshot1, 'lineno')
                memory_trace['top_changes'] = [
                    {
                        'file': stat.traceback.format()[-1],
                        'size_diff': stat.size_diff,
                        'count_diff': stat.count_diff
                    }
                    for stat in top_stats[:10]
                ]
            
            # Log significant memory changes
            if abs(memory_delta) > 50 * 1024 * 1024:  # 50MB
                logger.warning(f"Significant memory change in {context_name}: "
                             f"{memory_delta / 1024 / 1024:.1f}MB")
    
    def profile_memory_usage(self, func: Callable) -> Callable:
        """Decorator to profile memory usage of a function."""
        if not self.config.enabled:
            return func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.memory_context(func.__name__):
                return func(*args, **kwargs)
        
        return wrapper
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        process = psutil.Process()
        return process.memory_info().rss
    
    def _get_gc_stats(self) -> Dict[str, int]:
        """Get garbage collection statistics."""
        return {
            'collections': gc.get_stats()[0]['collections'] if gc.get_stats() else 0,
            'collected': gc.get_stats()[0]['collected'] if gc.get_stats() else 0,
            'uncollectable': gc.get_stats()[0]['uncollectable'] if gc.get_stats() else 0
        }
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report."""
        if not self.memory_traces:
            return {'error': 'No memory traces available'}
        
        memory_deltas = [trace['memory_delta'] for trace in self.memory_traces]
        gc_collections = [trace['gc_delta']['collections'] for trace in self.memory_traces]
        
        return {
            'total_traces': len(self.memory_traces),
            'memory_statistics': {
                'total_memory_delta': sum(memory_deltas),
                'avg_memory_delta': np.mean(memory_deltas),
                'max_memory_delta': np.max(memory_deltas),
                'min_memory_delta': np.min(memory_deltas),
                'std_memory_delta': np.std(memory_deltas)
            },
            'gc_statistics': {
                'total_collections': sum(gc_collections),
                'avg_collections': np.mean(gc_collections),
                'max_collections': np.max(gc_collections)
            },
            'memory_intensive_contexts': [
                trace for trace in self.memory_traces
                if abs(trace['memory_delta']) > 100 * 1024 * 1024  # 100MB
            ],
            'traces': self.memory_traces
        }

# =============================================================================
# GPU PROFILER
# =============================================================================

class GPUProfiler:
    """GPU profiling for CUDA operations."""
    
    def __init__(self, config: ProfilerConfig):
        self.config = config
        self.gpu_events = []
        self.cuda_memory_usage = []
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, GPU profiling disabled")
            self.config.enable_gpu_profiler = False
        
        logger.info("GPU profiler initialized")
    
    @contextmanager
    def cuda_context(self, context_name: str):
        """Context manager for CUDA profiling."""
        if not self.config.enabled or not self.config.enable_gpu_profiler:
            yield
            return
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        start_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        
        start_event.record()
        
        try:
            yield
        finally:
            end_event.record()
            torch.cuda.synchronize()
            
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            end_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            
            elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
            memory_delta = end_memory - start_memory
            reserved_delta = end_reserved - start_reserved
            
            gpu_event = {
                'context': context_name,
                'elapsed_time_ms': elapsed_time,
                'memory_delta': memory_delta,
                'reserved_delta': reserved_delta,
                'start_memory': start_memory,
                'end_memory': end_memory,
                'timestamp': time.time()
            }
            
            self.gpu_events.append(gpu_event)
            
            # Log slow GPU operations
            if elapsed_time > 100:  # 100ms
                logger.warning(f"Slow GPU operation in {context_name}: {elapsed_time:.2f}ms")
    
    def profile_cuda_operation(self, func: Callable) -> Callable:
        """Decorator to profile CUDA operations."""
        if not self.config.enabled or not self.config.enable_gpu_profiler:
            return func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.cuda_context(func.__name__):
                return func(*args, **kwargs)
        
        return wrapper
    
    def get_gpu_report(self) -> Dict[str, Any]:
        """Get comprehensive GPU report."""
        if not self.gpu_events:
            return {'error': 'No GPU events available'}
        
        elapsed_times = [event['elapsed_time_ms'] for event in self.gpu_events]
        memory_deltas = [event['memory_delta'] for event in self.gpu_events]
        
        return {
            'total_events': len(self.gpu_events),
            'gpu_statistics': {
                'total_time_ms': sum(elapsed_times),
                'avg_time_ms': np.mean(elapsed_times),
                'max_time_ms': np.max(elapsed_times),
                'min_time_ms': np.min(elapsed_times),
                'std_time_ms': np.std(elapsed_times)
            },
            'memory_statistics': {
                'total_memory_delta': sum(memory_deltas),
                'avg_memory_delta': np.mean(memory_deltas),
                'max_memory_delta': np.max(memory_deltas),
                'min_memory_delta': np.min(memory_deltas)
            },
            'slow_gpu_operations': [
                event for event in self.gpu_events
                if event['elapsed_time_ms'] > 100  # 100ms
            ],
            'memory_intensive_operations': [
                event for event in self.gpu_events
                if abs(event['memory_delta']) > 100 * 1024 * 1024  # 100MB
            ],
            'events': self.gpu_events
        }

# =============================================================================
# DATA LOADING PROFILER
# =============================================================================

class DataLoadingProfiler:
    """Specialized profiler for data loading and preprocessing."""
    
    def __init__(self, config: ProfilerConfig):
        self.config = config
        self.data_loading_stats = defaultdict(list)
        self.preprocessing_stats = defaultdict(list)
        self.augmentation_stats = defaultdict(list)
        
        logger.info("Data loading profiler initialized")
    
    def profile_dataset(self, dataset: data.Dataset) -> data.Dataset:
        """Profile a dataset with custom __getitem__ method."""
        if not self.config.enabled:
            return dataset
        
        class ProfiledDataset:
            def __init__(self, dataset, profiler):
                self.dataset = dataset
                self.profiler = profiler
                self.access_count = 0
            
            def __getitem__(self, index):
                start_time = time.time()
                start_memory = self.profiler._get_memory_usage()
                
                try:
                    item = self.dataset[index]
                except Exception as e:
                    logger.error(f"Error loading item {index}: {e}")
                    raise
                
                end_time = time.time()
                end_memory = self.profiler._get_memory_usage()
                
                load_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                # Record statistics
                self.profiler.data_loading_stats['load_times'].append(load_time)
                self.profiler.data_loading_stats['memory_deltas'].append(memory_delta)
                self.profiler.data_loading_stats['indices'].append(index)
                
                self.access_count += 1
                
                # Log slow loads
                if load_time > 1.0:
                    logger.warning(f"Slow data loading for index {index}: {load_time:.2f}s")
                
                return item
            
            def __len__(self):
                return len(self.dataset)
            
            def get_stats(self):
                if not self.profiler.data_loading_stats['load_times']:
                    return {}
                
                load_times = self.profiler.data_loading_stats['load_times']
                memory_deltas = self.profiler.data_loading_stats['memory_deltas']
                
                return {
                    'total_accesses': self.access_count,
                    'avg_load_time': np.mean(load_times),
                    'max_load_time': np.max(load_times),
                    'min_load_time': np.min(load_times),
                    'std_load_time': np.std(load_times),
                    'avg_memory_delta': np.mean(memory_deltas),
                    'max_memory_delta': np.max(memory_deltas),
                    'slow_loads': len([t for t in load_times if t > 1.0])
                }
        
        return ProfiledDataset(dataset, self)
    
    def profile_preprocessing(self, preprocessing_func: Callable) -> Callable:
        """Profile preprocessing operations."""
        if not self.config.enabled or not self.config.profile_preprocessing:
            return preprocessing_func
        
        @functools.wraps(preprocessing_func)
        def wrapper(data, *args, **kwargs):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                result = preprocessing_func(data, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in preprocessing: {e}")
                raise
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            preprocess_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Record statistics
            self.preprocessing_stats['preprocess_times'].append(preprocess_time)
            self.preprocessing_stats['memory_deltas'].append(memory_delta)
            self.preprocessing_stats['function_name'].append(preprocessing_func.__name__)
            
            # Log slow preprocessing
            if preprocess_time > 0.5:  # 500ms
                logger.warning(f"Slow preprocessing in {preprocessing_func.__name__}: {preprocess_time:.2f}s")
            
            return result
        
        return wrapper
    
    def profile_augmentation(self, augmentation_func: Callable) -> Callable:
        """Profile data augmentation operations."""
        if not self.config.enabled or not self.config.profile_augmentation:
            return augmentation_func
        
        @functools.wraps(augmentation_func)
        def wrapper(data, *args, **kwargs):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                result = augmentation_func(data, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in augmentation: {e}")
                raise
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            augment_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Record statistics
            self.augmentation_stats['augment_times'].append(augment_time)
            self.augmentation_stats['memory_deltas'].append(memory_delta)
            self.augmentation_stats['function_name'].append(augmentation_func.__name__)
            
            # Log slow augmentation
            if augment_time > 0.1:  # 100ms
                logger.warning(f"Slow augmentation in {augmentation_func.__name__}: {augment_time:.2f}s")
            
            return result
        
        return wrapper
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        process = psutil.Process()
        return process.memory_info().rss
    
    def get_data_loading_report(self) -> Dict[str, Any]:
        """Get comprehensive data loading report."""
        reports = {}
        
        # Data loading statistics
        if self.data_loading_stats['load_times']:
            load_times = self.data_loading_stats['load_times']
            memory_deltas = self.data_loading_stats['memory_deltas']
            
            reports['data_loading'] = {
                'total_loads': len(load_times),
                'avg_load_time': np.mean(load_times),
                'max_load_time': np.max(load_times),
                'min_load_time': np.min(load_times),
                'std_load_time': np.std(load_times),
                'avg_memory_delta': np.mean(memory_deltas),
                'max_memory_delta': np.max(memory_deltas),
                'slow_loads': len([t for t in load_times if t > 1.0]),
                'bottleneck_indices': [
                    idx for idx, time in zip(self.data_loading_stats['indices'], load_times)
                    if time > 1.0
                ]
            }
        
        # Preprocessing statistics
        if self.preprocessing_stats['preprocess_times']:
            preprocess_times = self.preprocessing_stats['preprocess_times']
            memory_deltas = self.preprocessing_stats['memory_deltas']
            function_names = self.preprocessing_stats['function_name']
            
            reports['preprocessing'] = {
                'total_operations': len(preprocess_times),
                'avg_preprocess_time': np.mean(preprocess_times),
                'max_preprocess_time': np.max(preprocess_times),
                'min_preprocess_time': np.min(preprocess_times),
                'std_preprocess_time': np.std(preprocess_times),
                'avg_memory_delta': np.mean(memory_deltas),
                'max_memory_delta': np.max(memory_deltas),
                'slow_operations': len([t for t in preprocess_times if t > 0.5]),
                'function_breakdown': {
                    name: {
                        'count': function_names.count(name),
                        'avg_time': np.mean([t for t, n in zip(preprocess_times, function_names) if n == name])
                    }
                    for name in set(function_names)
                }
            }
        
        # Augmentation statistics
        if self.augmentation_stats['augment_times']:
            augment_times = self.augmentation_stats['augment_times']
            memory_deltas = self.augmentation_stats['memory_deltas']
            function_names = self.augmentation_stats['function_name']
            
            reports['augmentation'] = {
                'total_operations': len(augment_times),
                'avg_augment_time': np.mean(augment_times),
                'max_augment_time': np.max(augment_times),
                'min_augment_time': np.min(augment_times),
                'std_augment_time': np.std(augment_times),
                'avg_memory_delta': np.mean(memory_deltas),
                'max_memory_delta': np.max(memory_deltas),
                'slow_operations': len([t for t in augment_times if t > 0.1]),
                'function_breakdown': {
                    name: {
                        'count': function_names.count(name),
                        'avg_time': np.mean([t for t, n in zip(augment_times, function_names) if n == name])
                    }
                    for name in set(function_names)
                }
            }
        
        return reports

# =============================================================================
# MAIN PROFILER CLASS
# =============================================================================

class VideoOpusClipProfiler:
    """Main profiler class that combines all profiling capabilities."""
    
    def __init__(self, config: ProfilerConfig):
        self.config = config
        self.performance_profiler = PerformanceProfiler(config)
        self.memory_profiler = MemoryProfiler(config)
        self.gpu_profiler = GPUProfiler(config)
        self.data_loading_profiler = DataLoadingProfiler(config)
        
        self.start_time = time.time()
        self.profiling_active = False
        
        logger.info("Video-OpusClip profiler initialized")
    
    def start_profiling(self):
        """Start profiling session."""
        self.profiling_active = True
        self.start_time = time.time()
        logger.info("Profiling session started")
    
    def stop_profiling(self):
        """Stop profiling session."""
        self.profiling_active = False
        end_time = time.time()
        session_duration = end_time - self.start_time
        logger.info(f"Profiling session stopped. Duration: {session_duration:.2f}s")
    
    def profile_function(self, func: Callable) -> Callable:
        """Profile a function with all available profilers."""
        if not self.config.enabled:
            return func
        
        # Apply all profilers
        func = self.performance_profiler.profile_function(func)
        func = self.memory_profiler.profile_memory_usage(func)
        
        if self.config.enable_gpu_profiler:
            func = self.gpu_profiler.profile_cuda_operation(func)
        
        return func
    
    def profile_class(self, cls: type) -> type:
        """Profile all methods in a class."""
        if not self.config.enabled:
            return cls
        
        return self.performance_profiler.profile_class(cls)
    
    def profile_data_loader(self, data_loader: data.DataLoader) -> data.DataLoader:
        """Profile a data loader."""
        if not self.config.enabled:
            return data_loader
        
        return self.performance_profiler.profile_data_loader(data_loader)
    
    def profile_dataset(self, dataset: data.Dataset) -> data.Dataset:
        """Profile a dataset."""
        if not self.config.enabled:
            return dataset
        
        return self.data_loading_profiler.profile_dataset(dataset)
    
    def profile_preprocessing(self, preprocessing_func: Callable) -> Callable:
        """Profile preprocessing operations."""
        if not self.config.enabled:
            return preprocessing_func
        
        return self.data_loading_profiler.profile_preprocessing(preprocessing_func)
    
    def profile_augmentation(self, augmentation_func: Callable) -> Callable:
        """Profile augmentation operations."""
        if not self.config.enabled:
            return augmentation_func
        
        return self.data_loading_profiler.profile_augmentation(augmentation_func)
    
    @contextmanager
    def profiling_context(self, context_name: str):
        """Context manager for comprehensive profiling."""
        if not self.config.enabled:
            yield
            return
        
        with self.memory_profiler.memory_context(context_name):
            with self.gpu_profiler.cuda_context(context_name):
                yield
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive profiling report."""
        if not self.profiling_active:
            logger.warning("No active profiling session")
            return {}
        
        report = {
            'session_info': {
                'start_time': self.start_time,
                'duration': time.time() - self.start_time,
                'config': self.config.__dict__
            },
            'performance': self.performance_profiler.get_profile_stats(),
            'memory': self.memory_profiler.get_memory_report(),
            'gpu': self.gpu_profiler.get_gpu_report(),
            'data_loading': self.data_loading_profiler.get_data_loading_report()
        }
        
        return report
    
    def save_comprehensive_report(self, filename: str = None):
        """Save comprehensive profiling report."""
        if not self.config.save_profiles:
            return
        
        if filename is None:
            filename = f"comprehensive_profile_{int(time.time())}.json"
        
        filepath = Path(self.config.output_dir) / filename
        
        report = self.get_comprehensive_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive profile report saved to {filepath}")
        return filepath
    
    def identify_bottlenecks(self) -> Dict[str, List[str]]:
        """Identify performance bottlenecks."""
        bottlenecks = {
            'slow_functions': [],
            'memory_intensive_functions': [],
            'slow_data_loading': [],
            'slow_preprocessing': [],
            'slow_gpu_operations': [],
            'recommendations': []
        }
        
        # Analyze performance data
        perf_stats = self.performance_profiler.get_profile_stats()
        for func in perf_stats.get('slow_functions', []):
            bottlenecks['slow_functions'].append(
                f"{func['function']}: {func['execution_time']:.2f}s"
            )
        
        for func in perf_stats.get('memory_intensive_functions', []):
            bottlenecks['memory_intensive_functions'].append(
                f"{func['function']}: {func['memory_delta'] / 1024 / 1024:.1f}MB"
            )
        
        # Analyze data loading
        data_report = self.data_loading_profiler.get_data_loading_report()
        if 'data_loading' in data_report:
            dl_stats = data_report['data_loading']
            if dl_stats['slow_loads'] > 0:
                bottlenecks['slow_data_loading'].append(
                    f"{dl_stats['slow_loads']} slow loads, avg: {dl_stats['avg_load_time']:.2f}s"
                )
        
        if 'preprocessing' in data_report:
            pre_stats = data_report['preprocessing']
            if pre_stats['slow_operations'] > 0:
                bottlenecks['slow_preprocessing'].append(
                    f"{pre_stats['slow_operations']} slow operations, avg: {pre_stats['avg_preprocess_time']:.2f}s"
                )
        
        # Analyze GPU operations
        gpu_report = self.gpu_profiler.get_gpu_report()
        if 'gpu_statistics' in gpu_report:
            gpu_stats = gpu_report['gpu_statistics']
            if gpu_stats['avg_time_ms'] > 50:  # 50ms
                bottlenecks['slow_gpu_operations'].append(
                    f"Avg GPU time: {gpu_stats['avg_time_ms']:.2f}ms"
                )
        
        # Generate recommendations
        if bottlenecks['slow_functions']:
            bottlenecks['recommendations'].append(
                "Consider optimizing slow functions or using caching"
            )
        
        if bottlenecks['memory_intensive_functions']:
            bottlenecks['recommendations'].append(
                "Consider memory optimization or garbage collection"
            )
        
        if bottlenecks['slow_data_loading']:
            bottlenecks['recommendations'].append(
                "Consider using num_workers in DataLoader or caching"
            )
        
        if bottlenecks['slow_preprocessing']:
            bottlenecks['recommendations'].append(
                "Consider preprocessing data offline or using faster operations"
            )
        
        return bottlenecks

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_profiler_config(
    profile_level: str = "detailed",
    enable_gpu_profiling: bool = True,
    save_profiles: bool = True
) -> ProfilerConfig:
    """Create profiler configuration."""
    if profile_level == "basic":
        return ProfilerConfig(
            enabled=True,
            profile_level="basic",
            enable_cprofile=True,
            enable_line_profiler=False,
            enable_memory_profiler=False,
            enable_gpu_profiler=enable_gpu_profiling,
            save_profiles=save_profiles
        )
    elif profile_level == "detailed":
        return ProfilerConfig(
            enabled=True,
            profile_level="detailed",
            enable_cprofile=True,
            enable_line_profiler=True,
            enable_memory_profiler=True,
            enable_gpu_profiler=enable_gpu_profiling,
            save_profiles=save_profiles
        )
    else:  # comprehensive
        return ProfilerConfig(
            enabled=True,
            profile_level="comprehensive",
            enable_cprofile=True,
            enable_line_profiler=True,
            enable_memory_profiler=True,
            enable_gpu_profiler=enable_gpu_profiling,
            enable_memory_tracking=True,
            enable_gc_profiling=True,
            memory_snapshots=True,
            enable_cuda_profiling=True,
            profile_cuda_memory=True,
            profile_cuda_operations=True,
            save_profiles=save_profiles
        )

def profile_function(profile_level: str = "detailed"):
    """Decorator to profile a function."""
    def decorator(func):
        config = create_profiler_config(profile_level)
        profiler = VideoOpusClipProfiler(config)
        return profiler.profile_function(func)
    return decorator

def profile_class(profile_level: str = "detailed"):
    """Decorator to profile a class."""
    def decorator(cls):
        config = create_profiler_config(profile_level)
        profiler = VideoOpusClipProfiler(config)
        return profiler.profile_class(cls)
    return decorator

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_profiling():
    """Example of how to use the profiler."""
    # Create profiler
    config = create_profiler_config("detailed")
    profiler = VideoOpusClipProfiler(config)
    
    # Start profiling
    profiler.start_profiling()
    
    # Profile a function
    @profiler.profile_function
    def slow_function():
        time.sleep(0.1)
        return "result"
    
    # Use the function
    for _ in range(10):
        slow_function()
    
    # Stop profiling and get report
    profiler.stop_profiling()
    report = profiler.get_comprehensive_report()
    bottlenecks = profiler.identify_bottlenecks()
    
    print("Profiling Report:")
    print(json.dumps(report, indent=2))
    print("\nBottlenecks:")
    print(json.dumps(bottlenecks, indent=2))

if __name__ == "__main__":
    example_profiling() 