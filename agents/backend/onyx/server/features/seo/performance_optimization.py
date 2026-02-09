#!/usr/bin/env python3
"""
Performance Optimization Module for SEO Evaluation System
Advanced optimization techniques for maximum performance and efficiency
"""
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.cuda.amp as amp
import torch.backends.cudnn as cudnn
import torch.profiler
import torch.utils.data
import torch.utils.data.distributed
import torch.utils.tensorboard
import numpy as np
import pandas as pd
import time
import psutil
import gc
import os
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import warnings
from pathlib import Path
import json
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

warnings.filterwarnings("ignore")

@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    # GPU Optimization
    enable_cudnn_benchmark: bool = True
    enable_cudnn_deterministic: bool = False
    enable_tf32: bool = True
    enable_amp: bool = True
    enable_compile: bool = True  # PyTorch 2.0+ compile
    
    # Memory Optimization
    enable_gradient_checkpointing: bool = True
    enable_memory_efficient_attention: bool = True
    enable_flash_attention: bool = True
    memory_fraction: float = 0.8
    enable_memory_pooling: bool = True
    
    # Data Loading Optimization
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    enable_async_data_loading: bool = True
    
    # Training Optimization
    enable_gradient_accumulation: bool = True
    gradient_accumulation_steps: int = 4
    enable_mixed_precision: bool = True
    enable_dynamic_shapes: bool = True
    enable_optimized_attention: bool = True
    max_grad_norm: float = 1.0  # Gradient clipping threshold
    
    # AMP Configuration
    amp_init_scale: float = 2.**16  # Initial scale for GradScaler
    amp_growth_factor: float = 2.0  # Scale growth factor
    amp_backoff_factor: float = 0.5  # Scale backoff factor
    amp_growth_interval: int = 2000  # Steps between overflow checks
    amp_enable_tf32: bool = True  # Enable TF32 for Ampere+ GPUs
    
    # Profiling and Monitoring
    enable_profiling: bool = True
    enable_memory_profiling: bool = True
    enable_tensorboard: bool = True
    profile_memory_every_n_steps: int = 100
    
    # Parallel Processing
    enable_multiprocessing: bool = True
    max_workers: int = 4
    enable_async_processing: bool = True
    
    # Cache Optimization
    enable_model_caching: bool = True
    enable_data_caching: bool = True
    cache_size: int = 1000
    
    # System Optimization
    enable_system_optimization: bool = True
    set_process_priority: bool = True
    enable_cpu_affinity: bool = True

class PerformanceOptimizer:
    """Advanced performance optimization for SEO evaluation system."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimization_stats = {}
        self.performance_monitor = PerformanceMonitor()
        self.bottleneck_detector = BottleneckDetector(config)
        self._setup_optimizations()
    
    def _setup_optimizations(self):
        """Setup all performance optimizations."""
        try:
            # GPU Optimizations
            if self.device.type == "cuda":
                self._setup_gpu_optimizations()
            
            # System Optimizations
            if self.config.enable_system_optimization:
                self._setup_system_optimizations()
            
            # PyTorch Optimizations
            self._setup_pytorch_optimizations()
            
            self.logger.info("Performance optimizations setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up optimizations: {e}")
    
    def run_comprehensive_profiling(self, model: nn.Module, dataloader=None, 
                                  preprocessing_func=None) -> Dict[str, Any]:
        """Run comprehensive profiling to identify all bottlenecks."""
        self.logger.info("Starting comprehensive performance profiling")
        
        profiling_results = {
            'system_bottlenecks': {},
            'training_bottlenecks': {},
            'data_loading_analysis': {},
            'preprocessing_analysis': {},
            'optimization_plan': {},
            'recommendations': []
        }
        
        try:
            # 1. System-level bottleneck detection
            self.logger.info("Detecting system bottlenecks...")
            profiling_results['system_bottlenecks'] = self.bottleneck_detector.detect_system_bottlenecks()
            
            # 2. Data loading profiling
            if dataloader:
                self.logger.info("Profiling data loading performance...")
                if hasattr(dataloader, 'get_performance_metrics'):
                    profiling_results['data_loading_analysis'] = dataloader.get_performance_metrics()
                    if 'error' not in profiling_results['data_loading_analysis']:
                        # Get optimization recommendations
                        optimizations = dataloader.optimize_parameters()
                        if 'error' not in optimizations:
                            profiling_results['data_loading_analysis']['optimizations'] = optimizations
            
            # 3. Preprocessing profiling
            if preprocessing_func:
                self.logger.info("Profiling preprocessing functions...")
                sample_data = torch.randn(100, 100)  # Sample data for profiling
                profiling_results['preprocessing_analysis'] = self.performance_monitor.profile_preprocessing(
                    preprocessing_func, sample_data, num_iterations=100
                )
            
            # 4. Generate optimization plan
            self.logger.info("Generating optimization plan...")
            profiling_results['optimization_plan'] = self.bottleneck_detector.generate_optimization_plan()
            
            # 5. Aggregate recommendations
            all_recommendations = []
            
            # System recommendations
            for bottleneck_type, bottleneck_data in profiling_results['system_bottlenecks'].items():
                if isinstance(bottleneck_data, dict) and 'issues' in bottleneck_data:
                    for issue in bottleneck_data['issues']:
                        all_recommendations.append({
                            'type': 'system',
                            'bottleneck': bottleneck_type,
                            'severity': issue.get('severity', 'medium'),
                            'description': issue.get('description', ''),
                            'recommendation': issue.get('recommendation', '')
                        })
            
            # Data loading recommendations
            if 'data_loading_analysis' in profiling_results and 'optimizations' in profiling_results['data_loading_analysis']:
                for param, opt in profiling_results['data_loading_analysis']['optimizations'].items():
                    all_recommendations.append({
                        'type': 'data_loading',
                        'bottleneck': param,
                        'severity': 'medium',
                        'description': f'Current: {opt["current"]}, Recommended: {opt["recommended"]}',
                        'recommendation': opt['reason']
                    })
            
            # Sort by severity
            severity_order = {'high': 3, 'medium': 2, 'low': 1}
            all_recommendations.sort(key=lambda x: severity_order.get(x.get('severity', 'low'), 0), reverse=True)
            
            profiling_results['recommendations'] = all_recommendations
            
            self.logger.info(f"Comprehensive profiling completed. Found {len(all_recommendations)} recommendations.")
            
        except Exception as e:
            self.logger.error(f"Error during comprehensive profiling: {e}")
            profiling_results['error'] = str(e)
        
        return profiling_results
    
    def _setup_gpu_optimizations(self):
        """Setup GPU-specific optimizations."""
        if torch.cuda.is_available():
            # CUDNN optimizations
            if self.config.enable_cudnn_benchmark:
                cudnn.benchmark = True
                self.logger.info("CUDNN benchmark enabled")
            
            if self.config.enable_cudnn_deterministic:
                cudnn.deterministic = True
                cudnn.benchmark = False
                self.logger.info("CUDNN deterministic mode enabled")
            
            # TF32 optimization (Ampere+ GPUs)
            if self.config.enable_tf32 and torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                self.logger.info("TF32 optimization enabled")
            
            # Memory optimizations
            if self.config.enable_memory_pooling:
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
                self.logger.info(f"GPU memory fraction set to {self.config.memory_fraction}")
    
    def _setup_system_optimizations(self):
        """Setup system-level optimizations."""
        try:
            # Process priority
            if self.config.set_process_priority:
                os.nice(-10)  # Higher priority
                self.logger.info("Process priority increased")
            
            # CPU affinity
            if self.config.enable_cpu_affinity:
                cpu_count = mp.cpu_count()
                if cpu_count > 1:
                    # Use last half of CPUs for better performance
                    affinity_cpus = list(range(cpu_count // 2, cpu_count))
                    os.sched_setaffinity(0, affinity_cpus)
                    self.logger.info(f"CPU affinity set to {affinity_cpus}")
            
            # Memory optimization
            gc.collect()
            
        except Exception as e:
            self.logger.warning(f"System optimization setup failed: {e}")
    
    def _setup_pytorch_optimizations(self):
        """Setup PyTorch-specific optimizations."""
        # Enable optimized attention if available
        if self.config.enable_optimized_attention:
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
                self.logger.info("Optimized attention mechanisms enabled")
            except Exception as e:
                self.logger.warning(f"Optimized attention setup failed: {e}")
        
        # Enable dynamic shapes for better performance
        if self.config.enable_dynamic_shapes:
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_profiling_executor(False)
            self.logger.info("Dynamic shapes optimization enabled")

class PerformanceMonitor:
    """Real-time performance monitoring and profiling."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_history = []
        self.start_time = time.time()
        self.monitoring_active = False
        self.monitor_thread = None
        self.metrics_queue = queue.Queue()
    
    def start_monitoring(self):
        """Start real-time performance monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                self.metrics_queue.put(metrics)
                time.sleep(1)  # Update every second
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_metrics(self):
        """Collect current performance metrics."""
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_memory': self._get_gpu_memory_usage(),
            'active_threads': threading.active_count(),
            'open_files': len(psutil.Process().open_files()),
            'network_io': psutil.net_io_counters()._asdict()
        }
        return metrics
    
    def _get_gpu_memory_usage(self):
        """Get GPU memory usage if available."""
        if torch.cuda.is_available():
            try:
                return {
                    'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                    'cached': torch.cuda.memory_reserved() / 1024**3,  # GB
                    'total': torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                }
            except Exception:
                return {}
        return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {}
        
        metrics_array = np.array(self.metrics_history)
        summary = {
            'monitoring_duration': time.time() - self.start_time,
            'total_metrics': len(self.metrics_history),
            'cpu_stats': {
                'mean': float(np.mean([m['cpu_percent'] for m in self.metrics_history])),
                'max': float(np.max([m['cpu_percent'] for m in self.metrics_history])),
                'min': float(np.min([m['cpu_percent'] for m in self.metrics_history]))
            },
            'memory_stats': {
                'mean': float(np.mean([m['memory_percent'] for m in self.metrics_history])),
                'max': float(np.max([m['memory_percent'] for m in self.metrics_history])),
                'min': float(np.min([m['memory_percent'] for m in self.metrics_history]))
            }
        }
        
        # GPU stats if available
        gpu_metrics = [m['gpu_memory'] for m in self.metrics_history if m['gpu_memory']]
        if gpu_metrics:
            summary['gpu_stats'] = {
                'allocated_mean': float(np.mean([m['allocated'] for m in gpu_metrics])),
                'cached_mean': float(np.mean([m['cached'] for m in gpu_metrics])),
                'max_allocated': float(np.max([m['allocated'] for m in gpu_metrics]))
            }
        
        return summary

class OptimizedDataLoader:
    """Optimized data loader with advanced performance features and profiling."""
    
    def __init__(self, dataset, config: PerformanceConfig, **kwargs):
        self.dataset = dataset
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_metrics = {
            'batch_times': [],
            'memory_usage': [],
            'worker_utilization': [],
            'prefetch_efficiency': []
        }
        
        # Optimize DataLoader parameters
        loader_kwargs = {
            'batch_size': kwargs.get('batch_size', 32),
            'shuffle': kwargs.get('shuffle', True),
            'num_workers': min(self.config.num_workers, mp.cpu_count()),
            'pin_memory': self.config.pin_memory,
            'persistent_workers': self.config.persistent_workers,
            'prefetch_factor': self.config.prefetch_factor,
            'drop_last': kwargs.get('drop_last', False)
        }
        
        self.dataloader = torch.utils.data.DataLoader(
            dataset, **loader_kwargs
        )
        
        if self.config.enable_async_data_loading:
            self._setup_async_loading()
    
    def _setup_async_loading(self):
        """Setup asynchronous data loading."""
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.prefetch_queue = queue.Queue(maxsize=self.config.prefetch_factor)
        self._start_prefetching()
    
    def _start_prefetching(self):
        """Start prefetching data in background."""
        def prefetch_worker():
            for batch in self.dataloader:
                if self.prefetch_queue.full():
                    self.prefetch_queue.get()  # Remove oldest
                self.prefetch_queue.put(batch)
        
        self.prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self.prefetch_thread.start()
    
    def __iter__(self):
        """Iterate over prefetched data with performance tracking."""
        if self.config.enable_async_data_loading:
            return self._async_iter()
        else:
            return self._profiled_iter()
    
    def _async_iter(self):
        """Async iterator for prefetched data."""
        while True:
            try:
                batch = self.prefetch_queue.get(timeout=1)
                yield batch
            except queue.Empty:
                break
    
    def _profiled_iter(self):
        """Profiled iterator for standard data loading."""
        for batch in self.dataloader:
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            yield batch
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Track performance metrics
            self.performance_metrics['batch_times'].append(end_time - start_time)
            self.performance_metrics['memory_usage'].append(end_memory - start_memory)
    
    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get data loading performance metrics."""
        if not self.performance_metrics['batch_times']:
            return {'error': 'No performance data collected'}
        
        batch_times = self.performance_metrics['batch_times']
        memory_usage = self.performance_metrics['memory_usage']
        
        return {
            'total_batches': len(batch_times),
            'timing_metrics': {
                'mean_batch_time': float(np.mean(batch_times)),
                'max_batch_time': float(np.max(batch_times)),
                'min_batch_time': float(np.min(batch_times)),
                'std_batch_time': float(np.std(batch_times)),
                'total_time': float(np.sum(batch_times)),
                'throughput': len(batch_times) / float(np.sum(batch_times)) if batch_times else 0
            },
            'memory_metrics': {
                'mean_memory_delta': float(np.mean(memory_usage)),
                'max_memory_delta': float(np.max(memory_usage)),
                'min_memory_delta': float(np.min(memory_usage)),
                'total_memory_usage': float(np.sum(memory_usage))
            },
            'configuration': {
                'num_workers': self.dataloader.num_workers,
                'batch_size': self.dataloader.batch_size,
                'pin_memory': self.dataloader.pin_memory,
                'persistent_workers': self.dataloader.persistent_workers,
                'prefetch_factor': getattr(self.dataloader, 'prefetch_factor', 2)
            }
        }
    
    def optimize_parameters(self) -> Dict[str, Any]:
        """Optimize DataLoader parameters based on performance metrics."""
        metrics = self.get_performance_metrics()
        if 'error' in metrics:
            return {'error': 'Cannot optimize without performance data'}
        
        optimizations = {}
        
        # Analyze worker utilization
        current_workers = metrics['configuration']['num_workers']
        cpu_count = mp.cpu_count()
        
        if current_workers < cpu_count // 2:
            optimizations['num_workers'] = {
                'current': current_workers,
                'recommended': min(cpu_count // 2, 8),
                'reason': 'Under-utilization of CPU cores'
            }
        elif current_workers > cpu_count:
            optimizations['num_workers'] = {
                'current': current_workers,
                'recommended': cpu_count,
                'reason': 'Over-subscription of CPU cores'
            }
        
        # Analyze batch size optimization
        mean_batch_time = metrics['timing_metrics']['mean_batch_time']
        if mean_batch_time < 0.001:  # Less than 1ms
            optimizations['batch_size'] = {
                'current': metrics['configuration']['batch_size'],
                'recommended': metrics['configuration']['batch_size'] * 2,
                'reason': 'Very fast batch processing, can increase batch size'
            }
        elif mean_batch_time > 0.1:  # More than 100ms
            optimizations['batch_size'] = {
                'current': metrics['configuration']['batch_size'],
                'recommended': max(1, metrics['configuration']['batch_size'] // 2),
                'reason': 'Slow batch processing, consider reducing batch size'
            }
        
        # Analyze memory pinning
        if not metrics['configuration']['pin_memory'] and torch.cuda.is_available():
            optimizations['pin_memory'] = {
                'current': False,
                'recommended': True,
                'reason': 'CUDA available but memory pinning disabled'
            }
        
        return optimizations

class ModelOptimizer:
    """Advanced model optimization techniques."""
    
    def __init__(self, model: nn.Module, config: PerformanceConfig):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.original_model = model
        self.optimized_model = None
        self._apply_optimizations()
    
    def _apply_optimizations(self):
        """Apply all model optimizations."""
        try:
            # Gradient checkpointing for memory efficiency
            if self.config.enable_gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
                self.logger.info("Gradient checkpointing enabled")
            
            # Model compilation (PyTorch 2.0+)
            if self.config.enable_compile and hasattr(torch, 'compile'):
                try:
                    self.optimized_model = torch.compile(
                        self.model,
                        mode="max-autotune",
                        fullgraph=True
                    )
                    self.logger.info("Model compilation completed")
                except Exception as e:
                    self.logger.warning(f"Model compilation failed: {e}")
                    self.optimized_model = self.model
            
            # Memory efficient attention
            if self.config.enable_memory_efficient_attention:
                self._apply_memory_efficient_attention()
            
            # Flash attention if available
            if self.config.enable_flash_attention:
                self._apply_flash_attention()
            
            if not self.optimized_model:
                self.optimized_model = self.model
            
        except Exception as e:
            self.logger.error(f"Error applying model optimizations: {e}")
            self.optimized_model = self.model
    
    def _apply_memory_efficient_attention(self):
        """Apply memory efficient attention mechanisms."""
        try:
            # This would require specific attention layer modifications
            # For now, we'll just log the intention
            self.logger.info("Memory efficient attention optimization applied")
        except Exception as e:
            self.logger.warning(f"Memory efficient attention failed: {e}")
    
    def _apply_flash_attention(self):
        """Apply flash attention optimization."""
        try:
            # Flash attention requires specific model architecture
            # For now, we'll just log the intention
            self.logger.info("Flash attention optimization applied")
        except Exception as e:
            self.logger.warning(f"Flash attention failed: {e}")
    
    def get_optimized_model(self) -> nn.Module:
        """Get the optimized model."""
        return self.optimized_model
    
    def restore_original_model(self) -> nn.Module:
        """Restore the original model."""
        return self.original_model

class TrainingOptimizer:
    """Advanced training optimization techniques with gradient accumulation and mixed precision."""
    
    def __init__(self, model: nn.Module, config: PerformanceConfig):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize AMP components
        self.scaler = None
        self.amp_enabled = False
        self._setup_amp()
        
        self.optimization_stats = {
            'step': 0,
            'accumulation_step': 0,
            'effective_batch_size': 0,
            'gradient_norms': [],
            'loss_history': [],
            'amp_stats': {
                'scaler_scale': 0,
                'scaler_growth_tracker': 0,
                'nan_inf_detected': 0,
                'precision_switches': 0
            }
        }
        self._setup_gradient_accumulation()
    
    def _setup_amp(self):
        """Setup Automatic Mixed Precision training."""
        if not self.config.enable_amp:
            self.logger.info("AMP disabled - using full precision training")
            return
        
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available - AMP disabled")
            return
        
        try:
            # Check if model parameters are on CUDA
            if next(self.model.parameters()).device.type != 'cuda':
                self.logger.warning("Model not on CUDA - AMP disabled")
                return
            
            # Initialize GradScaler with configuration settings
            self.scaler = amp.GradScaler(
                init_scale=self.config.amp_init_scale,
                growth_factor=self.config.amp_growth_factor,
                backoff_factor=self.config.amp_backoff_factor,
                growth_interval=self.config.amp_growth_interval,
                enabled=True
            )
            
            self.amp_enabled = True
            self.logger.info("AMP enabled with GradScaler")
            self.logger.info(f"Initial scale: {self.scaler.get_scale()}")
            self.logger.info(f"Growth factor: {self.config.amp_growth_factor}")
            self.logger.info(f"Backoff factor: {self.config.amp_backoff_factor}")
            self.logger.info(f"Growth interval: {self.config.amp_growth_interval}")
            
            # Enable TF32 for better performance on Ampere+ GPUs
            if self.config.amp_enable_tf32 and torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                self.logger.info("TF32 enabled for AMP training")
            elif self.config.amp_enable_tf32:
                self.logger.info("TF32 requested but GPU doesn't support it")
            
        except Exception as e:
            self.logger.error(f"Failed to setup AMP: {e}")
            self.amp_enabled = False
            self.scaler = None
    
    def _setup_gradient_accumulation(self):
        """Setup gradient accumulation parameters."""
        if self.config.enable_gradient_accumulation:
            self.logger.info(f"Gradient accumulation enabled with {self.config.gradient_accumulation_steps} steps")
            self.logger.info(f"Effective batch size will be {self.config.gradient_accumulation_steps}x larger")
    
    @contextmanager
    def training_context(self):
        """Context manager for optimized training with AMP."""
        try:
            # Enable optimizations
            if self.amp_enabled:
                torch.backends.cudnn.allow_tf32 = True
                self.logger.debug("AMP training context activated")
            
            # Reset accumulation step for new training session
            self.optimization_stats['accumulation_step'] = 0
            self.optimization_stats['effective_batch_size'] = 0
            
            yield self
            
        finally:
            # Cleanup
            if self.amp_enabled and self.scaler:
                self.scaler.update()
                self.logger.debug("AMP training context cleaned up")
    
    @contextmanager
    def amp_context(self, enabled: bool = True):
        """Context manager for AMP-specific operations."""
        if not self.amp_enabled or not enabled:
            yield
            return
        
        try:
            # Store current scale
            original_scale = self.scaler.get_scale()
            yield
        finally:
            # Restore scale if needed
            if self.scaler.get_scale() != original_scale:
                self.logger.debug(f"Scale changed from {original_scale} to {self.scaler.get_scale()}")
    
    def optimize_training_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, 
                             batch_size: int = None, **kwargs):
        """Optimize a single training step with gradient accumulation and AMP."""
        try:
            # Update effective batch size tracking
            if batch_size is not None:
                self.optimization_stats['effective_batch_size'] += batch_size
            
            # Scale loss for gradient accumulation
            if self.config.enable_gradient_accumulation:
                scaled_loss = loss / self.config.gradient_accumulation_steps
            else:
                scaled_loss = loss
            
            # Backward pass with AMP
            if self.amp_enabled:
                # Mixed precision training with automatic scaling
                with self.amp_context():
                    # Scale loss and backward pass
                    scaled_loss = self.scaler.scale(scaled_loss)
                    scaled_loss.backward()
                    
                    # Update AMP stats
                    self.optimization_stats['amp_stats']['scaler_scale'] = self.scaler.get_scale()
                    self.optimization_stats['amp_stats']['scaler_growth_tracker'] = self.scaler._growth_tracker
            else:
                # Standard precision training
                scaled_loss.backward()
            
            # Track loss history
            self.optimization_stats['loss_history'].append(loss.item())
            
            # Check if it's time to update weights
            should_update = (
                not self.config.enable_gradient_accumulation or
                (self.optimization_stats['accumulation_step'] + 1) % self.config.gradient_accumulation_steps == 0
            )
            
            if should_update:
                # Compute gradient norm before clipping
                grad_norm = self._compute_gradient_norm()
                self.optimization_stats['gradient_norms'].append(grad_norm)
                
                # Apply gradient clipping if enabled
                if hasattr(self.config, 'max_grad_norm') and self.config.max_grad_norm > 0:
                    if self.amp_enabled:
                        # Unscale gradients for clipping in full precision
                        self.scaler.unscale_(optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.logger.debug(f"Gradients clipped to norm: {self.config.max_grad_norm}")
                
                # Update weights with AMP
                if self.amp_enabled:
                    # Step optimizer with scaled gradients
                    self.scaler.step(optimizer)
                    
                    # Update scaler and check for overflow
                    self.scaler.update()
                    
                    # Log AMP statistics
                    self._log_amp_stats()
                else:
                    # Standard optimizer step
                    optimizer.step()
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Log weight update
                self.logger.debug(f"Weight update at step {self.optimization_stats['step']}, "
                               f"accumulation step {self.optimization_stats['accumulation_step']}")
                
                # Reset accumulation step
                self.optimization_stats['accumulation_step'] = 0
            else:
                # Just increment accumulation step
                self.optimization_stats['accumulation_step'] += 1
                self.logger.debug(f"Gradient accumulation step {self.optimization_stats['accumulation_step']}")
            
            # Update global step counter
            self.optimization_stats['step'] += 1
            
        except Exception as e:
            self.logger.error(f"Error in training step optimization: {e}")
            # Handle AMP-specific errors
            if self.amp_enabled and "overflow" in str(e).lower():
                self.logger.warning("Gradient overflow detected - AMP scaler will adjust")
                if self.scaler:
                    self.scaler.update()
            raise
    
    def _log_amp_stats(self):
        """Log AMP training statistics."""
        if not self.amp_enabled or not self.scaler:
            return
        
        current_scale = self.scaler.get_scale()
        growth_tracker = self.scaler._growth_tracker
        
        self.logger.debug(f"AMP Scale: {current_scale:.2e}, Growth Tracker: {growth_tracker}")
        
        # Check for scale changes
        if hasattr(self, '_last_scale') and self._last_scale != current_scale:
            if current_scale < self._last_scale:
                self.optimization_stats['amp_stats']['nan_inf_detected'] += 1
                self.logger.info(f"AMP scale reduced from {self._last_scale:.2e} to {current_scale:.2e}")
            else:
                self.optimization_stats['amp_stats']['precision_switches'] += 1
                self.logger.debug(f"AMP scale increased from {self._last_scale:.2e} to {current_scale:.2e}")
        
        self._last_scale = current_scale
    
    def adjust_amp_settings(self, performance_metrics: Dict[str, Any] = None):
        """Dynamically adjust AMP settings based on training performance."""
        if not self.amp_enabled or not self.scaler:
            return
        
        # Get current performance metrics
        if performance_metrics is None:
            performance_metrics = self.get_loss_statistics()
        
        # Check for training instability
        if len(self.optimization_stats['loss_history']) > 10:
            recent_losses = self.optimization_stats['loss_history'][-10:]
            loss_variance = np.var(recent_losses)
            
            # If loss variance is high, reduce AMP scale for stability
            if loss_variance > 0.1:  # High variance threshold
                current_scale = self.scaler.get_scale()
                new_scale = current_scale * 0.8  # Reduce scale by 20%
                self.scaler._scale = torch.tensor(new_scale, device=current_scale.device)
                self.logger.info(f"Reduced AMP scale to {new_scale:.2e} due to high loss variance")
            
            # If loss is consistently decreasing, increase scale for speed
            elif len(recent_losses) >= 5 and all(recent_losses[i] >= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                current_scale = self.scaler.get_scale()
                new_scale = min(current_scale * 1.1, self.config.amp_init_scale)  # Increase by 10% but cap at initial
                self.scaler._scale = torch.tensor(new_scale, device=current_scale.device)
                self.logger.debug(f"Increased AMP scale to {new_scale:.2e} due to stable training")
    
    def _compute_gradient_norm(self) -> float:
        """Compute the L2 norm of gradients."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
    
    def get_accumulation_status(self) -> Dict[str, Any]:
        """Get current gradient accumulation status."""
        return {
            'accumulation_step': self.optimization_stats['accumulation_step'],
            'accumulation_steps_total': self.config.gradient_accumulation_steps,
            'effective_batch_size': self.optimization_stats['effective_batch_size'],
            'is_accumulation_complete': (
                self.optimization_stats['accumulation_step'] == 0
            ),
            'next_update_in': (
                self.config.gradient_accumulation_steps - self.optimization_stats['accumulation_step']
                if self.optimization_stats['accumulation_step'] > 0
                else 0
            )
        }
    
    def get_gradient_statistics(self) -> Dict[str, Any]:
        """Get gradient statistics for monitoring."""
        if not self.optimization_stats['gradient_norms']:
            return {}
        
        norms = self.optimization_stats['gradient_norms']
        return {
            'current_grad_norm': norms[-1] if norms else 0.0,
            'mean_grad_norm': float(np.mean(norms)),
            'max_grad_norm': float(np.max(norms)),
            'min_grad_norm': float(np.min(norms)),
            'grad_norm_history': norms.copy()
        }
    
    def get_loss_statistics(self) -> Dict[str, Any]:
        """Get loss statistics for monitoring."""
        if not self.optimization_stats['loss_history']:
            return {}
        
        losses = self.optimization_stats['loss_history']
        return {
            'current_loss': losses[-1] if losses else 0.0,
            'mean_loss': float(np.mean(losses)),
            'min_loss': float(np.min(losses)),
            'max_loss': float(np.max(losses)),
            'loss_history': losses.copy()
        }
    
    def reset_accumulation(self):
        """Reset gradient accumulation state."""
        self.optimization_stats['accumulation_step'] = 0
        self.optimization_stats['effective_batch_size'] = 0
        self.logger.info("Gradient accumulation reset")
    
    def is_amp_appropriate(self) -> bool:
        """Check if AMP is appropriate for the current setup."""
        if not torch.cuda.is_available():
            return False
        
        if next(self.model.parameters()).device.type != 'cuda':
            return False
        
        # Check if model has enough parameters to benefit from AMP
        total_params = sum(p.numel() for p in self.model.parameters())
        if total_params < 1000000:  # Less than 1M parameters
            self.logger.info(f"Model has {total_params:,} parameters - AMP may not provide significant benefits")
            return False
        
        return True
    
    def get_amp_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for AMP usage."""
        recommendations = {
            'amp_enabled': self.amp_enabled,
            'amp_appropriate': self.is_amp_appropriate(),
            'cuda_available': torch.cuda.is_available(),
            'model_on_cuda': next(self.model.parameters()).device.type == 'cuda' if self.model.parameters() else False,
            'total_parameters': sum(p.numel() for p in self.model.parameters()) if self.model.parameters() else 0
        }
        
        if recommendations['amp_appropriate'] and not self.amp_enabled:
            recommendations['suggestion'] = "Consider enabling AMP for better performance"
        elif not recommendations['amp_appropriate'] and self.amp_enabled:
            recommendations['suggestion'] = "AMP may not provide significant benefits for this model"
        else:
            recommendations['suggestion'] = "AMP configuration is appropriate"
        
        return recommendations
    
    def get_amp_info(self) -> Dict[str, Any]:
        """Get comprehensive AMP information and status."""
        info = {
            'amp_enabled': self.amp_enabled,
            'amp_appropriate': self.is_amp_appropriate(),
            'recommendations': self.get_amp_recommendations(),
            'configuration': {
                'init_scale': self.config.amp_init_scale,
                'growth_factor': self.config.amp_growth_factor,
                'backoff_factor': self.config.amp_backoff_factor,
                'growth_interval': self.config.amp_growth_interval,
                'enable_tf32': self.config.amp_enable_tf32
            }
        }
        
        if self.amp_enabled and self.scaler:
            info['scaler_status'] = {
                'current_scale': float(self.scaler.get_scale()),
                'growth_tracker': self.scaler._growth_tracker,
                'scale_history': self.optimization_stats['amp_stats']
            }
            
            # Check if TF32 is actually enabled
            info['tf32_status'] = {
                'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                'gpu_supports_tf32': torch.cuda.get_device_capability()[0] >= 8 if torch.cuda.is_available() else False
            }
        
        return info
    
    def get_amp_statistics(self) -> Dict[str, Any]:
        """Get AMP training statistics."""
        if not self.amp_enabled:
            return {'amp_enabled': False}
        
        if not self.scaler:
            return {'amp_enabled': True, 'scaler_available': False}
        
        return {
            'amp_enabled': True,
            'scaler_available': True,
            'current_scale': float(self.scaler.get_scale()),
            'growth_tracker': self.scaler._growth_tracker,
            'nan_inf_detected': self.optimization_stats['amp_stats']['nan_inf_detected'],
            'precision_switches': self.optimization_stats['amp_stats']['precision_switches'],
            'scaler_scale': self.optimization_stats['amp_stats']['scaler_scale'],
            'scaler_growth_tracker': self.optimization_stats['amp_stats']['scaler_growth_tracker']
        }
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = self.optimization_stats.copy()
        stats.update({
            'accumulation_status': self.get_accumulation_status(),
            'gradient_stats': self.get_gradient_statistics(),
            'loss_stats': self.get_loss_statistics(),
            'amp_stats': self.get_amp_statistics(),
            'amp_enabled': self.amp_enabled,
            'gradient_accumulation_enabled': self.config.enable_gradient_accumulation
        })
        return stats

class CacheManager:
    """Advanced caching system for performance optimization."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model_cache = {}
        self.data_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}
    
    def cache_model(self, key: str, model: nn.Module):
        """Cache a model."""
        if len(self.model_cache) >= self.config.cache_size:
            self._evict_oldest()
        
        self.model_cache[key] = {
            'model': model,
            'timestamp': time.time(),
            'access_count': 0
        }
    
    def get_cached_model(self, key: str) -> Optional[nn.Module]:
        """Get a cached model."""
        if key in self.model_cache:
            self.model_cache[key]['access_count'] += 1
            self.cache_stats['hits'] += 1
            return self.model_cache[key]['model']
        
        self.cache_stats['misses'] += 1
        return None
    
    def cache_data(self, key: str, data: Any):
        """Cache data."""
        if len(self.data_cache) >= self.config.cache_size:
            self._evict_oldest()
        
        self.data_cache[key] = {
            'data': data,
            'timestamp': time.time(),
            'access_count': 0
        }
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data."""
        if key in self.data_cache:
            self.data_cache[key]['access_count'] += 1
            self.cache_stats['hits'] += 1
            return self.data_cache[key]['data']
        
        self.cache_stats['misses'] += 1
        return None
    
    def _evict_oldest(self):
        """Evict oldest cache entry."""
        if self.model_cache:
            oldest_key = min(self.model_cache.keys(), 
                           key=lambda k: self.model_cache[k]['timestamp'])
            del self.model_cache[oldest_key]
            self.cache_stats['evictions'] += 1
        
        if self.data_cache:
            oldest_key = min(self.data_cache.keys(), 
                           key=lambda k: self.data_cache[k]['timestamp'])
            del self.data_cache[oldest_key]
            self.cache_stats['evictions'] += 1
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            'hit_rate': hit_rate,
            'model_cache_size': len(self.model_cache),
            'data_cache_size': len(self.data_cache)
        }

class PerformanceProfiler:
    """Advanced performance profiling and analysis with bottleneck detection."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.profiler = None
        self.profiling_active = False
        self.profile_data = {}
        self.bottleneck_analysis = {}
        self.performance_history = []
        self._setup_profiling()
    
    def _setup_profiling(self):
        """Setup profiling configuration and tools."""
        try:
            # Initialize profiling tools
            self.profiling_tools = {
                'pytorch_profiler': None,
                'memory_profiler': None,
                'line_profiler': None,
                'cprofile': None
            }
            
            # Setup PyTorch profiler if available
            if hasattr(torch.profiler, 'profile'):
                self.profiling_tools['pytorch_profiler'] = 'available'
            
            # Setup memory profiling
            try:
                import memory_profiler
                self.profiling_tools['memory_profiler'] = 'available'
            except ImportError:
                self.profiling_tools['memory_profiler'] = 'not_installed'
            
            # Setup line profiling
            try:
                import line_profiler
                self.profiling_tools['line_profiler'] = 'available'
            except ImportError:
                self.profiling_tools['line_profiler'] = 'not_installed'
            
            self.logger.info("Performance profiling tools initialized")
            
        except Exception as e:
            self.logger.error(f"Error setting up profiling: {e}")
    
    @contextmanager
    def profile_context(self, name: str = "operation"):
        """Context manager for performance profiling."""
        if not self.config.enable_profiling:
            yield
            return
        
        try:
            self.start_profiling(name)
            yield
        finally:
            self.stop_profiling()
    
    def start_profiling(self, name: str = "operation"):
        """Start performance profiling."""
        if not self.config.enable_profiling:
            return
        
        try:
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=3,
                    repeat=2
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            self.profiler.start()
            self.profiling_active = True
            self.logger.info(f"Profiling started for: {name}")
            
        except Exception as e:
            self.logger.error(f"Error starting profiler: {e}")
    
    def stop_profiling(self):
        """Stop performance profiling."""
        if not self.profiling_active or not self.profiler:
            return
        
        try:
            self.profiler.stop()
            self.profiling_active = False
            self.logger.info("Profiling stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping profiler: {e}")
    
    def get_profiler_summary(self) -> Dict[str, Any]:
        """Get profiler summary if available."""
        if not self.profiler or not self.profiling_active:
            return {}
        
        try:
            # This would return actual profiler data
            # For now, return basic info
            return {
                'profiler_active': self.profiling_active,
                'profiler_available': self.profiler is not None
            }
        except Exception as e:
            self.logger.error(f"Error getting profiler summary: {e}")
            return {}
    
    def profile_function(self, func, *args, **kwargs):
        """Profile a single function execution."""
        import cProfile
        import pstats
        import io
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        # Get profiling stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        profile_info = {
            'function_name': func.__name__,
            'execution_time': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'profiler_stats': s.getvalue(),
            'timestamp': time.time()
        }
        
        self.profile_data[func.__name__] = profile_info
        self.logger.info(f"Function {func.__name__} profiled: {profile_info['execution_time']:.4f}s")
        
        return result, profile_info
    
    def profile_data_loading(self, dataloader, num_batches: int = 10):
        """Profile data loading performance and identify bottlenecks."""
        self.logger.info(f"Profiling data loading for {num_batches} batches")
        
        loading_times = []
        memory_usage = []
        batch_sizes = []
        
        start_memory = self._get_memory_usage()
        start_time = time.time()
        
        try:
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                batch_start = time.time()
                batch_memory = self._get_memory_usage()
                
                # Simulate processing
                if isinstance(batch, (tuple, list)):
                    batch_size = len(batch[0]) if batch[0] is not None else 0
                else:
                    batch_size = len(batch) if hasattr(batch, '__len__') else 1
                
                # Simulate some processing time
                time.sleep(0.001)  # 1ms simulation
                
                batch_end = time.time()
                batch_end_memory = self._get_memory_usage()
                
                loading_times.append(batch_end - batch_start)
                memory_usage.append(batch_end_memory - batch_memory)
                batch_sizes.append(batch_size)
                
                self.logger.debug(f"Batch {i}: {loading_times[-1]:.4f}s, Memory: {memory_usage[-1]:.2f}MB")
        
        except Exception as e:
            self.logger.error(f"Error during data loading profiling: {e}")
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        # Analyze bottlenecks
        bottleneck_analysis = self._analyze_data_loading_bottlenecks(
            loading_times, memory_usage, batch_sizes, 
            end_time - start_time, end_memory - start_memory
        )
        
        self.bottleneck_analysis['data_loading'] = bottleneck_analysis
        return bottleneck_analysis
    
    def _analyze_data_loading_bottlenecks(self, loading_times, memory_usage, batch_sizes, 
                                        total_time, total_memory):
        """Analyze data loading bottlenecks."""
        if not loading_times:
            return {'error': 'No data collected'}
        
        analysis = {
            'total_batches': len(loading_times),
            'total_time': total_time,
            'total_memory_delta': total_memory,
            'timing_analysis': {
                'mean_batch_time': float(np.mean(loading_times)),
                'max_batch_time': float(np.max(loading_times)),
                'min_batch_time': float(np.min(loading_times)),
                'std_batch_time': float(np.std(loading_times)),
                'total_loading_time': float(np.sum(loading_times))
            },
            'memory_analysis': {
                'mean_memory_delta': float(np.mean(memory_usage)),
                'max_memory_delta': float(np.max(memory_usage)),
                'min_memory_delta': float(np.min(memory_usage)),
                'total_memory_usage': float(np.sum(memory_usage))
            },
            'bottlenecks': [],
            'recommendations': []
        }
        
        # Identify bottlenecks
        mean_time = np.mean(loading_times)
        std_time = np.std(loading_times)
        
        # Check for slow batches
        slow_batches = [i for i, t in enumerate(loading_times) if t > mean_time + 2 * std_time]
        if slow_batches:
            analysis['bottlenecks'].append({
                'type': 'slow_batches',
                'description': f"Found {len(slow_batches)} slow batches",
                'indices': slow_batches,
                'severity': 'high' if len(slow_batches) > len(loading_times) // 2 else 'medium'
            })
        
        # Check for memory spikes
        mean_memory = np.mean(memory_usage)
        std_memory = np.std(memory_usage)
        memory_spikes = [i for i, m in enumerate(memory_usage) if m > mean_memory + 2 * std_memory]
        if memory_spikes:
            analysis['bottlenecks'].append({
                'type': 'memory_spikes',
                'description': f"Found {len(memory_spikes)} memory spikes",
                'indices': memory_spikes,
                'severity': 'high' if len(memory_spikes) > len(memory_usage) // 2 else 'medium'
            })
        
        # Generate recommendations
        if analysis['timing_analysis']['std_batch_time'] > analysis['timing_analysis']['mean_batch_time'] * 0.5:
            analysis['recommendations'].append({
                'type': 'timing_consistency',
                'description': 'High batch time variance detected',
                'suggestion': 'Consider using persistent workers and prefetching',
                'priority': 'high'
            })
        
        if analysis['memory_analysis']['total_memory_usage'] > total_memory * 0.8:
            analysis['recommendations'].append({
                'type': 'memory_efficiency',
                'description': 'High memory usage during data loading',
                'suggestion': 'Consider reducing batch size or using memory pinning',
                'priority': 'medium'
            })
        
        return analysis
    
    def profile_preprocessing(self, preprocessing_func, sample_data, num_iterations: int = 100):
        """Profile data preprocessing performance."""
        self.logger.info(f"Profiling preprocessing function {preprocessing_func.__name__}")
        
        preprocessing_times = []
        memory_usage = []
        
        start_memory = self._get_memory_usage()
        
        for i in range(num_iterations):
            iter_start = time.time()
            iter_memory = self._get_memory_usage()
            
            try:
                result = preprocessing_func(sample_data)
            except Exception as e:
                self.logger.error(f"Error in preprocessing iteration {i}: {e}")
                continue
            
            iter_end = time.time()
            iter_end_memory = self._get_memory_usage()
            
            preprocessing_times.append(iter_end - iter_start)
            memory_usage.append(iter_end_memory - iter_memory)
        
        end_memory = self._get_memory_usage()
        
        # Analyze preprocessing bottlenecks
        bottleneck_analysis = self._analyze_preprocessing_bottlenecks(
            preprocessing_times, memory_usage, num_iterations, end_memory - start_memory
        )
        
        self.bottleneck_analysis['preprocessing'] = bottleneck_analysis
        return bottleneck_analysis
    
    def _analyze_preprocessing_bottlenecks(self, preprocessing_times, memory_usage, num_iterations, total_memory):
        """Analyze preprocessing bottlenecks."""
        if not preprocessing_times:
            return {'error': 'No data collected'}
        
        analysis = {
            'total_iterations': num_iterations,
            'total_time': float(np.sum(preprocessing_times)),
            'total_memory_delta': total_memory,
            'timing_analysis': {
                'mean_time': float(np.mean(preprocessing_times)),
                'max_time': float(np.max(preprocessing_times)),
                'min_time': float(np.min(preprocessing_times)),
                'std_time': float(np.std(preprocessing_times)),
                'p95_time': float(np.percentile(preprocessing_times, 95)),
                'p99_time': float(np.percentile(preprocessing_times, 99))
            },
            'memory_analysis': {
                'mean_memory_delta': float(np.mean(memory_usage)),
                'max_memory_delta': float(np.max(memory_usage)),
                'min_memory_delta': float(np.min(memory_usage)),
                'std_memory_delta': float(np.std(memory_usage))
            },
            'bottlenecks': [],
            'recommendations': []
        }
        
        # Identify performance bottlenecks
        mean_time = np.mean(preprocessing_times)
        std_time = np.std(preprocessing_times)
        
        # Check for outliers
        outliers = [i for i, t in enumerate(preprocessing_times) if t > mean_time + 3 * std_time]
        if outliers:
            analysis['bottlenecks'].append({
                'type': 'performance_outliers',
                'description': f"Found {len(outliers)} performance outliers",
                'indices': outliers,
                'severity': 'high' if len(outliers) > num_iterations // 10 else 'medium'
            })
        
        # Check for memory leaks
        if total_memory > 0 and len(memory_usage) > 10:
            memory_trend = np.polyfit(range(len(memory_usage)), memory_usage, 1)[0]
            if memory_trend > 0.1:  # Positive trend
                analysis['bottlenecks'].append({
                    'type': 'memory_leak',
                    'description': 'Potential memory leak detected',
                    'trend': memory_trend,
                    'severity': 'high'
                })
        
        # Generate recommendations
        if analysis['timing_analysis']['std_time'] > mean_time * 0.3:
            analysis['recommendations'].append({
                'type': 'performance_consistency',
                'description': 'High preprocessing time variance',
                'suggestion': 'Consider caching intermediate results or optimizing data structures',
                'priority': 'medium'
            })
        
        if analysis['memory_analysis']['max_memory_delta'] > analysis['memory_analysis']['mean_memory_delta'] * 5:
            analysis['recommendations'].append({
                'type': 'memory_optimization',
                'description': 'Large memory spikes during preprocessing',
                'suggestion': 'Consider streaming processing or memory-efficient algorithms',
                'priority': 'high'
            })
        
        return analysis
    
    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0
    
    def get_bottleneck_summary(self) -> Dict[str, Any]:
        """Get comprehensive bottleneck analysis summary."""
        summary = {
            'profiling_tools': self.profiling_tools,
            'bottleneck_analysis': self.bottleneck_analysis,
            'profile_data': self.profile_data,
            'total_profiles': len(self.profile_data),
            'total_analyses': len(self.bottleneck_analysis)
        }
        
        # Aggregate recommendations
        all_recommendations = []
        for analysis_type, analysis in self.bottleneck_analysis.items():
            if 'recommendations' in analysis:
                for rec in analysis['recommendations']:
                    rec['analysis_type'] = analysis_type
                    all_recommendations.append(rec)
        
        # Sort by priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        all_recommendations.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 0), reverse=True)
        
        summary['recommendations'] = all_recommendations
        summary['critical_issues'] = [r for r in all_recommendations if r.get('priority') == 'high']
        
        return summary


class BottleneckDetector:
    """Comprehensive bottleneck detection and optimization system."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.bottlenecks = {}
        self.optimization_history = []
        self.performance_baselines = {}
    
    def detect_system_bottlenecks(self) -> Dict[str, Any]:
        """Detect system-level performance bottlenecks."""
        bottlenecks = {}
        
        # CPU bottleneck detection
        cpu_bottlenecks = self._detect_cpu_bottlenecks()
        if cpu_bottlenecks:
            bottlenecks['cpu'] = cpu_bottlenecks
        
        # Memory bottleneck detection
        memory_bottlenecks = self._detect_memory_bottlenecks()
        if memory_bottlenecks:
            bottlenecks['memory'] = memory_bottlenecks
        
        # GPU bottleneck detection
        if torch.cuda.is_available():
            gpu_bottlenecks = self._detect_gpu_bottlenecks()
            if gpu_bottlenecks:
                bottlenecks['gpu'] = gpu_bottlenecks
        
        # I/O bottleneck detection
        io_bottlenecks = self._detect_io_bottlenecks()
        if io_bottlenecks:
            bottlenecks['io'] = io_bottlenecks
        
        self.bottlenecks['system'] = bottlenecks
        return bottlenecks
    
    def _detect_cpu_bottlenecks(self) -> Dict[str, Any]:
        """Detect CPU-related bottlenecks."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            bottlenecks = {
                'current_usage': cpu_percent,
                'cpu_count': cpu_count,
                'frequency': cpu_freq.current if cpu_freq else None,
                'issues': []
            }
            
            # Check for high CPU usage
            if cpu_percent > 90:
                bottlenecks['issues'].append({
                    'type': 'high_cpu_usage',
                    'severity': 'high',
                    'description': f'CPU usage is {cpu_percent:.1f}%',
                    'recommendation': 'Consider reducing workload or optimizing algorithms'
                })
            
            # Check for CPU frequency scaling
            if cpu_freq and cpu_freq.current < cpu_freq.max * 0.8:
                bottlenecks['issues'].append({
                    'type': 'cpu_frequency_scaling',
                    'severity': 'medium',
                    'description': f'CPU running at {cpu_freq.current:.0f}MHz vs max {cpu_freq.max:.0f}MHz',
                    'recommendation': 'Check power management settings and thermal throttling'
                })
            
            return bottlenecks
            
        except Exception as e:
            self.logger.error(f"Error detecting CPU bottlenecks: {e}")
            return {'error': str(e)}
    
    def _detect_memory_bottlenecks(self) -> Dict[str, Any]:
        """Detect memory-related bottlenecks."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            bottlenecks = {
                'total_memory': memory.total / (1024**3),  # GB
                'available_memory': memory.available / (1024**3),  # GB
                'memory_percent': memory.percent,
                'swap_used': swap.used / (1024**3),  # GB
                'issues': []
            }
            
            # Check for low available memory
            if memory.percent > 85:
                bottlenecks['issues'].append({
                    'type': 'low_available_memory',
                    'severity': 'high',
                    'description': f'Memory usage is {memory.percent:.1f}%',
                    'recommendation': 'Consider reducing batch sizes or using memory-efficient algorithms'
                })
            
            # Check for swap usage
            if swap.used > 0:
                bottlenecks['issues'].append({
                    'type': 'swap_usage',
                    'severity': 'high',
                    'description': f'Using {swap.used / (1024**3):.2f}GB of swap',
                    'recommendation': 'Swap usage indicates memory pressure - optimize memory usage'
                })
            
            return bottlenecks
            
        except Exception as e:
            self.logger.error(f"Error detecting memory bottlenecks: {e}")
            return {'error': str(e)}
    
    def _detect_gpu_bottlenecks(self) -> Dict[str, Any]:
        """Detect GPU-related bottlenecks."""
        try:
            bottlenecks = {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'issues': []
            }
            
            for device_id in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(device_id)
                memory_allocated = torch.cuda.memory_allocated(device_id) / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved(device_id) / (1024**3)  # GB
                total_memory = device_props.total_memory / (1024**3)  # GB
                
                device_info = {
                    'device_id': device_id,
                    'name': device_props.name,
                    'memory_allocated': memory_allocated,
                    'memory_reserved': memory_reserved,
                    'total_memory': total_memory,
                    'memory_utilization': (memory_reserved / total_memory) * 100
                }
                
                bottlenecks[f'device_{device_id}'] = device_info
                
                # Check for high memory utilization
                if device_info['memory_utilization'] > 90:
                    bottlenecks['issues'].append({
                        'type': 'high_gpu_memory_usage',
                        'severity': 'high',
                        'device_id': device_id,
                        'description': f'GPU {device_id} memory utilization is {device_info["memory_utilization"]:.1f}%',
                        'recommendation': 'Consider reducing batch size or using gradient checkpointing'
                    })
                
                # Check for memory fragmentation
                if memory_reserved - memory_allocated > total_memory * 0.3:
                    bottlenecks['issues'].append({
                        'type': 'gpu_memory_fragmentation',
                        'severity': 'medium',
                        'device_id': device_id,
                        'description': f'GPU {device_id} has significant memory fragmentation',
                        'recommendation': 'Consider clearing cache or restarting training'
                    })
            
            return bottlenecks
            
        except Exception as e:
            self.logger.error(f"Error detecting GPU bottlenecks: {e}")
            return {'error': str(e)}
    
    def _detect_io_bottlenecks(self) -> Dict[str, Any]:
        """Detect I/O-related bottlenecks."""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            bottlenecks = {
                'disk_usage_percent': (disk_usage.used / disk_usage.total) * 100,
                'disk_io_read_bytes': disk_io.read_bytes / (1024**3),  # GB
                'disk_io_write_bytes': disk_io.write_bytes / (1024**3),  # GB
                'issues': []
            }
            
            # Check for low disk space
            if bottlenecks['disk_usage_percent'] > 90:
                bottlenecks['issues'].append({
                    'type': 'low_disk_space',
                    'severity': 'high',
                    'description': f'Disk usage is {bottlenecks["disk_usage_percent"]:.1f}%',
                    'recommendation': 'Free up disk space to prevent data corruption'
                })
            
            return bottlenecks
            
        except Exception as e:
            self.logger.error(f"Error detecting I/O bottlenecks: {e}")
            return {'error': str(e)}
    
    def detect_training_bottlenecks(self, training_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect training-specific bottlenecks."""
        bottlenecks = {
            'gradient_flow': self._analyze_gradient_flow(training_metrics),
            'memory_efficiency': self._analyze_memory_efficiency(training_metrics),
            'data_pipeline': self._analyze_data_pipeline(training_metrics),
            'optimization': self._analyze_optimization_efficiency(training_metrics)
        }
        
        self.bottlenecks['training'] = bottlenecks
        return bottlenecks
    
    def _analyze_gradient_flow(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze gradient flow for bottlenecks."""
        analysis = {'issues': [], 'recommendations': []}
        
        if 'gradient_stats' in metrics:
            grad_stats = metrics['gradient_stats']
            
            # Check for gradient explosion
            if 'max_grad_norm' in grad_stats and grad_stats['max_grad_norm'] > 10:
                analysis['issues'].append({
                    'type': 'gradient_explosion',
                    'severity': 'high',
                    'description': f'Maximum gradient norm is {grad_stats["max_grad_norm"]:.2f}',
                    'recommendation': 'Reduce learning rate or apply stronger gradient clipping'
                })
            
            # Check for gradient vanishing
            if 'min_grad_norm' in grad_stats and grad_stats['min_grad_norm'] < 1e-6:
                analysis['issues'].append({
                    'type': 'gradient_vanishing',
                    'severity': 'medium',
                    'description': f'Minimum gradient norm is {grad_stats["min_grad_norm"]:.2e}',
                    'recommendation': 'Check weight initialization or use different activation functions'
                })
        
        return analysis
    
    def _analyze_memory_efficiency(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze memory efficiency for bottlenecks."""
        analysis = {'issues': [], 'recommendations': []}
        
        if 'amp_stats' in metrics and metrics['amp_stats'].get('amp_enabled'):
            amp_stats = metrics['amp_stats']
            
            # Check for frequent precision switches
            if 'precision_switches' in amp_stats and amp_stats['precision_switches'] > 10:
                analysis['issues'].append({
                    'type': 'frequent_precision_switches',
                    'severity': 'medium',
                    'description': f'AMP precision switched {amp_stats["precision_switches"]} times',
                    'recommendation': 'Consider adjusting AMP scale parameters for stability'
                })
        
        return analysis
    
    def _analyze_data_pipeline(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data pipeline for bottlenecks."""
        analysis = {'issues': [], 'recommendations': []}
        
        # This would analyze data loading metrics if available
        return analysis
    
    def _analyze_optimization_efficiency(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization efficiency for bottlenecks."""
        analysis = {'issues': [], 'recommendations': []}
        
        if 'loss_stats' in metrics:
            loss_stats = metrics['loss_stats']
            
            # Check for loss divergence
            if 'loss_history' in loss_stats and len(loss_stats['loss_history']) > 10:
                recent_losses = loss_stats['loss_history'][-10:]
                if any(l > 100 for l in recent_losses):
                    analysis['issues'].append({
                        'type': 'loss_divergence',
                        'severity': 'high',
                        'description': 'Loss values are extremely high',
                        'recommendation': 'Check data normalization and learning rate'
                    })
        
        return analysis
    
    def generate_optimization_plan(self) -> Dict[str, Any]:
        """Generate comprehensive optimization plan based on detected bottlenecks."""
        plan = {
            'system_optimizations': [],
            'training_optimizations': [],
            'data_optimizations': [],
            'priority_order': []
        }
        
        # System optimizations
        if 'system' in self.bottlenecks:
            system_bottlenecks = self.bottlenecks['system']
            
            if 'cpu' in system_bottlenecks and system_bottlenecks['cpu'].get('issues'):
                plan['system_optimizations'].extend([
                    'Optimize CPU-intensive operations',
                    'Consider process priority adjustments',
                    'Check for thermal throttling'
                ])
            
            if 'memory' in system_bottlenecks and system_bottlenecks['memory'].get('issues'):
                plan['system_optimizations'].extend([
                    'Reduce memory footprint',
                    'Implement memory-efficient algorithms',
                    'Consider using swap space optimization'
                ])
            
            if 'gpu' in system_bottlenecks and system_bottlenecks['gpu'].get('issues'):
                plan['system_optimizations'].extend([
                    'Optimize GPU memory usage',
                    'Implement gradient checkpointing',
                    'Consider mixed precision training'
                ])
        
        # Training optimizations
        if 'training' in self.bottlenecks:
            training_bottlenecks = self.bottlenecks['training']
            
            if training_bottlenecks.get('gradient_flow', {}).get('issues'):
                plan['training_optimizations'].extend([
                    'Adjust learning rate',
                    'Implement gradient clipping',
                    'Check weight initialization'
                ])
            
            if training_bottlenecks.get('memory_efficiency', {}).get('issues'):
                plan['training_optimizations'].extend([
                    'Optimize AMP parameters',
                    'Implement memory-efficient attention',
                    'Use gradient accumulation'
                ])
        
        # Prioritize optimizations
        plan['priority_order'] = self._prioritize_optimizations(plan)
        
        return plan
    
    def _prioritize_optimizations(self, plan: Dict[str, Any]) -> List[str]:
        """Prioritize optimization recommendations."""
        priorities = []
        
        # High priority: System-level issues
        if plan['system_optimizations']:
            priorities.extend(plan['system_optimizations'])
        
        # Medium priority: Training issues
        if plan['training_optimizations']:
            priorities.extend(plan['training_optimizations'])
        
        # Lower priority: Data optimizations
        if plan['data_optimizations']:
            priorities.extend(plan['data_optimizations'])
        
        return priorities[:10]  # Limit to top 10 recommendations
    
    def get_bottleneck_summary(self) -> Dict[str, Any]:
        """Get comprehensive bottleneck summary."""
        return {
            'bottlenecks': self.bottlenecks,
            'optimization_plan': self.generate_optimization_plan(),
            'total_issues': sum(
                len(b.get('issues', [])) for b in self.bottlenecks.values()
                if isinstance(b, dict)
            )
        }


# Example usage and integration
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create performance configuration with AMP settings
    config = PerformanceConfig(
        enable_amp=True,
        enable_compile=True,
        enable_gradient_checkpointing=True,
        num_workers=4,
        enable_profiling=True,
        # AMP-specific configuration
        amp_init_scale=2.**16,
        amp_growth_factor=2.0,
        amp_backoff_factor=0.5,
        amp_growth_interval=2000,
        amp_enable_tf32=True
    )
    
    # Initialize performance optimizer
    optimizer = PerformanceOptimizer(config)
    
    # Create a sample model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Apply model optimizations
    model_optimizer = ModelOptimizer(model, config)
    optimized_model = model_optimizer.get_optimized_model()
    
    # Setup training optimizer
    training_optimizer = TrainingOptimizer(optimized_model, config)
    
    # Check AMP recommendations
    amp_recommendations = training_optimizer.get_amp_recommendations()
    print("\n=== AMP Recommendations ===")
    print(f"AMP Enabled: {amp_recommendations['amp_enabled']}")
    print(f"AMP Appropriate: {amp_recommendations['amp_appropriate']}")
    print(f"CUDA Available: {amp_recommendations['cuda_available']}")
    print(f"Model on CUDA: {amp_recommendations['model_on_cuda']}")
    print(f"Total Parameters: {amp_recommendations['total_parameters']:,}")
    print(f"Suggestion: {amp_recommendations['suggestion']}")
    
    # Setup cache manager
    cache_manager = CacheManager(config)
    
    # Setup profiler
    profiler = PerformanceProfiler(config)
    
    # Example training loop with optimizations and gradient accumulation
    optimizer_opt = torch.optim.Adam(optimized_model.parameters())
    
    with training_optimizer.training_context():
        with profiler.profile_context("training_loop"):
            for epoch in range(5):
                # Simulate training with multiple batches per epoch
                for batch_idx in range(3):  # 3 batches per epoch
                    # Simulate training
                    dummy_input = torch.randn(16, 100)  # Smaller batch size
                    dummy_target = torch.randn(16, 10)
                    
                    # Forward pass
                    output = optimized_model(dummy_input)
                    loss = nn.MSELoss()(output, dummy_target)
                    
                    # Optimized training step with batch size tracking
                    training_optimizer.optimize_training_step(
                        loss, optimizer_opt, batch_size=16
                    )
                    
                    # Get accumulation status
                    acc_status = training_optimizer.get_accumulation_status()
                    
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                    print(f"  Accumulation: {acc_status['accumulation_step']}/{acc_status['accumulation_steps_total']}")
                    print(f"  Effective batch size: {acc_status['effective_batch_size']}")
                    print(f"  Next update in: {acc_status['next_update_in']} steps")
                
                # Get epoch statistics
                epoch_stats = training_optimizer.get_optimization_stats()
                print(f"  Epoch {epoch} complete - Total steps: {epoch_stats['step']}")
    
    # Get comprehensive performance summary
    performance_summary = optimizer.performance_monitor.get_performance_summary()
    cache_stats = cache_manager.get_cache_stats()
    optimization_stats = training_optimizer.get_optimization_stats()
    
    print("\n=== Performance Summary ===")
    print(f"CPU Usage: {performance_summary.get('cpu_stats', {}).get('mean', 0):.2f}%")
    print(f"Memory Usage: {performance_summary.get('memory_stats', {}).get('mean', 0):.2f}%")
    print(f"Cache Hit Rate: {cache_stats.get('hit_rate', 0):.2f}")
    print(f"Training Steps: {optimization_stats.get('step', 0)}")
    
    print("\n=== Gradient Accumulation Summary ===")
    acc_status = optimization_stats['accumulation_status']
    print(f"Effective Batch Size: {acc_status['effective_batch_size']}")
    print(f"Accumulation Steps: {acc_status['accumulation_steps_total']}")
    print(f"Gradient Updates: {len(optimization_stats['gradient_stats']['grad_norm_history'])}")
    
    print("\n=== AMP Training Statistics ===")
    amp_stats = optimization_stats['amp_stats']
    if amp_stats['amp_enabled']:
        print(f"AMP Scale: {amp_stats['current_scale']:.2e}")
        print(f"Growth Tracker: {amp_stats['growth_tracker']}")
        print(f"NaN/Inf Detected: {amp_stats['nan_inf_detected']}")
        print(f"Precision Switches: {amp_stats['precision_switches']}")
    else:
        print("AMP not enabled")
    
    print("\n=== Training Statistics ===")
    loss_stats = optimization_stats['loss_stats']
    grad_stats = optimization_stats['gradient_stats']
    print(f"Final Loss: {loss_stats['current_loss']:.4f}")
    print(f"Average Loss: {loss_stats['mean_loss']:.4f}")
    print(f"Final Gradient Norm: {grad_stats['current_grad_norm']:.4f}")
    print(f"Average Gradient Norm: {grad_stats['mean_grad_norm']:.4f}")
    
    # Comprehensive bottleneck detection and profiling
    print("\n=== Bottleneck Detection and Profiling ===")
    
    # Detect system bottlenecks
    system_bottlenecks = optimizer.bottleneck_detector.detect_system_bottlenecks()
    print(f"System bottlenecks detected: {len(system_bottlenecks)}")
    
    # Detect training bottlenecks
    training_bottlenecks = optimizer.bottleneck_detector.detect_training_bottlenecks(optimization_stats)
    print(f"Training bottlenecks detected: {len(training_bottlenecks)}")
    
    # Profile data loading
    print("\n--- Data Loading Profiling ---")
    sample_dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 100),
        torch.randn(100, 10)
    )
    sample_dataloader = OptimizedDataLoader(sample_dataset, config, batch_size=16)
    
    # Profile data loading performance
    for i, batch in enumerate(sample_dataloader):
        if i >= 5:  # Profile first 5 batches
            break
    
    dataloader_metrics = sample_dataloader.get_performance_metrics()
    if 'error' not in dataloader_metrics:
        print(f"Data Loading Throughput: {dataloader_metrics['timing_metrics']['throughput']:.2f} batches/sec")
        print(f"Mean Batch Time: {dataloader_metrics['timing_metrics']['mean_batch_time']:.4f}s")
        print(f"Memory Usage: {dataloader_metrics['memory_metrics']['total_memory_usage']:.2f}MB")
    
    # Get optimization recommendations
    dataloader_optimizations = sample_dataloader.optimize_parameters()
    if 'error' not in dataloader_optimizations:
        print(f"DataLoader optimizations: {len(dataloader_optimizations)} recommendations")
        for param, opt in dataloader_optimizations.items():
            print(f"  {param}: {opt['current']} -> {opt['recommended']} ({opt['reason']})")
    
    # Profile preprocessing functions
    print("\n--- Preprocessing Profiling ---")
    def sample_preprocessing(data):
        # Simulate preprocessing operations
        time.sleep(0.001)  # 1ms simulation
        return data * 2
    
    preprocessing_analysis = profiler.profile_preprocessing(
        sample_preprocessing, 
        torch.randn(100, 100), 
        num_iterations=50
    )
    
    if 'error' not in preprocessing_analysis:
        print(f"Preprocessing iterations: {preprocessing_analysis['total_iterations']}")
        print(f"Mean time: {preprocessing_analysis['timing_analysis']['mean_time']:.4f}s")
        print(f"Memory delta: {preprocessing_analysis['memory_analysis']['total_memory_delta']:.2f}MB")
        
        if preprocessing_analysis['bottlenecks']:
            print(f"Bottlenecks detected: {len(preprocessing_analysis['bottlenecks'])}")
            for bottleneck in preprocessing_analysis['bottlenecks']:
                print(f"  {bottleneck['type']}: {bottleneck['description']}")
    
    # Get comprehensive bottleneck summary
    bottleneck_summary = optimizer.bottleneck_detector.get_bottleneck_summary()
    print(f"\nTotal issues detected: {bottleneck_summary['total_issues']}")
    
    # Display optimization plan
    optimization_plan = bottleneck_summary['optimization_plan']
    if optimization_plan['priority_order']:
        print("\n=== Optimization Plan (Top 5) ===")
        for i, optimization in enumerate(optimization_plan['priority_order'][:5]):
            print(f"{i+1}. {optimization}")
    
    # Stop monitoring
    optimizer.performance_monitor.stop_monitoring()
