from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import time
import gc
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, ContextManager
from dataclasses import dataclass, field
from datetime import datetime
import warnings
from abc import ABC, abstractmethod
import functools
from contextlib import contextmanager
import threading
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.cuda.amp import GradScaler
import torch.nn.utils as utils
            import psutil
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Gradient Accumulation for Large Batch Sizes
Comprehensive gradient accumulation system for handling large batch sizes with advanced features like dynamic accumulation, memory optimization, and monitoring.
"""


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class GradientAccumulationConfig:
    """Configuration for gradient accumulation."""
    # Basic accumulation parameters
    accumulation_steps: int = 4
    effective_batch_size: int = 128
    target_batch_size: int = 512
    
    # Dynamic accumulation parameters
    enable_dynamic_accumulation: bool = True
    dynamic_threshold: float = 0.1  # Memory usage threshold
    min_accumulation_steps: int = 1
    max_accumulation_steps: int = 16
    
    # Memory optimization parameters
    enable_memory_optimization: bool = True
    enable_gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    enable_gradient_scaling: bool = True
    gradient_scale_factor: float = 1.0
    
    # Mixed precision parameters
    enable_mixed_precision: bool = True
    enable_amp: bool = True
    amp_dtype: str = "float16"  # float16, bfloat16
    
    # Monitoring parameters
    enable_monitoring: bool = True
    enable_gradient_tracking: bool = True
    enable_memory_tracking: bool = True
    enable_performance_tracking: bool = True
    
    # Optimization parameters
    enable_optimizer_optimization: bool = True
    enable_scheduler_optimization: bool = True
    enable_loss_scaling: bool = True
    enable_dynamic_loss_scaling: bool = True
    
    # Advanced parameters
    enable_gradient_accumulation_hooks: bool = True
    enable_custom_accumulation: bool = False
    accumulation_strategy: str = "standard"  # standard, weighted, adaptive


class GradientAccumulator:
    """Advanced gradient accumulator for large batch training."""
    
    def __init__(self, config: GradientAccumulationConfig):
        
    """__init__ function."""
self.config = config
        self.current_step = 0
        self.accumulation_step = 0
        self.effective_batch_size = 0
        self.gradient_history = deque(maxlen=1000)
        self.memory_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        self.loss_history = deque(maxlen=1000)
        
        # Mixed precision setup
        self.scaler = None
        self.autocast_context = None
        if self.config.enable_amp:
            self._setup_mixed_precision()
        
        # Dynamic accumulation
        self.dynamic_accumulation_steps = self.config.accumulation_steps
        self.memory_monitor = MemoryMonitor() if self.config.enable_memory_tracking else None
        
        # Gradient tracking
        self.gradient_tracker = GradientTracker() if self.config.enable_gradient_tracking else None
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor() if self.config.enable_performance_tracking else None
    
    def _setup_mixed_precision(self) -> Any:
        """Setup mixed precision training."""
        if self.config.amp_dtype == "float16":
            self.scaler = GradScaler()
            self.autocast_context = torch.cuda.amp.autocast(dtype=torch.float16)
        elif self.config.amp_dtype == "bfloat16":
            self.scaler = GradScaler()
            self.autocast_context = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    
    @contextmanager
    def autocast(self) -> Any:
        """Context manager for automatic mixed precision."""
        if self.autocast_context is not None:
            with self.autocast_context:
                yield
        else:
            yield
    
    def should_accumulate(self) -> bool:
        """Check if gradients should be accumulated."""
        return self.accumulation_step < self.dynamic_accumulation_steps - 1
    
    def should_update(self) -> bool:
        """Check if parameters should be updated."""
        return self.accumulation_step >= self.dynamic_accumulation_steps - 1
    
    def step(self, model: nn.Module, optimizer: optim.Optimizer, 
             loss: torch.Tensor, batch_size: int = None) -> Dict[str, Any]:
        """Perform gradient accumulation step."""
        start_time = time.time()
        
        # Scale loss for mixed precision
        if self.scaler is not None:
            scaled_loss = self.scaler.scale(loss)
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update accumulation step
        self.accumulation_step += 1
        self.current_step += 1
        
        # Track loss
        self.loss_history.append(loss.item())
        
        # Track performance
        if self.performance_monitor:
            self.performance_monitor.record_step(
                batch_size or 1, 
                time.time() - start_time,
                self.accumulation_step,
                self.dynamic_accumulation_steps
            )
        
        # Check if we should update parameters
        if self.should_update():
            return self._update_parameters(model, optimizer, batch_size)
        else:
            return {
                'status': 'accumulated',
                'accumulation_step': self.accumulation_step,
                'total_steps': self.dynamic_accumulation_steps,
                'loss': loss.item()
            }
    
    def _update_parameters(self, model: nn.Module, optimizer: optim.Optimizer, 
                          batch_size: int = None) -> Dict[str, Any]:
        """Update model parameters after accumulation."""
        update_start_time = time.time()
        
        # Gradient clipping
        if self.config.enable_gradient_clipping:
            if self.scaler is not None:
                self.scaler.unscale_(optimizer)
            
            total_norm = utils.clip_grad_norm_(
                model.parameters(), 
                self.config.max_grad_norm
            )
        else:
            total_norm = self._compute_gradient_norm(model)
        
        # Track gradients
        if self.gradient_tracker:
            self.gradient_tracker.record_gradients(model, total_norm)
        
        # Step optimizer
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Reset accumulation step
        self.accumulation_step = 0
        
        # Update effective batch size
        self.effective_batch_size = (batch_size or 1) * self.dynamic_accumulation_steps
        
        # Track memory usage
        if self.memory_monitor:
            memory_info = self.memory_monitor.get_memory_usage()
            self.memory_history.append(memory_info)
        
        # Dynamic accumulation adjustment
        if self.config.enable_dynamic_accumulation:
            self._adjust_accumulation_steps()
        
        update_time = time.time() - update_start_time
        
        return {
            'status': 'updated',
            'accumulation_step': 0,
            'total_steps': self.dynamic_accumulation_steps,
            'effective_batch_size': self.effective_batch_size,
            'gradient_norm': total_norm.item() if hasattr(total_norm, 'item') else total_norm,
            'update_time': update_time,
            'loss': self.loss_history[-1] if self.loss_history else 0.0
        }
    
    def _compute_gradient_norm(self, model: nn.Module) -> float:
        """Compute gradient norm."""
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
        
        return total_norm
    
    def _adjust_accumulation_steps(self) -> Any:
        """Dynamically adjust accumulation steps based on memory usage."""
        if not self.memory_monitor:
            return
        
        memory_usage = self.memory_monitor.get_memory_usage()
        gpu_memory_used = memory_usage.get('gpu_memory_used_ratio', 0.0)
        
        # Adjust based on memory usage
        if gpu_memory_used > self.config.dynamic_threshold:
            # Increase accumulation steps to reduce memory usage
            new_steps = min(
                self.dynamic_accumulation_steps + 1,
                self.config.max_accumulation_steps
            )
            if new_steps != self.dynamic_accumulation_steps:
                logger.info(f"Increasing accumulation steps from {self.dynamic_accumulation_steps} to {new_steps}")
                self.dynamic_accumulation_steps = new_steps
        elif gpu_memory_used < self.config.dynamic_threshold * 0.5:
            # Decrease accumulation steps to improve efficiency
            new_steps = max(
                self.dynamic_accumulation_steps - 1,
                self.config.min_accumulation_steps
            )
            if new_steps != self.dynamic_accumulation_steps:
                logger.info(f"Decreasing accumulation steps from {self.dynamic_accumulation_steps} to {new_steps}")
                self.dynamic_accumulation_steps = new_steps
    
    def get_accumulation_stats(self) -> Dict[str, Any]:
        """Get gradient accumulation statistics."""
        return {
            'current_step': self.current_step,
            'accumulation_step': self.accumulation_step,
            'dynamic_accumulation_steps': self.dynamic_accumulation_steps,
            'effective_batch_size': self.effective_batch_size,
            'target_batch_size': self.config.target_batch_size,
            'accumulation_ratio': self.effective_batch_size / self.config.target_batch_size if self.config.target_batch_size > 0 else 0.0
        }


class MemoryMonitor:
    """Monitor memory usage for gradient accumulation."""
    
    def __init__(self) -> Any:
        self.memory_history = deque(maxlen=1000)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                total = torch.cuda.get_device_properties(i).total_memory
                
                memory_info[f'gpu_{i}_allocated_mb'] = allocated / (1024 * 1024)
                memory_info[f'gpu_{i}_reserved_mb'] = reserved / (1024 * 1024)
                memory_info[f'gpu_{i}_total_mb'] = total / (1024 * 1024)
                memory_info[f'gpu_{i}_memory_used_ratio'] = allocated / total
                memory_info[f'gpu_{i}_memory_reserved_ratio'] = reserved / total
        
        # System memory
        try:
            system_memory = psutil.virtual_memory()
            memory_info['system_memory_used_ratio'] = system_memory.percent / 100.0
            memory_info['system_memory_available_gb'] = system_memory.available / (1024 * 1024 * 1024)
        except ImportError:
            pass
        
        return memory_info
    
    def record_memory_usage(self) -> Any:
        """Record current memory usage."""
        memory_info = self.get_memory_usage()
        memory_info['timestamp'] = datetime.now().isoformat()
        self.memory_history.append(memory_info)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if not self.memory_history:
            return {}
        
        stats = {}
        for key in self.memory_history[0].keys():
            if key != 'timestamp':
                values = [m[key] for m in self.memory_history]
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_max'] = np.max(values)
                stats[f'{key}_min'] = np.min(values)
                stats[f'{key}_std'] = np.std(values)
        
        return stats


class GradientTracker:
    """Track gradient statistics during accumulation."""
    
    def __init__(self) -> Any:
        self.gradient_history = deque(maxlen=1000)
        self.norm_history = deque(maxlen=1000)
        self.parameter_stats = defaultdict(list)
    
    def record_gradients(self, model: nn.Module, total_norm: float):
        """Record gradient statistics."""
        # Record total norm
        self.norm_history.append(total_norm)
        
        # Record per-parameter statistics
        param_stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_mean = param.grad.data.mean().item()
                grad_std = param.grad.data.std().item()
                
                param_stats[name] = {
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std,
                    'min': param.grad.data.min().item(),
                    'max': param.grad.data.max().item()
                }
                
                self.parameter_stats[name].append(grad_norm)
        
        # Record overall statistics
        gradient_info = {
            'timestamp': datetime.now().isoformat(),
            'total_norm': total_norm,
            'parameter_count': len(param_stats),
            'param_stats': param_stats
        }
        
        self.gradient_history.append(gradient_info)
    
    def get_gradient_stats(self) -> Dict[str, Any]:
        """Get gradient statistics."""
        if not self.gradient_history:
            return {}
        
        # Overall statistics
        total_norms = [g['total_norm'] for g in self.gradient_history]
        
        stats = {
            'total_norm_mean': np.mean(total_norms),
            'total_norm_max': np.max(total_norms),
            'total_norm_min': np.min(total_norms),
            'total_norm_std': np.std(total_norms),
            'gradient_updates': len(self.gradient_history)
        }
        
        # Per-parameter statistics
        for param_name, norms in self.parameter_stats.items():
            if norms:
                stats[f'{param_name}_norm_mean'] = np.mean(norms)
                stats[f'{param_name}_norm_max'] = np.max(norms)
                stats[f'{param_name}_norm_min'] = np.min(norms)
                stats[f'{param_name}_norm_std'] = np.std(norms)
        
        return stats
    
    def plot_gradient_distribution(self, save_path: str = None):
        """Plot gradient distribution."""
        if not self.gradient_history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total gradient norm over time
        total_norms = [g['total_norm'] for g in self.gradient_history]
        axes[0, 0].plot(total_norms)
        axes[0, 0].set_title('Total Gradient Norm Over Time')
        axes[0, 0].set_ylabel('Gradient Norm')
        
        # Gradient norm distribution
        axes[0, 1].hist(total_norms, bins=50, alpha=0.7)
        axes[0, 1].set_title('Gradient Norm Distribution')
        axes[0, 1].set_xlabel('Gradient Norm')
        axes[0, 1].set_ylabel('Frequency')
        
        # Per-parameter gradient norms (top 10)
        param_norms = {}
        for g in self.gradient_history[-10:]:  # Last 10 updates
            for param_name, param_stats in g['param_stats'].items():
                if param_name not in param_norms:
                    param_norms[param_name] = []
                param_norms[param_name].append(param_stats['norm'])
        
        # Plot top parameters
        top_params = sorted(param_norms.items(), key=lambda x: np.mean(x[1]), reverse=True)[:10]
        param_names = [p[0] for p in top_params]
        param_means = [np.mean(p[1]) for p in top_params]
        
        axes[1, 0].barh(range(len(param_names)), param_means)
        axes[1, 0].set_yticks(range(len(param_names)))
        axes[1, 0].set_yticklabels([name.split('.')[-1] for name in param_names])
        axes[1, 0].set_title('Top 10 Parameters by Gradient Norm')
        axes[1, 0].set_xlabel('Average Gradient Norm')
        
        # Gradient norm vs parameter count
        param_counts = [g['parameter_count'] for g in self.gradient_history]
        axes[1, 1].scatter(param_counts, total_norms, alpha=0.6)
        axes[1, 1].set_xlabel('Parameter Count')
        axes[1, 1].set_ylabel('Total Gradient Norm')
        axes[1, 1].set_title('Gradient Norm vs Parameter Count')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class PerformanceMonitor:
    """Monitor performance during gradient accumulation."""
    
    def __init__(self) -> Any:
        self.performance_history = deque(maxlen=1000)
        self.start_time = time.time()
    
    def record_step(self, batch_size: int, step_time: float, 
                   accumulation_step: int, total_steps: int):
        """Record performance metrics for a step."""
        performance_info = {
            'timestamp': datetime.now().isoformat(),
            'batch_size': batch_size,
            'step_time': step_time,
            'accumulation_step': accumulation_step,
            'total_steps': total_steps,
            'effective_batch_size': batch_size * total_steps,
            'throughput': batch_size / step_time if step_time > 0 else 0,
            'gpu_utilization': self._get_gpu_utilization()
        }
        
        self.performance_history.append(performance_info)
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.utilization()
            return 0.0
        except:
            return 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.performance_history:
            return {}
        
        step_times = [p['step_time'] for p in self.performance_history]
        throughputs = [p['throughput'] for p in self.performance_history]
        effective_batch_sizes = [p['effective_batch_size'] for p in self.performance_history]
        gpu_utils = [p['gpu_utilization'] for p in self.performance_history]
        
        return {
            'total_steps': len(self.performance_history),
            'total_time': time.time() - self.start_time,
            'avg_step_time': np.mean(step_times),
            'max_step_time': np.max(step_times),
            'min_step_time': np.min(step_times),
            'avg_throughput': np.mean(throughputs),
            'max_throughput': np.max(throughputs),
            'avg_effective_batch_size': np.mean(effective_batch_sizes),
            'avg_gpu_utilization': np.mean(gpu_utils),
            'total_samples_processed': sum(p['batch_size'] for p in self.performance_history)
        }
    
    def plot_performance(self, save_path: str = None):
        """Plot performance metrics."""
        if not self.performance_history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Step time over time
        step_times = [p['step_time'] for p in self.performance_history]
        axes[0, 0].plot(step_times)
        axes[0, 0].set_title('Step Time Over Time')
        axes[0, 0].set_ylabel('Time (seconds)')
        
        # Throughput over time
        throughputs = [p['throughput'] for p in self.performance_history]
        axes[0, 1].plot(throughputs)
        axes[0, 1].set_title('Throughput Over Time')
        axes[0, 1].set_ylabel('Samples/Second')
        
        # Effective batch size over time
        effective_batch_sizes = [p['effective_batch_size'] for p in self.performance_history]
        axes[1, 0].plot(effective_batch_sizes)
        axes[1, 0].set_title('Effective Batch Size Over Time')
        axes[1, 0].set_ylabel('Batch Size')
        
        # GPU utilization over time
        gpu_utils = [p['gpu_utilization'] for p in self.performance_history]
        axes[1, 1].plot(gpu_utils)
        axes[1, 1].set_title('GPU Utilization Over Time')
        axes[1, 1].set_ylabel('Utilization %')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class AdvancedGradientAccumulator:
    """Advanced gradient accumulator with comprehensive features."""
    
    def __init__(self, config: GradientAccumulationConfig):
        
    """__init__ function."""
self.config = config
        self.accumulator = GradientAccumulator(config)
        self.memory_monitor = MemoryMonitor() if config.enable_memory_tracking else None
        self.gradient_tracker = GradientTracker() if config.enable_gradient_tracking else None
        self.performance_monitor = PerformanceMonitor() if config.enable_performance_tracking else None
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.current_epoch = 0
        self.training_history = []
    
    def setup_training(self, model: nn.Module, optimizer: optim.Optimizer, 
                      scheduler: optim.lr_scheduler._LRScheduler = None):
        """Setup training components."""
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        logger.info(f"Gradient accumulation setup: {self.config.accumulation_steps} steps")
    
    def training_step(self, data_batch: Any, loss_fn: Callable, 
                     batch_size: int = None) -> Dict[str, Any]:
        """Perform training step with gradient accumulation."""
        start_time = time.time()
        
        # Forward pass with mixed precision
        with self.accumulator.autocast():
            output = self.model(data_batch)
            loss = loss_fn(output, data_batch)
        
        # Perform gradient accumulation step
        step_result = self.accumulator.step(
            self.model, self.optimizer, loss, batch_size
        )
        
        # Update scheduler if available
        if self.scheduler and step_result['status'] == 'updated':
            self.scheduler.step()
        
        # Record training history
        training_info = {
            'epoch': self.current_epoch,
            'step': self.accumulator.current_step,
            'loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'status': step_result['status'],
            'effective_batch_size': step_result.get('effective_batch_size', 0),
            'gradient_norm': step_result.get('gradient_norm', 0.0),
            'step_time': time.time() - start_time
        }
        
        self.training_history.append(training_info)
        
        return step_result
    
    def set_epoch(self, epoch: int):
        """Set current epoch."""
        self.current_epoch = epoch
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        summary = {
            'accumulation_stats': self.accumulator.get_accumulation_stats(),
            'training_history': self.training_history[-100:] if self.training_history else [],
            'current_epoch': self.current_epoch,
            'total_steps': self.accumulator.current_step
        }
        
        # Add monitoring statistics
        if self.memory_monitor:
            summary['memory_stats'] = self.memory_monitor.get_memory_stats()
        
        if self.gradient_tracker:
            summary['gradient_stats'] = self.gradient_tracker.get_gradient_stats()
        
        if self.performance_monitor:
            summary['performance_stats'] = self.performance_monitor.get_performance_stats()
        
        return summary
    
    def plot_training_metrics(self, save_path: str = None):
        """Plot comprehensive training metrics."""
        if not self.training_history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss over time
        losses = [t['loss'] for t in self.training_history]
        steps = range(len(losses))
        axes[0, 0].plot(steps, losses)
        axes[0, 0].set_title('Training Loss Over Time')
        axes[0, 0].set_ylabel('Loss')
        
        # Learning rate over time
        lrs = [t['learning_rate'] for t in self.training_history]
        axes[0, 1].plot(steps, lrs)
        axes[0, 1].set_title('Learning Rate Over Time')
        axes[0, 1].set_ylabel('Learning Rate')
        
        # Effective batch size over time
        batch_sizes = [t['effective_batch_size'] for t in self.training_history]
        axes[1, 0].plot(steps, batch_sizes)
        axes[1, 0].set_title('Effective Batch Size Over Time')
        axes[1, 0].set_ylabel('Batch Size')
        
        # Gradient norm over time
        grad_norms = [t['gradient_norm'] for t in self.training_history]
        axes[1, 1].plot(steps, grad_norms)
        axes[1, 1].set_title('Gradient Norm Over Time')
        axes[1, 1].set_ylabel('Gradient Norm')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


# Utility functions
def create_gradient_accumulator(config: GradientAccumulationConfig = None) -> AdvancedGradientAccumulator:
    """Create gradient accumulator with default configuration."""
    if config is None:
        config = GradientAccumulationConfig()
    
    return AdvancedGradientAccumulator(config)


def calculate_optimal_accumulation_steps(target_batch_size: int, 
                                       current_batch_size: int,
                                       max_memory_usage: float = 0.8) -> int:
    """Calculate optimal accumulation steps based on target batch size and memory constraints."""
    if current_batch_size >= target_batch_size:
        return 1
    
    # Calculate required accumulation steps
    required_steps = target_batch_size // current_batch_size
    
    # Adjust based on memory constraints
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        current_memory = torch.cuda.memory_allocated(0)
        memory_ratio = current_memory / total_memory
        
        # If memory usage is high, increase accumulation steps
        if memory_ratio > max_memory_usage:
            required_steps = int(required_steps * (memory_ratio / max_memory_usage))
    
    return max(1, required_steps)


def optimize_batch_size_for_memory(model: nn.Module, 
                                 target_batch_size: int,
                                 max_memory_usage: float = 0.8) -> Tuple[int, int]:
    """Optimize batch size and accumulation steps for memory constraints."""
    if not torch.cuda.is_available():
        return target_batch_size, 1
    
    # Start with small batch size
    current_batch_size = 1
    accumulation_steps = target_batch_size
    
    # Gradually increase batch size until memory limit
    while current_batch_size < target_batch_size:
        # Clear memory
        torch.cuda.empty_cache()
        
        # Try larger batch size
        test_batch_size = min(current_batch_size * 2, target_batch_size)
        
        try:
            # Create dummy input
            dummy_input = torch.randn(test_batch_size, *model.input_shape if hasattr(model, 'input_shape') else (100,))
            dummy_input = dummy_input.cuda()
            
            # Forward pass
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Check memory usage
            memory_usage = torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory
            
            if memory_usage <= max_memory_usage:
                current_batch_size = test_batch_size
                accumulation_steps = target_batch_size // current_batch_size
            else:
                break
                
        except RuntimeError:
            # Out of memory, break
            break
    
    return current_batch_size, accumulation_steps


# Example usage
if __name__ == "__main__":
    # Create gradient accumulation configuration
    config = GradientAccumulationConfig(
        accumulation_steps=4,
        target_batch_size=512,
        enable_dynamic_accumulation=True,
        enable_mixed_precision=True,
        enable_gradient_clipping=True,
        enable_monitoring=True
    )
    
    # Create gradient accumulator
    accumulator = create_gradient_accumulator(config)
    
    # Example model and optimizer
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    # Setup training
    accumulator.setup_training(model, optimizer, scheduler)
    
    # Example training loop
    for epoch in range(10):
        accumulator.set_epoch(epoch)
        
        for batch_idx in range(100):
            # Create dummy data
            input_data = torch.randn(32, 100)
            target = torch.randn(32, 10)
            
            def loss_fn(output, data) -> Any:
                return nn.MSELoss()(output, target)
            
            # Training step
            step_result = accumulator.training_step(input_data, loss_fn, batch_size=32)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: {step_result}")
        
        # Get training summary
        summary = accumulator.get_training_summary()
        print(f"Epoch {epoch} summary: {summary}")
    
    # Plot training metrics
    accumulator.plot_training_metrics() 