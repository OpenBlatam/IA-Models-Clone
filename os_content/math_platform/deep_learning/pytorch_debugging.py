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
import torch.autograd as autograd
import torch.profiler as profiler
import torch.utils.tensorboard as tensorboard
import numpy as np
import logging
import time
import os
import sys
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, ContextManager
from dataclasses import dataclass, field
from datetime import datetime
import warnings
from abc import ABC, abstractmethod
import functools
import traceback
import gc
from contextlib import contextmanager
import threading
import queue
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
        import json
            import psutil
        import json
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
PyTorch Debugging Tools
Comprehensive debugging utilities using PyTorch's built-in debugging tools including autograd.detect_anomaly().
"""


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class PyTorchDebugConfig:
    """Configuration for PyTorch debugging tools."""
    # Autograd debugging
    enable_autograd_anomaly_detection: bool = True
    enable_autograd_profiler: bool = False
    enable_grad_check_nan_inf: bool = True
    
    # Memory debugging
    enable_memory_profiling: bool = True
    enable_memory_leak_detection: bool = True
    enable_gpu_memory_tracking: bool = True
    
    # Performance debugging
    enable_performance_profiling: bool = True
    enable_tensorboard_logging: bool = True
    enable_flops_counting: bool = True
    
    # Model debugging
    enable_model_parameter_tracking: bool = True
    enable_gradient_norm_tracking: bool = True
    enable_weight_distribution_tracking: bool = True
    
    # Debugging parameters
    debug_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    save_debug_info: bool = True
    debug_output_dir: str = "./debug_outputs"
    max_debug_history: int = 1000
    
    # Profiling parameters
    profile_memory: bool = True
    profile_cpu: bool = True
    profile_cuda: bool = True
    profile_with_stack: bool = True
    profile_record_shapes: bool = True
    profile_with_flops: bool = True
    
    # TensorBoard parameters
    tensorboard_log_dir: str = "./tensorboard_logs"
    tensorboard_update_freq: int = 100
    
    # Anomaly detection parameters
    anomaly_detection_mode: str = "default"  # default, nan, inf, both
    anomaly_detection_verbose: bool = True
    anomaly_detection_check_nan: bool = True
    anomaly_detection_check_inf: bool = True


class AutogradDebugger:
    """Debug autograd operations using PyTorch's built-in tools."""
    
    def __init__(self, config: PyTorchDebugConfig):
        
    """__init__ function."""
self.config = config
        self.anomaly_detected = False
        self.anomaly_history = deque(maxlen=config.max_debug_history)
        self.gradient_history = deque(maxlen=config.max_debug_history)
        
        # Setup output directory
        os.makedirs(config.debug_output_dir, exist_ok=True)
    
    @contextmanager
    def detect_anomaly(self, mode: str = None, verbose: bool = None):
        """Context manager for autograd anomaly detection."""
        if mode is None:
            mode = self.config.anomaly_detection_mode
        if verbose is None:
            verbose = self.config.anomaly_detection_verbose
        
        try:
            # Enable anomaly detection
            with autograd.detect_anomaly(mode=mode, check_nan=self.config.anomaly_detection_check_nan, 
                                       check_inf=self.config.anomaly_detection_check_inf):
                yield
        except Exception as e:
            self.anomaly_detected = True
            self._log_anomaly(e, mode)
            raise
    
    def _log_anomaly(self, error: Exception, mode: str):
        """Log detected anomaly."""
        anomaly_info = {
            'timestamp': datetime.now().isoformat(),
            'mode': mode,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc()
        }
        
        self.anomaly_history.append(anomaly_info)
        
        logger.error(f"Autograd anomaly detected in mode '{mode}': {error}")
        
        if self.config.save_debug_info:
            self._save_anomaly_info(anomaly_info)
    
    def _save_anomaly_info(self, anomaly_info: Dict[str, Any]):
        """Save anomaly information to file."""
        filename = f"anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.config.debug_output_dir, filename)
        
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(anomaly_info, f, indent=2)
    
    @contextmanager
    def grad_check_nan_inf(self) -> Any:
        """Context manager for gradient NaN/Inf checking."""
        if not self.config.enable_grad_check_nan_inf:
            yield
            return
        
        try:
            yield
        except Exception as e:
            logger.error(f"Gradient check failed: {e}")
            raise
    
    def check_gradients(self, model: nn.Module, loss: torch.Tensor) -> Dict[str, Any]:
        """Check gradients for anomalies."""
        gradient_info = {}
        
        # Check for NaN/Inf in loss
        if torch.isnan(loss).any():
            gradient_info['loss_nan'] = True
            logger.warning("NaN detected in loss")
        
        if torch.isinf(loss).any():
            gradient_info['loss_inf'] = True
            logger.warning("Inf detected in loss")
        
        # Check gradients for each parameter
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                
                if torch.isnan(grad).any():
                    gradient_info[f'{name}_grad_nan'] = True
                    logger.warning(f"NaN detected in gradients of {name}")
                
                if torch.isinf(grad).any():
                    gradient_info[f'{name}_grad_inf'] = True
                    logger.warning(f"Inf detected in gradients of {name}")
                
                # Calculate gradient norm
                grad_norm = torch.norm(grad).item()
                gradient_info[f'{name}_grad_norm'] = grad_norm
        
        self.gradient_history.append(gradient_info)
        return gradient_info
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies."""
        return {
            'total_anomalies': len(self.anomaly_history),
            'anomaly_detected': self.anomaly_detected,
            'recent_anomalies': list(self.anomaly_history)[-10:],
            'gradient_checks': len(self.gradient_history)
        }


class MemoryDebugger:
    """Debug memory usage and detect memory leaks."""
    
    def __init__(self, config: PyTorchDebugConfig):
        
    """__init__ function."""
self.config = config
        self.memory_history = deque(maxlen=config.max_debug_history)
        self.peak_memory = 0
        self.memory_leaks = []
    
    @contextmanager
    def track_memory(self, context_name: str = "default"):
        """Context manager for memory tracking."""
        if not self.config.enable_memory_profiling:
            yield
            return
        
        # Record initial memory state
        initial_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            # Record final memory state
            final_memory = self._get_memory_usage()
            
            # Calculate memory difference
            memory_diff = final_memory - initial_memory
            
            memory_info = {
                'context': context_name,
                'timestamp': datetime.now().isoformat(),
                'initial_memory': initial_memory,
                'final_memory': final_memory,
                'memory_diff': memory_diff,
                'peak_memory': max(self.peak_memory, final_memory)
            }
            
            self.memory_history.append(memory_info)
            self.peak_memory = max(self.peak_memory, final_memory)
            
            # Check for potential memory leak
            if memory_diff > 100 * 1024 * 1024:  # 100MB threshold
                self.memory_leaks.append(memory_info)
                logger.warning(f"Potential memory leak detected in {context_name}: {memory_diff / (1024*1024):.2f}MB")
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = {}
        
        # CPU memory
        if self.config.profile_cpu:
            process = psutil.Process()
            memory_info['cpu_memory_mb'] = process.memory_info().rss / (1024 * 1024)
        
        # GPU memory
        if self.config.profile_cuda and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_info[f'gpu_{i}_allocated_mb'] = torch.cuda.memory_allocated(i) / (1024 * 1024)
                memory_info[f'gpu_{i}_reserved_mb'] = torch.cuda.memory_reserved(i) / (1024 * 1024)
                memory_info[f'gpu_{i}_max_allocated_mb'] = torch.cuda.max_memory_allocated(i) / (1024 * 1024)
        
        return memory_info
    
    def clear_memory(self) -> Any:
        """Clear memory and reset tracking."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        
        logger.info("Memory cleared")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        return {
            'peak_memory': self.peak_memory,
            'total_memory_checks': len(self.memory_history),
            'potential_memory_leaks': len(self.memory_leaks),
            'recent_memory_usage': list(self.memory_history)[-10:],
            'memory_leaks': self.memory_leaks
        }


class PerformanceProfiler:
    """Profile performance using PyTorch's profiler."""
    
    def __init__(self, config: PyTorchDebugConfig):
        
    """__init__ function."""
self.config = config
        self.profiler_history = deque(maxlen=config.max_debug_history)
        self.current_profiler = None
    
    @contextmanager
    def profile(self, name: str = "default", record_shapes: bool = None, 
                profile_memory: bool = None, with_stack: bool = None):
        """Context manager for performance profiling."""
        if not self.config.enable_performance_profiling:
            yield
            return
        
        if record_shapes is None:
            record_shapes = self.config.profile_record_shapes
        if profile_memory is None:
            profile_memory = self.config.profile_memory
        if with_stack is None:
            with_stack = self.config.profile_with_stack
        
        try:
            with profiler.profile(
                activities=[
                    profiler.ProfilerActivity.CPU,
                    profiler.ProfilerActivity.CUDA,
                ] if torch.cuda.is_available() else [profiler.ProfilerActivity.CPU],
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=with_stack,
                with_flops=self.config.profile_with_flops
            ) as prof:
                self.current_profiler = prof
                yield prof
        finally:
            self.current_profiler = None
    
    def save_profile(self, prof: profiler.profile, filename: str = None):
        """Save profiling results."""
        if filename is None:
            filename = f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save to file
        prof.export_chrome_trace(f"{filename}.json")
        
        # Save summary
        summary = prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total")
        
        with open(f"{filename}_summary.txt", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(summary)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        logger.info(f"Profiling results saved to {filename}")
    
    def get_profile_summary(self, prof: profiler.profile) -> Dict[str, Any]:
        """Get profiling summary."""
        if prof is None:
            return {}
        
        summary = {}
        
        # Get top operations by time
        top_ops = prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=10)
        summary['top_operations'] = str(top_ops)
        
        # Get memory usage
        if self.config.profile_memory:
            memory_stats = prof.key_averages().table(sort_by="self_cuda_memory_usage" if torch.cuda.is_available() else "self_cpu_memory_usage", row_limit=10)
            summary['memory_usage'] = str(memory_stats)
        
        return summary


class ModelDebugger:
    """Debug model parameters, gradients, and weights."""
    
    def __init__(self, config: PyTorchDebugConfig):
        
    """__init__ function."""
self.config = config
        self.parameter_history = deque(maxlen=config.max_debug_history)
        self.weight_distribution_history = deque(maxlen=config.max_debug_history)
    
    def track_parameters(self, model: nn.Module, step: int = 0) -> Dict[str, Any]:
        """Track model parameters."""
        if not self.config.enable_model_parameter_tracking:
            return {}
        
        parameter_info = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'parameters': {}
        }
        
        for name, param in model.named_parameters():
            param_data = {
                'shape': list(param.shape),
                'dtype': str(param.dtype),
                'device': str(param.device),
                'requires_grad': param.requires_grad,
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item(),
                'norm': torch.norm(param.data).item()
            }
            
            if param.grad is not None:
                param_data['grad_mean'] = param.grad.mean().item()
                param_data['grad_std'] = param.grad.std().item()
                param_data['grad_norm'] = torch.norm(param.grad).item()
            
            parameter_info['parameters'][name] = param_data
        
        self.parameter_history.append(parameter_info)
        return parameter_info
    
    def track_weight_distributions(self, model: nn.Module, step: int = 0):
        """Track weight distributions."""
        if not self.config.enable_weight_distribution_tracking:
            return
        
        distribution_info = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'distributions': {}
        }
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                weights = param.data.cpu().numpy().flatten()
                
                distribution_info['distributions'][name] = {
                    'mean': float(np.mean(weights)),
                    'std': float(np.std(weights)),
                    'skewness': float(self._calculate_skewness(weights)),
                    'kurtosis': float(self._calculate_kurtosis(weights)),
                    'percentiles': {
                        '1%': float(np.percentile(weights, 1)),
                        '5%': float(np.percentile(weights, 5)),
                        '25%': float(np.percentile(weights, 25)),
                        '50%': float(np.percentile(weights, 50)),
                        '75%': float(np.percentile(weights, 75)),
                        '95%': float(np.percentile(weights, 95)),
                        '99%': float(np.percentile(weights, 99))
                    }
                }
        
        self.weight_distribution_history.append(distribution_info)
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def check_gradient_norms(self, model: nn.Module) -> Dict[str, float]:
        """Check gradient norms for all parameters."""
        if not self.config.enable_gradient_norm_tracking:
            return {}
        
        gradient_norms = {}
        total_norm = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = torch.norm(param.grad).item()
                gradient_norms[name] = norm
                total_norm += norm ** 2
        
        gradient_norms['total_norm'] = total_norm ** 0.5
        return gradient_norms
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model debugging summary."""
        return {
            'parameter_checks': len(self.parameter_history),
            'distribution_checks': len(self.weight_distribution_history),
            'recent_parameters': list(self.parameter_history)[-5:],
            'recent_distributions': list(self.weight_distribution_history)[-5:]
        }


class TensorBoardLogger:
    """Log debugging information to TensorBoard."""
    
    def __init__(self, config: PyTorchDebugConfig):
        
    """__init__ function."""
self.config = config
        self.writer = None
        self.step = 0
        
        if self.config.enable_tensorboard_logging:
            os.makedirs(self.config.tensorboard_log_dir, exist_ok=True)
            self.writer = tensorboard.SummaryWriter(self.config.tensorboard_log_dir)
    
    def log_scalar(self, tag: str, value: float, step: int = None):
        """Log scalar value to TensorBoard."""
        if self.writer is None:
            return
        
        if step is None:
            step = self.step
        
        self.writer.add_scalar(tag, value, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int = None):
        """Log histogram to TensorBoard."""
        if self.writer is None:
            return
        
        if step is None:
            step = self.step
        
        self.writer.add_histogram(tag, values, step)
    
    def log_model_graph(self, model: nn.Module, input_tensor: torch.Tensor):
        """Log model graph to TensorBoard."""
        if self.writer is None:
            return
        
        self.writer.add_graph(model, input_tensor)
    
    def log_parameter_distributions(self, model: nn.Module, step: int = None):
        """Log parameter distributions to TensorBoard."""
        if self.writer is None:
            return
        
        if step is None:
            step = self.step
        
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"parameters/{name}", param.data, step)
            if param.grad is not None:
                self.writer.add_histogram(f"gradients/{name}", param.grad, step)
    
    def log_gradient_norms(self, gradient_norms: Dict[str, float], step: int = None):
        """Log gradient norms to TensorBoard."""
        if self.writer is None:
            return
        
        if step is None:
            step = self.step
        
        for name, norm in gradient_norms.items():
            self.log_scalar(f"gradient_norms/{name}", norm, step)
    
    def log_memory_usage(self, memory_info: Dict[str, float], step: int = None):
        """Log memory usage to TensorBoard."""
        if self.writer is None:
            return
        
        if step is None:
            step = self.step
        
        for key, value in memory_info.items():
            self.log_scalar(f"memory/{key}", value, step)
    
    def increment_step(self) -> Any:
        """Increment the global step counter."""
        self.step += 1
    
    def close(self) -> Any:
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()


class PyTorchDebugManager:
    """Main manager for PyTorch debugging tools."""
    
    def __init__(self, config: PyTorchDebugConfig):
        
    """__init__ function."""
self.config = config
        self.autograd_debugger = AutogradDebugger(config)
        self.memory_debugger = MemoryDebugger(config)
        self.performance_profiler = PerformanceProfiler(config)
        self.model_debugger = ModelDebugger(config)
        self.tensorboard_logger = TensorBoardLogger(config)
        
        # Setup output directory
        os.makedirs(config.debug_output_dir, exist_ok=True)
    
    @contextmanager
    def debug_training_step(self, model: nn.Module, context_name: str = "training_step"):
        """Context manager for debugging a training step."""
        with self.autograd_debugger.detect_anomaly():
            with self.memory_debugger.track_memory(context_name):
                with self.performance_profiler.profile(context_name) as prof:
                    yield prof
    
    def debug_model_forward(self, model: nn.Module, input_data: torch.Tensor, 
                           context_name: str = "model_forward"):
        """Debug model forward pass."""
        with self.autograd_debugger.detect_anomaly():
            with self.memory_debugger.track_memory(context_name):
                with self.performance_profiler.profile(context_name) as prof:
                    output = model(input_data)
                    
                    # Track model parameters
                    self.model_debugger.track_parameters(model)
                    self.model_debugger.track_weight_distributions(model)
                    
                    return output, prof
    
    def debug_backward_pass(self, loss: torch.Tensor, model: nn.Module):
        """Debug backward pass."""
        with self.autograd_debugger.detect_anomaly():
            with self.memory_debugger.track_memory("backward_pass"):
                loss.backward()
                
                # Check gradients
                gradient_info = self.autograd_debugger.check_gradients(model, loss)
                gradient_norms = self.model_debugger.check_gradient_norms(model)
                
                return gradient_info, gradient_norms
    
    def log_debug_info(self, step: int, model: nn.Module, loss: torch.Tensor = None, 
                      gradient_info: Dict[str, Any] = None, gradient_norms: Dict[str, float] = None):
        """Log debugging information to TensorBoard."""
        # Log parameter distributions
        self.tensorboard_logger.log_parameter_distributions(model, step)
        
        # Log gradient norms
        if gradient_norms:
            self.tensorboard_logger.log_gradient_norms(gradient_norms, step)
        
        # Log loss
        if loss is not None:
            self.tensorboard_logger.log_scalar("loss", loss.item(), step)
        
        # Log memory usage
        memory_info = self.memory_debugger._get_memory_usage()
        self.tensorboard_logger.log_memory_usage(memory_info, step)
        
        # Increment step
        self.tensorboard_logger.increment_step()
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Get comprehensive debugging summary."""
        return {
            'autograd_summary': self.autograd_debugger.get_anomaly_summary(),
            'memory_summary': self.memory_debugger.get_memory_summary(),
            'model_summary': self.model_debugger.get_model_summary(),
            'config': asdict(self.config)
        }
    
    def save_debug_report(self, filename: str = None):
        """Save comprehensive debug report."""
        if filename is None:
            filename = f"debug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.config.debug_output_dir, filename)
        
        report = self.get_debug_summary()
        
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(report, f, indent=2)
        
        logger.info(f"Debug report saved to {filepath}")
    
    def cleanup(self) -> Any:
        """Cleanup debugging resources."""
        self.tensorboard_logger.close()
        self.memory_debugger.clear_memory()


# Utility functions for PyTorch debugging
def debug_function(func: Callable) -> Callable:
    """Decorator to add debugging to a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Create debug manager
        config = PyTorchDebugConfig()
        debug_manager = PyTorchDebugManager(config)
        
        try:
            with debug_manager.debug_training_step(None, func.__name__):
                result = func(*args, **kwargs)
                return result
        finally:
            debug_manager.cleanup()
    
    return wrapper


@contextmanager
def debug_context(context_name: str, config: PyTorchDebugConfig = None):
    """Context manager for debugging operations."""
    if config is None:
        config = PyTorchDebugConfig()
    
    debug_manager = PyTorchDebugManager(config)
    
    try:
        with debug_manager.debug_training_step(None, context_name):
            yield debug_manager
    finally:
        debug_manager.cleanup()


def enable_autograd_anomaly_detection(mode: str = "default", check_nan: bool = True, 
                                    check_inf: bool = True, verbose: bool = True):
    """Enable autograd anomaly detection globally."""
    return autograd.detect_anomaly(mode=mode, check_nan=check_nan, check_inf=check_inf)


def check_tensor_anomalies(tensor: torch.Tensor, name: str = "tensor") -> Dict[str, bool]:
    """Check tensor for anomalies."""
    anomalies = {}
    
    if torch.isnan(tensor).any():
        anomalies['nan'] = True
        logger.warning(f"NaN detected in {name}")
    
    if torch.isinf(tensor).any():
        anomalies['inf'] = True
        logger.warning(f"Inf detected in {name}")
    
    if not torch.isfinite(tensor).all():
        anomalies['non_finite'] = True
        logger.warning(f"Non-finite values detected in {name}")
    
    return anomalies


def monitor_gradients(model: nn.Module, loss: torch.Tensor) -> Dict[str, Any]:
    """Monitor gradients for anomalies."""
    gradient_info = {}
    
    # Check loss
    loss_anomalies = check_tensor_anomalies(loss, "loss")
    gradient_info['loss_anomalies'] = loss_anomalies
    
    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_anomalies = check_tensor_anomalies(param.grad, f"grad_{name}")
            gradient_info[f'{name}_anomalies'] = grad_anomalies
            
            # Calculate gradient norm
            gradient_info[f'{name}_norm'] = torch.norm(param.grad).item()
    
    return gradient_info


# Example usage
if __name__ == "__main__":
    # Create debugging configuration
    config = PyTorchDebugConfig(
        enable_autograd_anomaly_detection=True,
        enable_memory_profiling=True,
        enable_performance_profiling=True,
        enable_tensorboard_logging=True,
        debug_level="INFO"
    )
    
    # Create debug manager
    debug_manager = PyTorchDebugManager(config)
    
    # Example: Debug a simple model
    model = nn.Linear(10, 1)
    input_data = torch.randn(5, 10)
    
    # Debug forward pass
    output, prof = debug_manager.debug_model_forward(model, input_data)
    
    # Debug backward pass
    loss = output.mean()
    gradient_info, gradient_norms = debug_manager.debug_backward_pass(loss, model)
    
    # Log debug information
    debug_manager.log_debug_info(0, model, loss, gradient_info, gradient_norms)
    
    # Save debug report
    debug_manager.save_debug_report()
    
    # Get summary
    summary = debug_manager.get_debug_summary()
    print("Debugging completed!")
    print(f"Summary: {summary}")
    
    # Cleanup
    debug_manager.cleanup() 