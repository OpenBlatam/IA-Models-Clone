"""
Error Handling and Debugging System
Implements comprehensive error handling, logging, PyTorch debugging tools, and error recovery
"""

import torch
import torch.nn as nn
import logging
import traceback
import sys
import os
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from pathlib import Path
import warnings
import time
import functools
from datetime import datetime
import json
import gc
import psutil

class ErrorHandler:
    """Comprehensive error handling with automatic recovery"""
    
    def __init__(self, log_file: str = "error_log.txt", enable_debug: bool = True):
        self.log_file = log_file
        self.enable_debug = enable_debug
        self.error_counts = {}
        self.recovery_strategies = {}
        
        # Setup logging
        self._setup_logging()
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def _setup_logging(self):
        """Setup comprehensive logging system"""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG if self.enable_debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _register_default_strategies(self):
        """Register default error recovery strategies"""
        # Memory errors
        self.register_recovery_strategy(
            "CUDA out of memory",
            self._handle_memory_error
        )
        
        # Model loading errors
        self.register_recovery_strategy(
            "model loading",
            self._handle_model_loading_error
        )
        
        # Data loading errors
        self.register_recovery_strategy(
            "data loading",
            self._handle_data_loading_error
        )
    
    def register_recovery_strategy(self, error_pattern: str, strategy: Callable):
        """Register custom error recovery strategy"""
        self.recovery_strategies[error_pattern] = strategy
        self.logger.info(f"Registered recovery strategy for: {error_pattern}")
    
    def handle_error(self, error: Exception, context: str = "", 
                    max_retries: int = 3) -> Tuple[bool, Any]:
        """Handle error with automatic recovery attempts"""
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Log error
        self.logger.error(f"Error in {context}: {error_type}: {error_msg}")
        self.logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # Update error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Try recovery strategies
        for pattern, strategy in self.recovery_strategies.items():
            if pattern.lower() in error_msg.lower():
                self.logger.info(f"Attempting recovery with strategy: {pattern}")
                
                for attempt in range(max_retries):
                    try:
                        result = strategy(error, context, attempt)
                        if result:
                            self.logger.info(f"Recovery successful on attempt {attempt + 1}")
                            return True, result
                    except Exception as recovery_error:
                        self.logger.warning(f"Recovery attempt {attempt + 1} failed: {recovery_error}")
                
                self.logger.error(f"All recovery attempts failed for: {pattern}")
        
        return False, None
    
    def _handle_memory_error(self, error: Exception, context: str, attempt: int) -> Any:
        """Handle CUDA out of memory errors"""
        if torch.cuda.is_available():
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Reduce batch size if possible
            if hasattr(self, 'current_batch_size'):
                self.current_batch_size = max(1, self.current_batch_size // 2)
                self.logger.info(f"Reduced batch size to {self.current_batch_size}")
            
            return True
        
        return False
    
    def _handle_model_loading_error(self, error: Exception, context: str, attempt: int) -> Any:
        """Handle model loading errors"""
        # Try to reload model with different precision
        if attempt == 0:
            self.logger.info("Attempting to reload model with float16")
            return "try_float16"
        elif attempt == 1:
            self.logger.info("Attempting to reload model with CPU")
            return "try_cpu"
        
        return False
    
    def _handle_data_loading_error(self, error: Exception, context: str, attempt: int) -> Any:
        """Handle data loading errors"""
        # Try to reload data with different settings
        if attempt == 0:
            self.logger.info("Attempting to reload data with reduced workers")
            return "reduce_workers"
        elif attempt == 1:
            self.logger.info("Attempting to reload data with memory pinning disabled")
            return "disable_pinning"
        
        return False

class PyTorchDebugger:
    """PyTorch-specific debugging tools and utilities"""
    
    def __init__(self, enable_anomaly_detection: bool = False):
        self.enable_anomaly_detection = enable_anomaly_detection
        self.logger = logging.getLogger(__name__)
        
        if enable_anomaly_detection:
            self._enable_anomaly_detection()
    
    def _enable_anomaly_detection(self):
        """Enable PyTorch anomaly detection"""
        try:
            torch.autograd.set_detect_anomaly(True)
            self.logger.info("PyTorch anomaly detection enabled")
        except Exception as e:
            self.logger.warning(f"Failed to enable anomaly detection: {e}")
    
    def check_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """Check model gradients for issues"""
        gradient_info = {
            "has_gradients": False,
            "gradient_norms": {},
            "nan_gradients": [],
            "inf_gradients": [],
            "zero_gradients": []
        }
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_info["has_gradients"] = True
                
                grad = param.grad
                grad_norm = grad.norm().item()
                gradient_info["gradient_norms"][name] = grad_norm
                
                # Check for NaN gradients
                if torch.isnan(grad).any():
                    gradient_info["nan_gradients"].append(name)
                
                # Check for Inf gradients
                if torch.isinf(grad).any():
                    gradient_info["inf_gradients"].append(name)
                
                # Check for zero gradients
                if grad_norm == 0:
                    gradient_info["zero_gradients"].append(name)
        
        return gradient_info
    
    def check_model_parameters(self, model: nn.Module) -> Dict[str, Any]:
        """Check model parameters for issues"""
        parameter_info = {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "nan_parameters": [],
            "inf_parameters": [],
            "zero_parameters": []
        }
        
        for name, param in model.named_parameters():
            parameter_info["total_parameters"] += param.numel()
            
            if param.requires_grad:
                parameter_info["trainable_parameters"] += param.numel()
            
            # Check for NaN parameters
            if torch.isnan(param.data).any():
                parameter_info["nan_parameters"].append(name)
            
            # Check for Inf parameters
            if torch.isinf(param.data).any():
                parameter_info["inf_parameters"].append(name)
            
            # Check for zero parameters
            if param.data.norm() == 0:
                parameter_info["zero_parameters"].append(name)
        
        return parameter_info
    
    def profile_memory(self, model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Profile model memory usage"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        # Clear cache before profiling
        torch.cuda.empty_cache()
        
        # Get initial memory state
        initial_memory = torch.cuda.memory_allocated()
        initial_reserved = torch.cuda.memory_reserved()
        
        try:
            # Forward pass
            with torch.no_grad():
                output = model(input_tensor)
            
            # Get memory after forward pass
            forward_memory = torch.cuda.memory_allocated()
            forward_reserved = torch.cuda.memory_reserved()
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Get final memory state
            final_memory = torch.cuda.memory_allocated()
            final_reserved = torch.cuda.memory_reserved()
            
            memory_info = {
                "initial_allocated": initial_memory,
                "initial_reserved": initial_reserved,
                "forward_allocated": forward_memory,
                "forward_reserved": forward_reserved,
                "final_allocated": final_memory,
                "final_reserved": final_reserved,
                "peak_allocated": torch.cuda.max_memory_allocated(),
                "peak_reserved": torch.cuda.max_memory_reserved()
            }
            
            return memory_info
            
        except Exception as e:
            self.logger.error(f"Memory profiling failed: {e}")
            return {"error": str(e)}
    
    def enable_gradient_checkpointing(self, model: nn.Module):
        """Enable gradient checkpointing for memory efficiency"""
        try:
            model.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled")
        except Exception as e:
            self.logger.warning(f"Failed to enable gradient checkpointing: {e}")
    
    def disable_gradient_checkpointing(self, model: nn.Module):
        """Disable gradient checkpointing"""
        try:
            model.gradient_checkpointing_disable()
            self.logger.info("Gradient checkpointing disabled")
        except Exception as e:
            self.logger.warning(f"Failed to disable gradient checkpointing: {e}")

class DebugDecorator:
    """Decorator for automatic debugging and error handling"""
    
    def __init__(self, error_handler: ErrorHandler, max_retries: int = 3):
        self.error_handler = error_handler
        self.max_retries = max_retries
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(self.max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    success, result = self.error_handler.handle_error(
                        e, f"{func.__name__} (attempt {attempt + 1})", 
                        max_retries=1
                    )
                    
                    if not success and attempt == self.max_retries - 1:
                        raise e
            
            return None
        
        return wrapper

class PerformanceMonitor:
    """Monitor system performance and resource usage"""
    
    def __init__(self, log_interval: int = 60):
        self.log_interval = log_interval
        self.logger = logging.getLogger(__name__)
        self.last_log_time = time.time()
        self.performance_history = []
    
    def log_system_status(self) -> Dict[str, Any]:
        """Log current system status"""
        current_time = time.time()
        
        # Only log if enough time has passed
        if current_time - self.last_log_time < self.log_interval:
            return {}
        
        self.last_log_time = current_time
        
        # Get system information
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get GPU information if available
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_reserved": torch.cuda.memory_reserved(),
                "memory_cached": torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            }
        
        system_status = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available": memory.available,
            "disk_percent": disk.percent,
            "disk_free": disk.free,
            "gpu_info": gpu_info
        }
        
        # Log system status
        self.logger.info(f"System Status: CPU: {cpu_percent}%, "
                        f"Memory: {memory.percent}%, "
                        f"Disk: {disk.percent}%")
        
        # Store in history
        self.performance_history.append(system_status)
        
        # Keep only last 1000 entries
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        return system_status
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.performance_history:
            return {}
        
        cpu_values = [entry["cpu_percent"] for entry in self.performance_history]
        memory_values = [entry["memory_percent"] for entry in self.performance_history]
        
        summary = {
            "total_entries": len(self.performance_history),
            "time_span": {
                "start": self.performance_history[0]["timestamp"],
                "end": self.performance_history[-1]["timestamp"]
            },
            "cpu_stats": {
                "mean": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values)
            },
            "memory_stats": {
                "mean": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values)
            }
        }
        
        return summary

class ErrorRecoveryManager:
    """Manage error recovery strategies and automatic fixes"""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.logger = logging.getLogger(__name__)
        self.recovery_history = []
    
    def register_automatic_fix(self, error_pattern: str, fix_function: Callable):
        """Register automatic fix for specific error patterns"""
        self.error_handler.register_recovery_strategy(error_pattern, fix_function)
        self.logger.info(f"Registered automatic fix for: {error_pattern}")
    
    def apply_common_fixes(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """Apply common fixes for training issues"""
        fixes_applied = []
        
        # Check for NaN parameters
        param_info = self._check_parameters(model)
        if param_info["nan_parameters"]:
            self.logger.warning("NaN parameters detected, attempting to fix...")
            self._fix_nan_parameters(model)
            fixes_applied.append("nan_parameter_fix")
        
        # Check for exploding gradients
        grad_info = self._check_gradients(model)
        if grad_info["inf_gradients"]:
            self.logger.warning("Inf gradients detected, attempting to fix...")
            self._fix_exploding_gradients(model, optimizer)
            fixes_applied.append("exploding_gradient_fix")
        
        # Check for zero gradients
        if grad_info["zero_gradients"]:
            self.logger.warning("Zero gradients detected, attempting to fix...")
            self._fix_zero_gradients(model, optimizer)
            fixes_applied.append("zero_gradient_fix")
        
        if fixes_applied:
            self.logger.info(f"Applied fixes: {fixes_applied}")
        
        return fixes_applied
    
    def _check_parameters(self, model: nn.Module) -> Dict[str, Any]:
        """Check model parameters for issues"""
        param_info = {
            "nan_parameters": [],
            "inf_parameters": [],
            "zero_parameters": []
        }
        
        for name, param in model.named_parameters():
            if torch.isnan(param.data).any():
                param_info["nan_parameters"].append(name)
            if torch.isinf(param.data).any():
                param_info["inf_parameters"].append(name)
            if param.data.norm() == 0:
                param_info["zero_parameters"].append(name)
        
        return param_info
    
    def _check_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """Check model gradients for issues"""
        grad_info = {
            "nan_gradients": [],
            "inf_gradients": [],
            "zero_gradients": []
        }
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    grad_info["nan_gradients"].append(name)
                if torch.isinf(param.grad).any():
                    grad_info["inf_gradients"].append(name)
                if param.grad.norm() == 0:
                    grad_info["zero_gradients"].append(name)
        
        return grad_info
    
    def _fix_nan_parameters(self, model: nn.Module):
        """Fix NaN parameters by reinitializing them"""
        for name, param in model.named_parameters():
            if torch.isnan(param.data).any():
                self.logger.warning(f"Reinitializing NaN parameter: {name}")
                
                # Reinitialize with small random values
                if len(param.data.shape) == 1:
                    param.data = torch.randn_like(param.data) * 0.01
                else:
                    nn.init.xavier_uniform_(param.data)
    
    def _fix_exploding_gradients(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """Fix exploding gradients by clipping"""
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Reduce learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
        
        self.logger.info("Applied gradient clipping and reduced learning rate")
    
    def _fix_zero_gradients(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """Fix zero gradients by adjusting learning rate"""
        # Increase learning rate slightly
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 1.1
        
        self.logger.info("Increased learning rate to address zero gradients")

def main():
    """Example usage of the error handling and debugging system"""
    
    # Initialize components
    error_handler = ErrorHandler(enable_debug=True)
    debugger = PyTorchDebugger(enable_anomaly_detection=True)
    monitor = PerformanceMonitor(log_interval=30)
    recovery_manager = ErrorRecoveryManager(error_handler)
    
    # Example model
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    
    # Example usage of debugging tools
    print("Checking model parameters...")
    param_info = debugger.check_model_parameters(model)
    print(f"Parameter info: {param_info}")
    
    # Example usage of error handling
    @DebugDecorator(error_handler, max_retries=3)
    def example_function():
        # Simulate an error
        raise RuntimeError("Simulated error for testing")
    
    try:
        example_function()
    except Exception as e:
        print(f"Function failed after all retries: {e}")
    
    # Monitor system performance
    print("Monitoring system performance...")
    status = monitor.log_system_status()
    if status:
        print(f"System status: {status}")
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    if summary:
        print(f"Performance summary: {summary}")

if __name__ == "__main__":
    main()


