from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import gc
import warnings
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
import functools
import threading
from pathlib import Path
import json
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import detect_anomaly, profiler
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
import numpy as np
import psutil
import structlog
from error_handling_debugging import ErrorHandlingDebuggingSystem, ErrorSeverity, ErrorCategory
from robust_operations import RobustOperations, OperationResult
from training_logging_system import TrainingLogger, TrainingMetrics, LogLevel
from typing import Any, List, Dict, Optional
import logging
"""
PyTorch Debugging and Optimization System

This module provides comprehensive PyTorch debugging and optimization tools:
- autograd.detect_anomaly() integration for gradient debugging
- Memory profiling and optimization
- Performance monitoring and profiling
- Model compilation and optimization
- Debugging utilities for cybersecurity ML
- Integration with existing robust operations
"""




# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class DebugMode(Enum):
    """Debug modes for PyTorch operations."""
    NONE = "none"
    ANOMALY_DETECTION = "anomaly_detection"
    PROFILING = "profiling"
    MEMORY_PROFILING = "memory_profiling"
    GRADIENT_CHECKING = "gradient_checking"
    FULL_DEBUG = "full_debug"


class OptimizationMode(Enum):
    """Optimization modes for PyTorch operations."""
    NONE = "none"
    AMP = "amp"  # Automatic Mixed Precision
    COMPILATION = "compilation"
    MEMORY_EFFICIENT = "memory_efficient"
    GRADIENT_ACCUMULATION = "gradient_accumulation"
    FULL_OPTIMIZATION = "full_optimization"


@dataclass
class DebugMetrics:
    """Debug metrics data structure."""
    mode: DebugMode
    execution_time: float
    memory_usage: float
    gpu_memory: Optional[float] = None
    gradient_norm: Optional[float] = None
    gradient_anomalies: Optional[List[str]] = None
    profiling_data: Optional[Dict[str, Any]] = None
    memory_leaks: Optional[List[str]] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


@dataclass
class OptimizationMetrics:
    """Optimization metrics data structure."""
    mode: OptimizationMode
    execution_time: float
    memory_usage: float
    gpu_memory: Optional[float] = None
    speedup_factor: Optional[float] = None
    memory_savings: Optional[float] = None
    compilation_time: Optional[float] = None
    amp_enabled: bool = False
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


class PyTorchDebugger:
    """Comprehensive PyTorch debugging system."""
    
    def __init__(self, 
                 error_system: ErrorHandlingDebuggingSystem,
                 training_logger: Optional[TrainingLogger] = None,
                 debug_mode: DebugMode = DebugMode.NONE):
        """
        Initialize PyTorch debugger.
        
        Args:
            error_system: Error handling system for integration
            training_logger: Training logger for metrics
            debug_mode: Debug mode to enable
        """
        self.error_system = error_system
        self.training_logger = training_logger
        self.debug_mode = debug_mode
        
        # Debug state
        self.anomaly_detection_enabled = False
        self.profiling_enabled = False
        self.memory_profiling_enabled = False
        self.gradient_checking_enabled = False
        
        # Metrics storage
        self.debug_metrics: List[DebugMetrics] = []
        self.optimization_metrics: List[OptimizationMetrics] = []
        
        # Profiling data
        self.profiler_data: Dict[str, Any] = {}
        
        # Memory tracking
        self.memory_snapshots: List[Dict[str, float]] = []
        self.memory_leaks: List[str] = []
        
        # Setup debug mode
        self._setup_debug_mode(debug_mode)
        
        logger.info("PyTorchDebugger initialized", 
                   debug_mode=debug_mode.value,
                   cuda_available=torch.cuda.is_available())
    
    def _setup_debug_mode(self, mode: DebugMode):
        """Setup debug mode and enable appropriate features."""
        if mode == DebugMode.ANOMALY_DETECTION:
            self.anomaly_detection_enabled = True
            logger.info("Anomaly detection enabled")
            
        elif mode == DebugMode.PROFILING:
            self.profiling_enabled = True
            logger.info("Profiling enabled")
            
        elif mode == DebugMode.MEMORY_PROFILING:
            self.memory_profiling_enabled = True
            logger.info("Memory profiling enabled")
            
        elif mode == DebugMode.GRADIENT_CHECKING:
            self.gradient_checking_enabled = True
            logger.info("Gradient checking enabled")
            
        elif mode == DebugMode.FULL_DEBUG:
            self.anomaly_detection_enabled = True
            self.profiling_enabled = True
            self.memory_profiling_enabled = True
            self.gradient_checking_enabled = True
            logger.info("Full debug mode enabled")
    
    @contextmanager
    def debug_context(self, operation_name: str, **kwargs):
        """Context manager for debugging operations."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Enable anomaly detection if requested
        anomaly_context = detect_anomaly() if self.anomaly_detection_enabled else nullcontext()
        
        # Enable profiling if requested
        if self.profiling_enabled:
            profiler_context = profiler.profile(
                use_cuda=torch.cuda.is_available(),
                record_shapes=True,
                with_stack=True
            )
        else:
            profiler_context = nullcontext()
        
        try:
            with anomaly_context, profiler_context:
                yield
                
        except Exception as e:
            # Log error with debug context
            self.error_system.error_tracker.track_error(
                error=e,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.MODEL,
                context={
                    "operation": operation_name,
                    "debug_mode": self.debug_mode.value,
                    "anomaly_detection": self.anomaly_detection_enabled,
                    "profiling": self.profiling_enabled,
                    **kwargs
                }
            )
            
            if self.training_logger:
                self.training_logger.log_error(
                    e,
                    context={
                        "operation": operation_name,
                        "debug_mode": self.debug_mode.value,
                        **kwargs
                    }
                )
            raise
            
        finally:
            # Record debug metrics
            execution_time = time.time() - start_time
            end_memory = self._get_memory_usage()
            memory_usage = end_memory - start_memory
            
            debug_metrics = DebugMetrics(
                mode=self.debug_mode,
                execution_time=execution_time,
                memory_usage=memory_usage,
                gpu_memory=self._get_gpu_memory() if torch.cuda.is_available() else None,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Add profiling data if available
            if self.profiling_enabled and hasattr(profiler_context, 'key_averages'):
                debug_metrics.profiling_data = {
                    "cpu_time": profiler_context.key_averages().table(sort_by="cpu_time_total"),
                    "cuda_time": profiler_context.key_averages().table(sort_by="cuda_time_total") if torch.cuda.is_available() else None
                }
            
            self.debug_metrics.append(debug_metrics)
            
            # Log debug metrics
            if self.training_logger:
                self.training_logger.log_training_event(
                    "debug_metrics",
                    f"Debug metrics for {operation_name}",
                    level=LogLevel.DEBUG,
                    debug_metrics=debug_metrics.to_dict()
                )
    
    def check_gradients(self, model: nn.Module, loss: torch.Tensor) -> Dict[str, Any]:
        """Check gradients for anomalies and issues."""
        gradient_info = {
            "gradient_norm": None,
            "gradient_anomalies": [],
            "parameter_stats": {},
            "loss_value": loss.item()
        }
        
        try:
            # Calculate gradient norm
            total_norm = 0.0
            param_count = 0
            
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
                    
                    # Check for gradient anomalies
                    if torch.isnan(p.grad).any():
                        gradient_info["gradient_anomalies"].append(f"NaN gradients in parameter {p.shape}")
                    
                    if torch.isinf(p.grad).any():
                        gradient_info["gradient_anomalies"].append(f"Inf gradients in parameter {p.shape}")
                    
                    # Record parameter statistics
                    gradient_info["parameter_stats"][f"param_{param_count}"] = {
                        "shape": list(p.shape),
                        "grad_norm": param_norm.item(),
                        "grad_mean": p.grad.mean().item(),
                        "grad_std": p.grad.std().item()
                    }
            
            if param_count > 0:
                gradient_info["gradient_norm"] = total_norm ** 0.5
                
                # Check for gradient explosion/vanishing
                if gradient_info["gradient_norm"] > 10.0:
                    gradient_info["gradient_anomalies"].append("Gradient explosion detected")
                elif gradient_info["gradient_norm"] < 1e-6:
                    gradient_info["gradient_anomalies"].append("Gradient vanishing detected")
            
        except Exception as e:
            gradient_info["gradient_anomalies"].append(f"Gradient checking error: {str(e)}")
        
        return gradient_info
    
    def profile_memory(self, operation_name: str) -> Dict[str, Any]:
        """Profile memory usage for an operation."""
        memory_info = {
            "operation": operation_name,
            "cpu_memory": psutil.virtual_memory().percent,
            "gpu_memory": None,
            "memory_leaks": [],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if torch.cuda.is_available():
            memory_info["gpu_memory"] = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            memory_info["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        
        # Check for memory leaks
        if len(self.memory_snapshots) > 0:
            last_snapshot = self.memory_snapshots[-1]
            memory_increase = memory_info["cpu_memory"] - last_snapshot.get("cpu_memory", 0)
            
            if memory_increase > 10:  # 10% increase
                memory_info["memory_leaks"].append(f"Significant memory increase: {memory_increase:.2f}%")
        
        # Store snapshot
        self.memory_snapshots.append(memory_info.copy())
        
        return memory_info
    
    def optimize_model(self, 
                      model: nn.Module, 
                      optimization_mode: OptimizationMode = OptimizationMode.NONE) -> nn.Module:
        """Apply optimizations to the model."""
        start_time = time.time()
        original_model = model
        
        try:
            if optimization_mode == OptimizationMode.AMP:
                # Enable Automatic Mixed Precision
                model = self._enable_amp(model)
                
            elif optimization_mode == OptimizationMode.COMPILATION:
                # Compile model for optimization
                model = self._compile_model(model)
                
            elif optimization_mode == OptimizationMode.MEMORY_EFFICIENT:
                # Apply memory-efficient optimizations
                model = self._optimize_memory(model)
                
            elif optimization_mode == OptimizationMode.FULL_OPTIMIZATION:
                # Apply all optimizations
                model = self._enable_amp(model)
                model = self._compile_model(model)
                model = self._optimize_memory(model)
            
            optimization_time = time.time() - start_time
            
            # Record optimization metrics
            optimization_metrics = OptimizationMetrics(
                mode=optimization_mode,
                execution_time=optimization_time,
                memory_usage=self._get_memory_usage(),
                gpu_memory=self._get_gpu_memory() if torch.cuda.is_available() else None,
                compilation_time=optimization_time if optimization_mode == OptimizationMode.COMPILATION else None,
                amp_enabled=optimization_mode in [OptimizationMode.AMP, OptimizationMode.FULL_OPTIMIZATION],
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            self.optimization_metrics.append(optimization_metrics)
            
            if self.training_logger:
                self.training_logger.log_training_event(
                    "model_optimization",
                    f"Model optimized with {optimization_mode.value}",
                    level=LogLevel.INFO,
                    optimization_metrics=optimization_metrics.to_dict()
                )
            
            return model
            
        except Exception as e:
            self.error_system.error_tracker.track_error(
                error=e,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.MODEL,
                context={
                    "operation": "model_optimization",
                    "optimization_mode": optimization_mode.value
                }
            )
            
            logger.error(f"Model optimization failed: {str(e)}")
            return original_model
    
    def _enable_amp(self, model: nn.Module) -> nn.Module:
        """Enable Automatic Mixed Precision."""
        try:
            # Create GradScaler for AMP
            scaler = amp.GradScaler()
            
            # Enable AMP for the model
            model = model.half() if torch.cuda.is_available() else model
            
            logger.info("Automatic Mixed Precision enabled")
            return model
            
        except Exception as e:
            logger.warning(f"AMP enablement failed: {str(e)}")
            return model
    
    def _compile_model(self, model: nn.Module) -> nn.Module:
        """Compile model for optimization."""
        try:
            if hasattr(torch, 'compile'):
                # Use torch.compile for optimization
                compiled_model = torch.compile(model)
                logger.info("Model compiled successfully")
                return compiled_model
            else:
                logger.warning("torch.compile not available, skipping compilation")
                return model
                
        except Exception as e:
            logger.warning(f"Model compilation failed: {str(e)}")
            return model
    
    def _optimize_memory(self, model: nn.Module) -> nn.Module:
        """Apply memory-efficient optimizations."""
        try:
            # Enable gradient checkpointing for memory efficiency
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            
            # Optimize memory allocation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
                logger.info("CUDA memory optimizations applied")
            
            return model
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {str(e)}")
            return model
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return psutil.virtual_memory().used / 1024 / 1024
    
    def _get_gpu_memory(self) -> Optional[float]:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return None
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Get debug summary statistics."""
        if not self.debug_metrics:
            return {"error": "No debug metrics available"}
        
        execution_times = [m.execution_time for m in self.debug_metrics]
        memory_usages = [m.memory_usage for m in self.debug_metrics]
        
        summary = {
            "debug_mode": self.debug_mode.value,
            "total_operations": len(self.debug_metrics),
            "avg_execution_time": np.mean(execution_times),
            "max_execution_time": max(execution_times),
            "avg_memory_usage": np.mean(memory_usages),
            "max_memory_usage": max(memory_usages),
            "anomaly_detection_enabled": self.anomaly_detection_enabled,
            "profiling_enabled": self.profiling_enabled,
            "memory_profiling_enabled": self.memory_profiling_enabled,
            "gradient_checking_enabled": self.gradient_checking_enabled
        }
        
        # Add optimization summary if available
        if self.optimization_metrics:
            opt_times = [m.execution_time for m in self.optimization_metrics]
            summary.update({
                "total_optimizations": len(self.optimization_metrics),
                "avg_optimization_time": np.mean(opt_times),
                "optimization_modes_used": list(set(m.mode.value for m in self.optimization_metrics))
            })
        
        return summary
    
    def cleanup(self) -> Any:
        """Cleanup debug resources."""
        # Clear profiling data
        self.profiler_data.clear()
        
        # Clear memory snapshots
        self.memory_snapshots.clear()
        
        # Clear metrics
        self.debug_metrics.clear()
        self.optimization_metrics.clear()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("PyTorchDebugger cleanup completed")


class nullcontext:
    """Null context manager for conditional context usage."""
    def __enter__(self) -> Any:
        return None
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
        return False


class PyTorchOptimizer:
    """PyTorch optimization utilities."""
    
    def __init__(self, debugger: PyTorchDebugger):
        
    """__init__ function."""
self.debugger = debugger
    
    def optimize_training_loop(self, 
                             model: nn.Module,
                             dataloader: DataLoader,
                             optimizer: torch.optim.Optimizer,
                             criterion: nn.Module,
                             epochs: int = 1,
                             optimization_mode: OptimizationMode = OptimizationMode.NONE) -> Dict[str, Any]:
        """Optimize a complete training loop."""
        start_time = time.time()
        
        # Optimize model
        optimized_model = self.debugger.optimize_model(model, optimization_mode)
        
        # Enable AMP if requested
        scaler = None
        if optimization_mode in [OptimizationMode.AMP, OptimizationMode.FULL_OPTIMIZATION]:
            scaler = amp.GradScaler()
        
        training_metrics = []
        
        try:
            for epoch in range(epochs):
                epoch_start = time.time()
                
                for batch_idx, (data, target) in enumerate(dataloader):
                    batch_start = time.time()
                    
                    with self.debugger.debug_context(f"training_batch_{batch_idx}"):
                        optimizer.zero_grad()
                        
                        if scaler is not None:
                            # Use AMP
                            with amp.autocast():
                                output = optimized_model(data)
                                loss = criterion(output, target)
                            
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            # Standard training
                            output = optimized_model(data)
                            loss = criterion(output, target)
                            loss.backward()
                            optimizer.step()
                        
                        # Check gradients if enabled
                        if self.debugger.gradient_checking_enabled:
                            gradient_info = self.debugger.check_gradients(optimized_model, loss)
                            if gradient_info["gradient_anomalies"]:
                                logger.warning(f"Gradient anomalies detected: {gradient_info['gradient_anomalies']}")
                        
                        # Profile memory if enabled
                        if self.debugger.memory_profiling_enabled:
                            memory_info = self.debugger.profile_memory(f"batch_{batch_idx}")
                            if memory_info["memory_leaks"]:
                                logger.warning(f"Memory leaks detected: {memory_info['memory_leaks']}")
                        
                        batch_time = time.time() - batch_start
                        
                        # Record metrics
                        metrics = {
                            "epoch": epoch,
                            "batch": batch_idx,
                            "loss": loss.item(),
                            "batch_time": batch_time,
                            "memory_usage": self.debugger._get_memory_usage(),
                            "gpu_memory": self.debugger._get_gpu_memory()
                        }
                        
                        training_metrics.append(metrics)
                
                epoch_time = time.time() - epoch_start
                logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        except Exception as e:
            self.debugger.error_system.error_tracker.track_error(
                error=e,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.MODEL,
                context={"operation": "optimized_training_loop"}
            )
            raise
        
        finally:
            total_time = time.time() - start_time
            
            # Record optimization metrics
            optimization_metrics = OptimizationMetrics(
                mode=optimization_mode,
                execution_time=total_time,
                memory_usage=self.debugger._get_memory_usage(),
                gpu_memory=self.debugger._get_gpu_memory(),
                speedup_factor=total_time / (len(training_metrics) * 0.1),  # Rough estimate
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            self.debugger.optimization_metrics.append(optimization_metrics)
        
        return {
            "total_time": total_time,
            "training_metrics": training_metrics,
            "optimization_metrics": optimization_metrics.to_dict(),
            "model_optimized": optimized_model is not model
        }
    
    def benchmark_optimizations(self, 
                              model: nn.Module,
                              dataloader: DataLoader,
                              optimizer: torch.optim.Optimizer,
                              criterion: nn.Module,
                              epochs: int = 1) -> Dict[str, Any]:
        """Benchmark different optimization modes."""
        optimization_modes = [
            OptimizationMode.NONE,
            OptimizationMode.AMP,
            OptimizationMode.COMPILATION,
            OptimizationMode.MEMORY_EFFICIENT,
            OptimizationMode.FULL_OPTIMIZATION
        ]
        
        benchmark_results = {}
        
        for mode in optimization_modes:
            logger.info(f"Benchmarking {mode.value}")
            
            try:
                result = self.optimize_training_loop(
                    model=model,
                    dataloader=dataloader,
                    optimizer=optimizer,
                    criterion=criterion,
                    epochs=epochs,
                    optimization_mode=mode
                )
                
                benchmark_results[mode.value] = result
                
            except Exception as e:
                logger.error(f"Benchmark failed for {mode.value}: {str(e)}")
                benchmark_results[mode.value] = {"error": str(e)}
        
        return benchmark_results


# Utility decorators
def debug_operation(debug_mode: DebugMode = DebugMode.ANOMALY_DETECTION):
    """Decorator for debugging operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create debugger if not provided
            debugger = kwargs.get('debugger')
            if debugger is None:
                error_system = ErrorHandlingDebuggingSystem()
                debugger = PyTorchDebugger(error_system, debug_mode=debug_mode)
            
            with debugger.debug_context(func.__name__):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def optimize_model(optimization_mode: OptimizationMode = OptimizationMode.AMP):
    """Decorator for model optimization."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get model from args or kwargs
            model = kwargs.get('model') or (args[0] if args else None)
            
            if model is not None and isinstance(model, nn.Module):
                # Create debugger if not provided
                debugger = kwargs.get('debugger')
                if debugger is None:
                    error_system = ErrorHandlingDebuggingSystem()
                    debugger = PyTorchDebugger(error_system)
                
                # Optimize model
                optimized_model = debugger.optimize_model(model, optimization_mode)
                
                # Replace model in args/kwargs
                if 'model' in kwargs:
                    kwargs['model'] = optimized_model
                elif args:
                    args = (optimized_model,) + args[1:]
                
                return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    # Create debugger
    error_system = ErrorHandlingDebuggingSystem()
    training_logger = TrainingLogger()
    debugger = PyTorchDebugger(error_system, training_logger, DebugMode.FULL_DEBUG)
    
    # Create simple model
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Create dummy data
    data = torch.randn(32, 10)
    target = torch.randint(0, 2, (32,))
    
    # Test debugging
    with debugger.debug_context("test_operation"):
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # Test optimization
    optimizer = PyTorchOptimizer(debugger)
    optimized_model = debugger.optimize_model(model, OptimizationMode.AMP)
    
    # Get debug summary
    summary = debugger.get_debug_summary()
    print("Debug Summary:", json.dumps(summary, indent=2))
    
    debugger.cleanup() 