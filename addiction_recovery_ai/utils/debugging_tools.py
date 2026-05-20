"""
Advanced Debugging and Profiling Tools
For deep learning model development and optimization
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any, Callable
import logging
import time
import numpy as np
from contextlib import contextmanager
import traceback

logger = logging.getLogger(__name__)

try:
    from torch.profiler import profile, record_function, ProfilerActivity
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False
    logger.warning("PyTorch profiler not available")


class ModelDebugger:
    """
    Advanced debugging tools for PyTorch models
    - Gradient checking
    - NaN/Inf detection
    - Memory profiling
    - Performance profiling
    """
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Initialize model debugger
        
        Args:
            model: PyTorch model
            device: Device to use
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gradient_history = []
        self.memory_history = []
    
    @contextmanager
    def detect_anomaly(self, enabled: bool = True):
        """
        Context manager for anomaly detection
        
        Args:
            enabled: Enable anomaly detection
        """
        if enabled:
            torch.autograd.set_detect_anomaly(True)
        try:
            yield
        finally:
            if enabled:
                torch.autograd.set_detect_anomaly(False)
    
    def check_gradients(self, loss: torch.Tensor) -> Dict[str, Any]:
        """
        Check gradients for issues
        
        Args:
            loss: Loss tensor
            
        Returns:
            Dictionary with gradient statistics
        """
        stats = {
            "has_nan": False,
            "has_inf": False,
            "max_grad": 0.0,
            "min_grad": 0.0,
            "mean_grad": 0.0,
            "gradient_norm": 0.0,
            "exploding_gradients": False,
            "vanishing_gradients": False
        }
        
        total_norm = 0.0
        all_grads = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                # Check for NaN/Inf
                if torch.isnan(grad).any():
                    stats["has_nan"] = True
                    logger.warning(f"NaN gradient in {name}")
                
                if torch.isinf(grad).any():
                    stats["has_inf"] = True
                    logger.warning(f"Inf gradient in {name}")
                
                # Collect gradients
                param_norm = grad.norm()
                total_norm += param_norm.item() ** 2
                all_grads.append(grad.cpu().numpy().flatten())
        
        if all_grads:
            all_grads = np.concatenate(all_grads)
            stats["max_grad"] = float(np.max(np.abs(all_grads)))
            stats["min_grad"] = float(np.min(np.abs(all_grads)))
            stats["mean_grad"] = float(np.mean(np.abs(all_grads)))
            stats["gradient_norm"] = float(np.sqrt(total_norm))
            
            # Check for exploding/vanishing gradients
            if stats["gradient_norm"] > 100.0:
                stats["exploding_gradients"] = True
                logger.warning("Exploding gradients detected!")
            
            if stats["gradient_norm"] < 1e-6:
                stats["vanishing_gradients"] = True
                logger.warning("Vanishing gradients detected!")
        
        self.gradient_history.append(stats)
        return stats
    
    def check_outputs(self, outputs: torch.Tensor) -> Dict[str, Any]:
        """
        Check model outputs for issues
        
        Args:
            outputs: Model outputs
            
        Returns:
            Dictionary with output statistics
        """
        stats = {
            "has_nan": bool(torch.isnan(outputs).any().item()),
            "has_inf": bool(torch.isinf(outputs).any().item()),
            "max_value": float(outputs.max().item()) if outputs.numel() > 0 else 0.0,
            "min_value": float(outputs.min().item()) if outputs.numel() > 0 else 0.0,
            "mean_value": float(outputs.mean().item()) if outputs.numel() > 0 else 0.0,
            "std_value": float(outputs.std().item()) if outputs.numel() > 0 else 0.0
        }
        
        if stats["has_nan"]:
            logger.warning("NaN detected in outputs!")
        
        if stats["has_inf"]:
            logger.warning("Inf detected in outputs!")
        
        return stats
    
    def profile_memory(self) -> Dict[str, float]:
        """
        Profile memory usage
        
        Returns:
            Dictionary with memory statistics
        """
        stats = {}
        
        if self.device.type == "cuda":
            stats["allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            stats["reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            stats["max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            stats["max_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3
            
            if torch.cuda.get_device_properties(0).total_memory:
                stats["total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
                stats["usage_percent"] = (stats["allocated_gb"] / stats["total_gb"]) * 100
        else:
            import psutil
            process = psutil.Process()
            stats["memory_gb"] = process.memory_info().rss / 1024**3
        
        self.memory_history.append(stats)
        return stats
    
    def profile_forward(
        self, 
        input_tensor: torch.Tensor, 
        num_runs: int = 10,
        use_torch_profiler: bool = False
    ) -> Dict[str, float]:
        """
        Profile forward pass with optional PyTorch profiler
        
        Args:
            input_tensor: Input tensor
            num_runs: Number of runs for averaging
            use_torch_profiler: Use PyTorch's built-in profiler
            
        Returns:
            Dictionary with timing statistics
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        
        # Warmup
        with torch.inference_mode():
            for _ in range(3):
                _ = self.model(input_tensor)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        if use_torch_profiler and PROFILER_AVAILABLE:
            # Use PyTorch profiler for detailed analysis
            activities = [ProfilerActivity.CPU]
            if self.device.type == "cuda":
                activities.append(ProfilerActivity.CUDA)
            
            with profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                with record_function("model_forward"):
                    with torch.inference_mode():
                        _ = self.model(input_tensor)
            
            # Extract timing from profiler
            events = prof.key_averages()
            total_time = sum(event.cuda_time_total if self.device.type == "cuda" else event.cpu_time_total 
                           for event in events) / 1000  # Convert to ms
            
            return {
                "mean_time_ms": total_time,
                "profiler_events": len(events),
                "profiler_available": True
            }
        else:
            # Simple timing
            times = []
            for _ in range(num_runs):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                
                with torch.inference_mode():
                    _ = self.model(input_tensor)
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
            
            return {
                "mean_time_ms": float(np.mean(times)),
                "std_time_ms": float(np.std(times)),
                "min_time_ms": float(np.min(times)),
                "max_time_ms": float(np.max(times)),
                "median_time_ms": float(np.median(times)),
                "profiler_available": False
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary with model statistics
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "model_size_mb": total_params * 4 / 1024**2,  # Assuming float32
            "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype)
        }


class PerformanceProfiler:
    """
    Performance profiler for training and inference
    """
    
    def __init__(self):
        """Initialize profiler"""
        self.metrics = []
    
    @contextmanager
    def profile(self, name: str):
        """
        Context manager for profiling
        
        Args:
            name: Operation name
        """
        start_time = time.perf_counter()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            duration = (end_time - start_time) * 1000  # ms
            memory_delta = (end_memory - start_memory) / 1024**2  # MB
            
            self.metrics.append({
                "name": name,
                "duration_ms": duration,
                "memory_delta_mb": memory_delta,
                "timestamp": time.time()
            })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary"""
        if not self.metrics:
            return {}
        
        durations = [m["duration_ms"] for m in self.metrics]
        memory_deltas = [m["memory_delta_mb"] for m in self.metrics]
        
        return {
            "total_operations": len(self.metrics),
            "total_time_ms": sum(durations),
            "mean_time_ms": np.mean(durations),
            "max_time_ms": max(durations),
            "min_time_ms": min(durations),
            "total_memory_mb": sum(memory_deltas),
            "mean_memory_mb": np.mean(memory_deltas)
        }
    
    def clear(self):
        """Clear metrics"""
        self.metrics = []


class TrainingMonitor:
    """
    Monitor training progress and detect issues
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        """
        Initialize training monitor
        
        Args:
            patience: Patience for early stopping
            min_delta: Minimum change to be considered improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.history = []
    
    def update(self, loss: float, metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Update monitor with new loss
        
        Args:
            loss: Current loss
            metrics: Additional metrics
            
        Returns:
            Dictionary with monitoring results
        """
        improved = loss < (self.best_loss - self.min_delta)
        
        if improved:
            self.best_loss = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        result = {
            "loss": loss,
            "best_loss": self.best_loss,
            "improved": improved,
            "patience_counter": self.patience_counter,
            "should_stop": self.patience_counter >= self.patience,
            "metrics": metrics or {}
        }
        
        self.history.append(result)
        return result
    
    def get_history(self) -> List[Dict]:
        """Get training history"""
        return self.history


def create_model_debugger(model: nn.Module, device: Optional[torch.device] = None) -> ModelDebugger:
    """Factory for model debugger"""
    return ModelDebugger(model, device)


def create_profiler() -> PerformanceProfiler:
    """Factory for performance profiler"""
    return PerformanceProfiler()


def create_training_monitor(patience: int = 5, min_delta: float = 0.001) -> TrainingMonitor:
    """Factory for training monitor"""
    return TrainingMonitor(patience, min_delta)






