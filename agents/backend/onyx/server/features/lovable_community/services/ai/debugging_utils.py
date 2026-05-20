"""
Debugging Utilities for Deep Learning

Provides utilities for debugging training and inference:
- NaN/Inf detection
- Gradient checking
- Autograd anomaly detection
- Memory profiling
- Performance profiling
"""

import logging
import warnings
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from contextlib import contextmanager
import gc

logger = logging.getLogger(__name__)


class NaNInfDetector:
    """
    Detect NaN and Inf values in tensors and model parameters
    """
    
    @staticmethod
    def check_tensor(tensor: torch.Tensor, name: str = "tensor") -> bool:
        """
        Check if tensor contains NaN or Inf
        
        Args:
            tensor: Tensor to check
            name: Name for logging
            
        Returns:
            True if NaN/Inf found
        """
        if not isinstance(tensor, torch.Tensor):
            return False
        
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        
        if has_nan or has_inf:
            logger.warning(f"{name} contains NaN: {has_nan}, Inf: {has_inf}")
            if has_nan:
                logger.warning(f"NaN count: {torch.isnan(tensor).sum().item()}")
            if has_inf:
                logger.warning(f"Inf count: {torch.isinf(tensor).sum().item()}")
            return True
        
        return False
    
    @staticmethod
    def check_model(model: nn.Module, check_gradients: bool = False) -> Dict[str, bool]:
        """
        Check all model parameters for NaN/Inf
        
        Args:
            model: Model to check
            check_gradients: Whether to check gradients too
            
        Returns:
            Dictionary with parameter names and whether they contain NaN/Inf
        """
        issues = {}
        
        for name, param in model.named_parameters():
            has_issue = (
                NaNInfDetector.check_tensor(param.data, f"param.{name}") or
                (check_gradients and param.grad is not None and
                 NaNInfDetector.check_tensor(param.grad, f"grad.{name}"))
            )
            issues[name] = has_issue
        
        return issues
    
    @staticmethod
    def check_batch(batch: Dict[str, Any]) -> bool:
        """
        Check batch for NaN/Inf
        
        Args:
            batch: Batch dictionary
            
        Returns:
            True if NaN/Inf found
        """
        has_issue = False
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if NaNInfDetector.check_tensor(value, f"batch.{key}"):
                    has_issue = True
        return has_issue


class GradientChecker:
    """
    Check gradients for issues
    """
    
    @staticmethod
    def check_gradients(
        model: nn.Module,
        check_nan_inf: bool = True,
        check_exploding: bool = True,
        check_vanishing: bool = True,
        max_norm: float = 10.0,
        min_norm: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Check gradients for common issues
        
        Args:
            model: Model to check
            check_nan_inf: Check for NaN/Inf
            check_exploding: Check for exploding gradients
            check_vanishing: Check for vanishing gradients
            max_norm: Maximum gradient norm
            min_norm: Minimum gradient norm
            
        Returns:
            Dictionary with gradient statistics
        """
        stats = {
            "total_params": 0,
            "params_with_grad": 0,
            "grad_norms": [],
            "has_nan": False,
            "has_inf": False,
            "exploding": [],
            "vanishing": []
        }
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                stats["params_with_grad"] += 1
                grad = param.grad
                
                # Check NaN/Inf
                if check_nan_inf:
                    if torch.isnan(grad).any():
                        stats["has_nan"] = True
                        logger.warning(f"NaN in gradient: {name}")
                    if torch.isinf(grad).any():
                        stats["has_inf"] = True
                        logger.warning(f"Inf in gradient: {name}")
                
                # Check norm
                grad_norm = grad.norm().item()
                stats["grad_norms"].append((name, grad_norm))
                
                # Check exploding
                if check_exploding and grad_norm > max_norm:
                    stats["exploding"].append((name, grad_norm))
                    logger.warning(f"Exploding gradient: {name} (norm: {grad_norm:.2e})")
                
                # Check vanishing
                if check_vanishing and grad_norm < min_norm:
                    stats["vanishing"].append((name, grad_norm))
                    logger.debug(f"Vanishing gradient: {name} (norm: {grad_norm:.2e})")
            
            stats["total_params"] += 1
        
        # Overall gradient norm
        if stats["grad_norms"]:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            stats["total_grad_norm"] = total_norm.item()
        
        return stats


@contextmanager
def detect_anomaly():
    """
    Context manager for autograd anomaly detection
    
    Usage:
        with detect_anomaly():
            loss.backward()
    """
    torch.autograd.set_detect_anomaly(True)
    try:
        yield
    finally:
        torch.autograd.set_detect_anomaly(False)


@contextmanager
def gradient_checkpointing(model: nn.Module, enable: bool = True):
    """
    Context manager for gradient checkpointing (memory optimization)
    
    Args:
        model: Model to enable checkpointing for
        enable: Whether to enable checkpointing
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        if enable:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
    
    try:
        yield
    finally:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()


class MemoryProfiler:
    """
    Profile GPU memory usage
    """
    
    @staticmethod
    def get_memory_stats(device: Optional[torch.device] = None) -> Dict[str, float]:
        """
        Get current memory statistics
        
        Args:
            device: Device to check (None for current device)
            
        Returns:
            Dictionary with memory statistics
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if device.type != "cuda":
            return {"error": "Memory profiling only available for CUDA devices"}
        
        stats = {
            "allocated": torch.cuda.memory_allocated(device) / 1e9,  # GB
            "reserved": torch.cuda.memory_reserved(device) / 1e9,  # GB
            "max_allocated": torch.cuda.max_memory_allocated(device) / 1e9,  # GB
            "max_reserved": torch.cuda.max_memory_reserved(device) / 1e9,  # GB
        }
        
        return stats
    
    @staticmethod
    def log_memory_stats(device: Optional[torch.device] = None, prefix: str = "") -> None:
        """Log memory statistics"""
        stats = MemoryProfiler.get_memory_stats(device)
        if "error" not in stats:
            logger.info(
                f"{prefix}Memory - Allocated: {stats['allocated']:.2f}GB, "
                f"Reserved: {stats['reserved']:.2f}GB, "
                f"Max Allocated: {stats['max_allocated']:.2f}GB"
            )
    
    @staticmethod
    def clear_cache(device: Optional[torch.device] = None) -> None:
        """Clear GPU cache"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU cache cleared")


class PerformanceProfiler:
    """
    Profile performance of operations
    """
    
    @staticmethod
    @contextmanager
    def profile(operation_name: str = "operation"):
        """
        Profile an operation
        
        Usage:
            with PerformanceProfiler.profile("forward_pass"):
                output = model(input)
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        import time
        start_time = time.time()
        
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            logger.info(f"{operation_name} took {elapsed:.4f} seconds")
    
    @staticmethod
    def profile_model_forward(
        model: nn.Module,
        sample_input: Dict[str, torch.Tensor],
        num_runs: int = 10,
        warmup: int = 3
    ) -> Dict[str, float]:
        """
        Profile model forward pass
        
        Args:
            model: Model to profile
            sample_input: Sample input
            num_runs: Number of runs
            warmup: Number of warmup runs
            
        Returns:
            Dictionary with timing statistics
        """
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(**sample_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Profile
        import time
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.time()
                _ = model(**sample_input)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                elapsed = time.time() - start
                times.append(elapsed)
        
        return {
            "mean": sum(times) / len(times),
            "std": (sum((t - sum(times) / len(times))**2 for t in times) / len(times))**0.5,
            "min": min(times),
            "max": max(times),
            "median": sorted(times)[len(times) // 2]
        }


def enable_debug_mode():
    """
    Enable comprehensive debug mode
    
    Enables:
    - Anomaly detection
    - Detailed logging
    - Warnings
    """
    torch.autograd.set_detect_anomaly(True)
    logging.getLogger().setLevel(logging.DEBUG)
    warnings.filterwarnings("default")
    logger.info("Debug mode enabled")


def disable_debug_mode():
    """Disable debug mode"""
    torch.autograd.set_detect_anomaly(False)
    logging.getLogger().setLevel(logging.INFO)
    warnings.filterwarnings("ignore")
    logger.info("Debug mode disabled")















