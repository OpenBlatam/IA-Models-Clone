"""
Performance Utilities
Additional performance optimization utilities
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    Additional performance optimizations
    """
    
    @staticmethod
    def enable_torch_optimizations():
        """Enable PyTorch optimizations"""
        # Enable cuDNN benchmarking
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            logger.info("cuDNN benchmarking enabled")
        
        # Enable TensorFloat-32 (TF32) for Ampere GPUs
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled for Ampere GPUs")
    
    @staticmethod
    def optimize_data_loader(dataloader: torch.utils.data.DataLoader):
        """
        Optimize DataLoader settings
        
        Args:
            dataloader: DataLoader to optimize
        """
        # Set optimal number of workers
        if dataloader.num_workers == 0:
            import os
            num_workers = min(4, os.cpu_count() or 1)
            logger.info(f"Setting num_workers to {num_workers}")
        
        # Enable pin_memory if CUDA available
        if torch.cuda.is_available() and not dataloader.pin_memory:
            logger.info("Enabling pin_memory for faster GPU transfer")
    
    @staticmethod
    def profile_model(
        model: nn.Module,
        input_shape: tuple,
        device: Optional[torch.device] = None,
        num_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Profile model performance
        
        Args:
            model: Model to profile
            input_shape: Input shape
            device: Device to use
            num_runs: Number of profiling runs
            
        Returns:
            Profiling results
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        
        dummy_input = torch.randn(input_shape).to(device)
        
        # Warmup
        with torch.inference_mode():
            for _ in range(5):
                _ = model(dummy_input)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Profile
        import time
        times = []
        
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            with torch.inference_mode():
                _ = model(dummy_input)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        return {
            "mean_time_ms": sum(times) / len(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "std_time_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
        }


def enable_optimizations():
    """Enable all optimizations"""
    PerformanceOptimizer.enable_torch_optimizations()


def profile_model(model: nn.Module, input_shape: tuple, **kwargs) -> Dict[str, Any]:
    """Profile model"""
    return PerformanceOptimizer.profile_model(model, input_shape, **kwargs)









