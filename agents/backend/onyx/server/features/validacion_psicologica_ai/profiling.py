"""
Profiling and Optimization
==========================
Code profiling and performance optimization
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import structlog
import time
from contextlib import contextmanager
import numpy as np

try:
    from torch.profiler import profile, record_function, ProfilerActivity
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False

logger = structlog.get_logger()


class PerformanceProfiler:
    """
    Performance profiler for training and inference
    """
    
    def __init__(self, use_torch_profiler: bool = True):
        """
        Initialize profiler
        
        Args:
            use_torch_profiler: Use PyTorch profiler if available
        """
        self.use_torch_profiler = use_torch_profiler and PROFILER_AVAILABLE
        self.profiling_data = {}
        
        logger.info("PerformanceProfiler initialized", torch_profiler=self.use_torch_profiler)
    
    @contextmanager
    def profile_training_step(
        self,
        model: nn.Module,
        batch: Dict[str, Any]
    ):
        """
        Profile a training step
        
        Args:
            model: Model
            batch: Training batch
        """
        if self.use_torch_profiler:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True
            ) as prof:
                with record_function("training_step"):
                    yield prof
            
            # Export profiling data
            self.profiling_data["training_step"] = prof.key_averages().table(
                sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
            )
        else:
            start_time = time.time()
            start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            yield None
            
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            self.profiling_data["training_step"] = {
                "time": end_time - start_time,
                "memory_delta": end_memory - start_memory
            }
    
    def profile_data_loading(
        self,
        data_loader: DataLoader,
        num_batches: int = 10
    ) -> Dict[str, float]:
        """
        Profile data loading performance
        
        Args:
            data_loader: Data loader to profile
            num_batches: Number of batches to profile
            
        Returns:
            Profiling metrics
        """
        times = []
        
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break
            
            start_time = time.time()
            # Simulate processing
            _ = batch
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        return {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "total_time": np.sum(times)
        }
    
    def profile_model_inference(
        self,
        model: nn.Module,
        input_shape: tuple,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Profile model inference
        
        Args:
            model: Model to profile
            input_shape: Input shape (batch_size, seq_len)
            num_iterations: Number of iterations
            
        Returns:
            Inference metrics
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Warmup
        dummy_input = torch.randn(input_shape).to(device)
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Profile
        times = []
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                _ = model(dummy_input)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        return {
            "mean_inference_time": np.mean(times),
            "std_inference_time": np.std(times),
            "min_inference_time": np.min(times),
            "max_inference_time": np.max(times),
            "throughput": input_shape[0] / np.mean(times)  # samples per second
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics
        
        Returns:
            Memory statistics
        """
        stats = {}
        
        if torch.cuda.is_available():
            stats["cuda"] = {
                "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "reserved": torch.cuda.memory_reserved() / 1024**3,  # GB
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,  # GB
                "max_reserved": torch.cuda.max_memory_reserved() / 1024**3  # GB
            }
        
        return stats
    
    def reset_memory_stats(self) -> None:
        """Reset memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


class ModelOptimizer:
    """Model optimization utilities"""
    
    @staticmethod
    def optimize_for_inference(model: nn.Module) -> nn.Module:
        """
        Optimize model for inference
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        model.eval()
        
        # Fuse operations if possible
        try:
            if hasattr(torch.quantization, 'fuse_modules'):
                # Example: fuse Conv+BN+ReLU
                # This is model-specific
                pass
        except Exception as e:
            logger.warning("Could not fuse modules", error=str(e))
        
        # Set to inference mode
        with torch.no_grad():
            # JIT compile if beneficial
            try:
                # This is model-specific and may not always work
                # model = torch.jit.script(model)
                pass
            except Exception as e:
                logger.debug("JIT compilation not possible", error=str(e))
        
        return model
    
    @staticmethod
    def quantize_model(
        model: nn.Module,
        quantization_type: str = "dynamic"
    ) -> nn.Module:
        """
        Quantize model for faster inference
        
        Args:
            model: Model to quantize
            quantization_type: Type of quantization (dynamic, static, qat)
            
        Returns:
            Quantized model
        """
        if quantization_type == "dynamic":
            try:
                model = torch.quantization.quantize_dynamic(
                    model,
                    {nn.Linear, nn.LSTM, nn.GRU},
                    dtype=torch.qint8
                )
                logger.info("Model quantized (dynamic)")
            except Exception as e:
                logger.warning("Dynamic quantization failed", error=str(e))
        
        return model


# Global profiler instance
performance_profiler = PerformanceProfiler()




