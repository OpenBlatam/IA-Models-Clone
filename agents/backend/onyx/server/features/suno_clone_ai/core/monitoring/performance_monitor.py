"""
Performance Monitoring

Monitors model performance, inference time, throughput, etc.
"""

import logging
import time
from typing import Dict, Any, Optional
import torch
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor model performance metrics."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.inference_times = []
        self.throughput_samples = []
    
    def measure_inference(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        num_warmup: int = 5,
        num_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Measure inference performance.
        
        Args:
            model: Model to measure
            input_tensor: Input tensor
            num_warmup: Number of warmup iterations
            num_iterations: Number of measurement iterations
            
        Returns:
            Performance metrics
        """
        device = next(model.parameters()).device
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(input_tensor)
        
        # Synchronize
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if device.type == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    _ = model(input_tensor)
                    end_event.record()
                    torch.cuda.synchronize()
                    elapsed = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
                else:
                    start = time.time()
                    _ = model(input_tensor)
                    elapsed = time.time() - start
                
                times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = input_tensor.shape[0] / avg_time if avg_time > 0 else 0
        
        metrics = {
            'avg_inference_time_s': avg_time,
            'std_inference_time_s': std_time,
            'min_inference_time_s': np.min(times),
            'max_inference_time_s': np.max(times),
            'throughput_samples_per_sec': throughput
        }
        
        logger.info(f"Performance metrics: {metrics}")
        
        return metrics
    
    def measure_memory_usage(self, model: torch.nn.Module) -> Dict[str, float]:
        """
        Measure model memory usage.
        
        Args:
            model: Model to measure
            
        Returns:
            Memory usage metrics
        """
        device = next(model.parameters()).device
        
        metrics = {}
        
        if device.type == "cuda":
            metrics.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated(device) / (1024 ** 3),
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved(device) / (1024 ** 3),
                'gpu_memory_max_allocated_gb': torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            })
        
        return metrics



