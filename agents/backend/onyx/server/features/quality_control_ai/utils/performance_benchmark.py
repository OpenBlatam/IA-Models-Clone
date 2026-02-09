"""
Performance Benchmarking Tools
"""

import time
import torch
import numpy as np
from typing import Callable, Dict, List
import logging

logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Benchmark model performance"""
    
    @staticmethod
    def benchmark_inference(
        model: torch.nn.Module,
        input_shape: tuple,
        num_iterations: int = 100,
        warmup: int = 10,
        device: str = 'cuda'
    ) -> Dict:
        """Benchmark inference speed"""
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        # Warmup
        dummy_input = torch.randn(*input_shape).to(device)
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
        
        # Synchronize
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.time()
                _ = model(dummy_input)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1.0 / avg_time
        
        return {
            "avg_time_ms": avg_time * 1000,
            "std_time_ms": std_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "fps": fps,
            "throughput": fps * input_shape[0]  # batch size
        }
    
    @staticmethod
    def compare_models(models: Dict[str, torch.nn.Module], input_shape: tuple):
        """Compare multiple models"""
        results = {}
        for name, model in models.items():
            logger.info(f"Benchmarking {name}...")
            results[name] = PerformanceBenchmark.benchmark_inference(
                model, input_shape
            )
        return results
    
    @staticmethod
    def profile_model(model: torch.nn.Module, input_shape: tuple, device='cuda'):
        """Profile model with PyTorch profiler"""
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        dummy_input = torch.randn(*input_shape).to(device)
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True
        ) as prof:
            with torch.no_grad():
                _ = model(dummy_input)
        
        return prof

