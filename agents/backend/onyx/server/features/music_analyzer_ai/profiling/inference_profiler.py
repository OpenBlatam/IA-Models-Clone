"""
Inference Profiler
Profile inference performance
"""

from typing import Dict, Any, Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class InferenceProfiler:
    """Profile inference performance"""
    
    def __init__(self):
        self.profiling_data: Dict[str, list] = {}
    
    def profile_batch(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        device: str = "cuda",
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Profile batch inference
        
        Args:
            model: Model to profile
            batch: Input batch
            device: Device
            num_runs: Number of runs
            warmup_runs: Warmup runs
        
        Returns:
            Profiling results
        """
        if not TORCH_AVAILABLE:
            return {}
        
        model.eval()
        
        # Move batch to device
        device_batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(device_batch)
        
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Profile
        times = []
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                run_start = time.time()
                _ = model(device_batch)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - run_start)
        
        total_time = time.time() - start_time
        
        return {
            "avg_inference_time_ms": (total_time / num_runs) * 1000,
            "min_inference_time_ms": min(times) * 1000,
            "max_inference_time_ms": max(times) * 1000,
            "std_inference_time_ms": (sum((t - total_time/num_runs)**2 for t in times) / num_runs)**0.5 * 1000,
            "throughput_samples_per_sec": 1.0 / (total_time / num_runs) if total_time > 0 else 0.0,
            "num_runs": num_runs
        }
    
    def profile_latency(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str = "cuda",
        num_runs: int = 1000
    ) -> Dict[str, float]:
        """
        Profile inference latency
        
        Args:
            model: Model to profile
            input_shape: Input shape
            device: Device
            num_runs: Number of runs
        
        Returns:
            Latency statistics
        """
        if not TORCH_AVAILABLE:
            return {}
        
        dummy_input = torch.randn(input_shape).to(device)
        model.eval()
        
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.time()
                _ = model(dummy_input)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        
        times_ms = [t * 1000 for t in times]
        
        return {
            "p50_latency_ms": sorted(times_ms)[len(times_ms) // 2],
            "p95_latency_ms": sorted(times_ms)[int(len(times_ms) * 0.95)],
            "p99_latency_ms": sorted(times_ms)[int(len(times_ms) * 0.99)],
            "avg_latency_ms": sum(times_ms) / len(times_ms),
            "min_latency_ms": min(times_ms),
            "max_latency_ms": max(times_ms)
        }



