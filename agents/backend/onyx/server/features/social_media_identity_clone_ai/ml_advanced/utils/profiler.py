"""
Profiling y optimización de código
"""

import torch
import time
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Profiler de rendimiento"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.profiles = {}
    
    @contextmanager
    def profile(self, name: str, enabled: bool = True):
        """Context manager para profiling"""
        if not enabled:
            yield
            return
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        yield
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        self.profiles[name] = {
            "duration": duration,
            "memory_delta_mb": memory_delta,
            "start_memory_mb": start_memory,
            "end_memory_mb": end_memory
        }
        
        logger.info(
            f"Profile {name}: {duration:.4f}s, "
            f"Memory: {memory_delta:+.2f}MB"
        )
    
    def _get_memory_usage(self) -> float:
        """Obtiene uso de memoria en MB"""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de profiles"""
        if not self.profiles:
            return {}
        
        total_time = sum(p["duration"] for p in self.profiles.values())
        max_memory = max(p["end_memory_mb"] for p in self.profiles.values())
        
        return {
            "total_time": total_time,
            "max_memory_mb": max_memory,
            "profiles": self.profiles
        }
    
    def profile_function(
        self,
        func: Callable,
        *args,
        name: Optional[str] = None,
        **kwargs
    ) -> tuple:
        """Profilea una función"""
        profile_name = name or func.__name__
        
        with self.profile(profile_name):
            result = func(*args, **kwargs)
        
        return result


def profile_model_forward(
    model: torch.nn.Module,
    sample_input: Dict[str, torch.Tensor],
    num_runs: int = 10,
    warmup_runs: int = 3
) -> Dict[str, float]:
    """
    Profilea forward pass del modelo
    
    Args:
        model: Modelo a profilear
        sample_input: Input de ejemplo
        num_runs: Número de runs
        warmup_runs: Runs de warmup
        
    Returns:
        Métricas de performance
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(**sample_input)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Profile
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start = time.time()
            _ = model(**sample_input)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            times.append(time.time() - start)
    
    return {
        "mean_time": float(sum(times) / len(times)),
        "std_time": float(torch.tensor(times).std().item()),
        "min_time": float(min(times)),
        "max_time": float(max(times)),
        "throughput": float(1.0 / (sum(times) / len(times)))
    }




