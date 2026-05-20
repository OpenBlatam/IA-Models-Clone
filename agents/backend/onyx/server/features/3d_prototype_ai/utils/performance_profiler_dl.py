"""
Performance Profiler for Deep Learning - Profiler de rendimiento para DL
=========================================================================
Profiling de modelos, operaciones, y optimización
"""

import logging
import torch
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class DLPerformanceProfiler:
    """Profiler de rendimiento para deep learning"""
    
    def __init__(self):
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.memory_usage: Dict[str, List[float]] = defaultdict(list)
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager para perfilar operación"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.time()
        
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
            else:
                elapsed_time = time.time() - start_time
            
            end_memory = self._get_memory_usage()
            memory_used = end_memory - start_memory
            
            self.operation_times[operation_name].append(elapsed_time)
            self.memory_usage[operation_name].append(memory_used)
            
            logger.debug(f"{operation_name}: {elapsed_time:.4f}s, Memory: {memory_used:.2f}MB")
    
    def _get_memory_usage(self) -> float:
        """Obtiene uso de memoria"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        else:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # MB
    
    def profile_model(
        self,
        model: torch.nn.Module,
        input_shape: tuple,
        device: Optional[torch.device] = None,
        num_runs: int = 10
    ) -> Dict[str, Any]:
        """Perfila un modelo completo"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = model.to(device)
        model.eval()
        
        # Warmup
        dummy_input = torch.randn(input_shape).to(device)
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)
        
        # Profile inference
        inference_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                else:
                    start_time = time.time()
                
                _ = model(dummy_input)
                
                if torch.cuda.is_available():
                    end.record()
                    torch.cuda.synchronize()
                    inference_times.append(start.elapsed_time(end) / 1000.0)
                else:
                    inference_times.append(time.time() - start_time)
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        else:
            memory_allocated = self._get_memory_usage()
            memory_reserved = memory_allocated
        
        profile = {
            "model_name": model.__class__.__name__,
            "input_shape": input_shape,
            "device": str(device),
            "avg_inference_time": sum(inference_times) / len(inference_times),
            "min_inference_time": min(inference_times),
            "max_inference_time": max(inference_times),
            "std_inference_time": (sum((t - sum(inference_times)/len(inference_times))**2 for t in inference_times) / len(inference_times)) ** 0.5,
            "memory_allocated_mb": memory_allocated,
            "memory_reserved_mb": memory_reserved,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "num_runs": num_runs
        }
        
        profile_id = f"{model.__class__.__name__}_{len(self.profiles)}"
        self.profiles[profile_id] = profile
        
        return profile
    
    def get_operation_stats(self) -> Dict[str, Dict[str, float]]:
        """Obtiene estadísticas de operaciones"""
        stats = {}
        for op_name, times in self.operation_times.items():
            if times:
                stats[op_name] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_calls": len(times),
                    "total_time": sum(times)
                }
        return stats
    
    def get_memory_stats(self) -> Dict[str, Dict[str, float]]:
        """Obtiene estadísticas de memoria"""
        stats = {}
        for op_name, memory_list in self.memory_usage.items():
            if memory_list:
                stats[op_name] = {
                    "avg_memory": sum(memory_list) / len(memory_list),
                    "max_memory": max(memory_list),
                    "total_calls": len(memory_list)
                }
        return stats
    
    def clear_profiles(self):
        """Limpia perfiles"""
        self.profiles.clear()
        self.operation_times.clear()
        self.memory_usage.clear()
        logger.info("Profiles cleared")




