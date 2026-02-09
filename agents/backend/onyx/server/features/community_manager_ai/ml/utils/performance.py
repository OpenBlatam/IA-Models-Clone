"""
Performance Utils - Utilidades de Performance
=============================================

Utilidades para medir y optimizar performance.
"""

import time
import torch
import logging
from typing import Callable, Dict, Any, Optional
from contextlib import contextmanager
import functools

logger = logging.getLogger(__name__)


@contextmanager
def timer(operation_name: str):
    """Context manager para medir tiempo"""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"{operation_name} tomó {elapsed:.3f}s")


def benchmark_function(func: Callable, num_runs: int = 10, warmup: int = 2) -> Dict[str, float]:
    """
    Benchmark de función
    
    Args:
        func: Función a benchmarkear
        num_runs: Número de ejecuciones
        warmup: Ejecuciones de warmup
        
    Returns:
        Dict con estadísticas
    """
    # Warmup
    for _ in range(warmup):
        func()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        func()
        times.append(time.time() - start)
    
    return {
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
        "std": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
    }


def profile_model(model: torch.nn.Module, example_input: Dict[str, torch.Tensor]):
    """
    Profilear modelo con PyTorch profiler
    
    Args:
        model: Modelo a profilear
        example_input: Input de ejemplo
    """
    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True,
            profile_memory=True
        ) as prof:
            with torch.profiler.record_function("model_inference"):
                model(**example_input)
        
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    except Exception as e:
        logger.warning(f"Error en profiling: {e}")


def optimize_data_loading(dataloader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
    """
    Optimizar DataLoader
    
    Args:
        dataloader: DataLoader a optimizar
        
    Returns:
        DataLoader optimizado
    """
    # Ya debería estar optimizado, pero verificar config
    if dataloader.num_workers == 0:
        logger.warning("Considerar usar num_workers > 0 para mejor performance")
    
    if not dataloader.pin_memory and torch.cuda.is_available():
        logger.warning("Considerar usar pin_memory=True para transferencia rápida a GPU")
    
    return dataloader


def clear_cache():
    """Limpiar cache de PyTorch"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()




