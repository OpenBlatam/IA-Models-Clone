"""
Profiling Utilities
===================

Utilidades para profiling de código.
"""

import logging
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


class Profiler:
    """
    Profiler para código PyTorch.
    
    Usa torch.profiler para profiling avanzado.
    """
    
    def __init__(self, use_cuda: bool = True):
        """
        Inicializar profiler.
        
        Args:
            use_cuda: Usar CUDA profiling
        """
        self.use_cuda = use_cuda and TORCH_AVAILABLE and torch.cuda.is_available()
        self.profiler = None
    
    @contextmanager
    def profile(self, activities=None, record_shapes=False, profile_memory=False):
        """
        Context manager para profiling.
        
        Args:
            activities: Actividades a perfilar
            record_shapes: Registrar formas de tensores
            profile_memory: Perfilar memoria
        """
        if not TORCH_AVAILABLE:
            yield
            return
        
        if activities is None:
            activities = [
                torch.profiler.ProfilerActivity.CPU,
            ]
            if self.use_cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        with torch.profiler.profile(
            activities=activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory
        ) as prof:
            yield prof
    
    def export_chrome_trace(self, profiler, file_path: str):
        """
        Exportar trace a Chrome format.
        
        Args:
            profiler: Profiler instance
            file_path: Ruta del archivo
        """
        if TORCH_AVAILABLE:
            profiler.export_chrome_trace(file_path)
            logger.info(f"Exported Chrome trace to {file_path}")


@contextmanager
def profile_function(func_name: str = "function"):
    """
    Context manager simple para profiling de funciones.
    
    Args:
        func_name: Nombre de la función
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.debug(f"{func_name} took {elapsed:.4f} seconds")


def profile_model_forward(model, input_shape, device="cpu", num_runs=10):
    """
    Perfilar forward pass de modelo.
    
    Args:
        model: Modelo a perfilar
        input_shape: Forma del input
        device: Dispositivo
        num_runs: Número de runs
        
    Returns:
        Diccionario con estadísticas
    """
    if not TORCH_AVAILABLE:
        return {}
    
    model = model.to(device)
    model.eval()
    
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.time()
            x = torch.randn(input_shape).to(device)
            _ = model(x)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            elapsed = time.time() - start
            times.append(elapsed)
    
    return {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "std_time": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
    }

