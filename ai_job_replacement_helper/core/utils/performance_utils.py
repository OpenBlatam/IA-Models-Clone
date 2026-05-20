"""
Performance Utilities - Utilidades de performance
=================================================

Funciones para medir y optimizar performance.
"""

import logging
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from contextlib import contextmanager
import functools

logger = logging.getLogger(__name__)


@contextmanager
def timer(description: str = "Operation"):
    """
    Context manager para medir tiempo de ejecución.
    
    Args:
        description: Descripción de la operación
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{description} took {elapsed:.4f} seconds")


def profile_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, Any]:
    """
    Perfilar modelo para medir tiempo de inferencia.
    
    Args:
        model: Modelo a perfilar
        input_shape: Forma de entrada (sin batch dimension)
        num_runs: Número de runs para promediar
        warmup_runs: Número de runs de warmup
    
    Returns:
        Diccionario con estadísticas de performance
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Warmup
    dummy_input = torch.randn(1, *input_shape).to(device)
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    # Synchronize if CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Actual timing
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start = time.time()
            _ = model(dummy_input)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            elapsed = time.time() - start
            times.append(elapsed)
    
    times = np.array(times)
    
    return {
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "min_time": float(np.min(times)),
        "max_time": float(np.max(times)),
        "median_time": float(np.median(times)),
        "throughput": 1.0 / np.mean(times),  # samples per second
    }


def benchmark_dataloader(
    dataloader: torch.utils.data.DataLoader,
    num_batches: int = 100
) -> Dict[str, Any]:
    """
    Benchmark DataLoader para medir velocidad de carga de datos.
    
    Args:
        dataloader: DataLoader a benchmarkear
        num_batches: Número de batches a procesar
    
    Returns:
        Diccionario con estadísticas
    """
    times = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        start = time.time()
        # Simular procesamiento mínimo
        if isinstance(batch, (list, tuple)):
            _ = [b for b in batch]
        else:
            _ = batch
        elapsed = time.time() - start
        times.append(elapsed)
    
    times = np.array(times)
    
    return {
        "mean_batch_time": float(np.mean(times)),
        "std_batch_time": float(np.std(times)),
        "throughput": len(times) / np.sum(times),  # batches per second
    }


def get_memory_usage(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Obtener uso de memoria.
    
    Args:
        device: Dispositivo (None para CPU)
    
    Returns:
        Diccionario con información de memoria
    """
    if device is None:
        device = torch.device("cpu")
    
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)  # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
            "free_gb": reserved - allocated,
        }
    else:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_gb": memory_info.rss / (1024 ** 3),  # Resident Set Size
            "vms_gb": memory_info.vms / (1024 ** 3),  # Virtual Memory Size
        }


def clear_cache(device: Optional[torch.device] = None) -> None:
    """
    Limpiar caché de memoria.
    
    Args:
        device: Dispositivo (None para CPU)
    """
    if device is None:
        device = torch.device("cpu")
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")
    else:
        import gc
        gc.collect()
        logger.info("CPU cache cleared")


def optimize_model_for_inference(model: nn.Module) -> nn.Module:
    """
    Optimizar modelo para inferencia.
    
    Args:
        model: Modelo a optimizar
    
    Returns:
        Modelo optimizado
    """
    model.eval()
    
    # Fuse batch norm layers if possible
    try:
        if hasattr(torch.quantization, 'fuse_modules'):
            # Esto es específico para modelos que lo soporten
            pass
    except:
        pass
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    logger.info("Model optimized for inference")
    return model


def count_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...]
) -> int:
    """
    Contar FLOPs (Floating Point Operations) del modelo.
    
    Args:
        model: Modelo
        input_shape: Forma de entrada (sin batch dimension)
    
    Returns:
        Número de FLOPs
    """
    # Implementación simplificada
    # En producción, usaría thop o fvcore
    total_flops = 0
    
    dummy_input = torch.randn(1, *input_shape)
    
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # FLOPs = 2 * input_features * output_features
            total_flops += 2 * module.in_features * module.out_features
        elif isinstance(module, nn.Conv2d):
            # FLOPs = 2 * kernel_size^2 * in_channels * out_channels * output_size^2
            # Simplificado
            pass
    
    return total_flops

