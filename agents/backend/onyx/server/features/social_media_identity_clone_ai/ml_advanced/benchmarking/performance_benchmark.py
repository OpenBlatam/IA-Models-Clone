"""
Benchmarking de rendimiento
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Benchmark de rendimiento"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results = {}
    
    @contextmanager
    def benchmark(self, name: str):
        """Context manager para benchmarking"""
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
        
        self.results[name] = {
            "duration": duration,
            "memory_delta_mb": memory_delta,
            "start_memory_mb": start_memory,
            "end_memory_mb": end_memory
        }
    
    def _get_memory_usage(self) -> float:
        """Obtiene uso de memoria en MB"""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        return 0.0
    
    def benchmark_model(
        self,
        model: nn.Module,
        sample_input: Dict[str, torch.Tensor],
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark de modelo
        
        Args:
            model: Modelo a benchmarkear
            sample_input: Input de ejemplo
            num_runs: Número de runs
            warmup_runs: Runs de warmup
            
        Returns:
            Métricas de benchmark
        """
        model.eval()
        model = model.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(**sample_input)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        memory_usage = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                start_memory = self._get_memory_usage()
                start_time = time.time()
                
                _ = model(**sample_input)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                times.append(end_time - start_time)
                memory_usage.append(end_memory - start_memory)
        
        return {
            "mean_latency_ms": float(np.mean(times) * 1000),
            "std_latency_ms": float(np.std(times) * 1000),
            "min_latency_ms": float(np.min(times) * 1000),
            "max_latency_ms": float(np.max(times) * 1000),
            "p50_latency_ms": float(np.percentile(times, 50) * 1000),
            "p95_latency_ms": float(np.percentile(times, 95) * 1000),
            "p99_latency_ms": float(np.percentile(times, 99) * 1000),
            "throughput_samples_per_sec": float(1.0 / np.mean(times)),
            "mean_memory_mb": float(np.mean(memory_usage)),
            "max_memory_mb": float(np.max(memory_usage))
        }
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        sample_input: Dict[str, torch.Tensor],
        num_runs: int = 100
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compara múltiples modelos
        
        Args:
            models: Diccionario de modelos
            sample_input: Input de ejemplo
            num_runs: Número de runs
            
        Returns:
            Comparación de modelos
        """
        comparison = {}
        
        for name, model in models.items():
            logger.info(f"Benchmarking {name}...")
            comparison[name] = self.benchmark_model(model, sample_input, num_runs)
        
        return comparison




