"""
Profiling avanzado con torch.profiler

Mejoras:
- Profiling detallado de operaciones
- Análisis de memoria
- Análisis de tiempo
- Exportación a Chrome trace
- Recomendaciones de optimización
"""

import logging
import torch
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class AdvancedProfiler:
    """
    Profiler avanzado para análisis de performance
    
    Features:
    - CPU/GPU profiling
    - Memory profiling
    - Operator-level profiling
    - Export to Chrome trace
    - Optimization recommendations
    """
    
    def __init__(
        self,
        activities: Optional[List] = None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = True
    ):
        """
        Inicializa profiler avanzado
        
        Args:
            activities: Lista de actividades a perfilar
            record_shapes: Si registrar shapes de tensores
            profile_memory: Si perfilar memoria
            with_stack: Si incluir stack traces
            with_flops: Si calcular FLOPS
        """
        if activities is None:
            activities = [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ] if torch.cuda.is_available() else [
                torch.profiler.ProfilerActivity.CPU
            ]
        
        self.activities = activities
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        
        self.profiler: Optional[torch.profiler.profile] = None
        self.profiling_results: Optional[Dict[str, Any]] = None
    
    @contextmanager
    def profile(
        self,
        schedule: Optional[Callable] = None,
        on_trace_ready: Optional[Callable] = None
    ):
        """
        Context manager para profiling
        
        Args:
            schedule: Schedule function para profiling
            on_trace_ready: Callback cuando trace está listo
        """
        if schedule is None:
            schedule = torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=1
            )
        
        self.profiler = torch.profiler.profile(
            activities=self.activities,
            schedule=schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops
        )
        
        self.profiler.start()
        try:
            yield self.profiler
        finally:
            self.profiler.stop()
            self._analyze_results()
    
    def _analyze_results(self):
        """Analiza resultados del profiling"""
        if not self.profiler:
            return
        
        try:
            # Key metrics
            key_metrics = self.profiler.key_averages()
            
            # CPU time
            cpu_time_total = sum(
                event.self_cpu_time_total
                for event in key_metrics
            )
            
            # CUDA time
            cuda_time_total = sum(
                event.self_cuda_time_total
                for event in key_metrics
            ) if torch.cuda.is_available() else 0
            
            # Memory
            cpu_memory_total = sum(
                event.self_cpu_memory_usage
                for event in key_metrics
            )
            
            cuda_memory_total = sum(
                event.self_cuda_memory_usage
                for event in key_metrics
            ) if torch.cuda.is_available() else 0
            
            # Top operations
            top_cpu_ops = sorted(
                key_metrics,
                key=lambda x: x.self_cpu_time_total,
                reverse=True
            )[:10]
            
            top_cuda_ops = sorted(
                [e for e in key_metrics if e.self_cuda_time_total > 0],
                key=lambda x: x.self_cuda_time_total,
                reverse=True
            )[:10] if torch.cuda.is_available() else []
            
            self.profiling_results = {
                "cpu_time_total_us": cpu_time_total,
                "cuda_time_total_us": cuda_time_total,
                "cpu_memory_total_bytes": cpu_memory_total,
                "cuda_memory_total_bytes": cuda_memory_total,
                "top_cpu_operations": [
                    {
                        "name": op.key,
                        "cpu_time_us": op.self_cpu_time_total,
                        "count": op.count
                    }
                    for op in top_cpu_ops
                ],
                "top_cuda_operations": [
                    {
                        "name": op.key,
                        "cuda_time_us": op.self_cuda_time_total,
                        "count": op.count
                    }
                    for op in top_cuda_ops
                ]
            }
            
        except Exception as e:
            logger.error(f"Error analizando resultados: {e}", exc_info=True)
    
    def export_chrome_trace(self, path: str):
        """
        Exporta trace a formato Chrome
        
        Args:
            path: Path donde guardar el trace
        """
        if not self.profiler:
            raise ValueError("No hay profiling activo")
        
        try:
            self.profiler.export_chrome_trace(path)
            logger.info(f"Chrome trace exportado a {path}")
        except Exception as e:
            logger.error(f"Error exportando trace: {e}", exc_info=True)
    
    def export_stacks(self, path: str):
        """
        Exporta stack traces
        
        Args:
            path: Path donde guardar stacks
        """
        if not self.profiler:
            raise ValueError("No hay profiling activo")
        
        try:
            self.profiler.export_stacks(path)
            logger.info(f"Stack traces exportados a {path}")
        except Exception as e:
            logger.error(f"Error exportando stacks: {e}", exc_info=True)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del profiling
        
        Returns:
            Diccionario con resumen
        """
        if not self.profiling_results:
            return {}
        
        return {
            "summary": {
                "cpu_time_ms": self.profiling_results["cpu_time_total_us"] / 1000,
                "cuda_time_ms": self.profiling_results["cuda_time_total_us"] / 1000,
                "cpu_memory_mb": self.profiling_results["cpu_memory_total_bytes"] / (1024 ** 2),
                "cuda_memory_mb": self.profiling_results["cuda_memory_total_bytes"] / (1024 ** 2)
            },
            "top_operations": {
                "cpu": self.profiling_results["top_cpu_operations"],
                "cuda": self.profiling_results["top_cuda_operations"]
            }
        }
    
    def get_optimization_recommendations(self) -> List[str]:
        """
        Obtiene recomendaciones de optimización
        
        Returns:
            Lista de recomendaciones
        """
        if not self.profiling_results:
            return []
        
        recommendations = []
        
        # Analizar top operations
        top_cpu = self.profiling_results["top_cpu_operations"]
        top_cuda = self.profiling_results["top_cuda_operations"]
        
        # CPU bottlenecks
        if top_cpu:
            cpu_bottleneck = top_cpu[0]
            if cpu_bottleneck["cpu_time_us"] > 1000000:  # > 1s
                recommendations.append(
                    f"CPU bottleneck detectado: {cpu_bottleneck['name']} "
                    f"({cpu_bottleneck['cpu_time_us']/1000:.2f}ms). "
                    "Considera mover a GPU o optimizar."
                )
        
        # CUDA utilization
        if top_cuda:
            cuda_time = self.profiling_results["cuda_time_total_us"]
            cpu_time = self.profiling_results["cpu_time_total_us"]
            
            if cpu_time > 0:
                cuda_utilization = cuda_time / (cpu_time + cuda_time)
                if cuda_utilization < 0.5:
                    recommendations.append(
                        f"Baja utilización de GPU ({cuda_utilization*100:.1f}%). "
                        "Considera aumentar batch size o usar más operaciones en GPU."
                    )
        
        # Memory recommendations
        cuda_memory = self.profiling_results["cuda_memory_total_bytes"]
        if cuda_memory > 0:
            memory_gb = cuda_memory / (1024 ** 3)
            if memory_gb > 8:
                recommendations.append(
                    f"Alto uso de memoria GPU ({memory_gb:.2f}GB). "
                    "Considera reducir batch size o usar gradient checkpointing."
                )
        
        return recommendations


def profile_function(
    func: Callable,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """
    Profila una función
    
    Args:
        func: Función a perfilar
        *args: Argumentos para la función
        **kwargs: Keyword arguments para la función
        
    Returns:
        Resultados del profiling
    """
    profiler = AdvancedProfiler()
    
    with profiler.profile():
        result = func(*args, **kwargs)
    
    return {
        "result": result,
        "profiling": profiler.get_summary(),
        "recommendations": profiler.get_optimization_recommendations()
    }




