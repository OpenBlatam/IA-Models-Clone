"""
Model Efficiency Analyzer - Analizador de eficiencia de modelos
=================================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .common_utils import (
    get_device, move_to_device, count_parameters,
    calculate_model_size, estimate_flops, measure_inference_time
)
from .constants import DEFAULT_DEVICE

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyMetrics:
    """Métricas de eficiencia"""
    model_size_mb: float
    inference_time_ms: float
    throughput_qps: float
    memory_usage_mb: float
    flops: int
    parameters: int
    efficiency_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class ModelEfficiencyAnalyzer:
    """Analizador de eficiencia de modelos"""
    
    def __init__(self):
        self.efficiency_results: List[EfficiencyMetrics] = []
    
    def analyze_efficiency(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        device: Optional[str] = None,
        num_runs: int = 100
    ) -> EfficiencyMetrics:
        """Analiza eficiencia del modelo"""
        device_obj = get_device(device)
        model = model.to(device_obj)
        model.eval()
        
        # Tamaño del modelo usando utilidades compartidas
        model_size_mb = calculate_model_size(model)
        
        # Parámetros usando utilidades compartidas
        param_info = count_parameters(model)
        total_params = param_info["total"]
        
        # FLOPs usando utilidades compartidas
        input_shape = tuple(example_input.shape)
        flops = estimate_flops(model, input_shape, str(device_obj))
        
        # Latencia usando utilidades compartidas
        inference_time_ms = measure_inference_time(
            model, example_input, num_runs=num_runs, device=str(device_obj)
        )
        
        # Throughput
        throughput_qps = 1000.0 / inference_time_ms if inference_time_ms > 0 else 0
        
        # Memoria
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                _ = model(example_input.to(device))
            memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            memory_mb = 0.0
        
        # Efficiency score (normalizado)
        efficiency_score = self._calculate_efficiency_score(
            model_size_mb, inference_time_ms, throughput_qps, memory_mb
        )
        
        metrics = EfficiencyMetrics(
            model_size_mb=model_size_mb,
            inference_time_ms=inference_time_ms,
            throughput_qps=throughput_qps,
            memory_usage_mb=memory_mb,
            flops=flops,
            parameters=total_params,
            efficiency_score=efficiency_score
        )
        
        self.efficiency_results.append(metrics)
        return metrics
    
    # Métodos removidos - ahora usan utilidades compartidas
    
    def _calculate_efficiency_score(
        self,
        model_size_mb: float,
        inference_time_ms: float,
        throughput_qps: float,
        memory_mb: float
    ) -> float:
        """Calcula score de eficiencia (0-1, mayor es mejor)"""
        # Normalizar métricas (inversas para tamaño, tiempo, memoria)
        size_score = 1.0 / (1.0 + model_size_mb / 100.0)  # Normalizar a ~100MB
        time_score = 1.0 / (1.0 + inference_time_ms / 100.0)  # Normalizar a ~100ms
        throughput_score = min(1.0, throughput_qps / 100.0)  # Normalizar a ~100 QPS
        memory_score = 1.0 / (1.0 + memory_mb / 1000.0)  # Normalizar a ~1GB
        
        # Score combinado (promedio ponderado)
        efficiency_score = (
            0.2 * size_score +
            0.3 * time_score +
            0.3 * throughput_score +
            0.2 * memory_score
        )
        
        return efficiency_score
    
    def get_efficiency_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de eficiencia"""
        if not self.efficiency_results:
            return {}
        
        latest = self.efficiency_results[-1]
        
        return {
            "model_size_mb": latest.model_size_mb,
            "inference_time_ms": latest.inference_time_ms,
            "throughput_qps": latest.throughput_qps,
            "memory_usage_mb": latest.memory_usage_mb,
            "efficiency_score": latest.efficiency_score,
            "parameters": latest.parameters,
            "flops": latest.flops
        }

