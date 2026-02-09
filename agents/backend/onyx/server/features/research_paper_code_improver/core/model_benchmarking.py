"""
Model Benchmarking Suite - Suite de benchmarking de modelos
=============================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from .common_utils import (
    get_device, move_to_device, count_parameters,
    estimate_flops, get_model_output
)
from .constants import DEFAULT_DEVICE

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Resultado de benchmark"""
    model_name: str
    latency_ms: float
    throughput_qps: float
    memory_mb: float
    flops: int
    parameters: int
    accuracy: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class ModelBenchmarkingSuite:
    """Suite de benchmarking"""
    
    def __init__(self):
        self.benchmark_results: List[BenchmarkResult] = []
    
    def benchmark_model(
        self,
        model: nn.Module,
        model_name: str,
        test_loader: Any,
        device: Optional[str] = None,
        num_warmup: int = 10,
        num_runs: int = 100
    ) -> BenchmarkResult:
        """Benchmark completo de modelo"""
        device_obj = get_device(device)
        model = model.to(device_obj)
        model.eval()
        
        # Contar parámetros usando utilidades compartidas
        param_info = count_parameters(model)
        total_params = param_info["total"]
        
        # Estimar FLOPs - obtener input shape del loader
        try:
            batch = next(iter(test_loader))
            if isinstance(batch, dict):
                inputs = batch.get("input_ids") or batch.get("inputs")
            else:
                inputs = batch[0] if isinstance(batch, tuple) else batch
            input_shape = tuple(inputs.shape)
            flops = estimate_flops(model, input_shape, str(device_obj))
        except Exception as e:
            logger.warning(f"Error estimando FLOPs: {e}")
            flops = 0
        
        # Benchmark latency
        latency_ms = self._benchmark_latency(model, test_loader, device, num_warmup, num_runs)
        
        # Benchmark throughput
        throughput_qps = 1000.0 / latency_ms if latency_ms > 0 else 0
        
        # Medir memoria
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            _ = self._run_inference(model, test_loader, device, num_runs=1)
            memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            memory_mb = 0.0
        
        result = BenchmarkResult(
            model_name=model_name,
            latency_ms=latency_ms,
            throughput_qps=throughput_qps,
            memory_mb=memory_mb,
            flops=flops,
            parameters=total_params
        )
        
        self.benchmark_results.append(result)
        return result
    
    def _benchmark_latency(
        self,
        model: nn.Module,
        test_loader: Any,
        device: torch.device,
        num_warmup: int,
        num_runs: int
    ) -> float:
        """Benchmark de latencia"""
        # Obtener ejemplo de input
        try:
            batch = next(iter(test_loader))
            if isinstance(batch, dict):
                example_input = batch.get("input_ids") or batch.get("inputs")
            else:
                example_input = batch[0] if isinstance(batch, tuple) else batch
            
            # Usar utilidades compartidas para medir latencia
            from .common_utils import measure_inference_time
            return measure_inference_time(
                model, example_input, num_runs=num_runs, warmup=num_warmup, device=str(device)
            )
        except Exception as e:
            logger.warning(f"Error en benchmark de latencia: {e}")
            return 0.0
    
    def _run_inference(
        self,
        model: nn.Module,
        test_loader: Any,
        device: torch.device,
        num_runs: int = 1
    ):
        """Ejecuta inferencia"""
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= num_runs:
                    break
                
                if isinstance(batch, dict):
                    inputs = batch.get("input_ids") or batch.get("inputs")
                    inputs = inputs.to(device)
                    _ = model(**{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                               for k, v in batch.items()})
                else:
                    inputs = batch[0] if isinstance(batch, tuple) else batch
                    inputs = inputs.to(device)
                    _ = model(inputs)
    
    # _estimate_flops removido - ahora usa estimate_flops de common_utils
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        test_loader: Any,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compara múltiples modelos"""
        results = {}
        
        for name, model in models.items():
            result = self.benchmark_model(model, name, test_loader, device)
            results[name] = {
                "latency_ms": result.latency_ms,
                "throughput_qps": result.throughput_qps,
                "memory_mb": result.memory_mb,
                "parameters": result.parameters,
                "flops": result.flops
            }
        
        return results
    
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de benchmarks"""
        if not self.benchmark_results:
            return {}
        
        return {
            "total_benchmarks": len(self.benchmark_results),
            "results": [
                {
                    "model_name": r.model_name,
                    "latency_ms": r.latency_ms,
                    "throughput_qps": r.throughput_qps,
                    "memory_mb": r.memory_mb,
                    "parameters": r.parameters
                }
                for r in self.benchmark_results
            ]
        }

