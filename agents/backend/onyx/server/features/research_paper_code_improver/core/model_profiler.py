"""
Model Profiler - Profiler de modelos para análisis de performance
===================================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from .common_utils import (
    get_device, move_to_device, count_parameters,
    estimate_flops, measure_inference_time, create_dummy_input
)
from .constants import DEFAULT_DEVICE

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Resultado de profiling"""
    layer_name: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    parameters: int
    flops: int
    forward_time: float
    backward_time: Optional[float] = None
    memory_allocated: int = 0
    memory_reserved: int = 0


class ModelProfiler:
    """Profiler de modelos"""
    
    def __init__(self, device: Optional[str] = None):
        self.device = get_device(device)
        self.profile_results: List[ProfileResult] = []
    
    def profile_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_runs: int = 10
    ) -> Dict[str, Any]:
        """Profila un modelo completo"""
        model = model.to(self.device)
        model.eval()
        
        # Crear input dummy
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)
        
        # Profile forward usando utilidades compartidas
        avg_forward_time = measure_inference_time(
            model, dummy_input, num_runs=num_runs, warmup=3, device=str(self.device)
        ) / 1000.0  # Convertir de ms a s
        
        # Calcular parámetros y FLOPs usando utilidades compartidas
        param_info = count_parameters(model)
        total_params = param_info["total"]
        trainable_params = param_info["trainable"]
        
        # Estimar FLOPs usando utilidades compartidas
        flops = estimate_flops(model, input_shape, str(self.device))
        
        # Memory
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device)
            memory_reserved = torch.cuda.memory_reserved(self.device)
        else:
            memory_allocated = 0
            memory_reserved = 0
        
        result = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "estimated_flops": flops,
            "average_forward_time_ms": avg_forward_time * 1000,
            "throughput_samples_per_sec": 1.0 / avg_forward_time if avg_forward_time > 0 else 0,
            "memory_allocated_mb": memory_allocated / 1024**2,
            "memory_reserved_mb": memory_reserved / 1024**2,
            "input_shape": input_shape,
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        }
        
        self.profile_results.append(ProfileResult(
            layer_name="model",
            input_shape=input_shape,
            output_shape=input_shape,  # Simplified
            parameters=total_params,
            flops=flops,
            forward_time=avg_forward_time,
            memory_allocated=memory_allocated,
            memory_reserved=memory_reserved
        ))
        
        return result
    
    # _estimate_flops removido - ahora usa estimate_flops de common_utils
    
    def profile_layer(
        self,
        layer: nn.Module,
        input_shape: Tuple[int, ...],
        num_runs: int = 100
    ) -> ProfileResult:
        """Profila una capa individual"""
        layer = layer.to(self.device)
        layer.eval()
        
        dummy_input = create_dummy_input(input_shape, str(self.device))
        
        # Profile usando utilidades compartidas
        avg_time = measure_inference_time(
            layer, dummy_input, num_runs=num_runs, warmup=3, device=str(self.device)
        ) / 1000.0  # Convertir de ms a s
        
        with torch.no_grad():
            output = layer(dummy_input)
        
        result = ProfileResult(
            layer_name=layer.__class__.__name__,
            input_shape=input_shape,
            output_shape=output.shape if hasattr(output, 'shape') else (),
            parameters=sum(p.numel() for p in layer.parameters()),
            flops=0,  # Simplified
            forward_time=avg_time
        )
        
        return result
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Obtiene resumen del profiling"""
        if not self.profile_results:
            return {}
        
        total_params = sum(r.parameters for r in self.profile_results)
        total_flops = sum(r.flops for r in self.profile_results)
        total_time = sum(r.forward_time for r in self.profile_results)
        
        return {
            "total_layers_profiled": len(self.profile_results),
            "total_parameters": total_params,
            "total_flops": total_flops,
            "total_forward_time_ms": total_time * 1000,
            "layers": [
                {
                    "name": r.layer_name,
                    "parameters": r.parameters,
                    "forward_time_ms": r.forward_time * 1000
                }
                for r in self.profile_results
            ]
        }

