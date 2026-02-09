"""
Model Serving Optimizer - Optimizador de modelos para producción
==================================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Estrategias de optimización"""
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    FUSION = "fusion"
    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


@dataclass
class ServingConfig:
    """Configuración de serving"""
    batch_size: int = 1
    max_batch_size: int = 32
    timeout: float = 30.0
    max_queue_size: int = 100
    num_workers: int = 1
    use_batching: bool = True
    use_caching: bool = True


class ModelServingOptimizer:
    """Optimizador de modelos para producción"""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self.optimization_history: List[Dict[str, Any]] = []
    
    def optimize_for_serving(
        self,
        model: nn.Module,
        strategy: OptimizationStrategy,
        example_input: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """Optimiza modelo para serving"""
        if strategy == OptimizationStrategy.QUANTIZATION:
            return self._quantize_model(model)
        elif strategy == OptimizationStrategy.PRUNING:
            return self._prune_model(model)
        elif strategy == OptimizationStrategy.FUSION:
            return self._fuse_model(model)
        elif strategy == OptimizationStrategy.TORCHSCRIPT:
            return self._torchscript_model(model, example_input)
        else:
            raise ValueError(f"Estrategia {strategy} no soportada")
    
    def _quantize_model(self, model: nn.Module) -> nn.Module:
        """Cuantiza modelo"""
        try:
            model.eval()
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            logger.info("Modelo cuantizado exitosamente")
            return quantized_model
        except Exception as e:
            logger.warning(f"Error en cuantización: {e}")
            return model
    
    def _prune_model(self, model: nn.Module) -> nn.Module:
        """Poda modelo"""
        # Pruning simplificado
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Pruning de 20% de conexiones
                weight = module.weight.data
                threshold = torch.quantile(torch.abs(weight), 0.2)
                mask = torch.abs(weight) > threshold
                module.weight.data = weight * mask.float()
        
        logger.info("Modelo podado exitosamente")
        return model
    
    def _fuse_model(self, model: nn.Module) -> nn.Module:
        """Fusiona capas del modelo"""
        try:
            model.eval()
            fused_model = torch.quantization.fuse_modules(
                model,
                [['conv', 'bn', 'relu']]  # Ejemplo
            )
            logger.info("Modelo fusionado exitosamente")
            return fused_model
        except Exception as e:
            logger.warning(f"Error en fusión: {e}")
            return model
    
    def _torchscript_model(
        self,
        model: nn.Module,
        example_input: Optional[torch.Tensor]
    ) -> torch.jit.ScriptModule:
        """Convierte a TorchScript"""
        if example_input is None:
            raise ValueError("Se requiere example_input para TorchScript")
        
        model.eval()
        traced_model = torch.jit.trace(model, example_input)
        logger.info("Modelo convertido a TorchScript")
        return traced_model
    
    def benchmark_serving(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark de serving"""
        model.eval()
        device = next(model.parameters()).device
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(example_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()
                _ = model(example_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        throughput = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            "avg_inference_time_ms": avg_time * 1000,
            "throughput_qps": throughput,
            "p50_ms": sorted(times)[len(times) // 2] * 1000,
            "p95_ms": sorted(times)[int(len(times) * 0.95)] * 1000,
            "p99_ms": sorted(times)[int(len(times) * 0.99)] * 1000
        }




