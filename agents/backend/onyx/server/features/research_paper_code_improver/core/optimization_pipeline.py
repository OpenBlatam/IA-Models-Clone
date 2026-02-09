"""
Model Optimization Pipeline - Pipeline completo de optimización
=================================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationStep(Enum):
    """Pasos de optimización"""
    PRUNING = "pruning"
    QUANTIZATION = "quantization"
    DISTILLATION = "distillation"
    FINE_TUNING = "fine_tuning"
    COMPRESSION = "compression"


@dataclass
class OptimizationConfig:
    """Configuración de optimización"""
    steps: List[OptimizationStep] = field(default_factory=list)
    target_size_mb: Optional[float] = None
    target_latency_ms: Optional[float] = None
    target_accuracy: Optional[float] = None
    preserve_accuracy: bool = True


class ModelOptimizationPipeline:
    """Pipeline de optimización de modelos"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.optimization_history: List[Dict[str, Any]] = []
    
    def optimize(
        self,
        model: nn.Module,
        example_input: Optional[torch.Tensor] = None,
        eval_fn: Optional[Callable] = None
    ) -> nn.Module:
        """Ejecuta pipeline de optimización"""
        optimized_model = model
        initial_accuracy = None
        
        if eval_fn:
            initial_accuracy = eval_fn(model)
        
        for step in self.config.steps:
            logger.info(f"Ejecutando paso: {step.value}")
            
            if step == OptimizationStep.PRUNING:
                from .advanced_pruning import AdvancedPruner, PruningConfig, PruningMethod
                pruner = AdvancedPruner(PruningConfig(method=PruningMethod.MAGNITUDE, sparsity=0.3))
                optimized_model = pruner.prune_model(optimized_model, example_input)
            
            elif step == OptimizationStep.QUANTIZATION:
                from .advanced_quantization import AdvancedQuantizer, QuantizationConfig, QuantizationType
                quantizer = AdvancedQuantizer(QuantizationConfig(quantization_type=QuantizationType.DYNAMIC))
                optimized_model = quantizer.quantize_model(optimized_model, example_input)
            
            elif step == OptimizationStep.COMPRESSION:
                from .advanced_compression import AdvancedModelCompressor, CompressionTechnique
                compressor = AdvancedModelCompressor()
                optimized_model, result = compressor.compress_model(
                    optimized_model,
                    CompressionTechnique.PRUNING,
                    {"sparsity": 0.2},
                    example_input
                )
            
            # Verificar objetivos
            if self.config.target_size_mb:
                current_size = self._get_model_size(optimized_model)
                if current_size <= self.config.target_size_mb:
                    logger.info(f"Objetivo de tamaño alcanzado: {current_size:.2f}MB")
                    break
            
            if self.config.target_latency_ms and example_input is not None:
                current_latency = self._measure_latency(optimized_model, example_input)
                if current_latency <= self.config.target_latency_ms:
                    logger.info(f"Objetivo de latencia alcanzado: {current_latency:.2f}ms")
                    break
            
            if self.config.preserve_accuracy and eval_fn:
                current_accuracy = eval_fn(optimized_model)
                if initial_accuracy and current_accuracy < initial_accuracy * 0.95:  # 5% drop
                    logger.warning("Precisión cayó significativamente, deteniendo optimización")
                    break
        
        return optimized_model
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Obtiene tamaño del modelo"""
        total_params = sum(p.numel() for p in model.parameters())
        return (total_params * 4) / (1024 ** 2)  # MB
    
    def _measure_latency(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        num_runs: int = 10
    ) -> float:
        """Mide latencia"""
        import time
        device = next(model.parameters()).device
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(example_input.to(device))
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()
                _ = model(example_input.to(device))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append((time.time() - start) * 1000)
        
        return sum(times) / len(times)




