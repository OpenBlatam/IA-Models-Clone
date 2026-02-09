"""
Model Performance Predictor - Predictor de performance de modelos
===================================================================
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from .base_classes import BaseManager, BaseConfig
from .common_utils import (
    get_device, count_parameters, calculate_model_size,
    estimate_flops, measure_inference_time, create_dummy_input
)

logger = logging.getLogger(__name__)


@dataclass
class PerformancePrediction:
    """Predicción de performance"""
    predicted_accuracy: float
    predicted_latency_ms: float
    predicted_memory_mb: float
    confidence: float
    factors: Dict[str, float] = field(default_factory=dict)


class ModelPerformancePredictor(BaseManager):
    """Predictor de performance de modelos"""
    
    def __init__(self, config: Optional[BaseConfig] = None):
        super().__init__(config or BaseConfig())
        self.prediction_history: List[PerformancePrediction] = []
        # Modelos simples de predicción (en producción usar modelos ML)
        self.accuracy_model = None
        self.latency_model = None
        self.memory_model = None
    
    def predict_performance(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: Optional[str] = None
    ) -> PerformancePrediction:
        """Predice performance del modelo"""
        device_obj = get_device(device)
        
        # Extraer características del modelo usando utilidades compartidas
        features = self._extract_features(model, input_shape, str(device_obj))
        
        # Predecir accuracy (simplificado)
        predicted_accuracy = self._predict_accuracy(features)
        
        # Predecir latency usando medición real si es posible
        try:
            dummy_input = create_dummy_input(input_shape, str(device_obj))
            predicted_latency = measure_inference_time(
                model, dummy_input, num_runs=10, device=str(device_obj)
            )
        except Exception as e:
            logger.warning(f"Error midiendo latencia real, usando predicción: {e}")
            predicted_latency = self._predict_latency(features)
        
        # Predecir memoria usando cálculo real
        try:
            predicted_memory = calculate_model_size(model)
            # Agregar memoria de activaciones estimada
            input_size = np.prod(input_shape)
            activation_memory = (input_size * 4) / (1024 ** 2)  # MB
            predicted_memory += activation_memory * 2  # Forward + backward
        except Exception as e:
            logger.warning(f"Error calculando memoria real, usando predicción: {e}")
            predicted_memory = self._predict_memory(features)
        
        # Calcular confianza (simplificado)
        confidence = 0.7  # En producción se calcularía basado en datos históricos
        
        prediction = PerformancePrediction(
            predicted_accuracy=predicted_accuracy,
            predicted_latency_ms=predicted_latency,
            predicted_memory_mb=predicted_memory,
            confidence=confidence,
            factors=features
        )
        
        self.prediction_history.append(prediction)
        self.log_event("performance_prediction", {"model": model.__class__.__name__})
        return prediction
    
    def _extract_features(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str
    ) -> Dict[str, float]:
        """Extrae características del modelo usando utilidades compartidas"""
        # Usar utilidades compartidas para contar parámetros
        param_info = count_parameters(model)
        
        # Contar tipos de capas
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        num_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
        num_layers = len(list(model.modules()))
        
        # Estimar FLOPs usando utilidades compartidas
        try:
            flops = estimate_flops(model, input_shape, device)
        except Exception:
            flops = 0
        
        return {
            "total_params": float(param_info["total"]),
            "trainable_params": float(param_info["trainable"]),
            "num_linear": float(num_linear),
            "num_conv": float(num_conv),
            "num_layers": float(num_layers),
            "input_size": float(np.prod(input_shape)),
            "flops": float(flops)
        }
    
    def _predict_accuracy(self, features: Dict[str, float]) -> float:
        """Predice accuracy (modelo simplificado)"""
        # Modelo heurístico simple
        params = features["total_params"]
        
        # Más parámetros generalmente = mejor accuracy (hasta cierto punto)
        if params < 1e6:
            accuracy = 0.7 + (params / 1e6) * 0.2
        elif params < 1e7:
            accuracy = 0.9 + (params / 1e7) * 0.05
        else:
            accuracy = 0.95
        
        return min(0.99, max(0.5, accuracy))
    
    def _predict_latency(self, features: Dict[str, float]) -> float:
        """Predice latency (modelo simplificado)"""
        params = features["total_params"]
        input_size = features["input_size"]
        
        # Latencia estimada basada en parámetros e input size
        base_latency = 10.0  # ms
        param_factor = params / 1e6 * 0.1  # 0.1ms por millón de parámetros
        input_factor = input_size / 1e6 * 5.0  # 5ms por millón de elementos de input
        
        latency = base_latency + param_factor + input_factor
        return latency
    
    def _predict_memory(self, features: Dict[str, float]) -> float:
        """Predice uso de memoria"""
        params = features["total_params"]
        input_size = features["input_size"]
        
        # Memoria = parámetros + activaciones
        param_memory = (params * 4) / (1024 ** 2)  # MB (float32)
        activation_memory = (input_size * 4) / (1024 ** 2)  # MB
        
        total_memory = param_memory + activation_memory * 2  # Forward + backward
        return total_memory

