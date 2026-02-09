"""
Model Cost Estimator - Estimador de costos de modelos
=======================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .base_classes import BaseManager, BaseConfig
from .common_utils import calculate_model_size
from .constants import (
    TRAINING_COST_PER_GPU_HOUR, INFERENCE_COST_PER_1K,
    STORAGE_COST_PER_GB_MONTH
)

logger = logging.getLogger(__name__)


@dataclass
class CostEstimate:
    """Estimación de costos"""
    training_cost: float
    inference_cost_per_1k: float
    storage_cost_per_month: float
    total_cost: float
    breakdown: Dict[str, float] = field(default_factory=dict)


class ModelCostEstimator(BaseManager):
    """Estimador de costos de modelos"""
    
    def __init__(self, config: Optional[BaseConfig] = None):
        super().__init__(config or BaseConfig())
        # Precios por defecto (en USD) - usar constantes
        self.training_cost_per_hour = TRAINING_COST_PER_GPU_HOUR
        self.inference_cost_per_1k = INFERENCE_COST_PER_1K
        self.storage_cost_per_gb_month = STORAGE_COST_PER_GB_MONTH
        self.cost_estimates: List[CostEstimate] = []
    
    def estimate_costs(
        self,
        model: nn.Module,
        training_hours: float = 10.0,
        num_gpus: int = 1,
        expected_inferences_per_month: int = 1000000,
        device: str = "cuda"
    ) -> CostEstimate:
        """Estima costos del modelo"""
        # Costo de entrenamiento
        training_cost = training_hours * num_gpus * self.training_cost_per_hour
        
        # Tamaño del modelo usando utilidades compartidas
        model_size_mb = calculate_model_size(model)
        model_size_gb = model_size_mb / 1024.0
        
        # Costo de almacenamiento
        storage_cost = model_size_gb * self.storage_cost_per_gb_month
        
        # Costo de inferencia
        inference_cost = (expected_inferences_per_month / 1000) * self.inference_cost_per_1k
        
        # Costo total (primer mes)
        total_cost = training_cost + storage_cost + inference_cost
        
        breakdown = {
            "training": training_cost,
            "storage": storage_cost,
            "inference": inference_cost
        }
        
        estimate = CostEstimate(
            training_cost=training_cost,
            inference_cost_per_1k=self.inference_cost_per_1k,
            storage_cost_per_month=storage_cost,
            total_cost=total_cost,
            breakdown=breakdown
        )
        
        self.cost_estimates.append(estimate)
        self.log_event("cost_estimate", {"model_size_gb": model_size_gb})
        return estimate
    
    def estimate_monthly_cost(
        self,
        model: nn.Module,
        expected_inferences_per_month: int = 1000000
    ) -> float:
        """Estima costo mensual usando utilidades compartidas"""
        model_size_mb = calculate_model_size(model)
        model_size_gb = model_size_mb / 1024.0
        storage_cost = model_size_gb * self.storage_cost_per_gb_month
        inference_cost = (expected_inferences_per_month / 1000) * self.inference_cost_per_1k
        
        return storage_cost + inference_cost
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de costos"""
        if not self.cost_estimates:
            return {}
        
        latest = self.cost_estimates[-1]
        
        return {
            "training_cost": latest.training_cost,
            "storage_cost_per_month": latest.storage_cost_per_month,
            "inference_cost_per_1k": latest.inference_cost_per_1k,
            "total_cost": latest.total_cost,
            "breakdown": latest.breakdown
        }

