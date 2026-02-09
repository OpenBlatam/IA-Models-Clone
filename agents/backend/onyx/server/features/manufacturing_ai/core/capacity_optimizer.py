"""
Capacity Optimization System
=============================

Sistema de optimización de capacidad y recursos.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


@dataclass
class CapacityPlan:
    """Plan de capacidad."""
    plan_id: str
    resource_id: str
    current_capacity: float
    optimal_capacity: float
    utilization_rate: float
    recommendations: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ResourceAllocation:
    """Asignación de recursos."""
    allocation_id: str
    resource_id: str
    task_id: str
    allocated_capacity: float
    start_time: str
    end_time: str
    efficiency_score: float = 0.0


class CapacityOptimizerModel(nn.Module):
    """
    Modelo para optimización de capacidad.
    
    Usa MLP para predecir capacidad óptima.
    """
    
    def __init__(
        self,
        input_size: int = 10,  # features: load, demand, efficiency, etc.
        hidden_sizes: List[int] = [128, 64, 32],
        output_size: int = 1  # optimal capacity
    ):
        """
        Inicializar modelo.
        
        Args:
            input_size: Tamaño de entrada
            hidden_sizes: Tamaños de capas ocultas
            output_size: Tamaño de salida
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.ReLU())  # Capacidad no negativa
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada [batch, input_size]
            
        Returns:
            Capacidad óptima [batch, output_size]
        """
        return self.network(x)


class CapacityOptimizer:
    """
    Optimizador de capacidad.
    
    Optimiza asignación de recursos y capacidad.
    """
    
    def __init__(self):
        """Inicializar optimizador."""
        self.models: Dict[str, CapacityOptimizerModel] = {}
        self.plans: Dict[str, CapacityPlan] = {}
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
    
    def create_model(
        self,
        resource_id: str,
        input_size: int = 10
    ) -> str:
        """
        Crear modelo para recurso.
        
        Args:
            resource_id: ID del recurso
            input_size: Tamaño de entrada
            
        Returns:
            ID del modelo
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        model = CapacityOptimizerModel(input_size=input_size)
        model = model.to(self.device)
        self.models[resource_id] = model
        
        logger.info(f"Created capacity optimizer model for {resource_id}")
        return resource_id
    
    def optimize_capacity(
        self,
        resource_id: str,
        current_load: float,
        demand_forecast: float,
        efficiency: float,
        other_features: Optional[List[float]] = None
    ) -> CapacityPlan:
        """
        Optimizar capacidad.
        
        Args:
            resource_id: ID del recurso
            current_load: Carga actual
            demand_forecast: Pronóstico de demanda
            efficiency: Eficiencia actual
            other_features: Otras características (opcional)
            
        Returns:
            Plan de capacidad
        """
        if resource_id not in self.models:
            raise ValueError(f"Model not found for resource: {resource_id}")
        
        model = self.models[resource_id]
        model.eval()
        
        # Preparar features
        features = [
            current_load,
            demand_forecast,
            efficiency,
            current_load / max(demand_forecast, 0.001),  # ratio
            efficiency * current_load,  # effective capacity
        ]
        
        if other_features:
            features.extend(other_features[:5])  # Máximo 5 features adicionales
        
        # Rellenar hasta input_size
        while len(features) < 10:
            features.append(0.0)
        
        input_tensor = torch.FloatTensor([features]).to(self.device)
        
        with torch.no_grad():
            optimal_capacity = float(model(input_tensor).item())
        
        utilization_rate = current_load / max(optimal_capacity, 0.001)
        
        # Generar recomendaciones
        recommendations = []
        if utilization_rate > 0.9:
            recommendations.append("Consider increasing capacity - high utilization")
        elif utilization_rate < 0.5:
            recommendations.append("Consider reducing capacity - low utilization")
        
        if efficiency < 0.7:
            recommendations.append("Improve efficiency - current efficiency is low")
        
        plan = CapacityPlan(
            plan_id=str(uuid.uuid4()),
            resource_id=resource_id,
            current_capacity=current_load,
            optimal_capacity=optimal_capacity,
            utilization_rate=utilization_rate,
            recommendations=recommendations
        )
        
        self.plans[plan.plan_id] = plan
        logger.info(f"Optimized capacity for {resource_id}: {optimal_capacity:.2f}")
        
        return plan
    
    def allocate_resource(
        self,
        resource_id: str,
        task_id: str,
        required_capacity: float,
        duration: float
    ) -> ResourceAllocation:
        """
        Asignar recurso a tarea.
        
        Args:
            resource_id: ID del recurso
            task_id: ID de la tarea
            required_capacity: Capacidad requerida
            duration: Duración (horas)
            
        Returns:
            Asignación
        """
        allocation = ResourceAllocation(
            allocation_id=str(uuid.uuid4()),
            resource_id=resource_id,
            task_id=task_id,
            allocated_capacity=required_capacity,
            start_time=datetime.now().isoformat(),
            end_time=(datetime.now().timestamp() + duration * 3600).isoformat(),
            efficiency_score=0.85  # Simplificado
        )
        
        self.allocations[allocation.allocation_id] = allocation
        logger.info(f"Allocated resource {resource_id} to task {task_id}")
        
        return allocation
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        avg_utilization = sum(
            p.utilization_rate for p in self.plans.values()
        ) / len(self.plans) if self.plans else 0.0
        
        return {
            "total_models": len(self.models),
            "total_plans": len(self.plans),
            "total_allocations": len(self.allocations),
            "average_utilization": avg_utilization
        }


# Instancia global
_capacity_optimizer = None


def get_capacity_optimizer() -> CapacityOptimizer:
    """Obtener instancia global."""
    global _capacity_optimizer
    if _capacity_optimizer is None:
        _capacity_optimizer = CapacityOptimizer()
    return _capacity_optimizer

