"""
Production Route Optimizer
===========================

Optimizador de rutas de producción usando deep learning.
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
class ProductionStep:
    """Paso de producción."""
    step_id: str
    name: str
    machine_id: str
    duration: float
    dependencies: List[str] = field(default_factory=list)
    position: Tuple[float, float] = (0.0, 0.0)  # x, y coordinates


@dataclass
class ProductionRoute:
    """Ruta de producción."""
    route_id: str
    steps: List[ProductionStep]
    total_duration: float
    total_distance: float
    efficiency: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class RouteOptimizerModel(nn.Module):
    """
    Modelo para optimización de rutas.
    
    Usa Graph Neural Network para optimizar secuencias de pasos.
    """
    
    def __init__(
        self,
        node_features: int = 5,  # duration, x, y, dependencies_count, machine_load
        hidden_size: int = 64,
        num_layers: int = 3
    ):
        """
        Inicializar modelo.
        
        Args:
            node_features: Features por nodo
            hidden_size: Tamaño oculto
            num_layers: Número de capas
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        super().__init__()
        
        # Graph convolution layers (simplificado como MLP)
        layers = []
        prev_size = node_features
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size)
            ])
            prev_size = hidden_size
        
        self.graph_layers = nn.Sequential(*layers)
        
        # Output: score para cada paso
        self.output_layer = nn.Linear(hidden_size, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            node_features: Features de nodos [num_nodes, node_features]
            
        Returns:
            Scores [num_nodes, 1]
        """
        # Graph processing
        x = self.graph_layers(node_features)
        
        # Output scores
        scores = self.output_layer(x)
        
        return scores


class ProductionRouteOptimizer:
    """
    Optimizador de rutas de producción.
    
    Optimiza secuencias de pasos de producción.
    """
    
    def __init__(self):
        """Inicializar optimizador."""
        self.routes: Dict[str, ProductionRoute] = {}
        self.steps: Dict[str, ProductionStep] = {}
        self.models: Dict[str, RouteOptimizerModel] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
    
    def register_step(
        self,
        step_id: str,
        name: str,
        machine_id: str,
        duration: float,
        dependencies: Optional[List[str]] = None,
        position: Optional[Tuple[float, float]] = None
    ):
        """
        Registrar paso de producción.
        
        Args:
            step_id: ID del paso
            name: Nombre
            machine_id: ID de máquina
            duration: Duración (horas)
            dependencies: Dependencias (opcional)
            position: Posición (x, y) (opcional)
        """
        step = ProductionStep(
            step_id=step_id,
            name=name,
            machine_id=machine_id,
            duration=duration,
            dependencies=dependencies or [],
            position=position or (0.0, 0.0)
        )
        
        self.steps[step_id] = step
        logger.info(f"Registered production step: {step_id}")
    
    def create_route(
        self,
        step_ids: List[str],
        optimize: bool = True
    ) -> ProductionRoute:
        """
        Crear ruta de producción.
        
        Args:
            step_ids: IDs de pasos
            optimize: Optimizar orden (opcional)
            
        Returns:
            Ruta de producción
        """
        # Validar pasos
        for step_id in step_ids:
            if step_id not in self.steps:
                raise ValueError(f"Step not found: {step_id}")
        
        # Optimizar si se solicita
        if optimize:
            step_ids = self._optimize_sequence(step_ids)
        
        # Crear ruta
        steps = [self.steps[sid] for sid in step_ids]
        
        # Calcular duración total
        total_duration = sum(step.duration for step in steps)
        
        # Calcular distancia total
        total_distance = 0.0
        for i in range(len(steps) - 1):
            pos1 = steps[i].position
            pos2 = steps[i + 1].position
            distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            total_distance += distance
        
        # Calcular eficiencia
        efficiency = 1.0 / (1.0 + total_distance * 0.1)  # Simplificado
        
        route = ProductionRoute(
            route_id=str(uuid.uuid4()),
            steps=steps,
            total_duration=total_duration,
            total_distance=total_distance,
            efficiency=efficiency
        )
        
        self.routes[route.route_id] = route
        logger.info(f"Created production route: {route.route_id}")
        
        return route
    
    def _optimize_sequence(self, step_ids: List[str]) -> List[str]:
        """
        Optimizar secuencia de pasos.
        
        Args:
            step_ids: IDs de pasos
            
        Returns:
            Secuencia optimizada
        """
        # Respetar dependencias
        ordered = []
        remaining = set(step_ids)
        
        while remaining:
            # Encontrar pasos sin dependencias pendientes
            available = [
                sid for sid in remaining
                if all(dep in ordered for dep in self.steps[sid].dependencies)
            ]
            
            if not available:
                # Si hay dependencias circulares, usar orden original
                ordered.extend(list(remaining))
                break
            
            # Ordenar por posición (minimizar distancia)
            if ordered:
                last_pos = self.steps[ordered[-1]].position
                available.sort(
                    key=lambda sid: np.sqrt(
                        (self.steps[sid].position[0] - last_pos[0])**2 +
                        (self.steps[sid].position[1] - last_pos[1])**2
                    )
                )
            
            ordered.append(available[0])
            remaining.remove(available[0])
        
        return ordered
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "total_steps": len(self.steps),
            "total_routes": len(self.routes),
            "avg_route_duration": sum(r.total_duration for r in self.routes.values()) / len(self.routes) if self.routes else 0.0,
            "avg_route_efficiency": sum(r.efficiency for r in self.routes.values()) / len(self.routes) if self.routes else 0.0
        }


# Instancia global
_production_route_optimizer = None


def get_production_route_optimizer() -> ProductionRouteOptimizer:
    """Obtener instancia global."""
    global _production_route_optimizer
    if _production_route_optimizer is None:
        _production_route_optimizer = ProductionRouteOptimizer()
    return _production_route_optimizer

