"""
RRT Algorithm - Rapidly-exploring Random Tree
==============================================

Implementación del algoritmo RRT para optimización de trayectorias.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any

from .base_algorithm import BaseOptimizationAlgorithm
from ..types.types import TrajectoryPoint
from ..constants import (
    DEFAULT_RRT_MAX_ITERATIONS,
    DEFAULT_RRT_STEP_SIZE,
    RRT_GOAL_BIAS
)
from ..utils.quaternion_utils import quaternion_slerp
from ..utils.math_utils import point_in_obstacle

logger = logging.getLogger(__name__)


class RRTAlgorithm(BaseOptimizationAlgorithm):
    """
    Algoritmo RRT para optimización de trayectorias.
    
    RRT es útil para espacios con muchos obstáculos.
    """
    
    def __init__(
        self,
        max_iterations: int = DEFAULT_RRT_MAX_ITERATIONS,
        step_size: float = DEFAULT_RRT_STEP_SIZE
    ):
        """
        Inicializar algoritmo RRT.
        
        Args:
            max_iterations: Máximo de iteraciones
            step_size: Tamaño del paso
        """
        super().__init__("RRT")
        self.max_iterations = max_iterations
        self.step_size = step_size
    
    def optimize(
        self,
        start: TrajectoryPoint,
        goal: TrajectoryPoint,
        obstacles: Optional[List[np.ndarray]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[TrajectoryPoint]:
        """
        Optimizar usando RRT.
        
        Args:
            start: Punto inicial
            goal: Punto objetivo
            obstacles: Obstáculos
            constraints: Restricciones
            **kwargs: Parámetros adicionales (max_iterations, step_size pueden sobrescribir)
            
        Returns:
            Trayectoria optimizada
        """
        if not self.validate_inputs(start, goal):
            raise ValueError("Invalid inputs for RRT algorithm")
        
        logger.debug("Using RRT algorithm for trajectory optimization")
        
        # Permitir sobrescribir parámetros
        max_iterations = kwargs.get("max_iterations", self.max_iterations)
        step_size = kwargs.get("step_size", self.step_size)
        
        # Árbol RRT
        tree = {tuple(start.position): None}  # {position: parent_position}
        nodes = [start.position]
        
        for _ in range(max_iterations):
            # Generar punto aleatorio (con bias hacia el objetivo)
            if np.random.random() < RRT_GOAL_BIAS:
                random_point = goal.position
            else:
                # Punto aleatorio en el espacio
                bounds = np.array([
                    np.minimum(start.position, goal.position) - 0.5,
                    np.maximum(start.position, goal.position) + 0.5
                ])
                random_point = np.random.uniform(bounds[0], bounds[1])
            
            # Encontrar nodo más cercano
            nearest_node = min(nodes, key=lambda n: np.linalg.norm(n - random_point))
            
            # Extender hacia el punto aleatorio
            direction = random_point - nearest_node
            distance = np.linalg.norm(direction)
            if distance > step_size:
                direction = direction / distance * step_size
            
            new_node = nearest_node + direction
            
            # Verificar colisión
            if obstacles and any(point_in_obstacle(new_node, obs) for obs in obstacles):
                continue
            
            # Agregar al árbol
            tree[tuple(new_node)] = tuple(nearest_node)
            nodes.append(new_node)
            
            # Verificar si llegamos cerca del objetivo
            if np.linalg.norm(new_node - goal.position) < step_size:
                # Reconstruir camino
                path = []
                node = tuple(new_node)
                while node is not None:
                    path.append(np.array(node))
                    node = tree.get(node)
                path.reverse()
                
                # Agregar objetivo
                path.append(goal.position)
                
                # Convertir a trayectoria
                trajectory = []
                for i, pos in enumerate(path):
                    alpha = i / max(len(path) - 1, 1)
                    orientation = quaternion_slerp(start.orientation, goal.orientation, alpha)
                    trajectory.append(TrajectoryPoint(
                        position=pos,
                        orientation=orientation,
                        timestamp=i * 0.01
                    ))
                
                return trajectory
        
        # No se encontró camino, usar optimización estándar
        logger.warning("RRT could not find path, using direct path")
        return [start, goal]

