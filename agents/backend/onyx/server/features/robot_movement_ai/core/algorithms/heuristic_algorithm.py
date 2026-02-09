"""
Heuristic Algorithm
===================

Algoritmo heurístico para optimización de trayectorias (fallback).
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any

from .base_algorithm import BaseOptimizationAlgorithm
from ..types.types import TrajectoryPoint
from ..utils.trajectory_utils import smooth_trajectory
from ..utils.math_utils import point_in_obstacle, normalize_vector
from ..constants import MIN_OBSTACLE_DISTANCE, SAFE_OBSTACLE_DISTANCE

logger = logging.getLogger(__name__)


class HeuristicAlgorithm(BaseOptimizationAlgorithm):
    """
    Algoritmo heurístico para optimización de trayectorias.
    
    Usa métodos heurísticos cuando no hay modelo RL disponible.
    """
    
    def __init__(self):
        """Inicializar algoritmo heurístico."""
        super().__init__("Heuristic")
    
    def optimize(
        self,
        start: TrajectoryPoint,
        goal: TrajectoryPoint,
        obstacles: Optional[List[np.ndarray]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[TrajectoryPoint]:
        """
        Optimizar usando métodos heurísticos.
        
        Args:
            start: Punto inicial
            goal: Punto objetivo
            obstacles: Obstáculos
            constraints: Restricciones
            **kwargs: Parámetros adicionales
            
        Returns:
            Trayectoria optimizada
        """
        if not self.validate_inputs(start, goal):
            raise ValueError("Invalid inputs for Heuristic algorithm")
        
        logger.debug("Applying heuristic optimization")
        
        # Obtener trayectoria inicial
        initial_trajectory = kwargs.get("initial_trajectory")
        if initial_trajectory is None:
            from ..optimization.trajectory_optimizer import TrajectoryOptimizer
            optimizer = TrajectoryOptimizer()
            initial_trajectory = optimizer._generate_initial_trajectory(start, goal)
        
        # Suavizar trayectoria usando utilidad
        trajectory = smooth_trajectory(initial_trajectory)
        
        # Evitar obstáculos
        if obstacles:
            trajectory = self._avoid_obstacles(trajectory, obstacles)
        
        return trajectory
    
    def _avoid_obstacles(
        self,
        trajectory: List[TrajectoryPoint],
        obstacles: List[np.ndarray]
    ) -> List[TrajectoryPoint]:
        """Modificar trayectoria para evitar obstáculos."""
        adjusted = []
        
        for point in trajectory:
            adjusted_point = point
            
            for obstacle in obstacles:
                # Calcular distancia al obstáculo
                obstacle_center = (obstacle[:3] + obstacle[3:]) / 2
                distance = np.linalg.norm(point.position - obstacle_center)
                
                # Si está muy cerca, ajustar posición
                if distance < MIN_OBSTACLE_DISTANCE:
                    # Mover punto lejos del obstáculo
                    direction = point.position - obstacle_center
                    direction = normalize_vector(direction)
                    adjusted_point = TrajectoryPoint(
                        position=point.position + direction * SAFE_OBSTACLE_DISTANCE,
                        orientation=point.orientation,
                        velocity=point.velocity,
                        acceleration=point.acceleration,
                        timestamp=point.timestamp
                    )
                    break  # Solo ajustar una vez por punto
            
            adjusted.append(adjusted_point)
        
        return adjusted

