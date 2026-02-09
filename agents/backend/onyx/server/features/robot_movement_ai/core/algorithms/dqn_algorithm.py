"""
DQN Algorithm - Deep Q-Network
================================

Implementación del algoritmo DQN para optimización de trayectorias.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any

from .base_algorithm import BaseOptimizationAlgorithm
from ..types.types import TrajectoryPoint

logger = logging.getLogger(__name__)


class DQNAlgorithm(BaseOptimizationAlgorithm):
    """
    Algoritmo DQN para optimización de trayectorias.
    
    DQN aprende una función de valor Q que estima la calidad de acciones.
    """
    
    def __init__(self):
        """Inicializar algoritmo DQN."""
        super().__init__("DQN")
    
    def optimize(
        self,
        start: TrajectoryPoint,
        goal: TrajectoryPoint,
        obstacles: Optional[List[np.ndarray]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[TrajectoryPoint]:
        """
        Optimizar usando DQN.
        
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
            raise ValueError("Invalid inputs for DQN algorithm")
        
        logger.debug("Applying DQN optimization")
        
        # Obtener trayectoria inicial
        initial_trajectory = kwargs.get("initial_trajectory")
        if initial_trajectory is None:
            from ..optimization.trajectory_optimizer import TrajectoryOptimizer
            optimizer = TrajectoryOptimizer()
            initial_trajectory = optimizer._generate_initial_trajectory(start, goal)
        
        optimized = []
        
        for i, point in enumerate(initial_trajectory):
            if i == 0 or i == len(initial_trajectory) - 1:
                optimized.append(point)
                continue
            
            # Calcular Q-value para diferentes acciones
            best_action = self._select_best_action_dqn(point, obstacles)
            
            # Aplicar mejor acción
            new_position = point.position + best_action[:3] * 0.01  # Pequeño paso
            new_orientation = point.orientation
            
            optimized_point = TrajectoryPoint(
                position=new_position,
                orientation=new_orientation,
                velocity=point.velocity,
                acceleration=point.acceleration,
                timestamp=point.timestamp
            )
            optimized.append(optimized_point)
        
        return optimized if optimized else initial_trajectory
    
    def _select_best_action_dqn(
        self,
        point: TrajectoryPoint,
        obstacles: Optional[List[np.ndarray]]
    ) -> np.ndarray:
        """Seleccionar mejor acción usando DQN."""
        # Generar acciones candidatas (direcciones)
        actions = [
            np.array([0.01, 0, 0]),
            np.array([-0.01, 0, 0]),
            np.array([0, 0.01, 0]),
            np.array([0, -0.01, 0]),
            np.array([0, 0, 0.01]),
            np.array([0, 0, -0.01]),
        ]
        
        # Calcular Q-value para cada acción
        best_action = actions[0]
        best_q = float('-inf')
        
        for action in actions:
            # Q-value simplificado basado en distancia a obstáculos
            q_value = 1.0
            
            if obstacles:
                new_pos = point.position + action
                for obstacle in obstacles:
                    obstacle_center = (obstacle[:3] + obstacle[3:]) / 2
                    dist = np.linalg.norm(new_pos - obstacle_center)
                    if dist < 0.1:
                        q_value -= 10.0  # Penalizar acercarse a obstáculos
                    else:
                        q_value += 1.0 / (dist + 0.1)  # Recompensar distancia
            
            if q_value > best_q:
                best_q = q_value
                best_action = action
        
        return best_action






