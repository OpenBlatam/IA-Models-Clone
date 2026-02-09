"""
PPO Algorithm - Proximal Policy Optimization
=============================================

Implementación del algoritmo PPO para optimización de trayectorias.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any

from .base_algorithm import BaseOptimizationAlgorithm
from ..types.types import TrajectoryPoint
from ..constants import PPO_CLIP_RATIO, DEFAULT_LEARNING_RATE
from ..utils.math_utils import normalize_vector

logger = logging.getLogger(__name__)


class PPOAlgorithm(BaseOptimizationAlgorithm):
    """
    Algoritmo PPO para optimización de trayectorias.
    
    PPO es un algoritmo de política que optimiza directamente
    la política de movimiento del robot.
    """
    
    def __init__(self, learning_rate: float = DEFAULT_LEARNING_RATE):
        """
        Inicializar algoritmo PPO.
        
        Args:
            learning_rate: Tasa de aprendizaje
        """
        super().__init__("PPO")
        self.learning_rate = learning_rate
        self.clip_ratio = PPO_CLIP_RATIO
    
    def optimize(
        self,
        start: TrajectoryPoint,
        goal: TrajectoryPoint,
        obstacles: Optional[List[np.ndarray]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[TrajectoryPoint]:
        """
        Optimizar usando PPO.
        
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
            raise ValueError("Invalid inputs for PPO algorithm")
        
        logger.debug("Applying PPO optimization")
        
        # Obtener trayectoria inicial (de constraints o generar)
        initial_trajectory = kwargs.get("initial_trajectory")
        if initial_trajectory is None:
            # Generar trayectoria inicial simple
            initial_trajectory = self._generate_initial_trajectory(start, goal)
        
        # Calcular gradiente de política
        policy_gradient = self._compute_policy_gradient(initial_trajectory, obstacles)
        
        # Aplicar actualización de política con clipping PPO
        updated_trajectory = []
        
        for i, point in enumerate(initial_trajectory):
            if i == 0 or i == len(initial_trajectory) - 1:
                updated_trajectory.append(point)
                continue
            
            # Calcular cambio sugerido por política
            if i - 1 < len(policy_gradient):
                delta = policy_gradient[i - 1] * self.learning_rate
                
                # Clipping PPO para estabilidad
                delta = np.clip(delta, -self.clip_ratio, self.clip_ratio)
                
                # Aplicar cambio
                new_position = point.position + delta[:3]
                new_orientation = point.orientation
                
                updated_point = TrajectoryPoint(
                    position=new_position,
                    orientation=new_orientation,
                    velocity=point.velocity,
                    acceleration=point.acceleration,
                    timestamp=point.timestamp
                )
                updated_trajectory.append(updated_point)
            else:
                updated_trajectory.append(point)
        
        return updated_trajectory if updated_trajectory else initial_trajectory
    
    def _generate_initial_trajectory(
        self,
        start: TrajectoryPoint,
        goal: TrajectoryPoint,
        num_points: int = 50
    ) -> List[TrajectoryPoint]:
        """Generar trayectoria inicial."""
        from ..optimization.trajectory_optimizer import TrajectoryOptimizer
        
        # Usar método del optimizador principal
        optimizer = TrajectoryOptimizer()
        return optimizer._generate_initial_trajectory(start, goal, num_points)
    
    def _compute_policy_gradient(
        self,
        trajectory: List[TrajectoryPoint],
        obstacles: Optional[List[np.ndarray]]
    ) -> List[np.ndarray]:
        """Calcular gradiente de política."""
        gradients = []
        
        for i in range(1, len(trajectory) - 1):
            prev_point = trajectory[i - 1]
            curr_point = trajectory[i]
            next_point = trajectory[i + 1]
            
            # Calcular gradiente basado en recompensa esperada
            direction = normalize_vector(next_point.position - prev_point.position)
            
            # Ajustar según obstáculos
            if obstacles:
                for obstacle in obstacles:
                    obstacle_center = (obstacle[:3] + obstacle[3:]) / 2
                    dist_vec = curr_point.position - obstacle_center
                    dist = np.linalg.norm(dist_vec)
                    if dist < 0.2:
                        # Empujar lejos del obstáculo
                        obstacle_direction = normalize_vector(dist_vec)
                        direction += obstacle_direction * 0.5
            
            gradients.append(direction)
        
        return gradients

