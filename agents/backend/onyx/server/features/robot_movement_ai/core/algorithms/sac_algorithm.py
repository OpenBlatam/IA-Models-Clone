"""
SAC Algorithm - Soft Actor-Critic
==================================

Implementación del algoritmo SAC para optimización de trayectorias.
SAC maximiza un objetivo de "máxima entropía" que fomenta la exploración robusta.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any

from .base_algorithm import BaseOptimizationAlgorithm
from ..types.types import TrajectoryPoint
from ..constants import DEFAULT_LEARNING_RATE
from ..utils.math_utils import normalize_vector

logger = logging.getLogger(__name__)


class SACAlgorithm(BaseOptimizationAlgorithm):
    """
    Algoritmo SAC (Soft Actor-Critic) para optimización de trayectorias.
    
    SAC es un algoritmo off-policy que optimiza una política estocástica
    de manera off-policy. Utiliza la regularización de entropía para
    mejorar la exploración y la estabilidad.
    """
    
    def __init__(self, learning_rate: float = DEFAULT_LEARNING_RATE, alpha: float = 0.2):
        """
        Inicializar algoritmo SAC.
        
        Args:
            learning_rate: Tasa de aprendizaje
            alpha: Coeficiente de entropía (temperatura)
        """
        super().__init__("SAC")
        self.learning_rate = learning_rate
        self.alpha = alpha  # Coeficiente de entropía
        
    def optimize(
        self,
        start: TrajectoryPoint,
        goal: TrajectoryPoint,
        obstacles: Optional[List[np.ndarray]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[TrajectoryPoint]:
        """
        Optimizar usando SAC.
        
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
            raise ValueError("Invalid inputs for SAC algorithm")
        
        logger.debug("Applying SAC optimization with maximum entropy")
        
        # Obtener trayectoria inicial
        initial_trajectory = kwargs.get("initial_trajectory")
        if initial_trajectory is None:
            initial_trajectory = self._generate_initial_trajectory(start, goal)
            
        # En una implementación real de RL, aquí usaríamos las redes neuronales (Actor, Critic)
        # para inferir la acción óptima. Para esta simulación algorítmica,
        # aplicamos una optimización basada en gradientes con ruido de entropía.
        
        current_trajectory = [p for p in initial_trajectory]
        updated_trajectory = []
        
        # Calcular gradiente "soft" (incluyendo término de entropía simulado)
        soft_gradients = self._compute_soft_gradients(current_trajectory, goal, obstacles)
        
        for i, point in enumerate(current_trajectory):
            if i == 0 or i == len(current_trajectory) - 1:
                updated_trajectory.append(point)
                continue
                
            # Aplicar actualización con regularización de entropía
            # La "entropía" aquí se simula añadiendo ruido exploratorio adaptativo
            # En SAC real, esto es parte del objetivo de la política
            
            if i - 1 < len(soft_gradients):
                gradient = soft_gradients[i - 1]
                
                # Añadir ruido gaussiano escalado por alpha (simulando política estocástica)
                exploration_noise = np.random.normal(0, self.alpha, size=3)
                
                # Paso de actualización
                delta = (gradient + exploration_noise) * self.learning_rate
                
                new_position = point.position + delta
                
                updated_point = TrajectoryPoint(
                    position=new_position,
                    orientation=point.orientation,
                    velocity=point.velocity,
                    acceleration=point.acceleration,
                    timestamp=point.timestamp
                )
                updated_trajectory.append(updated_point)
            else:
                updated_trajectory.append(point)
                
        # Suavizar resultado final
        return self._smooth_trajectory(updated_trajectory)

    def _generate_initial_trajectory(self, start, goal):
        # Helper simple
        from ..utils.trajectory_utils import generate_linear_trajectory
        return generate_linear_trajectory(start, goal)

    def _compute_soft_gradients(
        self, 
        trajectory: List[TrajectoryPoint], 
        goal: TrajectoryPoint,
        obstacles: Optional[List[np.ndarray]]
    ) -> List[np.ndarray]:
        """Calcular gradientes considerando objetivo y obstáculos."""
        gradients = []
        
        for i in range(1, len(trajectory) - 1):
            current_pos = trajectory[i].position
            
            # 1. Atracción hacia el objetivo final (global)
            to_goal = normalize_vector(goal.position - current_pos)
            
            # 2. Suavidad (local) - mantenerse cerca del promedio de vecinos
            prev_pos = trajectory[i-1].position
            next_pos = trajectory[i+1].position
            desired_pos = (prev_pos + next_pos) / 2.0
            to_smooth = normalize_vector(desired_pos - current_pos)
            
            gradient = to_goal * 0.4 + to_smooth * 0.6
            
            # 3. Repulsión de obstáculos
            if obstacles:
                for obstacle in obstacles:
                    obs_center = (obstacle[:3] + obstacle[3:]) / 2
                    dist_vec = current_pos - obs_center
                    dist = np.linalg.norm(dist_vec)
                    
                    # Radio de influencia del obstáculo
                    if dist < 0.5:
                        repulsion = normalize_vector(dist_vec)
                        # Fuerza inversamente proporcional a la distancia
                        strength = 1.0 / (dist + 0.01)
                        gradient += repulsion * strength * 0.5
            
            gradients.append(gradient)
            
        return gradients
        
    def _smooth_trajectory(self, trajectory: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """Suavizado simple de media móvil."""
        if len(trajectory) < 3:
            return trajectory
            
        smoothed = [trajectory[0]]
        for i in range(1, len(trajectory) - 1):
            prev_p = trajectory[i-1].position
            curr_p = trajectory[i].position
            next_p = trajectory[i+1].position
            
            new_pos = (prev_p + curr_p + next_p) / 3.0
            
            new_point = TrajectoryPoint(
                position=new_pos,
                orientation=trajectory[i].orientation,
                velocity=trajectory[i].velocity,
                acceleration=trajectory[i].acceleration,
                timestamp=trajectory[i].timestamp
            )
            smoothed.append(new_point)
            
        smoothed.append(trajectory[-1])
        return smoothed
