"""
TD3 Algorithm - Twin Delayed DDPG
==================================

Implementación del algoritmo TD3 para optimización de trayectorias.
TD3 reduce la sobreestimación del valor Q utilizando dos críticos y actualización retardada.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any

from .base_algorithm import BaseOptimizationAlgorithm
from ..types.types import TrajectoryPoint
from ..constants import DEFAULT_LEARNING_RATE
from ..utils.math_utils import normalize_vector

logger = logging.getLogger(__name__)


class TD3Algorithm(BaseOptimizationAlgorithm):
    """
    Algoritmo TD3 (Twin Delayed Deep Deterministic Policy Gradient) para optimización.
    
    TD3 mejora DDPG utilizando:
    1. Clipped Double Q-Learning (dos críticos)
    2. Actualización retardada de la política (frecuencia menor)
    3. Ruido de suavizado del objetivo (target policy smoothing)
    """
    
    def __init__(
        self, 
        learning_rate: float = DEFAULT_LEARNING_RATE,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2
    ):
        """
        Inicializar algoritmo TD3.
        
        Args:
            learning_rate: Tasa de aprendizaje
            policy_noise: Ruido para suavizado de política objetivo
            noise_clip: Límite del ruido
            policy_freq: Frecuencia de actualización de política
        """
        super().__init__("TD3")
        self.learning_rate = learning_rate
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.iteration_count = 0
        
    def optimize(
        self,
        start: TrajectoryPoint,
        goal: TrajectoryPoint,
        obstacles: Optional[List[np.ndarray]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[TrajectoryPoint]:
        """
        Optimizar usando TD3.
        
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
            raise ValueError("Invalid inputs for TD3 algorithm")
        
        logger.debug("Applying TD3 optimization with delayed updates")
        
        # Obtener trayectoria inicial
        initial_trajectory = kwargs.get("initial_trajectory")
        if initial_trajectory is None:
            initial_trajectory = self._generate_initial_trajectory(start, goal)
            
        current_trajectory = [p for p in initial_trajectory]
        updated_trajectory = []
        
        # En una implementación real, aquí usaríamos dos redes Q y una red de política.
        # Simularemos el comportamiento de TD3: actualizaciones más conservadoras y robustas.
        
        # Calcular gradiente "conservador" (simulando el mínimo de dos críticos)
        conservative_gradients = self._compute_conservative_gradients(current_trajectory, goal, obstacles)
        
        for i, point in enumerate(current_trajectory):
            self.iteration_count += 1
            
            if i == 0 or i == len(current_trajectory) - 1:
                updated_trajectory.append(point)
                continue
                
            # Solo actualizar política cada 'policy_freq' pasos (Delayed Update)
            if self.iteration_count % self.policy_freq == 0:
                if i - 1 < len(conservative_gradients):
                    gradient = conservative_gradients[i - 1]
                    
                    # Target Policy Smoothing Noise
                    noise = np.random.normal(0, self.policy_noise, size=3)
                    noise = np.clip(noise, -self.noise_clip, self.noise_clip)
                    
                    # Actualización robusta
                    delta = (gradient + noise) * self.learning_rate
                    
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
            else:
                # Si no toca actualizar, mantener punto anterior
                updated_trajectory.append(point)
        
        # TD3 suele generar trayectorias menos "ruidosas" que DDPG, pero un suavizado final ayuda
        return self._smooth_trajectory(updated_trajectory)

    def _generate_initial_trajectory(self, start, goal):
        # Helper simple
        from ..utils.trajectory_utils import generate_linear_trajectory
        return generate_linear_trajectory(start, goal)

    def _compute_conservative_gradients(
        self, 
        trajectory: List[TrajectoryPoint], 
        goal: TrajectoryPoint,
        obstacles: Optional[List[np.ndarray]]
    ) -> List[np.ndarray]:
        """Calcular gradientes robustos/conservadores."""
        gradients = []
        
        for i in range(1, len(trajectory) - 1):
            current_pos = trajectory[i].position
            
            # 1. Atracción objetivo directa
            to_goal = normalize_vector(goal.position - current_pos)
            
            # 2. Penalización fuerte por desviaciones bruscas (estabilidad)
            prev_pos = trajectory[i-1].position
            next_pos = trajectory[i+1].position
            
            # Vector tangente estimado
            tangent = normalize_vector(next_pos - prev_pos)
            
            # Proyección para mantener suavidad direccional
            direction_consistency = np.dot(tangent, to_goal)
            
            # Gradiente base
            gradient = to_goal * 0.3
            
            # Si la dirección no es consistente, corregir fuertemente
            if direction_consistency < 0.5:
                correction = normalize_vector(tangent - to_goal)
                gradient += correction * 0.2
            
            # 3. Obstáculos: Conservador = mayor margen de seguridad
            if obstacles:
                for obstacle in obstacles:
                    obs_center = (obstacle[:3] + obstacle[3:]) / 2
                    dist_vec = current_pos - obs_center
                    dist = np.linalg.norm(dist_vec)
                    
                    # Margen de seguridad aumentado para TD3
                    margin = 0.8
                    if dist < margin:
                        repulsion = normalize_vector(dist_vec)
                        # Fuerza exponencial para evitar acercarse
                        strength = np.exp(-dist) 
                        gradient += repulsion * strength * 0.7
            
            gradients.append(gradient)
            
        return gradients
        
    def _smooth_trajectory(self, trajectory: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """Suavizado simple."""
        if len(trajectory) < 3:
            return trajectory
            
        smoothed = [trajectory[0]]
        for i in range(1, len(trajectory) - 1):
            # Media ponderada (más peso al punto actual para fidelidad)
            prev_p = trajectory[i-1].position
            curr_p = trajectory[i].position
            next_p = trajectory[i+1].position
            
            new_pos = (prev_p + 2*curr_p + next_p) / 4.0
            
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
