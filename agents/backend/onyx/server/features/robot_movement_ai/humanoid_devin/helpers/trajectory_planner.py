"""
Trajectory Planner for Humanoid Devin Robot (Optimizado)
=========================================================

Planificador de trayectorias avanzado con múltiples algoritmos.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


def ErrorCode(description: str):
    """Decorador para anotar excepciones con códigos de error y descripciones."""
    def decorator(cls):
        cls._error_description = description
        return cls
    return decorator


@ErrorCode(description="Error in trajectory planner system")
class TrajectoryPlannerError(Exception):
    """Excepción para errores del planificador de trayectorias."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in trajectory planner system")
        super().__init__(message)
        self.message = message


class TrajectoryPlanner:
    """
    Planificador de trayectorias para el robot humanoide.
    
    Soporta múltiples algoritmos de planificación.
    """
    
    def __init__(
        self,
        method: str = "linear",
        smooth: bool = True,
        max_acceleration: float = 1.0,
        max_jerk: float = 2.0
    ):
        """
        Inicializar planificador de trayectorias.
        
        Args:
            method: Método de planificación ("linear", "cubic", "quintic")
            smooth: Aplicar suavizado
            max_acceleration: Aceleración máxima (rad/s² o m/s²)
            max_jerk: Jerk máximo (rad/s³ o m/s³)
        """
        valid_methods = ["linear", "cubic", "quintic"]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {method}")
        
        if max_acceleration <= 0:
            raise ValueError("max_acceleration must be positive")
        if max_jerk <= 0:
            raise ValueError("max_jerk must be positive")
        
        self.method = method
        self.smooth = smooth
        self.max_acceleration = float(max_acceleration)
        self.max_jerk = float(max_jerk)
        
        logger.info(f"Trajectory planner initialized: method={method}, smooth={smooth}")
    
    def plan_joint_trajectory(
        self,
        start: Union[List[float], np.ndarray],
        end: Union[List[float], np.ndarray],
        duration: float = 1.0,
        num_points: int = 50
    ) -> np.ndarray:
        """
        Planificar trayectoria de articulaciones.
        
        Args:
            start: Posiciones iniciales
            end: Posiciones finales
            duration: Duración de la trayectoria (segundos)
            num_points: Número de puntos en la trayectoria
            
        Returns:
            Trayectoria (num_points, dof)
            
        Raises:
            TrajectoryPlannerError: Si hay error en la planificación
        """
        if duration <= 0:
            raise ValueError("duration must be positive")
        if num_points < 2:
            raise ValueError("num_points must be >= 2")
        
        try:
            start_array = np.array(start, dtype=np.float64)
            end_array = np.array(end, dtype=np.float64)
            
            if start_array.shape != end_array.shape:
                raise ValueError(
                    f"start and end must have the same shape, "
                    f"got {start_array.shape} and {end_array.shape}"
                )
            
            # Generar tiempos
            times = np.linspace(0, duration, num_points)
            
            # Planificar según método
            if self.method == "linear":
                trajectory = self._linear_interpolation(start_array, end_array, times)
            elif self.method == "cubic":
                trajectory = self._cubic_interpolation(start_array, end_array, times, duration)
            elif self.method == "quintic":
                trajectory = self._quintic_interpolation(start_array, end_array, times, duration)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            # Aplicar suavizado si está habilitado
            if self.smooth:
                trajectory = self._apply_smoothing(trajectory)
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Error planning joint trajectory: {e}", exc_info=True)
            raise TrajectoryPlannerError(f"Failed to plan trajectory: {str(e)}") from e
    
    def plan_cartesian_trajectory(
        self,
        start_position: np.ndarray,
        end_position: np.ndarray,
        start_orientation: Optional[np.ndarray] = None,
        end_orientation: Optional[np.ndarray] = None,
        duration: float = 1.0,
        num_points: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Planificar trayectoria cartesiana.
        
        Args:
            start_position: Posición inicial [x, y, z]
            end_position: Posición final [x, y, z]
            start_orientation: Orientación inicial [x, y, z, w] (opcional)
            end_orientation: Orientación final [x, y, z, w] (opcional)
            duration: Duración de la trayectoria (segundos)
            num_points: Número de puntos
            
        Returns:
            Dict con "positions" y "orientations" (si se proporcionaron)
        """
        if start_position.shape != (3,) or end_position.shape != (3,):
            raise ValueError("Positions must have shape (3,)")
        
        # Planificar posición
        position_trajectory = self.plan_joint_trajectory(
            start_position,
            end_position,
            duration,
            num_points
        )
        
        result = {"positions": position_trajectory}
        
        # Planificar orientación si se proporciona
        if start_orientation is not None and end_orientation is not None:
            # Interpolación SLERP para quaterniones
            orientation_trajectory = self._slerp_interpolation(
                start_orientation,
                end_orientation,
                num_points
            )
            result["orientations"] = orientation_trajectory
        
        return result
    
    def _linear_interpolation(
        self,
        start: np.ndarray,
        end: np.ndarray,
        times: np.ndarray
    ) -> np.ndarray:
        """Interpolación lineal."""
        normalized_times = times / times[-1] if times[-1] > 0 else times
        trajectory = start + np.outer(normalized_times, end - start)
        return trajectory
    
    def _cubic_interpolation(
        self,
        start: np.ndarray,
        end: np.ndarray,
        times: np.ndarray,
        duration: float
    ) -> np.ndarray:
        """Interpolación cúbica con velocidades cero en extremos."""
        normalized_times = times / duration if duration > 0 else times
        t = normalized_times
        
        # Polinomio cúbico: a*t³ + b*t² + c*t + d
        # Condiciones: p(0)=start, p(1)=end, v(0)=0, v(1)=0
        trajectory = (
            start[None, :] * (1 - 3*t**2 + 2*t**3)[:, None] +
            end[None, :] * (3*t**2 - 2*t**3)[:, None]
        )
        return trajectory
    
    def _quintic_interpolation(
        self,
        start: np.ndarray,
        end: np.ndarray,
        times: np.ndarray,
        duration: float
    ) -> np.ndarray:
        """Interpolación quíntica con velocidades y aceleraciones cero en extremos."""
        normalized_times = times / duration if duration > 0 else times
        t = normalized_times
        
        # Polinomio quíntico con condiciones en extremos
        trajectory = (
            start[None, :] * (1 - 10*t**3 + 15*t**4 - 6*t**5)[:, None] +
            end[None, :] * (10*t**3 - 15*t**4 + 6*t**5)[:, None]
        )
        return trajectory
    
    def _slerp_interpolation(
        self,
        start_quat: np.ndarray,
        end_quat: np.ndarray,
        num_points: int
    ) -> np.ndarray:
        """Interpolación SLERP para quaterniones."""
        # Normalizar quaterniones
        start_norm = start_quat / np.linalg.norm(start_quat)
        end_norm = end_quat / np.linalg.norm(end_quat)
        
        # Calcular ángulo entre quaterniones
        dot = np.clip(np.dot(start_norm, end_norm), -1.0, 1.0)
        angle = np.arccos(abs(dot))
        
        if angle < 1e-6:
            # Quaterniones muy cercanos, usar interpolación lineal
            t = np.linspace(0, 1, num_points)
            trajectory = start_norm[None, :] * (1 - t)[:, None] + end_norm[None, :] * t[:, None]
        else:
            # SLERP
            t = np.linspace(0, 1, num_points)
            sin_angle = np.sin(angle)
            trajectory = (
                start_norm[None, :] * (np.sin(angle * (1 - t)) / sin_angle)[:, None] +
                end_norm[None, :] * (np.sin(angle * t) / sin_angle)[:, None]
            )
        
        # Normalizar cada quaternion en la trayectoria
        norms = np.linalg.norm(trajectory, axis=1, keepdims=True)
        trajectory = trajectory / norms
        
        return trajectory
    
    def _apply_smoothing(self, trajectory: np.ndarray) -> np.ndarray:
        """Aplicar suavizado a la trayectoria."""
        if len(trajectory) < 3:
            return trajectory
        
        # Media móvil simple
        smoothed = trajectory.copy()
        for i in range(1, len(trajectory) - 1):
            smoothed[i] = 0.25 * trajectory[i-1] + 0.5 * trajectory[i] + 0.25 * trajectory[i+1]
        
        return smoothed
    
    def get_trajectory_velocity(
        self,
        trajectory: np.ndarray,
        dt: float = 0.01
    ) -> np.ndarray:
        """
        Calcular velocidades de una trayectoria.
        
        Args:
            trajectory: Trayectoria (num_points, dof)
            dt: Intervalo de tiempo entre puntos
            
        Returns:
            Velocidades (num_points-1, dof)
        """
        if len(trajectory) < 2:
            return np.array([])
        
        velocities = np.diff(trajectory, axis=0) / dt
        return velocities
    
    def get_trajectory_acceleration(
        self,
        trajectory: np.ndarray,
        dt: float = 0.01
    ) -> np.ndarray:
        """
        Calcular aceleraciones de una trayectoria.
        
        Args:
            trajectory: Trayectoria (num_points, dof)
            dt: Intervalo de tiempo entre puntos
            
        Returns:
            Aceleraciones (num_points-2, dof)
        """
        velocities = self.get_trajectory_velocity(trajectory, dt)
        
        if len(velocities) < 2:
            return np.array([])
        
        accelerations = np.diff(velocities, axis=0) / dt
        return accelerations

