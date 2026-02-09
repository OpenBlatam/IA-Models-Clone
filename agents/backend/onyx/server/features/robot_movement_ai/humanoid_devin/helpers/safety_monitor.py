"""
Safety Monitor for Humanoid Devin Robot (Optimizado)
=====================================================

Monitor de seguridad para prevenir movimientos peligrosos.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def ErrorCode(description: str):
    """Decorador para anotar excepciones con códigos de error y descripciones."""
    def decorator(cls):
        cls._error_description = description
        return cls
    return decorator


@ErrorCode(description="Error in safety monitoring system")
class SafetyError(Exception):
    """Excepción para errores de seguridad."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in safety monitoring system")
        super().__init__(message)
        self.message = message


class SafetyMonitor:
    """
    Monitor de seguridad para el robot humanoide.
    
    Valida movimientos y comandos para prevenir situaciones peligrosas.
    """
    
    def __init__(
        self,
        max_joint_velocity: float = 2.0,
        max_cartesian_velocity: float = 1.0,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
        workspace_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        emergency_stop_enabled: bool = True
    ):
        """
        Inicializar monitor de seguridad.
        
        Args:
            max_joint_velocity: Velocidad máxima de articulaciones (rad/s)
            max_cartesian_velocity: Velocidad máxima cartesiana (m/s)
            joint_limits: Límites de articulaciones [(min, max), ...]
            workspace_limits: Límites del espacio de trabajo {"x": (min, max), ...}
            emergency_stop_enabled: Habilitar parada de emergencia
        """
        # Validar parámetros
        if max_joint_velocity <= 0:
            raise ValueError("max_joint_velocity must be positive")
        if max_cartesian_velocity <= 0:
            raise ValueError("max_cartesian_velocity must be positive")
        
        self.max_joint_velocity = float(max_joint_velocity)
        self.max_cartesian_velocity = float(max_cartesian_velocity)
        self.joint_limits = joint_limits or []
        self.workspace_limits = workspace_limits or {}
        self.emergency_stop_enabled = emergency_stop_enabled
        self.emergency_stopped = False
        
        logger.info(
            f"Safety monitor initialized: "
            f"max_joint_vel={max_joint_velocity} rad/s, "
            f"max_cart_vel={max_cartesian_velocity} m/s"
        )
    
    def validate_joint_positions(
        self,
        positions: List[float],
        previous_positions: Optional[List[float]] = None,
        dt: float = 0.01
    ) -> Tuple[bool, Optional[str]]:
        """
        Validar posiciones de articulaciones.
        
        Args:
            positions: Posiciones objetivo
            previous_positions: Posiciones anteriores (para validar velocidad)
            dt: Intervalo de tiempo (segundos)
            
        Returns:
            Tupla (es_válido, mensaje_error)
        """
        if self.emergency_stopped:
            return False, "Emergency stop is active"
        
        # Validar límites de articulaciones
        if self.joint_limits:
            if len(positions) != len(self.joint_limits):
                return False, f"Position count ({len(positions)}) != limit count ({len(self.joint_limits)})"
            
            for i, (pos, (min_val, max_val)) in enumerate(zip(positions, self.joint_limits)):
                if pos < min_val or pos > max_val:
                    return False, f"Joint {i} position {pos} outside limits [{min_val}, {max_val}]"
        
        # Validar velocidad
        if previous_positions is not None:
            if len(positions) != len(previous_positions):
                return False, "Position arrays have different lengths"
            
            velocities = np.abs(np.array(positions) - np.array(previous_positions)) / dt
            
            if np.any(velocities > self.max_joint_velocity):
                max_vel = np.max(velocities)
                return False, f"Joint velocity {max_vel:.2f} rad/s exceeds maximum {self.max_joint_velocity} rad/s"
        
        return True, None
    
    def validate_cartesian_movement(
        self,
        start_position: np.ndarray,
        end_position: np.ndarray,
        dt: float = 0.01
    ) -> Tuple[bool, Optional[str]]:
        """
        Validar movimiento cartesiano.
        
        Args:
            start_position: Posición inicial [x, y, z]
            end_position: Posición final [x, y, z]
            dt: Intervalo de tiempo (segundos)
            
        Returns:
            Tupla (es_válido, mensaje_error)
        """
        if self.emergency_stopped:
            return False, "Emergency stop is active"
        
        # Validar límites del espacio de trabajo
        if self.workspace_limits:
            for axis, (min_val, max_val) in self.workspace_limits.items():
                idx = {"x": 0, "y": 1, "z": 2}.get(axis.lower())
                if idx is not None:
                    if end_position[idx] < min_val or end_position[idx] > max_val:
                        return False, f"{axis} position {end_position[idx]} outside workspace limits [{min_val}, {max_val}]"
        
        # Validar velocidad cartesiana
        distance = np.linalg.norm(end_position - start_position)
        velocity = distance / dt if dt > 0 else 0.0
        
        if velocity > self.max_cartesian_velocity:
            return False, f"Cartesian velocity {velocity:.2f} m/s exceeds maximum {self.max_cartesian_velocity} m/s"
        
        return True, None
    
    def validate_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        previous_position: Optional[np.ndarray] = None,
        dt: float = 0.01
    ) -> Tuple[bool, Optional[str]]:
        """
        Validar pose completa.
        
        Args:
            position: Posición [x, y, z]
            orientation: Orientación quaternion [x, y, z, w]
            previous_position: Posición anterior (para validar velocidad)
            dt: Intervalo de tiempo (segundos)
            
        Returns:
            Tupla (es_válido, mensaje_error)
        """
        # Validar posición
        if previous_position is not None:
            valid, error = self.validate_cartesian_movement(previous_position, position, dt)
            if not valid:
                return False, error
        
        # Validar límites del espacio de trabajo
        if self.workspace_limits:
            for axis, (min_val, max_val) in self.workspace_limits.items():
                idx = {"x": 0, "y": 1, "z": 2}.get(axis.lower())
                if idx is not None:
                    if position[idx] < min_val or position[idx] > max_val:
                        return False, f"{axis} position outside workspace limits"
        
        return True, None
    
    def emergency_stop(self) -> None:
        """Activar parada de emergencia."""
        if not self.emergency_stop_enabled:
            logger.warning("Emergency stop requested but not enabled")
            return
        
        self.emergency_stopped = True
        logger.critical("EMERGENCY STOP ACTIVATED")
    
    def clear_emergency_stop(self) -> None:
        """Limpiar parada de emergencia."""
        if self.emergency_stopped:
            self.emergency_stopped = False
            logger.info("Emergency stop cleared")
    
    def is_safe(self) -> bool:
        """
        Verificar si el sistema está en estado seguro.
        
        Returns:
            True si el sistema está seguro
        """
        return not self.emergency_stopped
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado del monitor de seguridad.
        
        Returns:
            Dict con estado del monitor
        """
        return {
            "emergency_stopped": self.emergency_stopped,
            "emergency_stop_enabled": self.emergency_stop_enabled,
            "max_joint_velocity": self.max_joint_velocity,
            "max_cartesian_velocity": self.max_cartesian_velocity,
            "joint_limits_configured": len(self.joint_limits) > 0,
            "workspace_limits_configured": len(self.workspace_limits) > 0,
            "is_safe": self.is_safe()
        }

