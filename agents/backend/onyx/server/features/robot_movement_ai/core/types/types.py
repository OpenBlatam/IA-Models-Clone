"""
Type Definitions
================

Definiciones de tipos para el sistema de movimiento robótico.
"""

from typing import (
    List, Tuple, Optional, Dict, Any, Union, Callable,
    Protocol, runtime_checkable
)
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray


# Type aliases para claridad
Position3D = NDArray[np.float64]  # [x, y, z]
Orientation = NDArray[np.float64]  # [qx, qy, qz, qw] (quaternion)

@dataclass
class TrajectoryPoint:
    """Punto en una trayectoria."""
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [qx, qy, qz, qw] (quaternion)
    velocity: Optional[np.ndarray] = None
    acceleration: Optional[np.ndarray] = None
    timestamp: float = 0.0

Velocity3D = NDArray[np.float64]  # [vx, vy, vz]
Acceleration3D = NDArray[np.float64]  # [ax, ay, az]
JointAngles = List[float]
JointVelocities = List[float]
Obstacle = NDArray[np.float64]  # [min_x, min_y, min_z, max_x, max_y, max_z]
BoundingBox = Tuple[Position3D, Position3D]  # (min_corner, max_corner)
Transform = NDArray[np.float64]  # 4x4 transformation matrix
RotationMatrix = NDArray[np.float64]  # 3x3 rotation matrix

# Config types
ConfigDict = Dict[str, Any]
MetricsDict = Dict[str, Any]
StatisticsDict = Dict[str, Any]

# Callback types
TrajectoryCallback = Callable[[List[Any]], None]
FeedbackCallback = Callable[[Any], None]
ErrorCallback = Callable[[Exception], None]


@runtime_checkable
class Optimizable(Protocol):
    """Protocolo para objetos optimizables."""
    
    def optimize(
        self,
        start: Any,
        goal: Any,
        obstacles: Optional[List[Any]] = None,
        **kwargs: Any
    ) -> List[Any]:
        """Optimizar trayectoria."""
        ...


@runtime_checkable
class Validatable(Protocol):
    """Protocolo para objetos validables."""
    
    def validate(self) -> bool:
        """Validar objeto."""
        ...


@runtime_checkable
class Serializable(Protocol):
    """Protocolo para objetos serializables."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable':
        """Crear desde diccionario."""
        ...


# Result types
class OptimizationResult:
    """Resultado de optimización."""
    
    def __init__(
        self,
        trajectory: List[Any],
        success: bool = True,
        message: str = "",
        metrics: Optional[Dict[str, float]] = None,
        iterations: int = 0,
        duration: float = 0.0
    ):
        self.trajectory = trajectory
        self.success = success
        self.message = message
        self.metrics = metrics or {}
        self.iterations = iterations
        self.duration = duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "success": self.success,
            "message": self.message,
            "trajectory_length": len(self.trajectory),
            "iterations": self.iterations,
            "duration": self.duration,
            "metrics": self.metrics
        }


class ValidationResult:
    """Resultado de validación."""
    
    def __init__(
        self,
        valid: bool,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None
    ):
        self.valid = valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings
        }






