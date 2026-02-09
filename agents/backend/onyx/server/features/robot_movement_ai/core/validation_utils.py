"""
Validation Utilities
====================

Utilidades de validación reutilizables para el sistema.
"""

from typing import Any, List, Optional, Union, Callable
import numpy as np
from .exceptions import ValidationError


def validate_quaternion(
    quaternion: List[float],
    name: str = "quaternion",
    tolerance: float = 0.1
) -> None:
    """
    Validar que un quaternion esté normalizado.
    
    Args:
        quaternion: Lista con 4 elementos [qx, qy, qz, qw]
        name: Nombre del parámetro para mensajes de error
        tolerance: Tolerancia para la normalización
    
    Raises:
        ValidationError: Si el quaternion es inválido
    """
    if not isinstance(quaternion, (list, tuple, np.ndarray)):
        raise ValidationError(
            f"{name} must be a list, tuple, or numpy array",
            error_code="INVALID_TYPE",
            details={"type": type(quaternion).__name__}
        )
    
    if len(quaternion) != 4:
        raise ValidationError(
            f"{name} must have exactly 4 elements",
            error_code="INVALID_LENGTH",
            details={"length": len(quaternion), "expected": 4}
        )
    
    quat_array = np.array(quaternion, dtype=float)
    norm = np.linalg.norm(quat_array)
    
    if abs(norm - 1.0) > tolerance:
        raise ValidationError(
            f"{name} must be normalized (norm={norm:.3f}, expected ~1.0)",
            error_code="INVALID_QUATERNION",
            details={"norm": float(norm), "tolerance": tolerance}
        )


def validate_position(
    position: Union[List[float], np.ndarray],
    name: str = "position",
    min_val: float = -10.0,
    max_val: float = 10.0,
    dimensions: int = 3
) -> None:
    """
    Validar posición 3D.
    
    Args:
        position: Posición como lista o array
        name: Nombre del parámetro
        min_val: Valor mínimo permitido
        max_val: Valor máximo permitido
        dimensions: Número de dimensiones esperadas
    
    Raises:
        ValidationError: Si la posición es inválida
    """
    if not isinstance(position, (list, tuple, np.ndarray)):
        raise ValidationError(
            f"{name} must be a list, tuple, or numpy array",
            error_code="INVALID_TYPE",
            details={"type": type(position).__name__}
        )
    
    pos_array = np.array(position, dtype=float)
    
    if len(pos_array) != dimensions:
        raise ValidationError(
            f"{name} must have {dimensions} dimensions",
            error_code="INVALID_DIMENSIONS",
            details={"dimensions": len(pos_array), "expected": dimensions}
        )
    
    if np.any(pos_array < min_val) or np.any(pos_array > max_val):
        out_of_range = np.any((pos_array < min_val) | (pos_array > max_val))
        if out_of_range:
            raise ValidationError(
                f"{name} coordinates must be between {min_val} and {max_val}",
                error_code="OUT_OF_RANGE",
                details={
                    "min": float(np.min(pos_array)),
                    "max": float(np.max(pos_array)),
                    "limits": {"min": min_val, "max": max_val}
                }
            )


def validate_trajectory_point(
    point: Any,
    name: str = "trajectory_point"
) -> None:
    """
    Validar punto de trayectoria.
    
    Args:
        point: Punto de trayectoria
        name: Nombre del parámetro
    
    Raises:
        ValidationError: Si el punto es inválido
    """
    if not hasattr(point, 'position'):
        raise ValidationError(
            f"{name} must have a 'position' attribute",
            error_code="MISSING_ATTRIBUTE",
            details={"attribute": "position"}
        )
    
    if not hasattr(point, 'orientation'):
        raise ValidationError(
            f"{name} must have an 'orientation' attribute",
            error_code="MISSING_ATTRIBUTE",
            details={"attribute": "orientation"}
        )
    
    validate_position(point.position, name=f"{name}.position")
    validate_quaternion(point.orientation, name=f"{name}.orientation")


def validate_obstacle(
    obstacle: Union[List[float], np.ndarray],
    name: str = "obstacle"
) -> None:
    """
    Validar obstáculo (bounding box).
    
    Args:
        obstacle: Obstáculo como [min_x, min_y, min_z, max_x, max_y, max_z]
        name: Nombre del parámetro
    
    Raises:
        ValidationError: Si el obstáculo es inválido
    """
    if not isinstance(obstacle, (list, tuple, np.ndarray)):
        raise ValidationError(
            f"{name} must be a list, tuple, or numpy array",
            error_code="INVALID_TYPE"
        )
    
    obs_array = np.array(obstacle, dtype=float)
    
    if len(obs_array) != 6:
        raise ValidationError(
            f"{name} must have 6 coordinates [min_x, min_y, min_z, max_x, max_y, max_z]",
            error_code="INVALID_LENGTH",
            details={"length": len(obs_array), "expected": 6}
        )
    
    min_coords = obs_array[:3]
    max_coords = obs_array[3:]
    
    if np.any(max_coords < min_coords):
        raise ValidationError(
            f"{name} max coordinates must be >= min coordinates",
            error_code="INVALID_BOUNDS",
            details={
                "min": min_coords.tolist(),
                "max": max_coords.tolist()
            }
        )


def validate_obstacles(
    obstacles: List[Union[List[float], np.ndarray]],
    name: str = "obstacles"
) -> None:
    """
    Validar lista de obstáculos.
    
    Args:
        obstacles: Lista de obstáculos
        name: Nombre del parámetro
    
    Raises:
        ValidationError: Si algún obstáculo es inválido
    """
    if not isinstance(obstacles, list):
        raise ValidationError(
            f"{name} must be a list",
            error_code="INVALID_TYPE"
        )
    
    if len(obstacles) > 1000:
        raise ValidationError(
            f"{name} list too long (max 1000)",
            error_code="LIST_TOO_LONG",
            details={"length": len(obstacles), "max": 1000}
        )
    
    for i, obstacle in enumerate(obstacles):
        try:
            validate_obstacle(obstacle, name=f"{name}[{i}]")
        except ValidationError as e:
            raise ValidationError(
                f"Invalid obstacle at index {i}: {e.message}",
                error_code=e.error_code,
                details={"index": i, **e.details}
            ) from e

