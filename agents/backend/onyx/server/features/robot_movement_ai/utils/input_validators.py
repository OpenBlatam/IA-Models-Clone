"""
Input Validators
================

Utilidades para validación de entrada con mensajes de error descriptivos.
"""

from typing import Any, List, Optional, Dict, Tuple
import numpy as np
from ..core.exceptions import ValidationError


def validate_position(
    x: float,
    y: float,
    z: float,
    min_value: float = -10.0,
    max_value: float = 10.0
) -> Tuple[float, float, float]:
    """
    Validar coordenadas de posición.
    
    Args:
        x: Coordenada X
        y: Coordenada Y
        z: Coordenada Z
        min_value: Valor mínimo permitido
        max_value: Valor máximo permitido
    
    Returns:
        Tupla con coordenadas validadas
    
    Raises:
        ValidationError: Si las coordenadas están fuera de rango
    """
    if not (min_value <= x <= max_value):
        raise ValidationError(
            f"X coordinate must be between {min_value} and {max_value}, got {x}",
            error_code="INVALID_X_COORDINATE",
            details={"x": x, "min": min_value, "max": max_value}
        )
    
    if not (min_value <= y <= max_value):
        raise ValidationError(
            f"Y coordinate must be between {min_value} and {max_value}, got {y}",
            error_code="INVALID_Y_COORDINATE",
            details={"y": y, "min": min_value, "max": max_value}
        )
    
    if not (min_value <= z <= max_value):
        raise ValidationError(
            f"Z coordinate must be between {min_value} and {max_value}, got {z}",
            error_code="INVALID_Z_COORDINATE",
            details={"z": z, "min": min_value, "max": max_value}
        )
    
    return (x, y, z)


def validate_quaternion(
    orientation: List[float],
    tolerance: float = 0.1
) -> np.ndarray:
    """
    Validar y normalizar quaternion.
    
    Args:
        orientation: Lista con 4 elementos [qx, qy, qz, qw]
        tolerance: Tolerancia para normalización
    
    Returns:
        Array numpy con quaternion normalizado
    
    Raises:
        ValidationError: Si el quaternion es inválido
    """
    if len(orientation) != 4:
        raise ValidationError(
            f"Orientation must have exactly 4 elements, got {len(orientation)}",
            error_code="INVALID_QUATERNION_LENGTH",
            details={"length": len(orientation), "orientation": orientation}
        )
    
    quat = np.array(orientation, dtype=np.float64)
    norm = np.linalg.norm(quat)
    
    if abs(norm - 1.0) > tolerance:
        raise ValidationError(
            f"Quaternion must be normalized (norm={norm:.6f}, expected ~1.0)",
            error_code="QUATERNION_NOT_NORMALIZED",
            details={"norm": float(norm), "orientation": orientation.tolist()}
        )
    
    if norm > 0:
        quat = quat / norm
    
    return quat


def validate_waypoints(
    waypoints: List[Dict[str, Any]],
    min_waypoints: int = 2,
    max_waypoints: int = 100
) -> List[Dict[str, Any]]:
    """
    Validar lista de waypoints.
    
    Args:
        waypoints: Lista de waypoints
        min_waypoints: Número mínimo de waypoints
        max_waypoints: Número máximo de waypoints
    
    Returns:
        Lista de waypoints validados
    
    Raises:
        ValidationError: Si los waypoints son inválidos
    """
    if len(waypoints) < min_waypoints:
        raise ValidationError(
            f"At least {min_waypoints} waypoints required, got {len(waypoints)}",
            error_code="INSUFFICIENT_WAYPOINTS",
            details={"count": len(waypoints), "min_required": min_waypoints}
        )
    
    if len(waypoints) > max_waypoints:
        raise ValidationError(
            f"Maximum {max_waypoints} waypoints allowed, got {len(waypoints)}",
            error_code="TOO_MANY_WAYPOINTS",
            details={"count": len(waypoints), "max_allowed": max_waypoints}
        )
    
    validated = []
    for i, wp in enumerate(waypoints):
        try:
            x = float(wp.get("x", wp.get("X", 0)))
            y = float(wp.get("y", wp.get("Y", 0)))
            z = float(wp.get("z", wp.get("Z", 0)))
            
            validate_position(x, y, z)
            
            orientation = wp.get("orientation")
            if orientation:
                orientation = validate_quaternion(orientation)
            
            validated.append({
                "x": x,
                "y": y,
                "z": z,
                "orientation": orientation.tolist() if orientation is not None else None
            })
        except (ValueError, KeyError, TypeError) as e:
            raise ValidationError(
                f"Invalid waypoint at index {i}: {str(e)}",
                error_code="INVALID_WAYPOINT",
                details={"index": i, "waypoint": wp, "error": str(e)}
            )
    
    return validated


def validate_message(
    message: str,
    min_length: int = 1,
    max_length: int = 10000
) -> str:
    """
    Validar mensaje de chat.
    
    Args:
        message: Mensaje a validar
        min_length: Longitud mínima
        max_length: Longitud máxima
    
    Returns:
        Mensaje validado (trimmed)
    
    Raises:
        ValidationError: Si el mensaje es inválido
    """
    if not message:
        raise ValidationError(
            "Message cannot be empty",
            error_code="EMPTY_MESSAGE",
            details={"message_length": 0}
        )
    
    message = message.strip()
    
    if len(message) < min_length:
        raise ValidationError(
            f"Message too short (min {min_length} characters), got {len(message)}",
            error_code="MESSAGE_TOO_SHORT",
            details={"length": len(message), "min_length": min_length}
        )
    
    if len(message) > max_length:
        raise ValidationError(
            f"Message too long (max {max_length} characters), got {len(message)}",
            error_code="MESSAGE_TOO_LONG",
            details={"length": len(message), "max_length": max_length}
        )
    
    return message


def validate_obstacles(
    obstacles: List[List[float]],
    max_obstacles: int = 1000
) -> List[np.ndarray]:
    """
    Validar lista de obstáculos.
    
    Args:
        obstacles: Lista de obstáculos, cada uno como [min_x, min_y, min_z, max_x, max_y, max_z]
        max_obstacles: Número máximo de obstáculos
    
    Returns:
        Lista de arrays numpy con obstáculos validados
    
    Raises:
        ValidationError: Si los obstáculos son inválidos
    """
    if len(obstacles) > max_obstacles:
        raise ValidationError(
            f"Maximum {max_obstacles} obstacles allowed, got {len(obstacles)}",
            error_code="TOO_MANY_OBSTACLES",
            details={"count": len(obstacles), "max_allowed": max_obstacles}
        )
    
    validated = []
    for i, obs in enumerate(obstacles):
        if len(obs) != 6:
            raise ValidationError(
                f"Obstacle at index {i} must have 6 coordinates [min_x, min_y, min_z, max_x, max_y, max_z], got {len(obs)}",
                error_code="INVALID_OBSTACLE_FORMAT",
                details={"index": i, "length": len(obs), "obstacle": obs}
            )
        
        try:
            obs_array = np.array(obs, dtype=np.float64)
            min_coords = obs_array[:3]
            max_coords = obs_array[3:]
            
            if np.any(max_coords < min_coords):
                raise ValidationError(
                    f"Obstacle at index {i}: max coordinates must be >= min coordinates",
                    error_code="INVALID_OBSTACLE_BOUNDS",
                    details={
                        "index": i,
                        "min_coords": min_coords.tolist(),
                        "max_coords": max_coords.tolist()
                    }
                )
            
            validated.append(obs_array)
        except (ValueError, TypeError) as e:
            raise ValidationError(
                f"Invalid obstacle at index {i}: {str(e)}",
                error_code="INVALID_OBSTACLE_DATA",
                details={"index": i, "obstacle": obs, "error": str(e)}
            )
    
    return validated

