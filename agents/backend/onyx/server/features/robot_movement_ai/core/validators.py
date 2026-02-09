"""
Validators
==========

Validadores para datos del sistema de movimiento robótico.
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

from .exceptions import (
    ValidationError,
    InvalidObstacleError,
    TrajectoryInvalidError
)
try:
    from .optimization.trajectory_optimizer import TrajectoryPoint
except ImportError:
    from ..optimization.trajectory_optimizer import TrajectoryPoint
try:
    from .constants.constants import (
        MAX_JUMP_DISTANCE,
        MAX_ACCELERATION_MAGNITUDE,
        MAX_JOINT_ANGLE,
        DEFAULT_MAX_JOINT_VELOCITY as MAX_JOINT_VELOCITY
    )
except ImportError:
    from ..constants.constants import (
        MAX_JUMP_DISTANCE,
        MAX_ACCELERATION_MAGNITUDE,
        MAX_JOINT_ANGLE,
        DEFAULT_MAX_JOINT_VELOCITY as MAX_JOINT_VELOCITY
    )

logger = logging.getLogger(__name__)


def validate_position(position: np.ndarray) -> None:
    """
    Validar posición 3D.
    
    Args:
        position: Array numpy de 3 elementos [x, y, z]
        
    Raises:
        ValidationError: Si la posición es inválida
    """
    if not isinstance(position, np.ndarray):
        raise ValidationError(
            "position",
            position,
            "Position must be a numpy array"
        )
    
    if position.shape != (3,):
        raise ValidationError(
            "position",
            position,
            f"Position must have shape (3,), got {position.shape}"
        )
    
    if not np.all(np.isfinite(position)):
        raise ValidationError(
            "position",
            position,
            "Position contains non-finite values"
        )


def validate_orientation(orientation: np.ndarray) -> None:
    """
    Validar orientación (quaternion).
    
    Args:
        orientation: Array numpy de 4 elementos [qx, qy, qz, qw]
        
    Raises:
        ValidationError: Si la orientación es inválida
    """
    if not isinstance(orientation, np.ndarray):
        raise ValidationError(
            "orientation",
            orientation,
            "Orientation must be a numpy array"
        )
    
    if orientation.shape != (4,):
        raise ValidationError(
            "orientation",
            orientation,
            f"Orientation must have shape (4,), got {orientation.shape}"
        )
    
    if not np.all(np.isfinite(orientation)):
        raise ValidationError(
            "orientation",
            orientation,
            "Orientation contains non-finite values"
        )
    
    # Verificar que sea un quaternion válido (norma cercana a 1)
    norm = np.linalg.norm(orientation)
    if abs(norm - 1.0) > 0.1:
        raise ValidationError(
            "orientation",
            orientation,
            f"Quaternion norm should be ~1.0, got {norm}"
        )


def validate_trajectory_point(point: TrajectoryPoint) -> None:
    """
    Validar punto de trayectoria.
    
    Args:
        point: Punto de trayectoria
        
    Raises:
        ValidationError: Si el punto es inválido
    """
    validate_position(point.position)
    validate_orientation(point.orientation)
    
    if point.velocity is not None:
        if not isinstance(point.velocity, np.ndarray) or point.velocity.shape != (3,):
            raise ValidationError(
                "velocity",
                point.velocity,
                "Velocity must be a numpy array of shape (3,)"
            )
        if not np.all(np.isfinite(point.velocity)):
            raise ValidationError(
                "velocity",
                point.velocity,
                "Velocity contains non-finite values"
            )
    
    if point.acceleration is not None:
        if not isinstance(point.acceleration, np.ndarray) or point.acceleration.shape != (3,):
            raise ValidationError(
                "acceleration",
                point.acceleration,
                "Acceleration must be a numpy array of shape (3,)"
            )
        if not np.all(np.isfinite(point.acceleration)):
            raise ValidationError(
                "acceleration",
                point.acceleration,
                "Acceleration contains non-finite values"
            )
        
        # Verificar magnitud de aceleración
        accel_magnitude = np.linalg.norm(point.acceleration)
        if accel_magnitude > MAX_ACCELERATION_MAGNITUDE:
            raise ValidationError(
                "acceleration",
                point.acceleration,
                f"Acceleration magnitude {accel_magnitude} exceeds maximum {MAX_ACCELERATION_MAGNITUDE}"
            )
    
    if point.timestamp < 0:
        raise ValidationError(
            "timestamp",
            point.timestamp,
            "Timestamp must be non-negative"
        )


def validate_trajectory(trajectory: List[TrajectoryPoint]) -> None:
    """
    Validar trayectoria completa.
    
    Args:
        trajectory: Lista de puntos de trayectoria
        
    Raises:
        TrajectoryInvalidError: Si la trayectoria es inválida
    """
    if not trajectory:
        raise TrajectoryInvalidError("Trajectory is empty")
    
    if len(trajectory) < 2:
        raise TrajectoryInvalidError("Trajectory must have at least 2 points")
    
    # Validar cada punto
    for i, point in enumerate(trajectory):
        try:
            validate_trajectory_point(point)
        except ValidationError as e:
            raise TrajectoryInvalidError(f"Invalid point at index {i}: {e}")
    
    # Validar continuidad
    for i in range(1, len(trajectory)):
        prev_pos = trajectory[i - 1].position
        curr_pos = trajectory[i].position
        distance = np.linalg.norm(curr_pos - prev_pos)
        
        if distance > MAX_JUMP_DISTANCE:
            raise TrajectoryInvalidError(
                f"Large jump detected at point {i}: {distance:.3f}m "
                f"(max: {MAX_JUMP_DISTANCE:.3f}m)"
            )
    
    # Validar timestamps crecientes
    for i in range(1, len(trajectory)):
        if trajectory[i].timestamp < trajectory[i - 1].timestamp:
            raise TrajectoryInvalidError(
                f"Non-monotonic timestamps at point {i}"
            )


def validate_obstacle(obstacle: np.ndarray) -> None:
    """
    Validar obstáculo (bounding box).
    
    Args:
        obstacle: Array numpy [min_x, min_y, min_z, max_x, max_y, max_z]
        
    Raises:
        InvalidObstacleError: Si el obstáculo es inválido
    """
    if not isinstance(obstacle, np.ndarray):
        raise InvalidObstacleError(obstacle)
    
    if obstacle.shape != (6,):
        raise InvalidObstacleError(
            f"Obstacle must have shape (6,), got {obstacle.shape}"
        )
    
    if not np.all(np.isfinite(obstacle)):
        raise InvalidObstacleError("Obstacle contains non-finite values")
    
    min_corner = obstacle[:3]
    max_corner = obstacle[3:]
    
    # Verificar que min < max para cada dimensión
    if not np.all(max_corner > min_corner):
        raise InvalidObstacleError(
            f"Invalid bounding box: min {min_corner} must be < max {max_corner}"
        )


def validate_obstacles(obstacles: List[np.ndarray]) -> None:
    """
    Validar lista de obstáculos.
    
    Args:
        obstacles: Lista de obstáculos
        
    Raises:
        InvalidObstacleError: Si algún obstáculo es inválido
    """
    if not isinstance(obstacles, list):
        raise InvalidObstacleError(f"Obstacles must be a list, got {type(obstacles)}")
    
    for i, obstacle in enumerate(obstacles):
        try:
            validate_obstacle(obstacle)
        except InvalidObstacleError as e:
            raise InvalidObstacleError(f"Invalid obstacle at index {i}: {e}")


def validate_joint_angles(angles: List[float], limits: Optional[List[Tuple[float, float]]] = None) -> None:
    """
    Validar ángulos de articulaciones.
    
    Args:
        angles: Lista de ángulos
        limits: Límites opcionales [(min, max), ...]
        
    Raises:
        ValidationError: Si los ángulos son inválidos
    """
    if not isinstance(angles, list):
        raise ValidationError(
            "angles",
            angles,
            "Angles must be a list"
        )
    
    if not angles:
        raise ValidationError(
            "angles",
            angles,
            "Angles list is empty"
        )
    
    for i, angle in enumerate(angles):
        if not isinstance(angle, (int, float)):
            raise ValidationError(
                f"angles[{i}]",
                angle,
                "Angle must be a number"
            )
        
        if not np.isfinite(angle):
            raise ValidationError(
                f"angles[{i}]",
                angle,
                "Angle must be finite"
            )
        
        if abs(angle) > MAX_JOINT_ANGLE:
            raise ValidationError(
                f"angles[{i}]",
                angle,
                f"Angle {angle} exceeds maximum {MAX_JOINT_ANGLE}"
            )
        
        # Validar límites si se proporcionan
        if limits and i < len(limits):
            min_angle, max_angle = limits[i]
            if angle < min_angle or angle > max_angle:
                raise ValidationError(
                    f"angles[{i}]",
                    angle,
                    f"Angle {angle} outside limits [{min_angle}, {max_angle}]"
                )


def validate_joint_velocities(velocities: List[float]) -> None:
    """
    Validar velocidades de articulaciones.
    
    Args:
        velocities: Lista de velocidades
        
    Raises:
        ValidationError: Si las velocidades son inválidas
    """
    if not isinstance(velocities, list):
        raise ValidationError(
            "velocities",
            velocities,
            "Velocities must be a list"
        )
    
    for i, velocity in enumerate(velocities):
        if not isinstance(velocity, (int, float)):
            raise ValidationError(
                f"velocities[{i}]",
                velocity,
                "Velocity must be a number"
            )
        
        if not np.isfinite(velocity):
            raise ValidationError(
                f"velocities[{i}]",
                velocity,
                "Velocity must be finite"
            )
        
        if abs(velocity) > MAX_JOINT_VELOCITY:
            raise ValidationError(
                f"velocities[{i}]",
                velocity,
                f"Velocity {velocity} exceeds maximum {MAX_JOINT_VELOCITY}"
            )






