"""
Test Helpers
============

Funciones helper para tests.
"""

import numpy as np
from typing import List
from ..core.trajectory_optimizer import TrajectoryPoint


def create_trajectory_point(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    qx: float = 0.0,
    qy: float = 0.0,
    qz: float = 0.0,
    qw: float = 1.0,
    timestamp: float = 0.0
) -> TrajectoryPoint:
    """
    Crear punto de trayectoria para tests.
    
    Args:
        x, y, z: Posición
        qx, qy, qz, qw: Orientación (quaternion)
        timestamp: Timestamp
        
    Returns:
        TrajectoryPoint
    """
    return TrajectoryPoint(
        position=np.array([x, y, z]),
        orientation=np.array([qx, qy, qz, qw]),
        timestamp=timestamp
    )


def create_linear_trajectory(
    start: TrajectoryPoint,
    goal: TrajectoryPoint,
    num_points: int = 10
) -> List[TrajectoryPoint]:
    """
    Crear trayectoria lineal para tests.
    
    Args:
        start: Punto inicial
        goal: Punto objetivo
        num_points: Número de puntos
        
    Returns:
        Lista de puntos de trayectoria
    """
    trajectory = []
    for i in range(num_points):
        alpha = i / (num_points - 1) if num_points > 1 else 0.0
        position = start.position + alpha * (goal.position - start.position)
        orientation = start.orientation  # Simplificado
        timestamp = start.timestamp + alpha * (goal.timestamp - start.timestamp)
        
        trajectory.append(TrajectoryPoint(
            position=position,
            orientation=orientation,
            timestamp=timestamp
        ))
    
    return trajectory


def assert_trajectory_valid(trajectory: List[TrajectoryPoint]):
    """
    Assert que una trayectoria es válida.
    
    Args:
        trajectory: Trayectoria a validar
    """
    assert len(trajectory) > 0, "Trajectory must not be empty"
    
    for i, point in enumerate(trajectory):
        assert point.position.shape == (3,), f"Point {i} position must be 3D"
        assert point.orientation.shape == (4,), f"Point {i} orientation must be quaternion"
        assert np.all(np.isfinite(point.position)), f"Point {i} position must be finite"
        assert np.all(np.isfinite(point.orientation)), f"Point {i} orientation must be finite"
    
    # Verificar continuidad básica
    for i in range(1, len(trajectory)):
        prev_pos = trajectory[i - 1].position
        curr_pos = trajectory[i].position
        distance = np.linalg.norm(curr_pos - prev_pos)
        assert distance < 1.0, f"Large jump at point {i}: {distance}m"


def assert_points_close(
    p1: np.ndarray,
    p2: np.ndarray,
    tolerance: float = 1e-6
):
    """
    Assert que dos puntos están cerca.
    
    Args:
        p1: Primer punto
        p2: Segundo punto
        tolerance: Tolerancia
    """
    distance = np.linalg.norm(p2 - p1)
    assert distance < tolerance, f"Points not close: distance={distance}, tolerance={tolerance}"


def create_obstacle(
    min_x: float,
    min_y: float,
    min_z: float,
    max_x: float,
    max_y: float,
    max_z: float
) -> np.ndarray:
    """
    Crear obstáculo (bounding box) para tests.
    
    Args:
        min_x, min_y, min_z: Esquina mínima
        max_x, max_y, max_z: Esquina máxima
        
    Returns:
        Array numpy [min_x, min_y, min_z, max_x, max_y, max_z]
    """
    return np.array([min_x, min_y, min_z, max_x, max_y, max_z])






