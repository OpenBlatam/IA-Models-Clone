"""
Testing Utilities
==================

Utilidades para testing del sistema.
"""

import numpy as np
from typing import List, Any, Optional, Dict
from unittest.mock import Mock, MagicMock, AsyncMock
import asyncio

from .trajectory_optimizer import TrajectoryPoint, TrajectoryOptimizer
from .movement_engine import RobotMovementEngine
from ..config.robot_config import RobotConfig, RobotBrand


def create_mock_trajectory_point(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0
) -> TrajectoryPoint:
    """
    Crear TrajectoryPoint mock para tests.
    
    Args:
        x, y, z: Posición
        
    Returns:
        TrajectoryPoint
    """
    return TrajectoryPoint(
        position=np.array([x, y, z]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        timestamp=0.0
    )


def create_mock_trajectory(
    start: TrajectoryPoint,
    goal: TrajectoryPoint,
    num_points: int = 10
) -> List[TrajectoryPoint]:
    """
    Crear trayectoria mock.
    
    Args:
        start: Punto inicial
        goal: Punto objetivo
        num_points: Número de puntos
        
    Returns:
        Lista de TrajectoryPoint
    """
    trajectory = []
    for i in range(num_points):
        alpha = i / (num_points - 1) if num_points > 1 else 0.0
        position = start.position + alpha * (goal.position - start.position)
        trajectory.append(TrajectoryPoint(
            position=position,
            orientation=start.orientation,
            timestamp=i * 0.01
        ))
    return trajectory


def create_mock_optimizer() -> TrajectoryOptimizer:
    """Crear optimizador mock."""
    return TrajectoryOptimizer()


def create_mock_movement_engine() -> RobotMovementEngine:
    """Crear motor de movimiento mock."""
    config = RobotConfig(
        robot_brand=RobotBrand.GENERIC,
        ros_enabled=False,
        camera_enabled=False
    )
    return RobotMovementEngine(config)


def create_mock_obstacle(
    center: np.ndarray,
    size: float = 0.2
) -> np.ndarray:
    """
    Crear obstáculo mock.
    
    Args:
        center: Centro del obstáculo [x, y, z]
        size: Tamaño del obstáculo
        
    Returns:
        Bounding box [min_x, min_y, min_z, max_x, max_y, max_z]
    """
    half_size = size / 2
    return np.array([
        center[0] - half_size,
        center[1] - half_size,
        center[2] - half_size,
        center[0] + half_size,
        center[1] + half_size,
        center[2] + half_size,
    ])


class AsyncTestCase:
    """
    Clase base para tests async.
    
    Proporciona utilidades para tests asíncronos.
    """
    
    @staticmethod
    def run_async(coro):
        """Ejecutar coroutine en test."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def assert_trajectory_valid(
    trajectory: List[TrajectoryPoint],
    min_points: int = 2
) -> None:
    """
    Assert que trayectoria es válida.
    
    Args:
        trajectory: Trayectoria a validar
        min_points: Número mínimo de puntos
    """
    assert len(trajectory) >= min_points, f"Trajectory must have at least {min_points} points"
    
    for i, point in enumerate(trajectory):
        assert point.position.shape == (3,), f"Point {i} must have 3D position"
        assert point.orientation.shape == (4,), f"Point {i} must have quaternion orientation"
        assert np.all(np.isfinite(point.position)), f"Point {i} position must be finite"
        assert np.all(np.isfinite(point.orientation)), f"Point {i} orientation must be finite"


def assert_points_close(
    p1: np.ndarray,
    p2: np.ndarray,
    tolerance: float = 1e-6
) -> None:
    """
    Assert que dos puntos están cerca.
    
    Args:
        p1: Primer punto
        p2: Segundo punto
        tolerance: Tolerancia
    """
    distance = np.linalg.norm(p2 - p1)
    assert distance < tolerance, f"Points not close: distance={distance}, tolerance={tolerance}"


def create_test_config(**overrides) -> RobotConfig:
    """
    Crear configuración de test.
    
    Args:
        **overrides: Valores a sobrescribir
        
    Returns:
        RobotConfig para tests
    """
    config = RobotConfig(
        robot_brand=RobotBrand.GENERIC,
        ros_enabled=False,
        camera_enabled=False,
        feedback_frequency=100,
        api_port=8011
    )
    
    # Aplicar overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config






