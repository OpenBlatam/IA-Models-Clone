"""
Test Utilities
==============

Utilidades para testing y desarrollo.
"""

from typing import Any, Callable, Optional, Dict, List
from contextlib import contextmanager
import asyncio
from unittest.mock import Mock, MagicMock, patch
import pytest


class AsyncMock(Mock):
    """Mock para funciones async."""
    
    async def __call__(self, *args, **kwargs):
        """Llamar mock async."""
        return super().__call__(*args, **kwargs)


def create_async_mock(**kwargs) -> AsyncMock:
    """
    Crear mock async.
    
    Args:
        **kwargs: Argumentos para el mock
    
    Returns:
        AsyncMock
    """
    return AsyncMock(**kwargs)


@contextmanager
def assert_raises(exception_type: type, message: Optional[str] = None):
    """
    Context manager para verificar excepciones.
    
    Args:
        exception_type: Tipo de excepción esperada
        message: Mensaje esperado (opcional)
    
    Yields:
        None
    
    Raises:
        AssertionError: Si no se lanza la excepción esperada
    """
    try:
        yield
        raise AssertionError(f"Expected {exception_type.__name__} but no exception was raised")
    except exception_type as e:
        if message and message not in str(e):
            raise AssertionError(f"Expected message '{message}' but got '{str(e)}'")
        return e


async def run_async_test(coro: Callable, timeout: float = 5.0) -> Any:
    """
    Ejecutar test async con timeout.
    
    Args:
        coro: Coroutine a ejecutar
        timeout: Timeout en segundos
    
    Returns:
        Resultado de la coroutine
    
    Raises:
        asyncio.TimeoutError: Si excede el timeout
    """
    return await asyncio.wait_for(coro, timeout=timeout)


def create_test_config(**overrides) -> Dict[str, Any]:
    """
    Crear configuración de prueba.
    
    Args:
        **overrides: Valores a sobrescribir
    
    Returns:
        Diccionario de configuración
    """
    default_config = {
        "robot_brand": "generic",
        "ros_enabled": False,
        "log_level": "DEBUG",
        "feedback_frequency": 100,
        "api_port": 8011,
        "robot_ip": "127.0.0.1",
        "robot_port": 30001,
    }
    default_config.update(overrides)
    return default_config


class MockRobotEngine:
    """Mock del RobotMovementEngine para tests."""
    
    def __init__(self):
        """Inicializar mock."""
        self.is_moving = False
        self.current_joint_state = None
        self.movement_history = []
        self.total_movements = 0
    
    async def initialize(self):
        """Mock initialize."""
        pass
    
    async def shutdown(self):
        """Mock shutdown."""
        pass
    
    async def move_to_pose(self, target_pose):
        """Mock move_to_pose."""
        self.is_moving = True
        self.total_movements += 1
        await asyncio.sleep(0.01)  # Simular movimiento
        self.is_moving = False
        return True
    
    async def stop_movement(self):
        """Mock stop_movement."""
        self.is_moving = False
    
    def get_status(self):
        """Mock get_status."""
        return {
            "is_moving": self.is_moving,
            "total_movements": self.total_movements
        }


def patch_config(**overrides):
    """
    Decorador para parchear configuración en tests.
    
    Args:
        **overrides: Valores a sobrescribir
    
    Returns:
        Decorador
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with patch('robot_movement_ai.config.robot_config.RobotConfig') as mock_config:
                config = create_test_config(**overrides)
                mock_config.return_value = type('Config', (), config)()
                return func(*args, **kwargs)
        return wrapper
    return decorator

