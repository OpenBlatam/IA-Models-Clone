"""
Testing Helpers - Utilidades adicionales para testing
======================================================

Funciones helper adicionales para facilitar testing del servidor MCP.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable, Awaitable
from contextlib import contextmanager
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@contextmanager
def override_settings(**settings):
    """
    Context manager para sobrescribir configuración temporalmente.
    
    Args:
        **settings: Configuración a sobrescribir
        
    Yields:
        None
    """
    import os
    original = {}
    
    try:
        # Guardar valores originales
        for key, value in settings.items():
            env_key = f"MCP_{key.upper()}"
            original[env_key] = os.environ.get(env_key)
            os.environ[env_key] = str(value)
        
        yield
    finally:
        # Restaurar valores originales
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_async(coro: Awaitable[Any]) -> Any:
    """
    Ejecutar coroutine en un nuevo event loop.
    
    Útil para testing de código asíncrono.
    
    Args:
        coro: Coroutine a ejecutar
        
    Returns:
        Resultado de la coroutine
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


def create_test_context(
    user_id: Optional[str] = None,
    resource_id: Optional[str] = None,
    operation: Optional[str] = None,
    additional_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Crear contexto de prueba para requests.
    
    Args:
        user_id: ID del usuario (opcional)
        resource_id: ID del recurso (opcional)
        operation: Operación (opcional)
        additional_data: Datos adicionales (opcional)
        
    Returns:
        Diccionario con contexto de prueba
    """
    context = {
        "timestamp": datetime.utcnow().isoformat(),
        "test": True,
    }
    
    if user_id:
        context["user_id"] = user_id
    if resource_id:
        context["resource_id"] = resource_id
    if operation:
        context["operation"] = operation
    if additional_data:
        context.update(additional_data)
    
    return context


def wait_for_condition(
    condition: Callable[[], bool],
    timeout: float = 5.0,
    interval: float = 0.1,
    error_message: Optional[str] = None
) -> bool:
    """
    Esperar hasta que una condición sea verdadera.
    
    Útil para testing de operaciones asíncronas o con delay.
    
    Args:
        condition: Función que retorna True cuando la condición se cumple
        timeout: Tiempo máximo de espera en segundos
        interval: Intervalo entre verificaciones en segundos
        error_message: Mensaje de error personalizado (opcional)
        
    Returns:
        True si la condición se cumplió, False si timeout
        
    Raises:
        TimeoutError: Si la condición no se cumple en el tiempo especificado
    """
    import time
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition():
            return True
        time.sleep(interval)
    
    if error_message:
        raise TimeoutError(error_message)
    raise TimeoutError(f"Condition not met within {timeout} seconds")


async def wait_for_condition_async(
    condition: Callable[[], bool],
    timeout: float = 5.0,
    interval: float = 0.1,
    error_message: Optional[str] = None
) -> bool:
    """
    Esperar hasta que una condición sea verdadera (async).
    
    Args:
        condition: Función que retorna True cuando la condición se cumple
        timeout: Tiempo máximo de espera en segundos
        interval: Intervalo entre verificaciones en segundos
        error_message: Mensaje de error personalizado (opcional)
        
    Returns:
        True si la condición se cumplió
        
    Raises:
        TimeoutError: Si la condición no se cumple en el tiempo especificado
    """
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        if condition():
            return True
        await asyncio.sleep(interval)
    
    if error_message:
        raise TimeoutError(error_message)
    raise TimeoutError(f"Condition not met within {timeout} seconds")


def mock_time(monkeypatch, target_time: datetime):
    """
    Mock de tiempo para testing.
    
    Args:
        monkeypatch: pytest monkeypatch fixture
        target_time: Tiempo a mockear
    """
    def mock_now():
        return target_time
    
    monkeypatch.setattr("datetime.datetime", "utcnow", mock_now)
    monkeypatch.setattr("datetime.datetime", "now", lambda tz=None: target_time)


def assert_response_time(response_time: float, max_time: float, operation: str = "operation") -> None:
    """
    Assert que el tiempo de respuesta esté dentro del límite.
    
    Args:
        response_time: Tiempo de respuesta en segundos
        max_time: Tiempo máximo permitido en segundos
        operation: Nombre de la operación (para mensaje de error)
        
    Raises:
        AssertionError: Si el tiempo excede el límite
    """
    assert response_time <= max_time, \
        f"{operation} took {response_time:.3f}s, expected <= {max_time:.3f}s"

