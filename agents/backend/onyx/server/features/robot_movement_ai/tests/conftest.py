"""
Pytest Configuration - Arquitectura Mejorada
============================================
"""

import pytest
import asyncio
import os

# Configurar variables de entorno para tests
os.environ['REPOSITORY_TYPE'] = 'in_memory'
os.environ['ENABLE_EVENT_BUS'] = 'false'


@pytest.fixture(scope="session")
def event_loop():
    """Crear event loop para tests async."""
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def reset_di():
    """Resetear DI antes de cada test."""
    from core.architecture.di_setup import _global_di_setup
    global _global_di_setup
    _global_di_setup = None
    yield
    _global_di_setup = None
