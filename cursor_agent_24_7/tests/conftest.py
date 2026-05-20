"""
Pytest Configuration
====================

Configuración para tests.
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Agregar al path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def event_loop():
    """Crear event loop para tests async"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()



