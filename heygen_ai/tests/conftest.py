import asyncio
import os
from typing import Any
import pytest

"""
Minimal, safe pytest configuration for the heygen_ai feature to avoid brittle imports.
Provides core fixtures used by unit tests in this feature without importing heavy app modules.
"""


def pytest_configure(config) -> Any:
    config.addinivalue_line("markers", "unit: marks tests as unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def caplog_structured(caplog) -> Any:
    # Basic passthrough for tests expecting this fixture
    return caplog
