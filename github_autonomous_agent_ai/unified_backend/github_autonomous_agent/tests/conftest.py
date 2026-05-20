"""
Pytest configuration y fixtures compartidas.
"""

import pytest
import asyncio
from typing import AsyncGenerator, Generator, Any
from unittest.mock import Mock, AsyncMock

from fastapi.testclient import TestClient
from config.settings import settings
from config.di_setup import setup_dependencies, get_service
from core.storage import TaskStorage
from core.github_client import GitHubClient
from core.services import CacheService, MetricsService, RateLimitService, LLMService


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Crear event loop para tests async."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def mock_storage() -> AsyncMock:
    """Mock de TaskStorage."""
    storage = AsyncMock(spec=TaskStorage)
    storage.init_db = AsyncMock(return_value=None)
    storage.save_task = AsyncMock(return_value=None)
    storage.get_task = AsyncMock(return_value={})
    storage.list_tasks = AsyncMock(return_value=[])
    storage.update_task_status = AsyncMock(return_value=None)
    return storage


@pytest.fixture(scope="function")
def mock_github_client() -> Mock:
    """Mock de GitHubClient."""
    client = Mock(spec=GitHubClient)
    client.get_repository = Mock(return_value=Mock())
    client.get_repository_info = Mock(return_value={
        "name": "test-repo",
        "full_name": "test/test-repo",
        "description": "Test repository"
    })
    client.create_file = Mock(return_value=True)
    client.update_file = Mock(return_value=True)
    client.create_branch = Mock(return_value=True)
    client.create_pull_request = Mock(return_value={"url": "https://github.com/test/test-repo/pull/1"})
    return client


@pytest.fixture(scope="function")
def mock_cache_service() -> Mock:
    """Mock de CacheService."""
    cache = Mock(spec=CacheService)
    cache.get = Mock(return_value=None)
    cache.set = Mock(return_value=None)
    cache.delete = Mock(return_value=True)
    cache.get_stats = Mock(return_value={
        "hits": 0,
        "misses": 0,
        "hit_rate": 0.0
    })
    return cache


@pytest.fixture(scope="function")
def mock_metrics_service() -> Mock:
    """Mock de MetricsService."""
    metrics = Mock(spec=MetricsService)
    metrics.record_task = Mock(return_value=None)
    metrics.record_api_request = Mock(return_value=None)
    metrics.record_error = Mock(return_value=None)
    metrics.start_timer = Mock(return_value=None)
    metrics.stop_timer = Mock(return_value=0.5)
    metrics.get_metrics = Mock(return_value={})
    return metrics


@pytest.fixture(scope="function")
def mock_rate_limit_service() -> Mock:
    """Mock de RateLimitService."""
    rate_limit = Mock(spec=RateLimitService)
    rate_limit.check_rate_limit = Mock(return_value=True)
    rate_limit.get_stats = Mock(return_value={
        "limit": 5000,
        "used": 0,
        "remaining": 5000
    })
    return rate_limit


@pytest.fixture(scope="function")
def mock_llm_service() -> Mock:
    """Mock de LLMService."""
    llm = Mock(spec=LLMService)
    llm.generate = AsyncMock(return_value=Mock(
        content="Test response",
        error=None,
        latency_ms=100.0
    ))
    llm.analyze_code = AsyncMock(return_value=Mock(
        content="Code analysis result",
        error=None
    ))
    llm.generate_instruction = AsyncMock(return_value=Mock(
        content="create file: test.py",
        error=None
    ))
    return llm


@pytest.fixture(scope="function")
def test_client() -> TestClient:
    """Cliente de test para FastAPI."""
    from main import create_app
    app = create_app()
    return TestClient(app)


@pytest.fixture(scope="function")
def sample_task() -> dict:
    """Tarea de ejemplo para tests."""
    return {
        "id": "test-task-123",
        "repository_owner": "test",
        "repository_name": "test-repo",
        "instruction": "create file: test.py",
        "status": "pending",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "metadata": {}
    }


@pytest.fixture(scope="function")
def sample_repository_info() -> dict:
    """Información de repositorio de ejemplo."""
    return {
        "name": "test-repo",
        "full_name": "test/test-repo",
        "description": "Test repository",
        "url": "https://github.com/test/test-repo",
        "default_branch": "main",
        "language": "Python",
        "stars": 100,
        "forks": 10,
        "is_private": False
    }


@pytest.fixture(autouse=True)
def reset_di_container():
    """Resetear contenedor DI antes de cada test."""
    from core.di import _container
    _container = None
    yield
    _container = None


@pytest.fixture(scope="function")
def temp_env(monkeypatch):
    """Fixture para modificar variables de entorno temporalmente."""
    def _set_env(**kwargs: Any):
        for key, value in kwargs.items():
            monkeypatch.setenv(key, str(value))
    return _set_env

