"""
Fixtures avanzadas para testing.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime

from fastapi.testclient import TestClient
from httpx import AsyncClient

from config.settings import settings
from core.storage import TaskStorage
from core.github_client import GitHubClient
from core.services import (
    CacheService, MetricsService, RateLimitService, LLMService,
    AuditService, NotificationService, MonitoringService
)


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Event loop para tests async."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def temp_storage_dir() -> Generator[Path, None, None]:
    """Directorio temporal para storage."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
async def real_storage(temp_storage_dir: Path) -> AsyncGenerator[TaskStorage, None]:
    """Storage real con base de datos temporal."""
    db_path = temp_storage_dir / "test.db"
    storage = TaskStorage(db_path=str(db_path))
    await storage.init_db()
    yield storage
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture(scope="function")
def mock_storage() -> AsyncMock:
    """Mock completo de TaskStorage."""
    storage = AsyncMock(spec=TaskStorage)
    storage.init_db = AsyncMock(return_value=None)
    storage.save_task = AsyncMock(return_value=True)
    storage.get_task = AsyncMock(return_value=None)
    storage.get_tasks = AsyncMock(return_value=[])
    storage.update_task_status = AsyncMock(return_value=True)
    storage.delete_task = AsyncMock(return_value=True)
    storage.get_agent_state = AsyncMock(return_value={"is_running": False})
    storage.save_agent_state = AsyncMock(return_value=True)
    return storage


@pytest.fixture(scope="function")
def mock_github_client() -> Mock:
    """Mock completo de GitHubClient."""
    client = Mock(spec=GitHubClient)
    client.token = "test-token"
    client.get_repository = AsyncMock(return_value=Mock())
    client.get_repository_info = AsyncMock(return_value={
        "name": "test-repo",
        "full_name": "test/test-repo",
        "description": "Test repository",
        "default_branch": "main",
        "is_private": False
    })
    client.create_file = AsyncMock(return_value=True)
    client.update_file = AsyncMock(return_value=True)
    client.delete_file = AsyncMock(return_value=True)
    client.create_branch = AsyncMock(return_value=True)
    client.create_pull_request = AsyncMock(return_value={
        "url": "https://github.com/test/test-repo/pull/1",
        "number": 1
    })
    client.get_file_content = AsyncMock(return_value="file content")
    return client


@pytest.fixture(scope="function")
def mock_cache_service() -> Mock:
    """Mock de CacheService."""
    cache = Mock(spec=CacheService)
    cache.get = Mock(return_value=None)
    cache.set = Mock(return_value=True)
    cache.delete = Mock(return_value=True)
    cache.clear = Mock(return_value=None)
    cache.get_stats = Mock(return_value={
        "hits": 0,
        "misses": 0,
        "sets": 0,
        "evictions": 0,
        "size": 0,
        "max_size": 1000,
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
    rate_limit.check_rate_limit = Mock(return_value=1000)
    rate_limit.get_stats = Mock(return_value={
        "limit": 5000,
        "used": 0,
        "remaining": 5000,
        "reset_at": None
    })
    return rate_limit


@pytest.fixture(scope="function")
def mock_llm_service() -> Mock:
    """Mock de LLMService."""
    llm = Mock(spec=LLMService)
    llm.generate = AsyncMock(return_value=Mock(
        content="Test response",
        error=None,
        latency_ms=100.0,
        tokens_used=50
    ))
    llm.analyze_code = AsyncMock(return_value=Mock(
        content="Code analysis result",
        error=None
    ))
    llm.generate_instruction = AsyncMock(return_value=Mock(
        content="create file: test.py",
        error=None
    ))
    llm.get_stats = Mock(return_value={
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0
    })
    return llm


@pytest.fixture(scope="function")
def mock_audit_service() -> Mock:
    """Mock de AuditService."""
    audit = Mock(spec=AuditService)
    audit.log_event = Mock(return_value=None)
    audit.get_events = Mock(return_value=[])
    audit.get_stats = Mock(return_value={
        "total_events": 0,
        "events_by_type": {}
    })
    return audit


@pytest.fixture(scope="function")
def mock_notification_service() -> Mock:
    """Mock de NotificationService."""
    notification = Mock(spec=NotificationService)
    notification.send = AsyncMock(return_value=Mock(
        id="notif-123",
        title="Test",
        message="Test message",
        level="info"
    ))
    notification.get_notifications = Mock(return_value=[])
    notification.get_stats = Mock(return_value={
        "total": 0,
        "by_level": {}
    })
    return notification


@pytest.fixture(scope="function")
def mock_monitoring_service() -> Mock:
    """Mock de MonitoringService."""
    monitoring = Mock(spec=MonitoringService)
    monitoring.record_metric = Mock(return_value=None)
    monitoring.set_gauge = Mock(return_value=None)
    monitoring.increment_counter = Mock(return_value=None)
    monitoring.get_current_metrics = Mock(return_value={})
    monitoring.get_stats = Mock(return_value={})
    return monitoring


@pytest.fixture(scope="function")
def test_client() -> TestClient:
    """Cliente de test síncrono para FastAPI."""
    from main import create_app
    app = create_app()
    return TestClient(app)


@pytest.fixture(scope="function")
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Cliente de test async para FastAPI."""
    from main import create_app
    app = create_app()
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture(scope="function")
def sample_task() -> dict:
    """Tarea de ejemplo."""
    return {
        "id": "test-task-123",
        "repository_owner": "test",
        "repository_name": "test-repo",
        "instruction": "create file: test.py",
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
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


@pytest.fixture(scope="function")
def sample_llm_response() -> dict:
    """Respuesta de LLM de ejemplo."""
    return {
        "content": "Test LLM response",
        "error": None,
        "latency_ms": 100.0,
        "tokens_used": 50,
        "model": "openai/gpt-4o-mini"
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
    """Fixture para modificar variables de entorno."""
    def _set_env(**kwargs):
        for key, value in kwargs.items():
            monkeypatch.setenv(key, str(value))
    return _set_env



