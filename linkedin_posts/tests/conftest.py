from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import pytest
import asyncio
from typing import AsyncGenerator, Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import os
from datetime import datetime, timedelta
import orjson
from fastapi.testclient import TestClient
from httpx import AsyncClient
import redis.asyncio as redis
import logging
import time
from typing import Any, List, Dict, Optional
from fastapi import FastAPI
from pathlib import Path
import sys

# Ensure repository root is on sys.path so absolute imports like 'agents.*' work
_this_file = Path(__file__).resolve()
_repo_root = None
for parent in _this_file.parents:
    if parent.name == "agents":
        _repo_root = parent.parent
        break
if _repo_root and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Attempt absolute imports; on failure, define stubs to avoid import-time errors
LINKEDIN_IMPORTS_OK = True
try:
    from agents.backend.onyx.server.features.linkedin_posts.core.domain.entities.linkedin_post import (
        LinkedInPost, PostStatus, PostType, PostTone,
    )
    from agents.backend.onyx.server.features.linkedin_posts.application.use_cases.linkedin_post_use_cases import (
        LinkedInPostUseCases,
    )
    from agents.backend.onyx.server.features.linkedin_posts.infrastructure.repositories.linkedin_post_repository import (
        LinkedInPostRepository,
    )
    from agents.backend.onyx.server.features.linkedin_posts.shared.cache import CacheManager
    from agents.backend.onyx.server.features.linkedin_posts.shared.config import Settings
    from agents.backend.onyx.server.features.linkedin_posts.presentation.api.linkedin_post_router_v2 import (
        router,
    )
except Exception:
    LINKEDIN_IMPORTS_OK = False
    LinkedInPost = None  # type: ignore
    PostStatus = None  # type: ignore
    PostType = None  # type: ignore
    PostTone = None  # type: ignore
    LinkedInPostRepository = None  # type: ignore
    LinkedInPostUseCases = None  # type: ignore
    CacheManager = None  # type: ignore
    Settings = None  # type: ignore
    router = None  # type: ignore
"""
Pytest Configuration and Fixtures
=================================

Advanced testing setup with fixtures, mocks, and test data.
"""



# Import our modules


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings():
    """Test settings with overrides."""
    if Settings is None:
        pytest.skip("Settings not available")
    return Settings(
        DATABASE_URL="sqlite:///./test.db",
        REDIS_URL="redis://localhost:6379/1",  # Use DB 1 for testing
        SECRET_KEY="test-secret-key-for-testing-only",
        ENVIRONMENT="testing",
        DEBUG=True,
        TESTING=True,
        ENABLE_CACHE=False,  # Disable cache for unit tests
        RATE_LIMIT_ENABLED=False,
        ENABLE_METRICS=False
    )


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    if redis is None:
        pytest.skip("redis not available")
    with patch('redis.asyncio.Redis') as mock:
        mock_client = AsyncMock()
        mock.from_url.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        mock_client.set.return_value = True
        mock_client.delete.return_value = 1
        mock_client.flushdb.return_value = True
        yield mock_client


@pytest.fixture
def mock_cache_manager(mock_redis) -> Any:
    """Mock cache manager."""
    with patch('linkedin_posts.shared.cache.cache_manager') as mock:
        mock.get.return_value = None
        mock.set.return_value = True
        mock.delete.return_value = True
        mock.clear.return_value = True
        mock.get_many.return_value = {}
        mock.set_many.return_value = True
        yield mock


@pytest.fixture
def sample_linkedin_post():
    """Sample LinkedIn post for testing."""
    if not LINKEDIN_IMPORTS_OK or LinkedInPost is None:
        pytest.skip("LinkedIn domain not available")
    return LinkedInPost(
        id="test-post-123",
        content="This is a test LinkedIn post for automated testing purposes.",
        post_type=PostType.ANNOUNCEMENT,
        tone=PostTone.PROFESSIONAL,
        target_audience="tech professionals",
        industry="technology",
        status=PostStatus.DRAFT,
        nlp_enhanced=True,
        nlp_processing_time=0.15,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )


@pytest.fixture
def sample_posts_batch():
    """Sample batch of LinkedIn posts for testing."""
    if not LINKEDIN_IMPORTS_OK or LinkedInPost is None:
        pytest.skip("LinkedIn domain not available")
    return [
        LinkedInPost(
            id=f"test-post-{i}",
            content=f"Test post number {i} for batch testing.",
            post_type=PostType.EDUCATIONAL,
            tone=PostTone.FRIENDLY,
            target_audience="professionals",
            industry="business",
            status=PostStatus.DRAFT,
            nlp_enhanced=False,
            nlp_processing_time=0.1,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        for i in range(1, 6)
    ]


@pytest.fixture
def mock_repository(sample_linkedin_post, sample_posts_batch) -> Any:
    """Mock repository with test data."""
    if LinkedInPostRepository is None:
        pytest.skip("Repository not available")
    mock_repo = AsyncMock(spec=LinkedInPostRepository)
    
    # Mock methods
    mock_repo.get_by_id.return_value = sample_linkedin_post
    mock_repo.list_posts.return_value = sample_posts_batch
    mock_repo.create.return_value = sample_linkedin_post
    mock_repo.update.return_value = sample_linkedin_post
    mock_repo.delete.return_value = True
    mock_repo.batch_create.return_value = sample_posts_batch
    mock_repo.batch_update.return_value = sample_posts_batch
    
    return mock_repo


@pytest.fixture
def mock_use_cases(mock_repository) -> Any:
    """Mock use cases with repository."""
    if LinkedInPostUseCases is None:
        pytest.skip("UseCases not available")
    return LinkedInPostUseCases(mock_repository)


@pytest.fixture
def mock_nlp_processor():
    """Mock NLP processor."""
    with patch('linkedin_posts.infrastructure.nlp.nlp_processor') as mock:
        mock.process_text.return_value = {
            "sentiment_score": 0.8,
            "readability_score": 75.5,
            "keywords": ["test", "linkedin", "post"],
            "entities": ["LinkedIn", "testing"],
            "processing_time": 0.12
        }
        mock.process_batch.return_value = [
            {
                "sentiment_score": 0.7,
                "readability_score": 70.0,
                "keywords": ["batch", "test"],
                "entities": ["batch"],
                "processing_time": 0.1
            }
        ] * 5
        yield mock


@pytest.fixture
def test_app():
    """Test FastAPI application."""
    if router is None:
        pytest.skip("router not available")
    app = FastAPI(title="LinkedIn Posts API Test")
    app.include_router(router)
    return app


@pytest.fixture
def test_client(test_app) -> Any:
    """Test client for FastAPI."""
    if TestClient is None:
        pytest.skip("TestClient not available")
    return TestClient(test_app)


@pytest.fixture
async def async_client(test_app) -> Any:
    """Async test client for FastAPI."""
    if AsyncClient is None:
        pytest.skip("AsyncClient not available")
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client


@pytest.fixture
def auth_headers():
    """Authentication headers for testing."""
    return {
        "Authorization": "Bearer test-jwt-token",
        "X-Request-ID": "test-request-123"
    }


@pytest.fixture
def sample_post_data():
    """Sample post data for API testing."""
    return {
        "content": "Exciting news! Our new AI-powered LinkedIn optimization tool is now live! 🚀",
        "post_type": "announcement",
        "tone": "professional",
        "target_audience": "tech professionals",
        "industry": "technology"
    }


@pytest.fixture
def sample_batch_data():
    """Sample batch data for API testing."""
    return [
        {
            "content": f"LinkedIn tip #{i}: Always engage with your network",
            "post_type": "educational",
            "tone": "friendly",
            "target_audience": "professionals",
            "industry": "business"
        }
        for i in range(1, 4)
    ]


@pytest.fixture
def mock_metrics():
    """Mock metrics collector."""
    with patch('linkedin_posts.shared.metrics.metrics_collector') as mock:
        mock.track_post_creation.return_value = None
        mock.track_batch_creation.return_value = None
        mock.get_performance_metrics.return_value = {
            "fast_nlp": {"avg_processing_time": 0.1, "cache_hit_rate": 0.85},
            "async_nlp": {"avg_processing_time": 0.05, "throughput": 100.0},
            "timestamp": datetime.utcnow()
        }
        yield mock


@pytest.fixture
def temp_db():
    """Temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    yield f"sqlite:///{db_path}"
    
    # Cleanup
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    with patch('linkedin_posts.shared.logging.get_logger') as mock:
        logger = MagicMock()
        mock.return_value = logger
        yield logger


@pytest.fixture
def performance_data():
    """Sample performance data for testing."""
    return {
        "request_count": 1000,
        "avg_response_time": 0.025,
        "p95_response_time": 0.080,
        "p99_response_time": 0.150,
        "error_rate": 0.001,
        "cache_hit_rate": 0.85,
        "throughput": 2000.0
    }


@pytest.fixture
def error_scenarios():
    """Common error scenarios for testing."""
    return {
        "database_connection_error": Exception("Database connection failed"),
        "redis_connection_error": Exception("Redis connection failed"),
        "nlp_service_error": Exception("NLP service unavailable"),
        "validation_error": ValueError("Invalid input data"),
        "rate_limit_error": Exception("Rate limit exceeded"),
        "authentication_error": Exception("Invalid token")
    }


@pytest.fixture
def mock_circuit_breaker():
    """Mock circuit breaker."""
    with patch('linkedin_posts.shared.middleware.circuit') as mock:
        mock.return_value = AsyncMock()
        yield mock


# Test data generators
class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def generate_posts(count: int = 10) -> List[Dict[str, Any]]:
        """Generate a list of test posts."""
        posts = []
        for i in range(count):
            posts.append({
                "id": f"post-{i}",
                "content": f"Test post content {i} with some keywords for testing purposes.",
                "post_type": PostType.EDUCATIONAL.value,
                "tone": PostTone.FRIENDLY.value,
                "target_audience": "professionals",
                "industry": "technology",
                "status": PostStatus.DRAFT.value,
                "nlp_enhanced": i % 2 == 0,
                "nlp_processing_time": 0.1 + (i * 0.01),
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            })
        return posts
    
    @staticmethod
    def generate_analytics_data() -> Dict[str, Any]:
        """Generate analytics test data."""
        return {
            "sentiment_score": 0.75,
            "readability_score": 78.5,
            "keywords": ["test", "analytics", "performance", "optimization"],
            "entities": ["LinkedIn", "API", "testing"],
            "processing_time": 0.125,
            "cached": False,
            "async_optimized": True
        }
    
    @staticmethod
    def generate_performance_metrics() -> Dict[str, Any]:
        """Generate performance metrics test data."""
        return {
            "fast_nlp_metrics": {
                "avg_processing_time": 0.08,
                "cache_hit_rate": 0.87,
                "throughput": 150.0,
                "error_rate": 0.002
            },
            "async_nlp_metrics": {
                "avg_processing_time": 0.04,
                "cache_hit_rate": 0.92,
                "throughput": 300.0,
                "error_rate": 0.001
            },
            "system_metrics": {
                "active_requests": 25,
                "total_requests": 15000,
                "cache_hit_rate": 0.85,
                "memory_usage_mb": 256,
                "cpu_usage_percent": 15
            },
            "timestamp": datetime.utcnow().isoformat()
        }


@pytest.fixture
def test_data_generator():
    """Test data generator fixture."""
    return TestDataGenerator()


# Async test utilities
class AsyncTestUtils:
    """Utilities for async testing."""
    
    @staticmethod
    async def wait_for_condition(condition_func, timeout=5.0, interval=0.1) -> Any:
        """Wait for a condition to be true."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await condition_func():
                return True
            await asyncio.sleep(interval)
        return False
    
    @staticmethod
    async def run_concurrent_requests(client, url, count=10, **kwargs) -> Any:
        """Run concurrent requests for load testing."""
        async def make_request():
            return await client.get(url, **kwargs)

        tasks = [make_request() for _ in range(count)]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    @staticmethod
    async def measure_performance(func, iterations=100) -> Any:
        """Measure function performance."""
        times = []
        for _ in range(iterations):
            start = time.time()
            await func()
            times.append(time.time() - start)
        
        return {
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "p95_time": sorted(times)[int(len(times) * 0.95)],
            "p99_time": sorted(times)[int(len(times) * 0.99)]
        }


@pytest.fixture
def async_utils():
    """Async test utilities fixture."""
    return AsyncTestUtils()


# Debug utilities
class DebugUtils:
    """Debug utilities for testing."""
    
    @staticmethod
    def print_response_details(response) -> Any:
        """Print detailed response information for debugging."""
        print(f"\n=== Response Details ===")
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Response Time: {response.headers.get('X-Response-Time', 'N/A')}")
        print(f"Cache Status: {response.headers.get('X-Cache', 'N/A')}")
        
        try:
            body = response.json()
            print(f"Response Body: {orjson.dumps(body, option=orjson.OPT_INDENT_2).decode()}")
        except:
            print(f"Response Body: {response.text}")
    
    @staticmethod
    def print_performance_metrics(metrics) -> Any:
        """Print performance metrics for debugging."""
        print(f"\n=== Performance Metrics ===")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
    
    @staticmethod
    def create_debug_logger():
        """Create a debug logger that prints everything."""
        
        logger = logging.getLogger("debug")
        logger.setLevel(logging.DEBUG)
        
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger


@pytest.fixture
def debug_utils():
    """Debug utilities fixture."""
    return DebugUtils()


# Export all fixtures
__all__ = [
    "test_settings",
    "mock_redis",
    "mock_cache_manager",
    "sample_linkedin_post",
    "sample_posts_batch",
    "mock_repository",
    "mock_use_cases",
    "mock_nlp_processor",
    "test_app",
    "test_client",
    "async_client",
    "auth_headers",
    "sample_post_data",
    "sample_batch_data",
    "mock_metrics",
    "temp_db",
    "mock_logger",
    "performance_data",
    "error_scenarios",
    "mock_circuit_breaker",
    "test_data_generator",
    "async_utils",
    "debug_utils"
] 