from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import pytest
import asyncio
import tempfile
import os
from typing import AsyncGenerator, Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
import factory
from faker import Faker
from hypothesis import given, strategies as st
import responses
import httpretty
from aioresponses import aioresponses
import freezegun
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs
import redis.asyncio as redis
from fastapi.testclient import TestClient
from httpx import AsyncClient
import pytest_httpx
import pytest_asyncio
import psutil
from memory_profiler import profile
import pytest_benchmark
from mimesis import Generic
from factory import Factory, Faker as FactoryFaker, SubFactory
from ..core.domain.entities.linkedin_post import LinkedInPost, PostStatus, PostType, PostTone
from ..application.use_cases.linkedin_post_use_cases import LinkedInPostUseCases
from ..infrastructure.repositories.linkedin_post_repository import LinkedInPostRepository
from ..shared.cache import CacheManager
from ..shared.config import Settings
        import uvloop
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from ..presentation.api.linkedin_post_router_v2 import router
        import logging
from typing import Any, List, Dict, Optional
"""
Advanced Pytest Configuration with Best Libraries
================================================

Ultra-modern testing setup using the best Python testing libraries.
"""


# Advanced testing libraries
# from hypothesis.extra.pytest import register_random  # Removed for compatibility

# FastAPI testing

# Performance testing

# Data generation

# Our modules

# Initialize Faker
fake = Faker()
generic = Generic()


# Factory Boy Models for Test Data Generation
class LinkedInPostFactory(Factory):
    """Factory for creating LinkedIn post test data."""
    
    @dataclass
class Meta:
        model = LinkedInPost
    
    id = FactoryFaker('uuid4')
    content = FactoryFaker('text', max_nb_chars=500)
    post_type = FactoryFaker('random_element', elements=list(PostType))
    tone = FactoryFaker('random_element', elements=list(PostTone))
    target_audience = FactoryFaker('random_element', elements=[
        'tech professionals', 'marketers', 'developers', 'business owners'
    ])
    industry = FactoryFaker('random_element', elements=[
        'technology', 'marketing', 'finance', 'healthcare', 'education'
    ])
    status = FactoryFaker('random_element', elements=list(PostStatus))
    nlp_enhanced = FactoryFaker('boolean')
    nlp_processing_time = FactoryFaker('pyfloat', min_value=0.01, max_value=2.0)
    created_at = FactoryFaker('date_time_this_year')
    updated_at = FactoryFaker('date_time_this_year')


class PostDataFactory(Factory):
    """Factory for creating post data dictionaries."""
    
    @dataclass
class Meta:
        model = dict
    
    content = FactoryFaker('text', max_nb_chars=500)
    post_type = FactoryFaker('random_element', elements=['announcement', 'educational', 'update'])
    tone = FactoryFaker('random_element', elements=['professional', 'casual', 'friendly'])
    target_audience = FactoryFaker('random_element', elements=[
        'tech professionals', 'marketers', 'developers'
    ])
    industry = FactoryFaker('random_element', elements=[
        'technology', 'marketing', 'finance'
    ])


# Hypothesis Strategies for Property-Based Testing
@st.composite
def linkedin_post_strategy(draw) -> Any:
    """Strategy for generating LinkedIn post data."""
    return {
        'content': draw(st.text(min_size=10, max_size=500)),
        'post_type': draw(st.sampled_from(['announcement', 'educational', 'update'])),
        'tone': draw(st.sampled_from(['professional', 'casual', 'friendly'])),
        'target_audience': draw(st.sampled_from(['tech professionals', 'marketers', 'developers'])),
        'industry': draw(st.sampled_from(['technology', 'marketing', 'finance']))
    }


@st.composite
def batch_post_strategy(draw) -> Any:
    """Strategy for generating batch post data."""
    size = draw(st.integers(min_value=1, max_value=10))
    return [draw(linkedin_post_strategy()) for _ in range(size)]


# Advanced Pytest Fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    try:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass
    
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings():
    """Advanced test settings with overrides."""
    return Settings(
        DATABASE_URL="sqlite:///./test.db",
        REDIS_URL="redis://localhost:6379/1",
        SECRET_KEY=fake.sha256(),
        ENVIRONMENT="testing",
        DEBUG=True,
        TESTING=True,
        ENABLE_CACHE=False,
        RATE_LIMIT_ENABLED=False,
        ENABLE_METRICS=False,
        ENABLE_TRACING=False
    )


@pytest.fixture(scope="session")
def redis_container():
    """Docker container for Redis testing."""
    with DockerContainer("redis:7-alpine") as container:
        container.with_exposed_ports(6379)
        container.start()
        wait_for_logs(container, "Ready to accept connections")
        
        redis_url = f"redis://{container.get_container_host_ip()}:{container.get_exposed_port(6379)}"
        yield redis_url


@pytest.fixture
async def redis_client(redis_container) -> Any:
    """Real Redis client for integration testing."""
    client = redis.from_url(redis_container, decode_responses=True)
    await client.ping()
    yield client
    await client.close()


@pytest.fixture
def mock_redis():
    """Advanced mock Redis client with comprehensive methods."""
    with patch('redis.asyncio.Redis') as mock:
        mock_client = AsyncMock()
        mock.from_url.return_value = mock_client
        
        # Basic operations
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        mock_client.set.return_value = True
        mock_client.setex.return_value = True
        mock_client.delete.return_value = 1
        mock_client.flushdb.return_value = True
        
        # Advanced operations
        mock_client.scan.return_value = (0, [])
        mock_client.pipeline.return_value.__aenter__.return_value = mock_client
        mock_client.pipeline.return_value.__aexit__.return_value = None
        
        # Pub/Sub
        mock_client.pubsub.return_value = AsyncMock()
        
        yield mock_client


@pytest.fixture
def mock_cache_manager(mock_redis) -> Any:
    """Advanced mock cache manager with comprehensive methods."""
    with patch('linkedin_posts.shared.cache.cache_manager') as mock:
        # Basic operations
        mock.get.return_value = None
        mock.set.return_value = True
        mock.delete.return_value = True
        mock.clear.return_value = True
        mock.get_many.return_value = {}
        mock.set_many.return_value = True
        
        # Advanced operations
        mock.delete_pattern.return_value = 0
        mock.warm_cache.return_value = None
        mock.get_stats.return_value = {
            'l1_memory': {'hits': 0, 'misses': 0, 'hit_rate': 0.0},
            'l2_redis': {'hits': 0, 'misses': 0, 'hit_rate': 0.0}
        }
        
        yield mock


@pytest.fixture
def sample_linkedin_post():
    """Generate sample LinkedIn post using Factory Boy."""
    return LinkedInPostFactory()


@pytest.fixture
def sample_posts_batch():
    """Generate batch of LinkedIn posts using Factory Boy."""
    return LinkedInPostFactory.build_batch(5)


@pytest.fixture
def mock_repository(sample_linkedin_post, sample_posts_batch) -> Any:
    """Advanced mock repository with comprehensive methods."""
    mock_repo = AsyncMock(spec=LinkedInPostRepository)
    
    # Basic CRUD operations
    mock_repo.get_by_id.return_value = sample_linkedin_post
    mock_repo.list_posts.return_value = sample_posts_batch
    mock_repo.create.return_value = sample_linkedin_post
    mock_repo.update.return_value = sample_linkedin_post
    mock_repo.delete.return_value = True
    
    # Batch operations
    mock_repo.batch_create.return_value = sample_posts_batch
    mock_repo.batch_update.return_value = sample_posts_batch
    mock_repo.batch_delete.return_value = len(sample_posts_batch)
    
    # Advanced operations
    mock_repo.search_posts.return_value = sample_posts_batch
    mock_repo.get_posts_by_user.return_value = sample_posts_batch
    mock_repo.get_posts_by_status.return_value = sample_posts_batch
    mock_repo.get_analytics.return_value = {
        'total_posts': len(sample_posts_batch),
        'avg_engagement': 0.75,
        'top_keywords': ['test', 'linkedin', 'post']
    }
    
    return mock_repo


@pytest.fixture
def mock_use_cases(mock_repository) -> Any:
    """Mock use cases with repository."""
    return LinkedInPostUseCases(mock_repository)


@pytest.fixture
def mock_nlp_processor():
    """Advanced mock NLP processor with comprehensive methods."""
    with patch('linkedin_posts.infrastructure.nlp.nlp_processor') as mock:
        # Basic processing
        mock.process_text.return_value = {
            "sentiment_score": fake.pyfloat(min_value=-1.0, max_value=1.0),
            "readability_score": fake.pyfloat(min_value=0.0, max_value=100.0),
            "keywords": fake.words(nb=5),
            "entities": fake.words(nb=3),
            "processing_time": fake.pyfloat(min_value=0.01, max_value=1.0)
        }
        
        # Batch processing
        mock.process_batch.return_value = [
            {
                "sentiment_score": fake.pyfloat(min_value=-1.0, max_value=1.0),
                "readability_score": fake.pyfloat(min_value=0.0, max_value=100.0),
                "keywords": fake.words(nb=3),
                "entities": fake.words(nb=2),
                "processing_time": fake.pyfloat(min_value=0.01, max_value=0.5)
            }
            for _ in range(5)
        ]
        
        # Advanced methods
        mock.optimize_content.return_value = "Optimized content"
        mock.extract_keywords.return_value = fake.words(nb=10)
        mock.analyze_sentiment.return_value = fake.pyfloat(min_value=-1.0, max_value=1.0)
        mock.detect_language.return_value = "en"
        mock.translate_text.return_value = "Translated text"
        
        yield mock


@pytest.fixture
def test_app():
    """Test FastAPI application with advanced configuration."""
    
    app = FastAPI(
        title="LinkedIn Posts API Test",
        description="Advanced testing application",
        version="2.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.include_router(router)
    
    return app


@pytest.fixture
def test_client(test_app) -> Any:
    """Test client for FastAPI with advanced configuration."""
    return TestClient(
        app=test_app,
        base_url="http://testserver"
    )


@pytest.fixture
async def async_client(test_app) -> Any:
    """Async test client for FastAPI with advanced configuration."""
    async with AsyncClient(
        app=test_app,
        base_url="http://test",
        timeout=30.0,
        follow_redirects=True
    ) as client:
        yield client


@pytest.fixture
def auth_headers():
    """Advanced authentication headers for testing."""
    return {
        "Authorization": f"Bearer {fake.sha256()}",
        "X-Request-ID": fake.uuid4(),
        "X-User-ID": fake.uuid4(),
        "X-Session-ID": fake.uuid4(),
        "Content-Type": "application/json",
        "Accept": "application/json"
    }


@pytest.fixture
def sample_post_data():
    """Generate sample post data using Factory Boy."""
    return PostDataFactory()


@pytest.fixture
def sample_batch_data():
    """Generate batch data using Factory Boy."""
    return PostDataFactory.build_batch(5)


@pytest.fixture
def mock_metrics():
    """Advanced mock metrics collector."""
    with patch('linkedin_posts.shared.metrics.metrics_collector') as mock:
        # Basic metrics
        mock.track_post_creation.return_value = None
        mock.track_batch_creation.return_value = None
        mock.track_api_request.return_value = None
        
        # Performance metrics
        mock.get_performance_metrics.return_value = {
            "fast_nlp": {
                "avg_processing_time": fake.pyfloat(min_value=0.01, max_value=0.5),
                "cache_hit_rate": fake.pyfloat(min_value=0.5, max_value=1.0),
                "throughput": fake.pyfloat(min_value=10.0, max_value=1000.0),
                "error_rate": fake.pyfloat(min_value=0.0, max_value=0.1)
            },
            "async_nlp": {
                "avg_processing_time": fake.pyfloat(min_value=0.01, max_value=0.3),
                "cache_hit_rate": fake.pyfloat(min_value=0.7, max_value=1.0),
                "throughput": fake.pyfloat(min_value=50.0, max_value=2000.0),
                "error_rate": fake.pyfloat(min_value=0.0, max_value=0.05)
            },
            "system": {
                "active_requests": fake.pyint(min_value=0, max_value=100),
                "total_requests": fake.pyint(min_value=1000, max_value=100000),
                "cache_hit_rate": fake.pyfloat(min_value=0.5, max_value=1.0),
                "memory_usage_mb": fake.pyint(min_value=100, max_value=1000),
                "cpu_usage_percent": fake.pyfloat(min_value=0.0, max_value=100.0)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Advanced metrics
        mock.get_error_metrics.return_value = {
            "total_errors": fake.pyint(min_value=0, max_value=100),
            "error_rate": fake.pyfloat(min_value=0.0, max_value=0.1),
            "top_errors": [
                {"error_type": "ValidationError", "count": fake.pyint(min_value=1, max_value=50)},
                {"error_type": "DatabaseError", "count": fake.pyint(min_value=1, max_value=20)}
            ]
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
    """Advanced mock logger with comprehensive methods."""
    with patch('linkedin_posts.shared.logging.get_logger') as mock:
        logger = MagicMock()
        
        # Basic logging methods
        logger.debug = MagicMock()
        logger.info = MagicMock()
        logger.warning = MagicMock()
        logger.error = MagicMock()
        logger.critical = MagicMock()
        
        # Advanced methods
        logger.exception = MagicMock()
        logger.log = MagicMock()
        
        # Structured logging
        logger.bind = MagicMock(return_value=logger)
        logger.new = MagicMock(return_value=logger)
        
        mock.return_value = logger
        yield logger


@pytest.fixture
def performance_data():
    """Generate performance test data."""
    return {
        "request_count": fake.pyint(min_value=100, max_value=10000),
        "avg_response_time": fake.pyfloat(min_value=0.01, max_value=1.0),
        "p95_response_time": fake.pyfloat(min_value=0.05, max_value=2.0),
        "p99_response_time": fake.pyfloat(min_value=0.1, max_value=5.0),
        "error_rate": fake.pyfloat(min_value=0.0, max_value=0.1),
        "cache_hit_rate": fake.pyfloat(min_value=0.5, max_value=1.0),
        "throughput": fake.pyfloat(min_value=10.0, max_value=1000.0),
        "memory_usage_mb": fake.pyint(min_value=100, max_value=2000),
        "cpu_usage_percent": fake.pyfloat(min_value=0.0, max_value=100.0)
    }


@pytest.fixture
def error_scenarios():
    """Comprehensive error scenarios for testing."""
    return {
        "database_connection_error": Exception("Database connection failed"),
        "redis_connection_error": Exception("Redis connection failed"),
        "nlp_service_error": Exception("NLP service unavailable"),
        "validation_error": ValueError("Invalid input data"),
        "rate_limit_error": Exception("Rate limit exceeded"),
        "authentication_error": Exception("Invalid token"),
        "authorization_error": Exception("Insufficient permissions"),
        "timeout_error": Exception("Request timeout"),
        "network_error": Exception("Network connection failed"),
        "memory_error": MemoryError("Out of memory"),
        "disk_error": OSError("Disk space full"),
        "service_unavailable": Exception("Service temporarily unavailable")
    }


@pytest.fixture
def mock_circuit_breaker():
    """Advanced mock circuit breaker."""
    with patch('linkedin_posts.shared.middleware.circuit') as mock:
        mock.return_value = AsyncMock()
        mock.return_value.__call__ = AsyncMock()
        yield mock


@pytest.fixture
def frozen_time():
    """Freeze time for deterministic testing."""
    with freezegun.freeze_time("2024-01-01 12:00:00"):
        yield


@pytest.fixture
def mock_responses():
    """Mock HTTP responses for external API testing."""
    with responses.RequestsMock() as rsps:
        yield rsps


@pytest.fixture
def mock_aio_responses():
    """Mock async HTTP responses."""
    with aioresponses() as m:
        yield m


@pytest.fixture
def benchmark():
    """Pytest benchmark fixture."""
    return pytest_benchmark.plugin.benchmark


# Advanced Test Data Generators
class AdvancedTestDataGenerator:
    """Advanced test data generator using multiple libraries."""
    
    @staticmethod
    def generate_posts(count: int = 10) -> List[Dict[str, Any]]:
        """Generate posts using Factory Boy."""
        return PostDataFactory.build_batch(count)
    
    @staticmethod
    def generate_analytics_data() -> Dict[str, Any]:
        """Generate analytics data using Faker."""
        return {
            "sentiment_score": fake.pyfloat(min_value=-1.0, max_value=1.0),
            "readability_score": fake.pyfloat(min_value=0.0, max_value=100.0),
            "keywords": fake.words(nb=10),
            "entities": fake.words(nb=5),
            "processing_time": fake.pyfloat(min_value=0.01, max_value=1.0),
            "cached": fake.boolean(),
            "async_optimized": fake.boolean(),
            "language": fake.language_code(),
            "confidence_score": fake.pyfloat(min_value=0.0, max_value=1.0)
        }
    
    @staticmethod
    def generate_performance_metrics() -> Dict[str, Any]:
        """Generate performance metrics using Faker."""
        return {
            "fast_nlp_metrics": {
                "avg_processing_time": fake.pyfloat(min_value=0.01, max_value=0.5),
                "cache_hit_rate": fake.pyfloat(min_value=0.5, max_value=1.0),
                "throughput": fake.pyfloat(min_value=10.0, max_value=1000.0),
                "error_rate": fake.pyfloat(min_value=0.0, max_value=0.1),
                "memory_usage_mb": fake.pyint(min_value=50, max_value=500),
                "cpu_usage_percent": fake.pyfloat(min_value=0.0, max_value=50.0)
            },
            "async_nlp_metrics": {
                "avg_processing_time": fake.pyfloat(min_value=0.01, max_value=0.3),
                "cache_hit_rate": fake.pyfloat(min_value=0.7, max_value=1.0),
                "throughput": fake.pyfloat(min_value=50.0, max_value=2000.0),
                "error_rate": fake.pyfloat(min_value=0.0, max_value=0.05),
                "memory_usage_mb": fake.pyint(min_value=30, max_value=300),
                "cpu_usage_percent": fake.pyfloat(min_value=0.0, max_value=30.0)
            },
            "system_metrics": {
                "active_requests": fake.pyint(min_value=0, max_value=100),
                "total_requests": fake.pyint(min_value=1000, max_value=100000),
                "cache_hit_rate": fake.pyfloat(min_value=0.5, max_value=1.0),
                "memory_usage_mb": fake.pyint(min_value=100, max_value=1000),
                "cpu_usage_percent": fake.pyfloat(min_value=0.0, max_value=100.0),
                "disk_usage_percent": fake.pyfloat(min_value=0.0, max_value=100.0),
                "network_io_mbps": fake.pyfloat(min_value=0.0, max_value=1000.0)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def generate_error_data() -> Dict[str, Any]:
        """Generate error data for testing."""
        return {
            "error_type": fake.random_element([
                "ValidationError", "DatabaseError", "NetworkError", 
                "TimeoutError", "AuthenticationError", "AuthorizationError"
            ]),
            "error_message": fake.sentence(),
            "error_code": fake.pyint(min_value=400, max_value=599),
            "stack_trace": fake.text(max_nb_chars=500),
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": fake.uuid4(),
            "user_id": fake.uuid4(),
            "endpoint": fake.url(),
            "method": fake.http_method()
        }


@pytest.fixture
def test_data_generator():
    """Advanced test data generator fixture."""
    return AdvancedTestDataGenerator()


# Advanced Async Test Utilities
class AdvancedAsyncTestUtils:
    """Advanced async testing utilities."""
    
    @staticmethod
    async def wait_for_condition(condition_func, timeout=5.0, interval=0.1) -> Any:
        """Wait for a condition to be true with advanced error handling."""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                if await condition_func():
                    return True
            except Exception as e:
                # Log error but continue waiting
                print(f"Condition check failed: {e}")
            
            await asyncio.sleep(interval)
        
        return False
    
    @staticmethod
    async async def run_concurrent_requests(client, url, count=10, **kwargs) -> Any:
        """Run concurrent requests with advanced error handling."""
        semaphore = asyncio.Semaphore(10)  # Limit concurrency
        
        async def make_request():
            
    """make_request function."""
async with semaphore:
                try:
                    return await client.get(url, **kwargs)
                except Exception as e:
                    return {"error": str(e)}
        
        tasks = [make_request() for _ in range(count)]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    @staticmethod
    async def measure_performance(func, iterations=100) -> Any:
        """Measure function performance with detailed metrics."""
        times = []
        memory_usage = []
        
        for _ in range(iterations):
            # Measure memory before
            process = psutil.Process()
            memory_before = process.memory_info().rss
            
            # Measure time
            start_time = asyncio.get_event_loop().time()
            await func()
            end_time = asyncio.get_event_loop().time()
            
            # Measure memory after
            memory_after = process.memory_info().rss
            
            times.append(end_time - start_time)
            memory_usage.append(memory_after - memory_before)
        
        return {
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "p50_time": sorted(times)[len(times) // 2],
            "p95_time": sorted(times)[int(len(times) * 0.95)],
            "p99_time": sorted(times)[int(len(times) * 0.99)],
            "avg_memory_delta": sum(memory_usage) / len(memory_usage),
            "max_memory_delta": max(memory_usage),
            "iterations": iterations
        }


@pytest.fixture
def async_utils():
    """Advanced async test utilities fixture."""
    return AdvancedAsyncTestUtils()


# Advanced Debug Utilities
class AdvancedDebugUtils:
    """Advanced debugging utilities."""
    
    @staticmethod
    def print_response_details(response) -> Any:
        """Print detailed response information for debugging."""
        print(f"\n=== Response Details ===")
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Response Time: {response.headers.get('X-Response-Time', 'N/A')}")
        print(f"Cache Status: {response.headers.get('X-Cache', 'N/A')}")
        print(f"Request ID: {response.headers.get('X-Request-ID', 'N/A')}")
        
        try:
            body = response.json()
            print(f"Response Body: {json.dumps(body, indent=2)}")
        except:
            print(f"Response Body: {response.text}")
    
    @staticmethod
    def print_performance_metrics(metrics) -> Any:
        """Print performance metrics for debugging."""
        print(f"\n=== Performance Metrics ===")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            elif isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float):
                        print(f"  {sub_key}: {sub_value:.3f}")
                    else:
                        print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
    
    @staticmethod
    def create_debug_logger():
        """Create an advanced debug logger."""
        
        logger = logging.getLogger("advanced_debug")
        logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler("advanced_debug.log")
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def profile_memory(func) -> Any:
        """Decorator to profile memory usage."""
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            process = psutil.Process()
            memory_before = process.memory_info().rss
            
            result = await func(*args, **kwargs)
            
            memory_after = process.memory_info().rss
            memory_delta = memory_after - memory_before
            
            print(f"Memory usage for {func.__name__}: {memory_delta / 1024 / 1024:.2f} MB")
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            process = psutil.Process()
            memory_before = process.memory_info().rss
            
            result = func(*args, **kwargs)
            
            memory_after = process.memory_info().rss
            memory_delta = memory_after - memory_before
            
            print(f"Memory usage for {func.__name__}: {memory_delta / 1024 / 1024:.2f} MB")
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


@pytest.fixture
def debug_utils():
    """Advanced debug utilities fixture."""
    return AdvancedDebugUtils()


# Register Hypothesis strategies
register_random(linkedin_post_strategy)
register_random(batch_post_strategy)

# Export all fixtures
__all__ = [
    "test_settings",
    "redis_container",
    "redis_client",
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
    "frozen_time",
    "mock_responses",
    "mock_aio_responses",
    "benchmark",
    "test_data_generator",
    "async_utils",
    "debug_utils",
    "LinkedInPostFactory",
    "PostDataFactory",
    "linkedin_post_strategy",
    "batch_post_strategy"
] 