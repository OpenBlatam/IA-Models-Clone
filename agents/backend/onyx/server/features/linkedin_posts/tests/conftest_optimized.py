from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import pytest
import asyncio
import tempfile
import os
import time
import statistics
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from faker import Faker
import factory
from factory import Factory, Faker as FactoryFaker
import psutil
from memory_profiler import profile
        import uvloop
from typing import Any, List, Dict, Optional
import logging
"""
Optimized Pytest Configuration
=============================

Clean, fast, and efficient testing setup with only essential dependencies.
"""


# Essential testing libraries

# Performance testing

# Initialize Faker
fake = Faker()


class OptimizedTestDataGenerator:
    """Optimized test data generator with caching and performance improvements."""
    
    def __init__(self) -> Any:
        self.fake = Faker()
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    def generate_post_data(self, **overrides) -> Dict[str, Any]:
        """Generate optimized post data with caching."""
        cache_key = f"post_data_{hash(str(overrides))}"
        
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_data.copy()
        
        data = {
            "id": self.fake.uuid4(),
            "content": self.fake.text(max_nb_chars=300),
            "post_type": self.fake.random_element(['announcement', 'educational', 'update']),
            "tone": self.fake.random_element(['professional', 'casual', 'friendly']),
            "target_audience": self.fake.random_element(['tech professionals', 'marketers', 'developers']),
            "industry": self.fake.random_element(['technology', 'marketing', 'finance']),
            "created_at": self.fake.date_time_this_year().isoformat(),
            "updated_at": self.fake.date_time_this_year().isoformat()
        }
        
        data.update(overrides)
        self._cache[cache_key] = (time.time(), data)
        return data
    
    def generate_batch_data(self, count: int, **overrides) -> List[Dict[str, Any]]:
        """Generate optimized batch data."""
        return [self.generate_post_data(**overrides) for _ in range(count)]
    
    def clear_cache(self) -> Any:
        """Clear the data cache."""
        self._cache.clear()


class OptimizedPerformanceMonitor:
    """Optimized performance monitoring with minimal overhead."""
    
    def __init__(self) -> Any:
        self.process = psutil.Process()
        self.metrics = {}
    
    def start_monitoring(self, operation_name: str):
        """Start monitoring an operation."""
        self.metrics[operation_name] = {
            "start_time": time.time(),
            "start_memory": self.process.memory_info().rss,
            "start_cpu": self.process.cpu_percent()
        }
    
    def stop_monitoring(self, operation_name: str) -> Dict[str, Any]:
        """Stop monitoring and return metrics."""
        if operation_name not in self.metrics:
            return {}
        
        start_metrics = self.metrics[operation_name]
        end_time = time.time()
        end_memory = self.process.memory_info().rss
        end_cpu = self.process.cpu_percent()
        
        metrics = {
            "duration": end_time - start_metrics["start_time"],
            "memory_delta_mb": (end_memory - start_metrics["start_memory"]) / 1024 / 1024,
            "cpu_usage": end_cpu,
            "operations_per_second": 1.0 / (end_time - start_metrics["start_time"])
        }
        
        del self.metrics[operation_name]
        return metrics


# Optimized Factory Boy Models
class OptimizedLinkedInPostFactory(Factory):
    """Optimized factory for LinkedIn post data."""
    
    class Meta:
        model = dict
    
    id = FactoryFaker('uuid4')
    content = FactoryFaker('text', max_nb_chars=300)
    post_type = FactoryFaker('random_element', elements=['announcement', 'educational', 'update'])
    tone = FactoryFaker('random_element', elements=['professional', 'casual', 'friendly'])
    target_audience = FactoryFaker('random_element', elements=[
        'tech professionals', 'marketers', 'developers'
    ])
    industry = FactoryFaker('random_element', elements=[
        'technology', 'marketing', 'finance'
    ])
    created_at = FactoryFaker('date_time_this_year')
    updated_at = FactoryFaker('date_time_this_year')


# Optimized Pytest Fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an optimized event loop for the test session."""
    try:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass
    
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_data_generator():
    """Optimized test data generator fixture."""
    generator = OptimizedTestDataGenerator()
    yield generator
    generator.clear_cache()


@pytest.fixture(scope="session")
def performance_monitor():
    """Optimized performance monitor fixture."""
    return OptimizedPerformanceMonitor()


@pytest.fixture
def sample_post_data():
    """Generate optimized sample post data."""
    return OptimizedLinkedInPostFactory()


@pytest.fixture
def sample_batch_data():
    """Generate optimized batch data."""
    return OptimizedLinkedInPostFactory.build_batch(5)


@pytest.fixture
def mock_repository():
    """Optimized mock repository."""
    mock_repo = AsyncMock()
    
    # Use factory for test data
    sample_post = OptimizedLinkedInPostFactory()
    sample_posts = OptimizedLinkedInPostFactory.build_batch(5)
    
    mock_repo.get_by_id.return_value = sample_post
    mock_repo.list_posts.return_value = sample_posts
    mock_repo.create.return_value = sample_post
    mock_repo.update.return_value = sample_post
    mock_repo.delete.return_value = True
    mock_repo.batch_create.return_value = sample_posts
    mock_repo.batch_update.return_value = sample_posts
    
    return mock_repo


@pytest.fixture
def mock_cache_manager():
    """Optimized mock cache manager."""
    mock_cache = AsyncMock()
    
    # Optimized cache operations
    mock_cache.get.return_value = None
    mock_cache.set.return_value = True
    mock_cache.delete.return_value = True
    mock_cache.clear.return_value = True
    mock_cache.get_many.return_value = {}
    mock_cache.set_many.return_value = True
    
    return mock_cache


@pytest.fixture
def mock_nlp_processor():
    """Optimized mock NLP processor."""
    mock_nlp = AsyncMock()
    
    # Optimized NLP responses
    mock_nlp.process_text.return_value = {
        "sentiment_score": fake.pyfloat(min_value=-1.0, max_value=1.0),
        "readability_score": fake.pyfloat(min_value=0.0, max_value=100.0),
        "keywords": fake.words(nb=5),
        "entities": fake.words(nb=3),
        "processing_time": fake.pyfloat(min_value=0.01, max_value=0.5)
    }
    
    mock_nlp.process_batch.return_value = [
        {
            "sentiment_score": fake.pyfloat(min_value=-1.0, max_value=1.0),
            "readability_score": fake.pyfloat(min_value=0.0, max_value=100.0),
            "keywords": fake.words(nb=3),
            "entities": fake.words(nb=2),
            "processing_time": fake.pyfloat(min_value=0.01, max_value=0.3)
        }
        for _ in range(5)
    ]
    
    return mock_nlp


@pytest.fixture
def temp_db():
    """Optimized temporary database fixture."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    yield f"sqlite:///{db_path}"
    
    # Cleanup
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def auth_headers():
    """Optimized authentication headers."""
    return {
        "Authorization": f"Bearer {fake.sha256()}",
        "X-Request-ID": fake.uuid4(),
        "Content-Type": "application/json",
        "Accept": "application/json"
    }


# Optimized test utilities
class OptimizedTestUtils:
    """Optimized test utilities with performance improvements."""
    
    @staticmethod
    async def run_concurrent_operations(operation_func, count: int, max_concurrent: int = 10):
        """Run operations concurrently with optimized concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_operation():
            
    """limited_operation function."""
async with semaphore:
                return await operation_func()
        
        tasks = [limited_operation() for _ in range(count)]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    @staticmethod
    def measure_performance(func, iterations: int = 100):
        """Measure function performance with optimized timing."""
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            result = func()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            "avg_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "p50_time": statistics.quantiles(times, n=2)[0] if len(times) > 1 else times[0],
            "p95_time": statistics.quantiles(times, n=20)[18] if len(times) > 19 else times[-1],
            "iterations": iterations
        }
    
    @staticmethod
    def profile_memory(func) -> Any:
        """Profile memory usage with minimal overhead."""
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss
        result = func()
        final_memory = process.memory_info().rss
        
        return {
            "memory_delta_mb": (final_memory - initial_memory) / 1024 / 1024,
            "result": result
        }


@pytest.fixture
def test_utils():
    """Optimized test utilities fixture."""
    return OptimizedTestUtils()


# Optimized async utilities
class OptimizedAsyncUtils:
    """Optimized async testing utilities."""
    
    @staticmethod
    async def wait_for_condition(condition_func, timeout: float = 5.0, interval: float = 0.1):
        """Wait for condition with optimized polling."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if await condition_func():
                    return True
            except Exception:
                pass
            
            await asyncio.sleep(interval)
        
        return False
    
    @staticmethod
    async def retry_operation(operation_func, max_retries: int = 3, delay: float = 0.1):
        """Retry operation with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return await operation_func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(delay * (2 ** attempt))


@pytest.fixture
def async_utils():
    """Optimized async utilities fixture."""
    return OptimizedAsyncUtils()


# Export all fixtures and utilities
__all__ = [
    "test_data_generator",
    "performance_monitor",
    "sample_post_data",
    "sample_batch_data",
    "mock_repository",
    "mock_cache_manager",
    "mock_nlp_processor",
    "temp_db",
    "auth_headers",
    "test_utils",
    "async_utils",
    "OptimizedTestDataGenerator",
    "OptimizedPerformanceMonitor",
    "OptimizedTestUtils",
    "OptimizedAsyncUtils"
] 