"""
Base test classes and utilities for HeyGen AI system.
Refactored for better organization and maintainability.
"""

import pytest
import time
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys
import os

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
sys.path.insert(0, str(parent_dir))

@dataclass
class TestConfig:
    """Configuration for test execution."""
    timeout: float = 30.0
    retry_attempts: int = 3
    log_level: str = "INFO"
    debug_mode: bool = False
    performance_threshold: float = 1.0
    memory_threshold: float = 100.0  # MB
    coverage_threshold: float = 80.0  # %

@dataclass
class TestResult:
    """Test result data structure."""
    name: str
    status: str  # passed, failed, skipped, error
    duration: float
    message: Optional[str] = None
    coverage: Optional[float] = None
    memory_usage: Optional[float] = None
    performance_score: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

class BaseTest(ABC):
    """Base test class with common functionality."""
    
    def __init__(self, config: TestConfig = None):
        self.config = config or TestConfig()
        self.logger = self._setup_logger()
        self.results: List[TestResult] = []
        self.start_time = None
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for test class."""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def setup_method(self):
        """Setup method called before each test."""
        self.start_time = time.time()
        self.logger.info(f"Starting test: {self.__class__.__name__}")
    
    def teardown_method(self):
        """Teardown method called after each test."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.logger.info(f"Test completed in {duration:.3f}s")
    
    def assert_performance(self, duration: float, operation: str = "operation"):
        """Assert that operation meets performance requirements."""
        if duration > self.config.performance_threshold:
            pytest.fail(f"{operation} took {duration:.3f}s, exceeds threshold {self.config.performance_threshold}s")
    
    def assert_memory_usage(self, memory_mb: float, operation: str = "operation"):
        """Assert that memory usage is within limits."""
        if memory_mb > self.config.memory_threshold:
            pytest.fail(f"{operation} used {memory_mb:.1f}MB, exceeds threshold {self.config.memory_threshold}MB")
    
    def measure_time(self, func: Callable, *args, **kwargs) -> tuple:
        """Measure execution time of a function."""
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        return result, duration
    
    async def measure_async_time(self, coro) -> tuple:
        """Measure execution time of an async function."""
        start_time = time.time()
        result = await coro
        duration = time.time() - start_time
        return result, duration
    
    def record_result(self, name: str, status: str, duration: float, **kwargs):
        """Record test result."""
        result = TestResult(
            name=name,
            status=status,
            duration=duration,
            **kwargs
        )
        self.results.append(result)
        return result
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of test results."""
        if not self.results:
            return {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0}
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "passed")
        failed = sum(1 for r in self.results if r.status == "failed")
        skipped = sum(1 for r in self.results if r.status == "skipped")
        errors = sum(1 for r in self.results if r.status == "error")
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,
            "success_rate": (passed / total * 100) if total > 0 else 0
        }

class PerformanceTest(BaseTest):
    """Base class for performance tests."""
    
    def __init__(self, config: TestConfig = None):
        super().__init__(config)
        self.performance_results: List[Dict[str, Any]] = []
    
    def benchmark_operation(self, name: str, func: Callable, iterations: int = 1, *args, **kwargs) -> Dict[str, Any]:
        """Benchmark an operation."""
        durations = []
        
        for _ in range(iterations):
            result, duration = self.measure_time(func, *args, **kwargs)
            durations.append(duration)
        
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        benchmark_result = {
            "name": name,
            "iterations": iterations,
            "avg_duration": avg_duration,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "throughput": iterations / avg_duration if avg_duration > 0 else 0
        }
        
        self.performance_results.append(benchmark_result)
        return benchmark_result
    
    def assert_benchmark(self, benchmark_result: Dict[str, Any], max_duration: float = None):
        """Assert benchmark meets performance requirements."""
        if max_duration is None:
            max_duration = self.config.performance_threshold
        
        if benchmark_result["avg_duration"] > max_duration:
            pytest.fail(f"Benchmark {benchmark_result['name']} failed: "
                       f"avg duration {benchmark_result['avg_duration']:.3f}s "
                       f"exceeds threshold {max_duration}s")

class IntegrationTest(BaseTest):
    """Base class for integration tests."""
    
    def __init__(self, config: TestConfig = None):
        super().__init__(config)
        self.test_data: Dict[str, Any] = {}
        self.mock_services: Dict[str, Any] = {}
    
    def setup_test_data(self, data: Dict[str, Any]):
        """Setup test data for integration tests."""
        self.test_data.update(data)
    
    def setup_mock_service(self, name: str, service: Any):
        """Setup mock service for integration tests."""
        self.mock_services[name] = service
    
    def get_test_data(self, key: str, default: Any = None) -> Any:
        """Get test data by key."""
        return self.test_data.get(key, default)
    
    def get_mock_service(self, name: str) -> Any:
        """Get mock service by name."""
        return self.mock_services.get(name)

class UnitTest(BaseTest):
    """Base class for unit tests."""
    
    def __init__(self, config: TestConfig = None):
        super().__init__(config)
        self.test_objects: Dict[str, Any] = {}
    
    def create_test_object(self, name: str, obj: Any):
        """Create test object."""
        self.test_objects[name] = obj
    
    def get_test_object(self, name: str) -> Any:
        """Get test object by name."""
        return self.test_objects.get(name)
    
    def assert_object_equals(self, actual: Any, expected: Any, message: str = None):
        """Assert two objects are equal."""
        if actual != expected:
            error_msg = message or f"Objects not equal: {actual} != {expected}"
            pytest.fail(error_msg)

# Test utilities
class TestUtilities:
    """Utility functions for tests."""
    
    @staticmethod
    def generate_test_data(data_type: str, count: int = 1) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Generate test data based on type."""
        if data_type == "user":
            return TestUtilities._generate_user_data(count)
        elif data_type == "service":
            return TestUtilities._generate_service_data(count)
        elif data_type == "api_request":
            return TestUtilities._generate_api_request_data(count)
        else:
            return {"type": data_type, "count": count}
    
    @staticmethod
    def _generate_user_data(count: int) -> List[Dict[str, Any]]:
        """Generate user test data."""
        users = []
        for i in range(count):
            users.append({
                "id": i + 1,
                "username": f"user_{i}",
                "email": f"user_{i}@example.com",
                "active": i % 2 == 0,
                "created_at": datetime.now().isoformat()
            })
        return users
    
    @staticmethod
    def _generate_service_data(count: int) -> List[Dict[str, Any]]:
        """Generate service test data."""
        services = []
        for i in range(count):
            services.append({
                "id": i + 1,
                "name": f"service_{i}",
                "status": "running",
                "port": 8000 + i,
                "dependencies": [f"dep_{j}" for j in range(i % 3)]
            })
        return services
    
    @staticmethod
    def _generate_api_request_data(count: int) -> List[Dict[str, Any]]:
        """Generate API request test data."""
        requests = []
        for i in range(count):
            requests.append({
                "method": "POST",
                "url": f"/api/v1/endpoint_{i}",
                "headers": {"Content-Type": "application/json"},
                "body": {"id": i, "data": f"test_data_{i}"}
            })
        return requests
    
    @staticmethod
    def assert_json_equals(actual: str, expected: Dict[str, Any]):
        """Assert JSON string equals expected dict."""
        actual_dict = json.loads(actual)
        assert actual_dict == expected, f"JSON not equal: {actual_dict} != {expected}"
    
    @staticmethod
    def assert_list_contains(list_data: List[Any], expected_item: Any):
        """Assert list contains expected item."""
        assert expected_item in list_data, f"Expected item {expected_item} not found in list"
    
    @staticmethod
    def assert_dict_contains(actual: Dict[str, Any], expected: Dict[str, Any]):
        """Assert actual dict contains all expected keys and values."""
        for key, value in expected.items():
            assert key in actual, f"Key '{key}' not found in actual dict"
            assert actual[key] == value, f"Value for key '{key}' doesn't match: {actual[key]} != {value}"

# Test fixtures
@pytest.fixture
def test_config():
    """Test configuration fixture."""
    return TestConfig()

@pytest.fixture
def base_test(test_config):
    """Base test fixture."""
    return BaseTest(test_config)

@pytest.fixture
def performance_test(test_config):
    """Performance test fixture."""
    return PerformanceTest(test_config)

@pytest.fixture
def integration_test(test_config):
    """Integration test fixture."""
    return IntegrationTest(test_config)

@pytest.fixture
def unit_test(test_config):
    """Unit test fixture."""
    return UnitTest(test_config)

@pytest.fixture
def test_utilities():
    """Test utilities fixture."""
    return TestUtilities

# Test markers
pytestmark = pytest.mark.usefixtures("test_config")

if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
