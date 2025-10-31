"""
Refactored test utilities for HeyGen AI system.
Consolidated and optimized for better maintainability.
"""

import time
import json
import random
import string
import asyncio
import logging
import subprocess
import psutil
import gc
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import sys
import os

T = TypeVar('T')

class DataType(Enum):
    """Test data types."""
    USER = "user"
    SERVICE = "service"
    API_REQUEST = "api_request"
    ERROR = "error"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"

class TestStatus(Enum):
    """Test status types."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    operation: str
    duration: float
    memory_usage: float
    cpu_usage: float
    iterations: int
    throughput: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TestData:
    """Test data structure."""
    data_type: DataType
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

class TestDataGenerator:
    """Refactored test data generator with improved functionality."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
        self.generated_data: List[TestData] = []
    
    def generate_string(self, length: int = 10, prefix: str = "", suffix: str = "") -> str:
        """Generate random string with prefix and suffix."""
        chars = string.ascii_letters + string.digits
        random_str = ''.join(random.choice(chars) for _ in range(length))
        return f"{prefix}{random_str}{suffix}"
    
    def generate_email(self, domain: str = "example.com") -> str:
        """Generate random email address."""
        username = self.generate_string(8)
        return f"{username}@{domain}"
    
    def generate_phone(self, country_code: str = "+1") -> str:
        """Generate random phone number."""
        number = random.randint(1000000000, 9999999999)
        return f"{country_code}{number}"
    
    def generate_date_range(self, days: int = 30, start_date: Optional[datetime] = None) -> tuple:
        """Generate date range."""
        if start_date is None:
            start_date = datetime.now()
        end_date = start_date + timedelta(days=days)
        return start_date, end_date
    
    def generate_json_data(self, complexity: int = 3, size: str = "small") -> Dict[str, Any]:
        """Generate complex JSON test data."""
        size_multipliers = {"small": 1, "medium": 5, "large": 20}
        multiplier = size_multipliers.get(size, 1)
        
        base_data = {
            "id": random.randint(1, 10000 * multiplier),
            "name": self.generate_string(10),
            "email": self.generate_email(),
            "phone": self.generate_phone(),
            "created_at": datetime.now().isoformat(),
            "active": random.choice([True, False]),
            "metadata": {
                "version": f"v{random.randint(1, 10)}.{random.randint(0, 9)}",
                "tags": [self.generate_string(5) for _ in range(3)],
                "settings": {
                    "notifications": random.choice([True, False]),
                    "theme": random.choice(["light", "dark", "auto"]),
                    "language": random.choice(["en", "es", "fr", "de"])
                }
            }
        }
        
        if complexity > 1:
            base_data["nested_data"] = {
                "items": [
                    {
                        "id": i,
                        "value": self.generate_string(8),
                        "score": random.uniform(0, 100)
                    }
                    for i in range(random.randint(1, 5 * multiplier))
                ],
                "statistics": {
                    "total": random.randint(100, 10000 * multiplier),
                    "average": random.uniform(0, 100),
                    "trend": random.choice(["up", "down", "stable"])
                }
            }
        
        if complexity > 2:
            base_data["relationships"] = {
                "parent_id": random.randint(1, 1000) if random.choice([True, False]) else None,
                "children": [random.randint(1, 1000) for _ in range(random.randint(0, 3))],
                "dependencies": {
                    "required": [self.generate_string(6) for _ in range(2)],
                    "optional": [self.generate_string(6) for _ in range(3)]
                }
            }
        
        return base_data
    
    def generate_test_data(self, data_type: DataType, count: int = 1, **kwargs) -> Union[TestData, List[TestData]]:
        """Generate test data based on type."""
        if data_type == DataType.USER:
            data = self._generate_user_data(count, **kwargs)
        elif data_type == DataType.SERVICE:
            data = self._generate_service_data(count, **kwargs)
        elif data_type == DataType.API_REQUEST:
            data = self._generate_api_request_data(count, **kwargs)
        elif data_type == DataType.ERROR:
            data = self._generate_error_data(count, **kwargs)
        elif data_type == DataType.PERFORMANCE:
            data = self._generate_performance_data(count, **kwargs)
        elif data_type == DataType.INTEGRATION:
            data = self._generate_integration_data(count, **kwargs)
        else:
            data = {"type": data_type.value, "count": count}
        
        if count == 1:
            test_data = TestData(data_type=data_type, data=data)
        else:
            test_data = [TestData(data_type=data_type, data=item) for item in data]
        
        self.generated_data.append(test_data if isinstance(test_data, list) else [test_data])
        return test_data
    
    def _generate_user_data(self, count: int, **kwargs) -> List[Dict[str, Any]]:
        """Generate user test data."""
        users = []
        for i in range(count):
            users.append({
                "id": i + 1,
                "username": self.generate_string(8),
                "email": self.generate_email(),
                "full_name": self.generate_string(15),
                "created_at": datetime.now().isoformat(),
                "is_active": random.choice([True, False]),
                "permissions": random.sample(["read", "write", "admin", "delete"], random.randint(1, 3)),
                "profile": {
                    "bio": self.generate_string(50),
                    "avatar_url": f"https://example.com/{self.generate_string(10)}.jpg",
                    "timezone": random.choice(["UTC", "EST", "PST", "GMT"])
                }
            })
        return users
    
    def _generate_service_data(self, count: int, **kwargs) -> List[Dict[str, Any]]:
        """Generate service test data."""
        services = []
        for i in range(count):
            services.append({
                "id": i + 1,
                "name": f"service-{self.generate_string(6)}",
                "version": f"{random.randint(1, 10)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                "status": random.choice(["running", "stopped", "starting", "error"]),
                "port": random.randint(8000, 9000),
                "dependencies": [self.generate_string(8) for _ in range(random.randint(1, 5))],
                "config": {
                    "max_connections": random.randint(10, 1000),
                    "timeout": random.randint(5, 300),
                    "retry_attempts": random.randint(1, 10)
                },
                "metrics": {
                    "cpu_usage": random.uniform(0, 100),
                    "memory_usage": random.uniform(0, 100),
                    "request_count": random.randint(0, 100000)
                }
            })
        return services
    
    def _generate_api_request_data(self, count: int, **kwargs) -> List[Dict[str, Any]]:
        """Generate API request test data."""
        requests = []
        for i in range(count):
            requests.append({
                "method": random.choice(["GET", "POST", "PUT", "DELETE", "PATCH"]),
                "url": f"/api/v1/{self.generate_string(8)}",
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.generate_string(32)}",
                    "User-Agent": f"TestClient/{random.randint(1, 10)}.{random.randint(0, 9)}"
                },
                "body": self.generate_json_data(2),
                "timestamp": datetime.now().isoformat()
            })
        return requests
    
    def _generate_error_data(self, count: int, **kwargs) -> List[Dict[str, Any]]:
        """Generate error test data."""
        errors = []
        for i in range(count):
            errors.append({
                "error_code": f"TEST_ERROR_{i:03d}",
                "error_message": f"Test error {i} occurred",
                "error_type": random.choice(["ValidationError", "RuntimeError", "ValueError", "TypeError"]),
                "stack_trace": [
                    f"File 'test_{i}.py', line {j}, in test_function_{i}"
                    for j in range(1, random.randint(2, 5))
                ],
                "context": {
                    "user_id": random.randint(1, 1000),
                    "request_id": f"req_{self.generate_string(8)}",
                    "timestamp": datetime.now().isoformat()
                },
                "severity": random.choice(["low", "medium", "high", "critical"])
            })
        return errors
    
    def _generate_performance_data(self, count: int, **kwargs) -> List[Dict[str, Any]]:
        """Generate performance test data."""
        performance_data = []
        for i in range(count):
            performance_data.append({
                "operation": f"test_operation_{i}",
                "duration": random.uniform(0.001, 1.0),
                "memory_usage": random.uniform(1.0, 100.0),
                "cpu_usage": random.uniform(0.0, 100.0),
                "iterations": random.randint(1, 1000),
                "throughput": random.uniform(10.0, 1000.0),
                "timestamp": datetime.now().isoformat()
            })
        return performance_data
    
    def _generate_integration_data(self, count: int, **kwargs) -> List[Dict[str, Any]]:
        """Generate integration test data."""
        integration_data = []
        for i in range(count):
            integration_data.append({
                "test_case": f"integration_test_{i}",
                "components": [self.generate_string(8) for _ in range(random.randint(2, 5))],
                "data_flow": [
                    {"from": self.generate_string(6), "to": self.generate_string(6), "type": "data"}
                    for _ in range(random.randint(1, 3))
                ],
                "expected_result": self.generate_string(20),
                "actual_result": self.generate_string(20),
                "status": random.choice(["passed", "failed", "pending"]),
                "timestamp": datetime.now().isoformat()
            })
        return integration_data

class PerformanceProfiler:
    """Refactored performance profiler with enhanced functionality."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.process = psutil.Process()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def measure_operation(self, operation_name: str, func: Callable, *args, **kwargs) -> PerformanceMetrics:
        """Measure performance of an operation."""
        gc.collect()  # Force garbage collection
        
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        start_time = time.perf_counter()
        start_cpu = self.process.cpu_percent()
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            end_time = time.perf_counter()
            end_cpu = self.process.cpu_percent()
        
        duration = end_time - start_time
        memory_usage = (self.process.memory_info().rss / 1024 / 1024) - initial_memory
        cpu_usage = (start_cpu + end_cpu) / 2
        throughput = 1 / duration if duration > 0 else float('inf')
        
        metrics = PerformanceMetrics(
            operation=operation_name,
            duration=duration,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            iterations=1,
            throughput=throughput
        )
        
        self.metrics.append(metrics)
        return metrics
    
    def measure_iterations(self, operation_name: str, func: Callable, iterations: int, *args, **kwargs) -> PerformanceMetrics:
        """Measure performance over multiple iterations."""
        gc.collect()
        
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        start_time = time.perf_counter()
        start_cpu = self.process.cpu_percent()
        
        try:
            for _ in range(iterations):
                func(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            end_time = time.perf_counter()
            end_cpu = self.process.cpu_percent()
        
        total_duration = end_time - start_time
        avg_duration = total_duration / iterations
        memory_usage = (self.process.memory_info().rss / 1024 / 1024) - initial_memory
        cpu_usage = (start_cpu + end_cpu) / 2
        throughput = iterations / total_duration if total_duration > 0 else float('inf')
        
        metrics = PerformanceMetrics(
            operation=operation_name,
            duration=avg_duration,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            iterations=iterations,
            throughput=throughput
        )
        
        self.metrics.append(metrics)
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics:
            return {}
        
        durations = [m.duration for m in self.metrics]
        memory_usage = [m.memory_usage for m in self.metrics]
        cpu_usage = [m.cpu_usage for m in self.metrics]
        throughput = [m.throughput for m in self.metrics]
        
        return {
            "total_operations": len(self.metrics),
            "total_duration": sum(durations),
            "average_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "total_memory_usage": sum(memory_usage),
            "average_memory_usage": sum(memory_usage) / len(memory_usage),
            "max_memory_usage": max(memory_usage),
            "average_cpu_usage": sum(cpu_usage) / len(cpu_usage),
            "max_cpu_usage": max(cpu_usage),
            "average_throughput": sum(throughput) / len(throughput),
            "max_throughput": max(throughput)
        }
    
    def reset(self):
        """Reset profiler metrics."""
        self.metrics.clear()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024

class TestAssertions:
    """Refactored test assertions with enhanced functionality."""
    
    @staticmethod
    def assert_performance(actual_duration: float, max_duration: float, operation: str = "operation"):
        """Assert performance meets requirements."""
        if actual_duration > max_duration:
            raise AssertionError(f"{operation} took {actual_duration:.3f}s, exceeds threshold {max_duration:.3f}s")
    
    @staticmethod
    def assert_memory_usage(actual_memory: float, max_memory: float, operation: str = "operation"):
        """Assert memory usage is within limits."""
        if actual_memory > max_memory:
            raise AssertionError(f"{operation} used {actual_memory:.1f}MB, exceeds threshold {max_memory:.1f}MB")
    
    @staticmethod
    def assert_json_equals(actual: str, expected: Dict[str, Any]):
        """Assert JSON string equals expected dict."""
        try:
            actual_dict = json.loads(actual)
        except json.JSONDecodeError as e:
            raise AssertionError(f"Invalid JSON: {e}")
        
        if actual_dict != expected:
            raise AssertionError(f"JSON not equal: {actual_dict} != {expected}")
    
    @staticmethod
    def assert_list_contains(list_data: List[Any], expected_item: Any):
        """Assert list contains expected item."""
        if expected_item not in list_data:
            raise AssertionError(f"Expected item {expected_item} not found in list")
    
    @staticmethod
    def assert_dict_contains(actual: Dict[str, Any], expected: Dict[str, Any]):
        """Assert actual dict contains all expected keys and values."""
        for key, value in expected.items():
            if key not in actual:
                raise AssertionError(f"Key '{key}' not found in actual dict")
            if actual[key] != value:
                raise AssertionError(f"Value for key '{key}' doesn't match: {actual[key]} != {value}")
    
    @staticmethod
    def assert_async_result(coro, expected_result: Any, timeout: float = 5.0):
        """Assert async coroutine returns expected result."""
        try:
            result = asyncio.run(asyncio.wait_for(coro, timeout=timeout))
            if result != expected_result:
                raise AssertionError(f"Async result {result} != expected {expected_result}")
        except asyncio.TimeoutError:
            raise AssertionError(f"Async operation timed out after {timeout}s")
        except Exception as e:
            raise AssertionError(f"Async operation failed: {e}")

# Create global instance for easy access
test_assertions = TestAssertions()

class TestRunner:
    """Refactored test runner with enhanced functionality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.results: List[Dict[str, Any]] = []
        self.start_time = None
    
    def run_test(self, test_func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Run a single test function."""
        test_name = getattr(test_func, '__name__', 'unknown_test')
        start_time = time.time()
        
        try:
            result = test_func(*args, **kwargs)
            status = TestStatus.PASSED
            message = None
        except AssertionError as e:
            status = TestStatus.FAILED
            message = str(e)
            result = None
        except Exception as e:
            status = TestStatus.ERROR
            message = str(e)
            result = None
        
        duration = time.time() - start_time
        
        test_result = {
            "name": test_name,
            "status": status.value,
            "duration": duration,
            "message": message,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results.append(test_result)
        return test_result
    
    def run_tests(self, test_functions: List[Callable], *args, **kwargs) -> List[Dict[str, Any]]:
        """Run multiple test functions."""
        self.start_time = time.time()
        results = []
        
        for test_func in test_functions:
            result = self.run_test(test_func, *args, **kwargs)
            results.append(result)
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test run summary."""
        if not self.results:
            return {"total": 0, "passed": 0, "failed": 0, "errors": 0}
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == TestStatus.PASSED.value)
        failed = sum(1 for r in self.results if r["status"] == TestStatus.FAILED.value)
        errors = sum(1 for r in self.results if r["status"] == TestStatus.ERROR.value)
        
        total_duration = sum(r["duration"] for r in self.results)
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "total_duration": total_duration,
            "average_duration": total_duration / total if total > 0 else 0
        }

# Global instances
data_generator = TestDataGenerator()
performance_profiler = PerformanceProfiler()
test_runner = TestRunner()

# Convenience functions
def generate_test_data(data_type: DataType, count: int = 1, **kwargs):
    """Generate test data using global generator."""
    return data_generator.generate_test_data(data_type, count, **kwargs)

def measure_performance(operation_name: str, func: Callable, *args, **kwargs):
    """Measure performance using global profiler."""
    return performance_profiler.measure_operation(operation_name, func, *args, **kwargs)

def run_test(test_func: Callable, *args, **kwargs):
    """Run test using global runner."""
    return test_runner.run_test(test_func, *args, **kwargs)

# Assertion convenience functions
def assert_performance(actual_duration: float, max_duration: float, operation: str = "operation"):
    """Assert performance meets requirements."""
    return test_assertions.assert_performance(actual_duration, max_duration, operation)

def assert_memory_usage(actual_memory: float, max_memory: float, operation: str = "operation"):
    """Assert memory usage is within limits."""
    return test_assertions.assert_memory_usage(actual_memory, max_memory, operation)

def assert_json_equals(actual: str, expected: Dict[str, Any]):
    """Assert JSON string equals expected dict."""
    return test_assertions.assert_json_equals(actual, expected)

def assert_list_contains(list_data: List[Any], expected_item: Any):
    """Assert list contains expected item."""
    return test_assertions.assert_list_contains(list_data, expected_item)

def assert_dict_contains(actual: Dict[str, Any], expected: Dict[str, Any]):
    """Assert actual dict contains all expected keys and values."""
    return test_assertions.assert_dict_contains(actual, expected)

if __name__ == "__main__":
    # Demo the utilities
    print("Testing refactored utilities...")
    
    # Test data generation
    users = generate_test_data(DataType.USER, 3)
    print(f"Generated {len(users)} users")
    
    # Test performance measurement
    def sample_operation():
        return sum(range(1000))
    
    metrics = measure_performance("sample_operation", sample_operation)
    print(f"Performance metrics: {metrics.duration:.3f}s, {metrics.memory_usage:.1f}MB")
    
    # Test assertions
    test_assertions.assert_performance(0.1, 1.0, "test_operation")
    print("All tests passed!")
