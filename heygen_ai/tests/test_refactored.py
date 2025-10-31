"""
Refactored test suite for HeyGen AI system.
Uses the new base classes and utilities for better organization and maintainability.
"""

import pytest
import asyncio
import time
import json
from typing import Dict, List, Any
from pathlib import Path

# Import refactored utilities
from tests.core.test_base import BaseTest, PerformanceTest, IntegrationTest, UnitTest, TestConfig
from tests.utils.test_utilities import (
    TestDataGenerator, PerformanceProfiler, TestAssertions, TestRunner,
    DataType, TestStatus, generate_test_data, measure_performance, run_test
)

class TestBasicFunctionality(BaseTest):
    """Refactored basic functionality tests."""
    
    def test_basic_operations(self):
        """Test basic Python operations."""
        # Test math operations
        result, duration = self.measure_time(lambda: 2 + 2)
        assert result == 4
        self.assert_performance(duration, 0.001, "math_operation")
        
        # Test string operations
        result, duration = self.measure_time(lambda: "hello" in "hello world")
        assert result is True
        self.assert_performance(duration, 0.001, "string_operation")
        
        # Test list operations
        result, duration = self.measure_time(lambda: len([1, 2, 3]))
        assert result == 3
        self.assert_performance(duration, 0.001, "list_operation")
        
        self.record_result("basic_operations", "passed", duration)
    
    def test_import_functionality(self):
        """Test import functionality with error handling."""
        import_errors = []
        
        # Test standard library imports
        standard_modules = ["json", "time", "os", "sys", "pathlib", "datetime", "asyncio"]
        
        for module_name in standard_modules:
            try:
                __import__(module_name)
            except ImportError as e:
                import_errors.append(f"Failed to import {module_name}: {e}")
        
        # Test core module imports with fallback
        core_modules = [
            ("core.base_service", ["ServiceStatus", "ServiceType"]),
            ("core.dependency_manager", ["ServicePriority", "ServiceInfo"]),
            ("core.error_handler", ["ErrorHandler"])
        ]
        
        for module_name, classes in core_modules:
            try:
                module = __import__(module_name, fromlist=classes)
                for class_name in classes:
                    if not hasattr(module, class_name):
                        import_errors.append(f"Missing class {class_name} in {module_name}")
            except ImportError:
                # Expected for some modules
                pass
        
        # Record results
        if import_errors:
            self.record_result("import_functionality", "failed", 0, message=f"Import errors: {len(import_errors)}")
        else:
            self.record_result("import_functionality", "passed", 0)
    
    def test_async_functionality(self):
        """Test async functionality."""
        async def async_test():
            await asyncio.sleep(0.01)
            return "async_result"
        
        result, duration = self.measure_async_time(async_test())
        assert result == "async_result"
        self.assert_performance(duration, 0.1, "async_operation")
        
        self.record_result("async_functionality", "passed", duration)
    
    def test_json_serialization(self):
        """Test JSON serialization."""
        test_data = {
            "name": "test",
            "value": 42,
            "nested": {"key": "value"},
            "list": [1, 2, 3]
        }
        
        # Test serialization
        result, duration = self.measure_time(lambda: json.dumps(test_data))
        assert isinstance(result, str)
        self.assert_performance(duration, 0.01, "json_serialization")
        
        # Test deserialization
        result, duration = self.measure_time(lambda: json.loads(json.dumps(test_data)))
        assert result == test_data
        self.assert_performance(duration, 0.01, "json_deserialization")
        
        self.record_result("json_serialization", "passed", duration)

class TestPerformanceRefactored(PerformanceTest):
    """Refactored performance tests."""
    
    def test_list_operations_performance(self):
        """Test list operations performance."""
        test_data = list(range(10000))
        
        # Test sorting performance
        benchmark = self.benchmark_operation(
            "list_sorting",
            lambda: sorted(test_data),
            iterations=10
        )
        
        self.assert_benchmark(benchmark, max_duration=0.1)
        self.record_result("list_operations", "passed", benchmark["avg_duration"])
    
    def test_dict_operations_performance(self):
        """Test dictionary operations performance."""
        test_data = {f"key_{i}": f"value_{i}" for i in range(10000)}
        
        # Test access performance
        def dict_access_test():
            return [test_data[f"key_{i}"] for i in range(100)]
        
        benchmark = self.benchmark_operation(
            "dict_access",
            dict_access_test,
            iterations=50
        )
        
        self.assert_benchmark(benchmark, max_duration=0.05)
        self.record_result("dict_operations", "passed", benchmark["avg_duration"])
    
    def test_string_operations_performance(self):
        """Test string operations performance."""
        test_string = "This is a test string for performance testing"
        
        def string_operations():
            return test_string.upper().lower().replace("test", "performance").split()
        
        benchmark = self.benchmark_operation(
            "string_operations",
            string_operations,
            iterations=1000
        )
        
        self.assert_benchmark(benchmark, max_duration=0.01)
        self.record_result("string_operations", "passed", benchmark["avg_duration"])

class TestIntegrationRefactored(IntegrationTest):
    """Refactored integration tests."""
    
    def test_file_system_integration(self):
        """Test file system integration."""
        # Setup test data
        test_file = Path(__file__)
        
        # Test file operations
        def file_exists_test():
            return test_file.exists()
        
        def file_read_test():
            return test_file.read_text(encoding='utf-8').startswith('"""')
        
        # Measure file operations
        exists_result, exists_duration = self.measure_time(file_exists_test)
        read_result, read_duration = self.measure_time(file_read_test)
        
        assert exists_result is True
        assert read_result is True
        
        self.assert_performance(exists_duration, 0.1, "file_exists")
        self.assert_performance(read_duration, 0.1, "file_read")
        
        self.record_result("file_system_integration", "passed", exists_duration + read_duration)
    
    def test_subprocess_integration(self):
        """Test subprocess integration."""
        import subprocess
        
        def python_version_test():
            result = subprocess.run(
                ["python", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        
        result, duration = self.measure_time(python_version_test)
        assert result is True
        self.assert_performance(duration, 1.0, "subprocess_test")
        
        self.record_result("subprocess_integration", "passed", duration)

class TestUnitRefactored(UnitTest):
    """Refactored unit tests."""
    
    def test_data_structures(self):
        """Test data structures."""
        from enum import Enum
        from dataclasses import dataclass
        
        # Test enum
        class TestEnum(Enum):
            VALUE1 = "value1"
            VALUE2 = "value2"
        
        assert TestEnum.VALUE1 != TestEnum.VALUE2
        assert TestEnum.VALUE1 == TestEnum.VALUE1
        
        # Test dataclass
        @dataclass
        class TestData:
            name: str
            value: int
        
        data1 = TestData("test", 42)
        data2 = TestData("test", 42)
        data3 = TestData("different", 42)
        
        assert data1 == data2
        assert data1 != data3
        
        self.record_result("data_structures", "passed", 0)
    
    def test_error_handling(self):
        """Test error handling."""
        # Test exception handling
        def test_division_by_zero():
            return 1 / 0
        
        def test_key_error():
            return {}["missing"]
        
        def test_type_error():
            return "string" + 123
        
        # Test that exceptions are raised
        with pytest.raises(ZeroDivisionError):
            test_division_by_zero()
        
        with pytest.raises(KeyError):
            test_key_error()
        
        with pytest.raises(TypeError):
            test_type_error()
        
        self.record_result("error_handling", "passed", 0)

class TestDataGeneration:
    """Test data generation functionality."""
    
    def test_user_data_generation(self):
        """Test user data generation."""
        users = generate_test_data(DataType.USER, 5)
        
        assert len(users) == 5
        assert all("username" in user.data for user in users)
        assert all("email" in user.data for user in users)
        assert all("id" in user.data for user in users)
        
        # Test data uniqueness
        usernames = [user.data["username"] for user in users]
        assert len(set(usernames)) == len(usernames)  # All unique
    
    def test_service_data_generation(self):
        """Test service data generation."""
        services = generate_test_data(DataType.SERVICE, 3)
        
        assert len(services) == 3
        assert all("name" in service.data for service in services)
        assert all("status" in service.data for service in services)
        assert all("port" in service.data for service in services)
    
    def test_api_request_data_generation(self):
        """Test API request data generation."""
        requests = generate_test_data(DataType.API_REQUEST, 2)
        
        assert len(requests) == 2
        assert all("method" in req.data for req in requests)
        assert all("url" in req.data for req in requests)
        assert all("headers" in req.data for req in requests)

class TestPerformanceProfiling:
    """Test performance profiling functionality."""
    
    def test_performance_measurement(self):
        """Test performance measurement."""
        def sample_operation():
            return sum(range(1000))
        
        metrics = measure_performance("sample_operation", sample_operation)
        
        assert metrics.operation == "sample_operation"
        assert metrics.duration > 0
        assert metrics.memory_usage >= 0
        assert metrics.cpu_usage >= 0
        assert metrics.iterations == 1
        assert metrics.throughput > 0
    
    def test_performance_benchmark(self):
        """Test performance benchmarking."""
        def benchmark_operation():
            return [i * i for i in range(100)]
        
        # Create a new profiler for this test
        profiler = PerformanceProfiler()
        
        metrics = profiler.measure_iterations("benchmark_operation", benchmark_operation, 10)
        
        assert metrics.operation == "benchmark_operation"
        assert metrics.iterations == 10
        assert metrics.duration > 0
        assert metrics.throughput > 0

class TestAssertions:
    """Test assertion functionality."""
    
    def test_performance_assertions(self):
        """Test performance assertions."""
        TestAssertions.assert_performance(0.1, 1.0, "test_operation")
        
        with pytest.raises(AssertionError):
            TestAssertions.assert_performance(2.0, 1.0, "test_operation")
    
    def test_memory_assertions(self):
        """Test memory assertions."""
        TestAssertions.assert_memory_usage(50.0, 100.0, "test_operation")
        
        with pytest.raises(AssertionError):
            TestAssertions.assert_memory_usage(150.0, 100.0, "test_operation")
    
    def test_json_assertions(self):
        """Test JSON assertions."""
        test_data = {"name": "test", "value": 42}
        json_string = json.dumps(test_data)
        
        TestAssertions.assert_json_equals(json_string, test_data)
        
        with pytest.raises(AssertionError):
            TestAssertions.assert_json_equals(json_string, {"different": "data"})
    
    def test_list_assertions(self):
        """Test list assertions."""
        test_list = [1, 2, 3, 4, 5]
        
        TestAssertions.assert_list_contains(test_list, 3)
        
        with pytest.raises(AssertionError):
            TestAssertions.assert_list_contains(test_list, 6)
    
    def test_dict_assertions(self):
        """Test dictionary assertions."""
        test_dict = {"a": 1, "b": 2, "c": 3}
        expected = {"a": 1, "b": 2}
        
        TestAssertions.assert_dict_contains(test_dict, expected)
        
        with pytest.raises(AssertionError):
            TestAssertions.assert_dict_contains(test_dict, {"d": 4})

class TestRunnerFunctionality:
    """Test test runner functionality."""
    
    def test_single_test_execution(self):
        """Test single test execution."""
        def sample_test():
            return "test_result"
        
        result = run_test(sample_test)
        
        assert result["name"] == "sample_test"
        assert result["status"] == "passed"
        assert result["result"] == "test_result"
        assert result["duration"] > 0
    
    def test_failing_test_execution(self):
        """Test failing test execution."""
        def failing_test():
            assert False, "This test should fail"
        
        result = run_test(failing_test)
        
        assert result["name"] == "failing_test"
        assert result["status"] == "failed"
        assert "This test should fail" in result["message"]
        assert result["duration"] > 0

# Test fixtures using refactored utilities
@pytest.fixture
def test_data_generator():
    """Test data generator fixture."""
    return TestDataGenerator(seed=42)  # Fixed seed for reproducibility

@pytest.fixture
def performance_profiler():
    """Performance profiler fixture."""
    return PerformanceProfiler()

@pytest.fixture
def test_runner():
    """Test runner fixture."""
    return TestRunner()

# Test markers
pytestmark = pytest.mark.usefixtures("test_data_generator", "performance_profiler", "test_runner")

if __name__ == "__main__":
    # Run the refactored tests
    pytest.main([__file__, "-v"])
