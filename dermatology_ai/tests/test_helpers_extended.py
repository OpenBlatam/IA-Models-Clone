"""
Extended Test Helpers
Additional helper utilities for testing
"""

import pytest
from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock, AsyncMock
import asyncio
import time
from datetime import datetime, timedelta


class PerformanceHelpers:
    """Helpers for performance testing"""
    
    @staticmethod
    async def measure_execution_time(func: Callable, *args, **kwargs) -> tuple[Any, float]:
        """Measure execution time of a function"""
        start_time = time.time()
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    
    @staticmethod
    def assert_performance_threshold(execution_time: float, max_time: float, operation: str = "operation"):
        """Assert that execution time is within threshold"""
        assert execution_time <= max_time, \
            f"{operation} took {execution_time:.3f}s, expected <= {max_time:.3f}s"
    
    @staticmethod
    async def run_benchmark(func: Callable, iterations: int = 10, *args, **kwargs) -> Dict[str, float]:
        """Run benchmark and return statistics"""
        times = []
        for _ in range(iterations):
            _, exec_time = await PerformanceHelpers.measure_execution_time(func, *args, **kwargs)
            times.append(exec_time)
        
        return {
            "min": min(times),
            "max": max(times),
            "avg": sum(times) / len(times),
            "total": sum(times)
        }


class MockHelpersExtended:
    """Extended mock helpers"""
    
    @staticmethod
    def create_chain_mock(chain_config: List[Dict[str, Any]]) -> Mock:
        """Create a mock with method chaining"""
        mock = Mock()
        current = mock
        
        for config in chain_config:
            method_name = config["method"]
            return_value = config.get("return_value")
            side_effect = config.get("side_effect")
            
            next_mock = Mock()
            if return_value is not None:
                next_mock.return_value = return_value
            if side_effect is not None:
                next_mock.side_effect = side_effect
            
            setattr(current, method_name, Mock(return_value=next_mock))
            current = next_mock
        
        return mock
    
    @staticmethod
    def create_async_mock_sequence(return_values: List[Any]) -> AsyncMock:
        """Create async mock that returns different values on each call"""
        mock = AsyncMock()
        mock.side_effect = return_values
        return mock
    
    @staticmethod
    def create_mock_with_call_tracking() -> tuple[Mock, List[Dict[str, Any]]]:
        """Create mock that tracks all calls"""
        calls = []
        mock = Mock()
        
        def track_call(*args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs, "timestamp": time.time()})
            return Mock()
        
        mock.side_effect = track_call
        return mock, calls


class DataHelpers:
    """Helpers for test data generation"""
    
    @staticmethod
    def generate_test_ids(count: int, prefix: str = "test") -> List[str]:
        """Generate list of test IDs"""
        return [f"{prefix}-{i}" for i in range(count)]
    
    @staticmethod
    def create_timestamp_range(start: datetime, count: int, interval_minutes: int = 60) -> List[datetime]:
        """Create list of timestamps with intervals"""
        return [start + timedelta(minutes=interval_minutes * i) for i in range(count)]
    
    @staticmethod
    def create_dict_with_defaults(keys: List[str], default_value: Any = None) -> Dict[str, Any]:
        """Create dictionary with default values for keys"""
        return {key: default_value for key in keys}
    
    @staticmethod
    def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple dictionaries"""
        result = {}
        for d in dicts:
            result.update(d)
        return result


class ValidationHelpers:
    """Helpers for validation testing"""
    
    @staticmethod
    def assert_dict_structure(data: Dict[str, Any], structure: Dict[str, type]):
        """Assert dictionary has expected structure with types"""
        for key, expected_type in structure.items():
            assert key in data, f"Missing key: {key}"
            assert isinstance(data[key], expected_type), \
                f"Key '{key}' has type {type(data[key])}, expected {expected_type}"
    
    @staticmethod
    def assert_list_items_type(items: List[Any], item_type: type):
        """Assert all items in list are of expected type"""
        assert all(isinstance(item, item_type) for item in items), \
            f"Not all items are of type {item_type}"
    
    @staticmethod
    def assert_range(value: float, min_val: float, max_val: float, name: str = "value"):
        """Assert value is within range"""
        assert min_val <= value <= max_val, \
            f"{name} ({value}) is not in range [{min_val}, {max_val}]"


class AsyncHelpersExtended:
    """Extended async helpers"""
    
    @staticmethod
    async def run_with_timeout(coro, timeout: float = 5.0):
        """Run coroutine with timeout"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise AssertionError(f"Operation timed out after {timeout}s")
    
    @staticmethod
    async def wait_for_async_condition(
        condition_func: Callable,
        timeout: float = 5.0,
        interval: float = 0.1,
        error_message: str = "Condition not met"
    ) -> bool:
        """Wait for async condition with better error handling"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if asyncio.iscoroutinefunction(condition_func):
                result = await condition_func()
            else:
                result = condition_func()
            
            if result:
                return True
            
            await asyncio.sleep(interval)
        
        raise AssertionError(f"{error_message} (timeout: {timeout}s)")
    
    @staticmethod
    async def run_parallel_with_results(tasks: List[Callable], max_workers: int = 5) -> List[Any]:
        """Run tasks in parallel and return results"""
        semaphore = asyncio.Semaphore(max_workers)
        
        async def run_with_limit(task):
            async with semaphore:
                if asyncio.iscoroutinefunction(task):
                    return await task()
                return task()
        
        return await asyncio.gather(*[run_with_limit(task) for task in tasks])


class ErrorHelpers:
    """Helpers for error testing"""
    
    @staticmethod
    def assert_error_contains(error: Exception, expected_text: str):
        """Assert error message contains expected text"""
        assert expected_text.lower() in str(error).lower(), \
            f"Error message '{str(error)}' does not contain '{expected_text}'"
    
    @staticmethod
    def assert_error_type(error: Exception, expected_type: type):
        """Assert error is of expected type"""
        assert isinstance(error, expected_type), \
            f"Error is {type(error)}, expected {expected_type}"
    
    @staticmethod
    async def assert_raises_async(exception_type: type, coro):
        """Assert coroutine raises expected exception"""
        with pytest.raises(exception_type):
            await coro


# Convenience exports
measure_execution_time = PerformanceHelpers.measure_execution_time
assert_performance_threshold = PerformanceHelpers.assert_performance_threshold
run_benchmark = PerformanceHelpers.run_benchmark

create_chain_mock = MockHelpersExtended.create_chain_mock
create_async_mock_sequence = MockHelpersExtended.create_async_mock_sequence
create_mock_with_call_tracking = MockHelpersExtended.create_mock_with_call_tracking

generate_test_ids = DataHelpers.generate_test_ids
create_timestamp_range = DataHelpers.create_timestamp_range
create_dict_with_defaults = DataHelpers.create_dict_with_defaults
merge_dicts = DataHelpers.merge_dicts

assert_dict_structure = ValidationHelpers.assert_dict_structure
assert_list_items_type = ValidationHelpers.assert_list_items_type
assert_range = ValidationHelpers.assert_range

run_with_timeout = AsyncHelpersExtended.run_with_timeout
wait_for_async_condition = AsyncHelpersExtended.wait_for_async_condition
run_parallel_with_results = AsyncHelpersExtended.run_parallel_with_results

assert_error_contains = ErrorHelpers.assert_error_contains
assert_error_type = ErrorHelpers.assert_error_type
assert_raises_async = ErrorHelpers.assert_raises_async

