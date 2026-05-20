"""
Test Utilities
Additional utility functions for testing
"""

import pytest
from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock, AsyncMock
import asyncio
import json
from datetime import datetime


class TestAssertions:
    """Extended test assertions"""
    
    @staticmethod
    def assert_dict_equals(expected: Dict[str, Any], actual: Dict[str, Any], ignore_keys: Optional[List[str]] = None):
        """Assert dictionaries are equal, optionally ignoring keys"""
        ignore_keys = ignore_keys or []
        filtered_expected = {k: v for k, v in expected.items() if k not in ignore_keys}
        filtered_actual = {k: v for k, v in actual.items() if k not in ignore_keys}
        assert filtered_expected == filtered_actual
    
    @staticmethod
    def assert_list_contains(items: List[Any], expected_item: Any, key: Optional[str] = None):
        """Assert list contains item, optionally by key"""
        if key:
            assert any(getattr(item, key, None) == expected_item for item in items), \
                f"List does not contain item with {key}={expected_item}"
        else:
            assert expected_item in items, f"List does not contain {expected_item}"
    
    @staticmethod
    def assert_list_length(items: List[Any], expected_length: int):
        """Assert list has expected length"""
        assert len(items) == expected_length, \
            f"List has length {len(items)}, expected {expected_length}"
    
    @staticmethod
    def assert_datetime_approx(actual: datetime, expected: datetime, tolerance_seconds: int = 5):
        """Assert datetime is approximately equal"""
        diff = abs((actual - expected).total_seconds())
        assert diff <= tolerance_seconds, \
            f"Datetime difference {diff}s exceeds tolerance {tolerance_seconds}s"


class AsyncTestUtils:
    """Utilities for async testing"""
    
    @staticmethod
    async def collect_results(tasks: List[Callable], timeout: Optional[float] = None) -> List[Any]:
        """Collect results from async tasks"""
        if timeout:
            return await asyncio.wait_for(
                asyncio.gather(*[task() if asyncio.iscoroutinefunction(task) else asyncio.to_thread(task) for task in tasks]),
                timeout=timeout
            )
        return await asyncio.gather(*[task() if asyncio.iscoroutinefunction(task) else asyncio.to_thread(task) for task in tasks])
    
    @staticmethod
    async def run_with_retry(
        func: Callable,
        max_retries: int = 3,
        retry_delay: float = 0.1,
        expected_exception: Optional[type] = None
    ) -> Any:
        """Run function with retry logic"""
        last_exception = None
        for attempt in range(max_retries):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func()
                return func()
            except Exception as e:
                last_exception = e
                if expected_exception and not isinstance(e, expected_exception):
                    raise
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    raise
        raise last_exception


class MockUtils:
    """Utilities for working with mocks"""
    
    @staticmethod
    def reset_mock(mock: Mock):
        """Reset mock to initial state"""
        mock.reset_mock()
        mock.called = False
        mock.call_count = 0
    
    @staticmethod
    def assert_called_with_args(mock: Mock, *args, **kwargs):
        """Assert mock was called with specific arguments"""
        assert mock.called, "Mock was not called"
        if args or kwargs:
            mock.assert_called_with(*args, **kwargs)
        else:
            assert mock.called
    
    @staticmethod
    def get_call_args(mock: Mock, call_index: int = 0) -> tuple:
        """Get arguments from specific call"""
        assert mock.call_count > call_index, f"Mock was not called {call_index + 1} times"
        return mock.call_args_list[call_index]
    
    @staticmethod
    def verify_call_order(mock: Mock, expected_calls: List[str]):
        """Verify mock methods were called in expected order"""
        actual_calls = [call[0][0] for call in mock.call_args_list if call[0]]
        assert actual_calls == expected_calls, \
            f"Call order {actual_calls} does not match expected {expected_calls}"


class JSONUtils:
    """Utilities for JSON testing"""
    
    @staticmethod
    def assert_json_valid(json_string: str):
        """Assert JSON string is valid"""
        try:
            json.loads(json_string)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON: {e}")
    
    @staticmethod
    def assert_json_contains(json_string: str, key: str, expected_value: Any = None):
        """Assert JSON contains key, optionally with value"""
        data = json.loads(json_string)
        assert key in data, f"JSON does not contain key '{key}'"
        if expected_value is not None:
            assert data[key] == expected_value, \
                f"JSON key '{key}' has value {data[key]}, expected {expected_value}"
    
    @staticmethod
    def compare_json(json1: str, json2: str, ignore_keys: Optional[List[str]] = None):
        """Compare two JSON strings"""
        data1 = json.loads(json1)
        data2 = json.loads(json2)
        TestAssertions.assert_dict_equals(data1, data2, ignore_keys)


class FileUtils:
    """Utilities for file testing"""
    
    @staticmethod
    def create_temp_file(content: bytes, suffix: str = ".tmp") -> str:
        """Create temporary file with content"""
        import tempfile
        import os
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            os.write(fd, content)
            return path
        finally:
            os.close(fd)
    
    @staticmethod
    def read_file_content(file_path: str) -> bytes:
        """Read file content"""
        with open(file_path, 'rb') as f:
            return f.read()
    
    @staticmethod
    def assert_file_exists(file_path: str):
        """Assert file exists"""
        import os
        assert os.path.exists(file_path), f"File {file_path} does not exist"
    
    @staticmethod
    def assert_file_size(file_path: str, expected_size: int, tolerance: int = 0):
        """Assert file has expected size"""
        import os
        actual_size = os.path.getsize(file_path)
        assert abs(actual_size - expected_size) <= tolerance, \
            f"File size {actual_size} does not match expected {expected_size} (tolerance: {tolerance})"


# Convenience exports
assert_dict_equals = TestAssertions.assert_dict_equals
assert_list_contains = TestAssertions.assert_list_contains
assert_list_length = TestAssertions.assert_list_length
assert_datetime_approx = TestAssertions.assert_datetime_approx

collect_results = AsyncTestUtils.collect_results
run_with_retry = AsyncTestUtils.run_with_retry

reset_mock = MockUtils.reset_mock
assert_called_with_args = MockUtils.assert_called_with_args
get_call_args = MockUtils.get_call_args
verify_call_order = MockUtils.verify_call_order

assert_json_valid = JSONUtils.assert_json_valid
assert_json_contains = JSONUtils.assert_json_contains
compare_json = JSONUtils.compare_json

create_temp_file = FileUtils.create_temp_file
read_file_content = FileUtils.read_file_content
assert_file_exists = FileUtils.assert_file_exists
assert_file_size = FileUtils.assert_file_size



