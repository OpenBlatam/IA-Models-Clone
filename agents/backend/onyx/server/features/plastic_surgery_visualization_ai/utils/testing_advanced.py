"""Advanced testing utilities."""

from typing import Any, Callable, Dict, List, Optional, TypeVar
from contextlib import contextmanager
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio
import time
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import io

T = TypeVar('T')


class MockBuilder:
    """Builder for creating mocks."""
    
    def __init__(self):
        self.mock = Mock()
    
    def with_return_value(self, value: Any) -> 'MockBuilder':
        """Set return value."""
        self.mock.return_value = value
        return self
    
    def with_side_effect(self, effect: Callable) -> 'MockBuilder':
        """Set side effect."""
        self.mock.side_effect = effect
        return self
    
    def with_attributes(self, **attrs) -> 'MockBuilder':
        """Set attributes."""
        for key, value in attrs.items():
            setattr(self.mock, key, value)
        return self
    
    def build(self) -> Mock:
        """Build mock."""
        return self.mock


class AsyncMockBuilder(MockBuilder):
    """Builder for async mocks."""
    
    def __init__(self):
        super().__init__()
        self.mock = MagicMock()
    
    def with_async_return(self, value: Any) -> 'AsyncMockBuilder':
        """Set async return value."""
        async def async_return():
            return value
        self.mock.return_value = async_return()
        return self
    
    def build(self) -> MagicMock:
        """Build async mock."""
        return self.mock


class TestFixture:
    """Test fixture manager."""
    
    def __init__(self):
        self.fixtures: Dict[str, Any] = {}
        self.setup_callbacks: List[Callable] = []
        self.teardown_callbacks: List[Callable] = []
    
    def register(self, name: str, value: Any):
        """Register fixture."""
        self.fixtures[name] = value
    
    def get(self, name: str) -> Any:
        """Get fixture."""
        return self.fixtures.get(name)
    
    def setup(self, callback: Callable):
        """Register setup callback."""
        self.setup_callbacks.append(callback)
    
    def teardown(self, callback: Callable):
        """Register teardown callback."""
        self.teardown_callbacks.append(callback)
    
    def run_setup(self):
        """Run setup callbacks."""
        for callback in self.setup_callbacks:
            callback()
    
    def run_teardown(self):
        """Run teardown callbacks."""
        for callback in reversed(self.teardown_callbacks):
            try:
                callback()
            except Exception as e:
                print(f"Error in teardown: {e}")


class AssertHelper:
    """Helper for assertions."""
    
    @staticmethod
    def assert_dict_contains(dict1: Dict, dict2: Dict):
        """Assert dict1 contains all keys from dict2."""
        for key, value in dict2.items():
            assert key in dict1, f"Key {key} not in dict1"
            assert dict1[key] == value, f"Value mismatch for {key}"
    
    @staticmethod
    def assert_approximately_equal(value1: float, value2: float, tolerance: float = 0.001):
        """Assert values are approximately equal."""
        assert abs(value1 - value2) < tolerance, f"{value1} not approximately equal to {value2}"
    
    @staticmethod
    def assert_list_contains(list1: List, list2: List):
        """Assert list1 contains all items from list2."""
        for item in list2:
            assert item in list1, f"Item {item} not in list1"
    
    @staticmethod
    def assert_async_result(coro, expected: Any):
        """Assert async coroutine result."""
        result = asyncio.run(coro)
        assert result == expected, f"Expected {expected}, got {result}"


class PerformanceTest:
    """Performance testing utilities."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
    
    @contextmanager
    def measure(self, name: str):
        """Measure execution time."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.results.append({
                'name': name,
                'time': elapsed,
                'timestamp': time.time()
            })
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get performance results."""
        return self.results.copy()
    
    def get_average_time(self, name: Optional[str] = None) -> float:
        """Get average time for test."""
        if name:
            times = [r['time'] for r in self.results if r['name'] == name]
        else:
            times = [r['time'] for r in self.results]
        
        return sum(times) / len(times) if times else 0.0


class TestDataFactory:
    """Factory for test data."""
    
    @staticmethod
    def create_dict(keys: List[str], values: Optional[List[Any]] = None) -> Dict:
        """Create test dictionary."""
        if values is None:
            values = [None] * len(keys)
        return dict(zip(keys, values))
    
    @staticmethod
    def create_list(count: int, factory: Callable) -> List:
        """Create list of test data."""
        return [factory() for _ in range(count)]
    
    @staticmethod
    def create_nested_dict(depth: int, width: int) -> Dict:
        """Create nested dictionary."""
        if depth == 0:
            return {}
        
        result = {}
        for i in range(width):
            key = f"key_{i}"
            if depth > 1:
                result[key] = TestDataFactory.create_nested_dict(depth - 1, width)
            else:
                result[key] = f"value_{i}"
        
        return result


class AsyncTestHelper:
    """Helper for async tests."""
    
    @staticmethod
    async def wait_for_condition(
        condition: Callable,
        timeout: float = 5.0,
        interval: float = 0.1
    ) -> bool:
        """Wait for condition to be true."""
        start = time.time()
        while time.time() - start < timeout:
            if condition():
                return True
            await asyncio.sleep(interval)
        return False
    
    @staticmethod
    async def run_with_timeout(coro, timeout: float = 5.0) -> Any:
        """Run coroutine with timeout."""
        return await asyncio.wait_for(coro, timeout=timeout)
    
    @staticmethod
    async def gather_with_errors(*coros) -> List[Any]:
        """Gather coroutines and return exceptions."""
        results = await asyncio.gather(*coros, return_exceptions=True)
        return results


class TestDouble:
    """Test double (stub/spy/mock)."""
    
    def __init__(self, target: Any):
        self.target = target
        self.calls: List[Dict[str, Any]] = []
    
    def __call__(self, *args, **kwargs):
        """Record call."""
        self.calls.append({
            'args': args,
            'kwargs': kwargs,
            'timestamp': time.time()
        })
        return self.target(*args, **kwargs)
    
    def get_calls(self) -> List[Dict[str, Any]]:
        """Get recorded calls."""
        return self.calls.copy()
    
    def was_called(self) -> bool:
        """Check if was called."""
        return len(self.calls) > 0
    
    def was_called_with(self, *args, **kwargs) -> bool:
        """Check if called with specific arguments."""
        for call in self.calls:
            if call['args'] == args and call['kwargs'] == kwargs:
                return True
        return False


def patch_and_verify(target: str, return_value: Any = None):
    """Patch and verify call."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with patch(target, return_value=return_value) as mock:
                result = func(*args, **kwargs)
                assert mock.called, f"{target} was not called"
                return result
        return wrapper
    return decorator

