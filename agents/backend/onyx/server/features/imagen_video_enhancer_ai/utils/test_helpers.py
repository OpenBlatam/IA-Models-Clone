"""
Test Helpers
============

Advanced testing utilities and fixtures.
"""

import asyncio
import tempfile
import shutil
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import Mock, AsyncMock, MagicMock
import pytest


class AsyncTestMixin:
    """Mixin for async test utilities."""
    
    @staticmethod
    def run_async(coro):
        """Run async coroutine in test."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    
    @staticmethod
    async def wait_for_condition(
        condition: Callable[[], bool],
        timeout: float = 5.0,
        interval: float = 0.1
    ) -> bool:
        """
        Wait for condition to be true.
        
        Args:
            condition: Function that returns bool
            timeout: Maximum time to wait
            interval: Check interval
            
        Returns:
            True if condition met, False if timeout
        """
        elapsed = 0.0
        while elapsed < timeout:
            if condition():
                return True
            await asyncio.sleep(interval)
            elapsed += interval
        return False


@contextmanager
def temp_directory():
    """Create temporary directory context manager."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@contextmanager
def temp_file(content: str = "", suffix: str = ".txt"):
    """Create temporary file context manager."""
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(content)
        temp_path = Path(f.name)
    
    try:
        yield temp_path
    finally:
        temp_path.unlink(missing_ok=True)


class MockResponse:
    """Mock HTTP response."""
    
    def __init__(
        self,
        status_code: int = 200,
        json_data: Optional[Dict[str, Any]] = None,
        text: str = ""
    ):
        """
        Initialize mock response.
        
        Args:
            status_code: HTTP status code
            json_data: JSON response data
            text: Text response
        """
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
    
    def json(self) -> Dict[str, Any]:
        """Return JSON data."""
        return self._json_data
    
    def raise_for_status(self):
        """Raise exception for bad status."""
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def create_mock_client(
    responses: Optional[Dict[str, MockResponse]] = None
) -> MagicMock:
    """
    Create mock HTTP client.
    
    Args:
        responses: Dictionary of URL -> MockResponse
        
    Returns:
        Mock client
    """
    client = MagicMock()
    responses = responses or {}
    
    async def mock_get(url: str, **kwargs):
        response = responses.get(url, MockResponse())
        return response
    
    async def mock_post(url: str, **kwargs):
        response = responses.get(url, MockResponse())
        return response
    
    client.get = AsyncMock(side_effect=mock_get)
    client.post = AsyncMock(side_effect=mock_post)
    
    return client


def create_mock_task(
    task_id: str = "test-task-123",
    status: str = "pending",
    **kwargs
) -> Dict[str, Any]:
    """
    Create mock task dictionary.
    
    Args:
        task_id: Task ID
        status: Task status
        **kwargs: Additional task fields
        
    Returns:
        Task dictionary
    """
    return {
        "id": task_id,
        "status": status,
        "service_type": "enhance_image",
        "parameters": {},
        "priority": 0,
        "created_at": "2024-01-01T00:00:00",
        **kwargs
    }


class AssertionHelpers:
    """Helper assertions for tests."""
    
    @staticmethod
    def assert_dict_contains(dict1: Dict[str, Any], dict2: Dict[str, Any]):
        """
        Assert that dict1 contains all keys/values from dict2.
        
        Args:
            dict1: Dictionary to check
            dict2: Dictionary with expected keys/values
        """
        for key, value in dict2.items():
            assert key in dict1, f"Key '{key}' not found in dict1"
            assert dict1[key] == value, f"Value for '{key}' doesn't match: {dict1[key]} != {value}"
    
    @staticmethod
    def assert_approx_equal(value1: float, value2: float, tolerance: float = 0.01):
        """
        Assert that two floats are approximately equal.
        
        Args:
            value1: First value
            value2: Second value
            tolerance: Tolerance for comparison
        """
        assert abs(value1 - value2) <= tolerance, f"{value1} != {value2} (tolerance: {tolerance})"


@pytest.fixture
def temp_dir():
    """Pytest fixture for temporary directory."""
    with temp_directory() as temp_path:
        yield temp_path


@pytest.fixture
def temp_file_fixture():
    """Pytest fixture for temporary file."""
    def _create(content: str = "", suffix: str = ".txt"):
        return temp_file(content, suffix)
    return _create


@pytest.fixture
def mock_http_client():
    """Pytest fixture for mock HTTP client."""
    return create_mock_client


@pytest.fixture
def mock_task():
    """Pytest fixture for mock task."""
    return create_mock_task




