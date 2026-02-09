"""
Test Utilities
==============

Advanced testing utilities and helpers.
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock, AsyncMock, MagicMock
from contextlib import contextmanager
import pytest


class TestUtils:
    """Test utility functions."""
    
    @staticmethod
    @contextmanager
    def temp_directory():
        """Create temporary directory context."""
        temp_dir = tempfile.mkdtemp()
        try:
            yield Path(temp_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @staticmethod
    @contextmanager
    def temp_file(content: str = "", suffix: str = ".txt"):
        """Create temporary file context."""
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)
        try:
            yield temp_path
        finally:
            temp_path.unlink(missing_ok=True)
    
    @staticmethod
    def create_mock_client(**kwargs) -> Mock:
        """
        Create mock HTTP client.
        
        Args:
            **kwargs: Mock attributes
            
        Returns:
            Mock client
        """
        mock = MagicMock()
        for key, value in kwargs.items():
            setattr(mock, key, value)
        return mock
    
    @staticmethod
    def create_async_mock(**kwargs) -> AsyncMock:
        """
        Create async mock.
        
        Args:
            **kwargs: Mock attributes
            
        Returns:
            Async mock
        """
        mock = AsyncMock()
        for key, value in kwargs.items():
            setattr(mock, key, value)
        return mock
    
    @staticmethod
    def create_mock_task(
        task_id: str = "test_task",
        status: str = "pending",
        service_type: str = "enhance_image",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create mock task dictionary.
        
        Args:
            task_id: Task ID
            status: Task status
            service_type: Service type
            **kwargs: Additional task fields
            
        Returns:
            Mock task dictionary
        """
        task = {
            "id": task_id,
            "status": status,
            "service_type": service_type,
            "parameters": {},
            "created_at": "2024-01-01T00:00:00",
            **kwargs
        }
        return task
    
    @staticmethod
    async def wait_for_condition(
        condition: Callable[[], bool],
        timeout: float = 5.0,
        interval: float = 0.1
    ) -> bool:
        """
        Wait for condition to be true.
        
        Args:
            condition: Condition function
            timeout: Timeout in seconds
            interval: Check interval in seconds
            
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
    
    @staticmethod
    def assert_dict_contains(actual: Dict[str, Any], expected: Dict[str, Any]):
        """
        Assert dictionary contains expected keys and values.
        
        Args:
            actual: Actual dictionary
            expected: Expected dictionary
        """
        for key, value in expected.items():
            assert key in actual, f"Key {key} not found in actual dict"
            assert actual[key] == value, f"Value for {key} mismatch: {actual[key]} != {value}"
    
    @staticmethod
    def assert_list_contains(items: List[Any], item: Any):
        """
        Assert list contains item.
        
        Args:
            items: List to check
            item: Item to find
        """
        assert item in items, f"Item {item} not found in list"
    
    @staticmethod
    def assert_file_exists(file_path: Path):
        """
        Assert file exists.
        
        Args:
            file_path: File path
        """
        assert file_path.exists(), f"File {file_path} does not exist"
        assert file_path.is_file(), f"{file_path} is not a file"
    
    @staticmethod
    def assert_directory_exists(dir_path: Path):
        """
        Assert directory exists.
        
        Args:
            dir_path: Directory path
        """
        assert dir_path.exists(), f"Directory {dir_path} does not exist"
        assert dir_path.is_dir(), f"{dir_path} is not a directory"


class AsyncTestMixin:
    """Mixin for async tests."""
    
    @pytest.fixture
    def event_loop(self):
        """Create event loop for async tests."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()
    
    async def run_async(self, coro):
        """Run async coroutine."""
        return await coro


class MockHelpers:
    """Helper functions for creating mocks."""
    
    @staticmethod
    def mock_openrouter_response(content: str = "Test response", model: str = "gpt-4"):
        """Create mock OpenRouter response."""
        return {
            "id": "test_id",
            "model": model,
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
    
    @staticmethod
    def mock_task_manager():
        """Create mock task manager."""
        mock = MagicMock()
        mock.create_task = AsyncMock(return_value={"id": "test_task"})
        mock.get_task = AsyncMock(return_value={"id": "test_task", "status": "pending"})
        mock.update_task_status = AsyncMock()
        mock.get_pending_tasks = AsyncMock(return_value=[])
        return mock
    
    @staticmethod
    def mock_agent():
        """Create mock agent."""
        mock = MagicMock()
        mock.enhance_image = AsyncMock(return_value={"task_id": "test_task"})
        mock.enhance_video = AsyncMock(return_value={"task_id": "test_task"})
        mock.get_task_status = AsyncMock(return_value={"status": "completed"})
        return mock




