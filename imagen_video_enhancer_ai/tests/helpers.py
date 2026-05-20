"""
Test Helpers
============

Helper utilities for testing.
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, MagicMock

from ..core.enhancer_agent import EnhancerAgent
from ..config.enhancer_config import EnhancerConfig
from ..core.types import FileInfo, ProcessingOptions, TaskContext


class TestHelpers:
    """Helper class for test utilities."""
    
    @staticmethod
    def create_temp_file(content: bytes = b"test content", suffix: str = ".txt") -> Path:
        """
        Create a temporary file for testing.
        
        Args:
            content: File content
            suffix: File suffix
            
        Returns:
            Path to temporary file
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(content)
        temp_file.close()
        return Path(temp_file.name)
    
    @staticmethod
    def create_temp_dir() -> Path:
        """
        Create a temporary directory for testing.
        
        Returns:
            Path to temporary directory
        """
        return Path(tempfile.mkdtemp())
    
    @staticmethod
    def cleanup_temp_path(path: Path):
        """
        Clean up temporary file or directory.
        
        Args:
            path: Path to clean up
        """
        if path.exists():
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path, ignore_errors=True)
    
    @staticmethod
    def create_mock_agent(config: Optional[EnhancerConfig] = None) -> Mock:
        """
        Create a mock enhancer agent.
        
        Args:
            config: Optional configuration
            
        Returns:
            Mock agent
        """
        agent = Mock(spec=EnhancerAgent)
        agent.config = config or EnhancerConfig()
        agent.enhance_image = AsyncMock(return_value="task_123")
        agent.enhance_video = AsyncMock(return_value="task_456")
        agent.get_task_status = AsyncMock(return_value={"status": "completed"})
        agent.get_task_result = AsyncMock(return_value={"result": "enhanced"})
        agent.get_stats = Mock(return_value={"total_tasks": 10})
        agent.output_dirs = {
            "uploads": "/tmp/uploads",
            "results": "/tmp/results",
            "cache": "/tmp/cache"
        }
        return agent
    
    @staticmethod
    def create_sample_file_info() -> FileInfo:
        """Create sample file info."""
        return FileInfo(
            path="/tmp/test.jpg",
            size_bytes=1024 * 1024,  # 1MB
            mime_type="image/jpeg",
            extension=".jpg"
        )
    
    @staticmethod
    def create_sample_processing_options() -> ProcessingOptions:
        """Create sample processing options."""
        return ProcessingOptions(
            enhancement_level="medium",
            preserve_quality=True
        )
    
    @staticmethod
    def create_sample_task_context() -> TaskContext:
        """Create sample task context."""
        return TaskContext(
            task_id="test_task_123",
            user_id="user_456",
            session_id="session_789"
        )
    
    @staticmethod
    async def run_async(coro):
        """
        Run async coroutine in test.
        
        Args:
            coro: Coroutine to run
            
        Returns:
            Coroutine result
        """
        return await coro
    
    @staticmethod
    def assert_file_exists(path: Path, message: Optional[str] = None):
        """
        Assert that a file exists.
        
        Args:
            path: File path
            message: Optional assertion message
        """
        assert path.exists(), message or f"File does not exist: {path}"
    
    @staticmethod
    def assert_file_not_exists(path: Path, message: Optional[str] = None):
        """
        Assert that a file does not exist.
        
        Args:
            path: File path
            message: Optional assertion message
        """
        assert not path.exists(), message or f"File should not exist: {path}"


class AsyncTestMixin:
    """Mixin for async test utilities."""
    
    @staticmethod
    async def wait_for_condition(
        condition: callable,
        timeout: float = 5.0,
        interval: float = 0.1
    ) -> bool:
        """
        Wait for a condition to become true.
        
        Args:
            condition: Condition function
            timeout: Maximum wait time
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




