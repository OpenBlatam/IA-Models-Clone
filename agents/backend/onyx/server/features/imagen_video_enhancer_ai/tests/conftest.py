"""
Pytest Configuration and Shared Fixtures
========================================

Shared fixtures and configuration for all tests.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from pathlib import Path
import tempfile
import shutil

from ..core.enhancer_agent import EnhancerAgent
from ..config.enhancer_config import EnhancerConfig
from ..core.types import FileInfo, ProcessingOptions, TaskContext


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    config = EnhancerConfig()
    config.openrouter.api_key = "test-api-key"
    return config


@pytest.fixture
def mock_agent(sample_config):
    """Create mock enhancer agent."""
    agent = Mock(spec=EnhancerAgent)
    agent.config = sample_config
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
    agent.video_processor = Mock()
    agent.video_processor.analyze_video = AsyncMock(return_value={"fps": 30})
    return agent


@pytest.fixture
def sample_file_info():
    """Create sample file info."""
    return FileInfo(
        path="/tmp/test.jpg",
        size_bytes=1024 * 1024,  # 1MB
        mime_type="image/jpeg",
        extension=".jpg"
    )


@pytest.fixture
def sample_processing_options():
    """Create sample processing options."""
    return ProcessingOptions(
        enhancement_level="medium",
        preserve_quality=True
    )


@pytest.fixture
def sample_task_context():
    """Create sample task context."""
    return TaskContext(
        task_id="test_task_123",
        user_id="user_456",
        session_id="session_789"
    )


@pytest.fixture
def mock_openrouter_client():
    """Create mock OpenRouter client."""
    client = Mock()
    client.chat_completion = AsyncMock(return_value={
        "choices": [{
            "message": {
                "content": "Test response"
            }
        }]
    })
    client.process_image = AsyncMock(return_value={
        "analysis": "Test analysis"
    })
    return client


@pytest.fixture
def mock_truthgpt_client():
    """Create mock TruthGPT client."""
    client = Mock()
    client.process_with_truthgpt = AsyncMock(return_value={
        "optimized": True
    })
    return client


@pytest.fixture
def sample_image_path(temp_dir):
    """Create sample image file path."""
    image_path = temp_dir / "test_image.jpg"
    image_path.write_bytes(b"fake image data")
    return str(image_path)


@pytest.fixture
def sample_video_path(temp_dir):
    """Create sample video file path."""
    video_path = temp_dir / "test_video.mp4"
    video_path.write_bytes(b"fake video data")
    return str(video_path)


# Import test utilities for use in all tests
from .test_utils import TestUtils, AsyncTestMixin, MockHelpers
from .assertions import AssertionHelpers

# Make utilities available to all tests
__all__ = [
    "TestUtils",
    "AsyncTestMixin",
    "MockHelpers",
    "AssertionHelpers"
]

