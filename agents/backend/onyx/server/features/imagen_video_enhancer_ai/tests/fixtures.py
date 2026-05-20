"""
Test Fixtures
=============

Advanced test fixtures for pytest.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Generator, AsyncGenerator
from unittest.mock import MagicMock, AsyncMock

from ..core.enhancer_agent import EnhancerAgent
from ..infrastructure.openrouter_client import OpenRouterClient
from ..infrastructure.truthgpt_client import TruthGPTClient
from ..config.enhancer_config import EnhancerConfig


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_image_path(temp_dir: Path) -> Path:
    """Create sample image file."""
    image_path = temp_dir / "sample.jpg"
    image_path.write_bytes(b"fake image data")
    return image_path


@pytest.fixture
def sample_video_path(temp_dir: Path) -> Path:
    """Create sample video file."""
    video_path = temp_dir / "sample.mp4"
    video_path.write_bytes(b"fake video data")
    return video_path


@pytest.fixture
def mock_openrouter_client() -> MagicMock:
    """Create mock OpenRouter client."""
    mock = MagicMock(spec=OpenRouterClient)
    mock.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Test response"}}]
    })
    mock.process_image = AsyncMock(return_value="Enhanced image")
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_truthgpt_client() -> MagicMock:
    """Create mock TruthGPT client."""
    mock = MagicMock(spec=TruthGPTClient)
    mock.process_with_truthgpt = AsyncMock(return_value="Optimized response")
    mock.optimize_query = AsyncMock(return_value="Optimized query")
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def sample_config() -> EnhancerConfig:
    """Create sample configuration."""
    from ..config.enhancer_config import OpenRouterConfig, TruthGPTConfig
    
    return EnhancerConfig(
        openrouter=OpenRouterConfig(
            api_key="test_key",
            base_url="https://api.openrouter.ai/v1"
        ),
        truthgpt=TruthGPTConfig(
            api_key="test_key",
            base_url="https://api.truthgpt.com"
        ),
        max_image_size_mb=10,
        max_video_size_mb=100
    )


@pytest.fixture
async def mock_agent(
    mock_openrouter_client: MagicMock,
    mock_truthgpt_client: MagicMock,
    sample_config: EnhancerConfig,
    temp_dir: Path
) -> AsyncGenerator[MagicMock, None]:
    """Create mock agent."""
    agent = MagicMock(spec=EnhancerAgent)
    agent.config = sample_config
    agent.output_dirs = {
        "uploads": temp_dir / "uploads",
        "results": temp_dir / "results",
        "logs": temp_dir / "logs"
    }
    agent.enhance_image = AsyncMock(return_value={"task_id": "test_task"})
    agent.enhance_video = AsyncMock(return_value={"task_id": "test_task"})
    agent.upscale = AsyncMock(return_value={"task_id": "test_task"})
    agent.denoise = AsyncMock(return_value={"task_id": "test_task"})
    agent.restore = AsyncMock(return_value={"task_id": "test_task"})
    agent.color_correction = AsyncMock(return_value={"task_id": "test_task"})
    agent.get_task_status = AsyncMock(return_value={"status": "completed"})
    agent.get_task_result = AsyncMock(return_value={"result": "test_result"})
    agent.start = AsyncMock()
    agent.stop = AsyncMock()
    agent.close = AsyncMock()
    yield agent


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_task_data() -> dict:
    """Create sample task data."""
    return {
        "id": "test_task_123",
        "service_type": "enhance_image",
        "status": "pending",
        "parameters": {
            "file_path": "/tmp/test.jpg",
            "enhancement_level": "high"
        },
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00"
    }


@pytest.fixture
def sample_batch_data() -> list:
    """Create sample batch data."""
    return [
        {"file": "image1.jpg", "type": "image"},
        {"file": "image2.jpg", "type": "image"},
        {"file": "video1.mp4", "type": "video"}
    ]




