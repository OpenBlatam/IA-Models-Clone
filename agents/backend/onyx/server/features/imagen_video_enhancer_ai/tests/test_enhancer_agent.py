"""
Tests for Enhancer Agent
========================
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

from imagen_video_enhancer_ai import EnhancerAgent, EnhancerConfig
from imagen_video_enhancer_ai.core.batch_processor import BatchItem
from imagen_video_enhancer_ai.core.webhook_manager import Webhook, WebhookEvent


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def config():
    """Create test configuration."""
    config = EnhancerConfig()
    config.openrouter.api_key = "test-key"
    config.truthgpt.enabled = False  # Disable TruthGPT for tests
    return config


@pytest.fixture
async def agent(config, temp_dir):
    """Create test agent."""
    agent = EnhancerAgent(
        config=config,
        output_dir=str(temp_dir / "output"),
        max_parallel_tasks=2
    )
    yield agent
    await agent.close()


@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """Test agent initialization."""
    assert agent is not None
    assert agent.task_manager is not None
    assert agent.parallel_executor is not None
    assert agent.cache_manager is not None
    assert agent.webhook_manager is not None


@pytest.mark.asyncio
async def test_enhance_image_task_creation(agent, temp_dir):
    """Test image enhancement task creation."""
    # Create a dummy image file
    test_image = temp_dir / "test.jpg"
    test_image.write_bytes(b"fake image data")
    
    task_id = await agent.enhance_image(
        file_path=str(test_image),
        enhancement_type="general",
        priority=5
    )
    
    assert task_id is not None
    assert isinstance(task_id, str)
    
    # Check task status
    status = await agent.get_task_status(task_id)
    assert status["status"] in ["pending", "processing", "completed", "failed"]


@pytest.mark.asyncio
async def test_batch_processing(agent, temp_dir):
    """Test batch processing."""
    # Create dummy files
    files = []
    for i in range(3):
        test_file = temp_dir / f"test_{i}.jpg"
        test_file.write_bytes(b"fake image data")
        files.append(str(test_file))
    
    # Create batch items
    batch_items = [
        BatchItem(
            file_path=files[0],
            service_type="enhance_image",
            enhancement_type="general"
        ),
        BatchItem(
            file_path=files[1],
            service_type="enhance_image",
            enhancement_type="sharpness"
        ),
    ]
    
    # Process batch
    result = await agent.process_batch(batch_items)
    
    assert result is not None
    assert result.total_items == 2
    assert result.completed + result.failed == result.total_items


@pytest.mark.asyncio
async def test_webhook_registration(agent):
    """Test webhook registration."""
    webhook = Webhook(
        url="https://example.com/webhook",
        events=[WebhookEvent.TASK_COMPLETED],
        secret="test-secret"
    )
    
    agent.webhook_manager.register(webhook)
    
    stats = agent.webhook_manager.get_stats()
    assert stats["registered_webhooks"] == 1


@pytest.mark.asyncio
async def test_cache_operations(agent, temp_dir):
    """Test cache operations."""
    test_file = temp_dir / "test.jpg"
    test_file.write_bytes(b"fake image data")
    
    # Set cache
    await agent.cache_manager.set(
        file_path=str(test_file),
        service_type="enhance_image",
        result={"test": "result"},
        enhancement_type="general"
    )
    
    # Get cache
    cached = await agent.cache_manager.get(
        file_path=str(test_file),
        service_type="enhance_image",
        enhancement_type="general"
    )
    
    assert cached is not None
    assert cached["test"] == "result"
    
    # Get stats
    stats = agent.cache_manager.get_stats()
    assert stats["hits"] >= 0
    assert stats["misses"] >= 0


def test_stats(agent):
    """Test statistics."""
    stats = agent.get_stats()
    
    assert "executor_stats" in stats
    assert "cache_stats" in stats
    assert "webhook_stats" in stats
    assert "running" in stats
    assert "max_parallel_tasks" in stats




