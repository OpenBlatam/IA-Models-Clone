"""
Tests for Queue Manager
========================
"""

import pytest
import asyncio
from ..core.queue_manager import QueueManager, QueuePriority


@pytest.fixture
def queue_manager():
    """Create queue manager for testing."""
    return QueueManager()


@pytest.mark.asyncio
async def test_enqueue_message(queue_manager):
    """Test enqueueing a message."""
    message_id = await queue_manager.enqueue(
        queue_name="test_queue",
        message={"content": "test message"},
        priority=QueuePriority.NORMAL
    )
    
    assert message_id is not None
    assert "test_queue" in queue_manager.queues


@pytest.mark.asyncio
async def test_dequeue_message(queue_manager):
    """Test dequeueing a message."""
    await queue_manager.enqueue("test_queue", {"content": "test"}, QueuePriority.NORMAL)
    
    message = await queue_manager.dequeue("test_queue")
    
    assert message is not None
    assert message["content"] == "test"


@pytest.mark.asyncio
async def test_message_priority(queue_manager):
    """Test message priority ordering."""
    await queue_manager.enqueue("test_queue", {"content": "low"}, QueuePriority.LOW)
    await queue_manager.enqueue("test_queue", {"content": "high"}, QueuePriority.HIGH)
    
    # High priority should be dequeued first
    first = await queue_manager.dequeue("test_queue")
    
    assert first["content"] == "high"


@pytest.mark.asyncio
async def test_get_queue_stats(queue_manager):
    """Test getting queue statistics."""
    await queue_manager.enqueue("test_queue", {"msg": "1"}, QueuePriority.NORMAL)
    await queue_manager.enqueue("test_queue", {"msg": "2"}, QueuePriority.NORMAL)
    
    stats = queue_manager.get_queue_stats("test_queue")
    
    assert stats is not None
    assert "queue_size" in stats or "total_messages" in stats


@pytest.mark.asyncio
async def test_clear_queue(queue_manager):
    """Test clearing a queue."""
    await queue_manager.enqueue("test_queue", {"msg": "1"}, QueuePriority.NORMAL)
    await queue_manager.enqueue("test_queue", {"msg": "2"}, QueuePriority.NORMAL)
    
    await queue_manager.clear_queue("test_queue")
    
    stats = queue_manager.get_queue_stats("test_queue")
    assert stats.get("queue_size", stats.get("total_messages", 0)) == 0


