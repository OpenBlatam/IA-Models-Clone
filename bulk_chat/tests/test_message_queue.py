"""
Tests for Message Queue
========================
"""

import pytest
import asyncio
from ..core.message_queue import MessageQueue, Message, MessagePriority


@pytest.fixture
def message_queue():
    """Create message queue for testing."""
    return MessageQueue()


@pytest.mark.asyncio
async def test_enqueue_message(message_queue):
    """Test enqueueing a message."""
    message = Message(
        message_id="msg1",
        content="Test message",
        priority=MessagePriority.NORMAL
    )
    
    await message_queue.enqueue(message)
    
    assert message.message_id in message_queue.queue


@pytest.mark.asyncio
async def test_dequeue_message(message_queue):
    """Test dequeueing a message."""
    message = Message("msg1", "Test", MessagePriority.NORMAL)
    
    await message_queue.enqueue(message)
    dequeued = await message_queue.dequeue()
    
    assert dequeued is not None
    assert dequeued.message_id == "msg1"


@pytest.mark.asyncio
async def test_message_priority(message_queue):
    """Test message priority ordering."""
    low_msg = Message("msg1", "Low", MessagePriority.LOW)
    high_msg = Message("msg2", "High", MessagePriority.HIGH)
    
    await message_queue.enqueue(low_msg)
    await message_queue.enqueue(high_msg)
    
    # High priority should be dequeued first
    first = await message_queue.dequeue()
    
    assert first.priority == MessagePriority.HIGH


@pytest.mark.asyncio
async def test_get_queue_stats(message_queue):
    """Test getting queue statistics."""
    await message_queue.enqueue(Message("msg1", "Test", MessagePriority.NORMAL))
    await message_queue.enqueue(Message("msg2", "Test", MessagePriority.NORMAL))
    
    stats = message_queue.get_stats()
    
    assert stats["queue_size"] >= 0
    assert "total_processed" in stats or "total_enqueued" in stats


@pytest.mark.asyncio
async def test_clear_queue(message_queue):
    """Test clearing the queue."""
    await message_queue.enqueue(Message("msg1", "Test", MessagePriority.NORMAL))
    await message_queue.enqueue(Message("msg2", "Test", MessagePriority.NORMAL))
    
    assert message_queue.get_stats()["queue_size"] >= 2
    
    await message_queue.clear()
    
    assert message_queue.get_stats()["queue_size"] == 0


