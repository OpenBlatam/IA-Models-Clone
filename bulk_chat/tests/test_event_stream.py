"""
Tests for Event Stream
=======================
"""

import pytest
import asyncio
from ..core.event_stream import EventStream, EventType


@pytest.fixture
def event_stream():
    """Create event stream for testing."""
    return EventStream()


@pytest.mark.asyncio
async def test_publish_event(event_stream):
    """Test publishing an event to stream."""
    event_id = await event_stream.publish(
        event_type=EventType.SESSION_CREATED,
        data={"session_id": "test_session"},
        metadata={"source": "test"}
    )
    
    assert event_id is not None
    assert event_id in event_stream.events


@pytest.mark.asyncio
async def test_subscribe_to_stream(event_stream):
    """Test subscribing to event stream."""
    events_received = []
    
    def handler(event):
        events_received.append(event)
    
    subscription_id = event_stream.subscribe(
        event_types=[EventType.SESSION_CREATED],
        handler=handler
    )
    
    assert subscription_id is not None
    
    # Publish event
    await event_stream.publish(EventType.SESSION_CREATED, {"test": "data"})
    
    await asyncio.sleep(0.1)
    
    assert len(events_received) >= 1


@pytest.mark.asyncio
async def test_filter_events(event_stream):
    """Test filtering events."""
    await event_stream.publish(EventType.SESSION_CREATED, {"session_id": "session1"})
    await event_stream.publish(EventType.SESSION_PAUSED, {"session_id": "session2"})
    await event_stream.publish(EventType.SESSION_CREATED, {"session_id": "session3"})
    
    filtered = event_stream.filter_events(
        event_types=[EventType.SESSION_CREATED],
        limit=10
    )
    
    assert len(filtered) >= 2
    assert all(e.event_type == EventType.SESSION_CREATED for e in filtered)


@pytest.mark.asyncio
async def test_get_event_stream_stats(event_stream):
    """Test getting event stream statistics."""
    await event_stream.publish(EventType.SESSION_CREATED, {})
    await event_stream.publish(EventType.SESSION_PAUSED, {})
    
    stats = event_stream.get_event_stream_stats()
    
    assert stats is not None
    assert "total_events" in stats or "events_by_type" in stats


@pytest.mark.asyncio
async def test_get_event_stream_summary(event_stream):
    """Test getting event stream summary."""
    await event_stream.publish(EventType.SESSION_CREATED, {})
    
    summary = event_stream.get_event_stream_summary()
    
    assert summary is not None
    assert "total_events" in summary or "total_subscriptions" in summary


