"""
Tests for Event System
======================
"""

import pytest
import asyncio
from ..core.event_system import EventBus, Event, EventType


@pytest.fixture
def event_bus():
    """Create event bus for testing."""
    return EventBus()


@pytest.mark.asyncio
async def test_subscribe_to_event(event_bus):
    """Test subscribing to an event."""
    events_received = []
    
    def handler(event):
        events_received.append(event)
    
    event_bus.subscribe(EventType.SESSION_CREATED, handler)
    
    # Publish event
    event = Event(
        event_type=EventType.SESSION_CREATED,
        data={"session_id": "test_session"}
    )
    await event_bus.publish(event)
    
    # Wait for async processing
    await asyncio.sleep(0.1)
    
    assert len(events_received) >= 1


@pytest.mark.asyncio
async def test_unsubscribe_from_event(event_bus):
    """Test unsubscribing from an event."""
    events_received = []
    
    def handler(event):
        events_received.append(event)
    
    subscription_id = event_bus.subscribe(EventType.SESSION_CREATED, handler)
    event_bus.unsubscribe(subscription_id)
    
    # Publish event
    event = Event(EventType.SESSION_CREATED, {"session_id": "test"})
    await event_bus.publish(event)
    
    await asyncio.sleep(0.1)
    
    # Should not receive event after unsubscribe
    assert len(events_received) == 0


@pytest.mark.asyncio
async def test_publish_event(event_bus):
    """Test publishing an event."""
    events_received = []
    
    def handler(event):
        events_received.append(event)
    
    event_bus.subscribe(EventType.SESSION_PAUSED, handler)
    
    event = Event(
        EventType.SESSION_PAUSED,
        {"session_id": "test_session", "reason": "User request"}
    )
    await event_bus.publish(event)
    
    await asyncio.sleep(0.1)
    
    assert len(events_received) >= 1
    assert events_received[0].data["session_id"] == "test_session"


@pytest.mark.asyncio
async def test_get_event_history(event_bus):
    """Test getting event history."""
    await event_bus.publish(Event(EventType.SESSION_CREATED, {"test": "data"}))
    await event_bus.publish(Event(EventType.SESSION_PAUSED, {"test": "data"}))
    
    await asyncio.sleep(0.1)
    
    history = event_bus.get_event_history(limit=10)
    
    assert len(history) >= 2


@pytest.mark.asyncio
async def test_get_event_bus_stats(event_bus):
    """Test getting event bus statistics."""
    await event_bus.publish(Event(EventType.SESSION_CREATED, {}))
    await event_bus.publish(Event(EventType.SESSION_PAUSED, {}))
    
    await asyncio.sleep(0.1)
    
    stats = event_bus.get_stats()
    
    assert stats["total_events_published"] >= 2


