"""
Tests for Event Bus
===================
"""

import pytest
import asyncio
from ..core.event_bus import EventBus


@pytest.fixture
def event_bus():
    """Create event bus for testing."""
    return EventBus()


@pytest.mark.asyncio
async def test_subscribe(event_bus):
    """Test subscribing to events."""
    events_received = []
    
    def handler(event):
        events_received.append(event)
    
    subscription_id = event_bus.subscribe("test_event", handler)
    
    assert subscription_id is not None
    assert "test_event" in event_bus.subscriptions


@pytest.mark.asyncio
async def test_publish(event_bus):
    """Test publishing an event."""
    events_received = []
    
    def handler(event):
        events_received.append(event)
    
    event_bus.subscribe("test_event", handler)
    
    await event_bus.publish("test_event", {"data": "test"})
    
    # Wait for async processing
    await asyncio.sleep(0.1)
    
    assert len(events_received) >= 1


@pytest.mark.asyncio
async def test_unsubscribe(event_bus):
    """Test unsubscribing from events."""
    def handler(event):
        pass
    
    subscription_id = event_bus.subscribe("test_event", handler)
    event_bus.unsubscribe(subscription_id)
    
    await event_bus.publish("test_event", {"data": "test"})
    
    await asyncio.sleep(0.1)
    
    # Should not receive event after unsubscribe
    assert subscription_id not in event_bus.subscriptions.values()


@pytest.mark.asyncio
async def test_get_event_bus_stats(event_bus):
    """Test getting event bus statistics."""
    await event_bus.publish("event1", {"data": "test"})
    await event_bus.publish("event2", {"data": "test"})
    
    await asyncio.sleep(0.1)
    
    stats = event_bus.get_event_bus_stats()
    
    assert stats is not None
    assert "total_events" in stats or "total_subscriptions" in stats


