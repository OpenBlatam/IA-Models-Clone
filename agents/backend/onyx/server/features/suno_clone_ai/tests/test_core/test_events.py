"""
Comprehensive Unit Tests for Events System

Tests cover event bus functionality with diverse test cases
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from core.events import EventBus, Event, EventType


class TestEvent:
    """Test cases for Event dataclass"""
    
    def test_event_creation(self):
        """Test creating an event"""
        event = Event(
            event_type=EventType.MUSIC_GENERATED,
            data={"song_id": "song123"}
        )
        
        assert event.event_type == EventType.MUSIC_GENERATED
        assert event.data == {"song_id": "song123"}
        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)
    
    def test_event_with_source(self):
        """Test event with source"""
        event = Event(
            event_type=EventType.GENERATION_STARTED,
            data={},
            source="music_generator"
        )
        
        assert event.source == "music_generator"
    
    def test_event_timestamp_auto_generated(self):
        """Test event timestamp is auto-generated"""
        event1 = Event(EventType.MUSIC_GENERATED, {})
        event2 = Event(EventType.MUSIC_GENERATED, {})
        
        # Timestamps should be close but may differ slightly
        assert isinstance(event1.timestamp, datetime)
        assert isinstance(event2.timestamp, datetime)


class TestEventBus:
    """Test cases for EventBus class"""
    
    def test_event_bus_init(self):
        """Test initializing event bus"""
        bus = EventBus()
        assert len(bus._subscribers) == 0
        assert len(bus._event_history) == 0
        assert bus._max_history == 1000
    
    def test_subscribe_handler(self):
        """Test subscribing a handler"""
        bus = EventBus()
        handler = Mock()
        
        bus.subscribe(EventType.MUSIC_GENERATED, handler)
        
        assert EventType.MUSIC_GENERATED in bus._subscribers
        assert handler in bus._subscribers[EventType.MUSIC_GENERATED]
    
    def test_subscribe_multiple_handlers(self):
        """Test subscribing multiple handlers to same event"""
        bus = EventBus()
        handler1 = Mock()
        handler2 = Mock()
        
        bus.subscribe(EventType.MUSIC_GENERATED, handler1)
        bus.subscribe(EventType.MUSIC_GENERATED, handler2)
        
        assert len(bus._subscribers[EventType.MUSIC_GENERATED]) == 2
    
    def test_subscribe_different_event_types(self):
        """Test subscribing handlers to different event types"""
        bus = EventBus()
        handler1 = Mock()
        handler2 = Mock()
        
        bus.subscribe(EventType.MUSIC_GENERATED, handler1)
        bus.subscribe(EventType.GENERATION_STARTED, handler2)
        
        assert len(bus._subscribers) == 2
    
    def test_unsubscribe_handler(self):
        """Test unsubscribing a handler"""
        bus = EventBus()
        handler = Mock()
        
        bus.subscribe(EventType.MUSIC_GENERATED, handler)
        bus.unsubscribe(EventType.MUSIC_GENERATED, handler)
        
        assert handler not in bus._subscribers[EventType.MUSIC_GENERATED]
    
    def test_unsubscribe_nonexistent_handler(self):
        """Test unsubscribing non-existent handler doesn't raise"""
        bus = EventBus()
        handler = Mock()
        
        # Should not raise
        bus.unsubscribe(EventType.MUSIC_GENERATED, handler)
    
    @pytest.mark.asyncio
    async def test_publish_event_notifies_subscribers(self):
        """Test publishing event notifies subscribers"""
        bus = EventBus()
        handler = AsyncMock()
        
        bus.subscribe(EventType.MUSIC_GENERATED, handler)
        
        event = Event(EventType.MUSIC_GENERATED, {"song_id": "song123"})
        await bus.publish(event)
        
        handler.assert_called_once_with(event)
    
    @pytest.mark.asyncio
    async def test_publish_event_sync_handler(self):
        """Test publishing event with sync handler"""
        bus = EventBus()
        handler = Mock()
        
        bus.subscribe(EventType.MUSIC_GENERATED, handler)
        
        event = Event(EventType.MUSIC_GENERATED, {})
        await bus.publish(event)
        
        handler.assert_called_once_with(event)
    
    @pytest.mark.asyncio
    async def test_publish_event_multiple_subscribers(self):
        """Test publishing event notifies all subscribers"""
        bus = EventBus()
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        
        bus.subscribe(EventType.MUSIC_GENERATED, handler1)
        bus.subscribe(EventType.MUSIC_GENERATED, handler2)
        
        event = Event(EventType.MUSIC_GENERATED, {})
        await bus.publish(event)
        
        handler1.assert_called_once_with(event)
        handler2.assert_called_once_with(event)
    
    @pytest.mark.asyncio
    async def test_publish_event_no_subscribers(self):
        """Test publishing event with no subscribers"""
        bus = EventBus()
        event = Event(EventType.MUSIC_GENERATED, {})
        
        # Should not raise
        await bus.publish(event)
    
    @pytest.mark.asyncio
    async def test_publish_event_adds_to_history(self):
        """Test publishing event adds to history"""
        bus = EventBus()
        event = Event(EventType.MUSIC_GENERATED, {})
        
        await bus.publish(event)
        
        assert len(bus._event_history) == 1
        assert bus._event_history[0] == event
    
    @pytest.mark.asyncio
    async def test_publish_event_history_limit(self):
        """Test event history respects max limit"""
        bus = EventBus()
        bus._max_history = 5
        
        # Publish more than max
        for i in range(10):
            event = Event(EventType.MUSIC_GENERATED, {"index": i})
            await bus.publish(event)
        
        # Should only keep last 5
        assert len(bus._event_history) == 5
        assert bus._event_history[0].data["index"] == 5  # First should be 5th event
    
    @pytest.mark.asyncio
    async def test_publish_event_handler_error_handling(self):
        """Test error in handler doesn't stop other handlers"""
        bus = EventBus()
        handler1 = Mock(side_effect=Exception("Handler error"))
        handler2 = AsyncMock()
        
        bus.subscribe(EventType.MUSIC_GENERATED, handler1)
        bus.subscribe(EventType.MUSIC_GENERATED, handler2)
        
        event = Event(EventType.MUSIC_GENERATED, {})
        
        # Should not raise, handler2 should still be called
        await bus.publish(event)
        
        handler2.assert_called_once_with(event)
    
    def test_get_event_history(self):
        """Test getting event history"""
        bus = EventBus()
        event1 = Event(EventType.MUSIC_GENERATED, {})
        event2 = Event(EventType.GENERATION_STARTED, {})
        
        bus._event_history = [event1, event2]
        
        history = bus.get_event_history()
        assert len(history) == 2
        assert history[0] == event1
    
    def test_get_event_history_by_type(self):
        """Test getting event history filtered by type"""
        bus = EventBus()
        event1 = Event(EventType.MUSIC_GENERATED, {})
        event2 = Event(EventType.GENERATION_STARTED, {})
        event3 = Event(EventType.MUSIC_GENERATED, {})
        
        bus._event_history = [event1, event2, event3]
        
        history = bus.get_event_history(EventType.MUSIC_GENERATED)
        assert len(history) == 2
        assert all(e.event_type == EventType.MUSIC_GENERATED for e in history)
    
    def test_get_event_history_limit(self):
        """Test getting event history with limit"""
        bus = EventBus()
        events = [Event(EventType.MUSIC_GENERATED, {}) for _ in range(10)]
        bus._event_history = events
        
        history = bus.get_event_history(limit=5)
        assert len(history) == 5
    
    def test_clear_history(self):
        """Test clearing event history"""
        bus = EventBus()
        bus._event_history = [
            Event(EventType.MUSIC_GENERATED, {}) for _ in range(10)
        ]
        
        bus.clear_history()
        
        assert len(bus._event_history) == 0















