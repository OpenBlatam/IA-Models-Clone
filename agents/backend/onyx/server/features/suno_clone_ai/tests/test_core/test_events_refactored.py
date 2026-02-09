"""
Tests refactorizados para Events System
Usando clases base y helpers para eliminar duplicación
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from core.events import EventBus, Event, EventType
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestEventRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para Event dataclass"""
    
    @pytest.mark.parametrize("event_type,data,source", [
        (EventType.MUSIC_GENERATED, {"song_id": "song123"}, None),
        (EventType.GENERATION_STARTED, {}, "music_generator"),
        (EventType.AUDIO_PROCESSED, {"audio_id": "audio456"}, "processor")
    ])
    def test_event_creation(self, event_type, data, source):
        """Test de creación de evento con diferentes configuraciones"""
        event = Event(
            event_type=event_type,
            data=data,
            source=source
        )
        
        assert event.event_type == event_type
        assert event.data == data
        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)
        if source:
            assert event.source == source
    
    def test_event_timestamp_auto_generated(self):
        """Test de que los timestamps se generan automáticamente"""
        event1 = Event(EventType.MUSIC_GENERATED, {})
        event2 = Event(EventType.MUSIC_GENERATED, {})
        
        assert isinstance(event1.timestamp, datetime)
        assert isinstance(event2.timestamp, datetime)


class TestEventBusRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para EventBus class"""
    
    @pytest.fixture
    def event_bus(self):
        """Fixture para EventBus"""
        return EventBus()
    
    def test_event_bus_init(self, event_bus):
        """Test de inicialización"""
        assert len(event_bus._subscribers) == 0
        assert len(event_bus._event_history) == 0
        assert event_bus._max_history == 1000
    
    def test_subscribe_handler(self, event_bus):
        """Test de suscripción de handler"""
        handler = Mock()
        
        event_bus.subscribe(EventType.MUSIC_GENERATED, handler)
        
        assert EventType.MUSIC_GENERATED in event_bus._subscribers
        assert handler in event_bus._subscribers[EventType.MUSIC_GENERATED]
    
    @pytest.mark.parametrize("num_handlers", [1, 2, 3, 5])
    def test_subscribe_multiple_handlers(self, event_bus, num_handlers):
        """Test de suscripción de múltiples handlers"""
        handlers = [Mock() for _ in range(num_handlers)]
        
        for handler in handlers:
            event_bus.subscribe(EventType.MUSIC_GENERATED, handler)
        
        assert len(event_bus._subscribers[EventType.MUSIC_GENERATED]) == num_handlers
    
    def test_subscribe_different_event_types(self, event_bus):
        """Test de suscripción a diferentes tipos de eventos"""
        handler1 = Mock()
        handler2 = Mock()
        
        event_bus.subscribe(EventType.MUSIC_GENERATED, handler1)
        event_bus.subscribe(EventType.GENERATION_STARTED, handler2)
        
        assert len(event_bus._subscribers) == 2
    
    def test_unsubscribe_handler(self, event_bus):
        """Test de desuscripción de handler"""
        handler = Mock()
        
        event_bus.subscribe(EventType.MUSIC_GENERATED, handler)
        event_bus.unsubscribe(EventType.MUSIC_GENERATED, handler)
        
        assert handler not in event_bus._subscribers[EventType.MUSIC_GENERATED]
    
    @pytest.mark.asyncio
    async def test_publish_event(self, event_bus):
        """Test de publicación de evento"""
        handler = AsyncMock()
        event_bus.subscribe(EventType.MUSIC_GENERATED, handler)
        
        event = Event(
            event_type=EventType.MUSIC_GENERATED,
            data={"song_id": "song123"}
        )
        
        await event_bus.publish(event)
        
        handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_publish_event_no_subscribers(self, event_bus):
        """Test de publicación sin subscribers"""
        event = Event(
            event_type=EventType.MUSIC_GENERATED,
            data={"song_id": "song123"}
        )
        
        await event_bus.publish(event)
        
        assert len(event_bus._event_history) == 1
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("event_type", [
        EventType.MUSIC_GENERATED,
        EventType.GENERATION_STARTED,
        EventType.AUDIO_PROCESSED
    ])
    async def test_publish_different_event_types(self, event_bus, event_type):
        """Test de publicación de diferentes tipos de eventos"""
        handler = AsyncMock()
        event_bus.subscribe(event_type, handler)
        
        event = Event(event_type=event_type, data={})
        await event_bus.publish(event)
        
        handler.assert_called_once()



