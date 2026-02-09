"""
Tests para event bus service
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from services.event_bus import (
    EventBus,
    Event,
    EventType
)


@pytest.fixture
def event_bus():
    """Fixture para EventBus"""
    return EventBus()


@pytest.fixture
def sample_event():
    """Evento de ejemplo"""
    return Event(
        event_type=EventType.MUSIC_GENERATED,
        payload={"song_id": "song-123", "user_id": "user-456"}
    )


@pytest.mark.unit
@pytest.mark.services
class TestEvent:
    """Tests para Event"""
    
    def test_event_creation(self):
        """Test de creación de evento"""
        event = Event(
            event_type=EventType.MUSIC_GENERATED,
            payload={"key": "value"}
        )
        
        assert event.event_type == EventType.MUSIC_GENERATED
        assert event.payload == {"key": "value"}
        assert event.timestamp is not None
        assert event.event_id is not None
        assert event.source == "suno-clone-ai"
        assert event.version == "1.0"
    
    def test_event_auto_timestamp(self):
        """Test de timestamp automático"""
        event = Event(
            event_type=EventType.AUDIO_PROCESSED,
            payload={}
        )
        
        assert isinstance(event.timestamp, datetime)
    
    def test_event_auto_id(self):
        """Test de ID automático"""
        event1 = Event(EventType.MUSIC_GENERATED, {})
        event2 = Event(EventType.MUSIC_GENERATED, {})
        
        assert event1.event_id != event2.event_id


@pytest.mark.unit
@pytest.mark.services
class TestEventBus:
    """Tests para EventBus"""
    
    def test_event_bus_init(self, event_bus):
        """Test de inicialización"""
        assert event_bus._subscribers == {}
        assert event_bus._event_history == []
        assert event_bus._enabled is True
    
    def test_subscribe(self, event_bus):
        """Test de suscripción a eventos"""
        handler = Mock()
        
        event_bus.subscribe(EventType.MUSIC_GENERATED, handler)
        
        assert EventType.MUSIC_GENERATED in event_bus._subscribers
        assert handler in event_bus._subscribers[EventType.MUSIC_GENERATED]
    
    def test_subscribe_multiple_handlers(self, event_bus):
        """Test de múltiples handlers para el mismo evento"""
        handler1 = Mock()
        handler2 = Mock()
        
        event_bus.subscribe(EventType.MUSIC_GENERATED, handler1)
        event_bus.subscribe(EventType.MUSIC_GENERATED, handler2)
        
        assert len(event_bus._subscribers[EventType.MUSIC_GENERATED]) == 2
    
    def test_unsubscribe(self, event_bus):
        """Test de desuscripción"""
        handler = Mock()
        
        event_bus.subscribe(EventType.MUSIC_GENERATED, handler)
        event_bus.unsubscribe(EventType.MUSIC_GENERATED, handler)
        
        assert handler not in event_bus._subscribers[EventType.MUSIC_GENERATED]
    
    @pytest.mark.asyncio
    async def test_publish_with_subscribers(self, event_bus, sample_event):
        """Test de publicación con subscribers"""
        handler = AsyncMock()
        event_bus.subscribe(EventType.MUSIC_GENERATED, handler)
        
        result = await event_bus.publish(sample_event)
        
        assert result is True
        handler.assert_called_once_with(sample_event)
    
    @pytest.mark.asyncio
    async def test_publish_no_subscribers(self, event_bus, sample_event):
        """Test de publicación sin subscribers"""
        result = await event_bus.publish(sample_event)
        
        assert result is True
        assert len(event_bus._event_history) == 1
    
    @pytest.mark.asyncio
    async def test_publish_disabled(self, event_bus, sample_event):
        """Test de publicación cuando está deshabilitado"""
        event_bus._enabled = False
        
        result = await event_bus.publish(sample_event)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_publish_multiple_handlers(self, event_bus, sample_event):
        """Test de publicación con múltiples handlers"""
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        
        event_bus.subscribe(EventType.MUSIC_GENERATED, handler1)
        event_bus.subscribe(EventType.MUSIC_GENERATED, handler2)
        
        result = await event_bus.publish(sample_event)
        
        assert result is True
        handler1.assert_called_once()
        handler2.assert_called_once()
    
    def test_get_event_history(self, event_bus, sample_event):
        """Test de obtención de historial"""
        # Publicar eventos
        asyncio.run(event_bus.publish(sample_event))
        
        history = event_bus.get_event_history()
        
        assert len(history) > 0
        assert history[-1].event_type == EventType.MUSIC_GENERATED
    
    def test_get_event_history_limit(self, event_bus):
        """Test de límite de historial"""
        # Publicar más eventos que el límite
        for i in range(1500):
            event = Event(EventType.MUSIC_GENERATED, {"index": i})
            asyncio.run(event_bus.publish(event))
        
        history = event_bus.get_event_history()
        
        assert len(history) <= event_bus._max_history
    
    def test_clear_history(self, event_bus, sample_event):
        """Test de limpieza de historial"""
        asyncio.run(event_bus.publish(sample_event))
        
        event_bus.clear_history()
        
        assert len(event_bus._event_history) == 0
    
    def test_enable_disable(self, event_bus):
        """Test de habilitar/deshabilitar"""
        assert event_bus._enabled is True
        
        event_bus.disable()
        assert event_bus._enabled is False
        
        event_bus.enable()
        assert event_bus._enabled is True



