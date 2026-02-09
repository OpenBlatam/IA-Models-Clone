"""
Tests refactorizados para event bus service
Usando clases base y helpers para eliminar duplicación
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
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestEventRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para Event"""
    
    @pytest.mark.parametrize("event_type,payload", [
        (EventType.MUSIC_GENERATED, {"song_id": "song-123"}),
        (EventType.AUDIO_PROCESSED, {"audio_id": "audio-456"}),
        (EventType.USER_CREATED, {"user_id": "user-789"})
    ])
    def test_event_creation(self, event_type, payload):
        """Test de creación de evento con diferentes tipos"""
        event = Event(event_type=event_type, payload=payload)
        
        assert event.event_type == event_type
        assert event.payload == payload
        assert event.timestamp is not None
        assert event.event_id is not None
        assert event.source == "suno-clone-ai"
        assert event.version == "1.0"
    
    def test_event_auto_timestamp(self):
        """Test de timestamp automático"""
        event = Event(EventType.AUDIO_PROCESSED, {})
        
        assert isinstance(event.timestamp, datetime)
    
    def test_event_auto_id_unique(self):
        """Test de que los IDs son únicos"""
        events = [Event(EventType.MUSIC_GENERATED, {}) for _ in range(10)]
        ids = [event.event_id for event in events]
        
        assert len(ids) == len(set(ids))  # Todos únicos


class TestEventBusRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para EventBus"""
    
    @pytest.fixture
    def event_bus(self):
        """Fixture para EventBus"""
        return EventBus()
    
    @pytest.fixture
    def sample_event(self):
        """Evento de ejemplo"""
        return Event(
            event_type=EventType.MUSIC_GENERATED,
            payload={"song_id": "song-123"}
        )
    
    def test_event_bus_init(self, event_bus):
        """Test de inicialización"""
        assert event_bus._subscribers == {}
        assert event_bus._event_history == []
        assert event_bus._enabled is True
    
    @pytest.mark.parametrize("event_type", [
        EventType.MUSIC_GENERATED,
        EventType.AUDIO_PROCESSED,
        EventType.USER_CREATED
    ])
    def test_subscribe(self, event_bus, event_type):
        """Test de suscripción a diferentes tipos de eventos"""
        handler = Mock()
        
        event_bus.subscribe(event_type, handler)
        
        assert event_type in event_bus._subscribers
        assert handler in event_bus._subscribers[event_type]
    
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
        asyncio.run(event_bus.publish(sample_event))
        
        history = event_bus.get_event_history()
        
        assert len(history) > 0
        assert history[-1].event_type == EventType.MUSIC_GENERATED
    
    def test_clear_history(self, event_bus, sample_event):
        """Test de limpieza de historial"""
        asyncio.run(event_bus.publish(sample_event))
        
        event_bus.clear_history()
        
        assert len(event_bus._event_history) == 0
    
    @pytest.mark.parametrize("enabled", [True, False])
    def test_enable_disable(self, event_bus, enabled):
        """Test de habilitar/deshabilitar"""
        if enabled:
            event_bus.enable()
        else:
            event_bus.disable()
        
        assert event_bus._enabled == enabled



