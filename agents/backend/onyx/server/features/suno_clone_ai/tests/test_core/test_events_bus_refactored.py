"""
Tests refactorizados para event bus en core/events
Usando clases base y helpers para eliminar duplicación
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

from core.events.event_bus import (
    EventBus,
    Event
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestEventRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para Event"""
    
    @pytest.mark.parametrize("event_type,data,source", [
        ("test_event", {"key": "value"}, None),
        ("music_generated", {"song_id": "123"}, "generator"),
        ("audio_processed", {"audio_id": "456"}, "processor")
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
        assert event.event_id is not None
        if source:
            assert event.source == source
    
    def test_event_auto_timestamp(self):
        """Test de timestamp automático"""
        event = Event("test_event", {})
        
        assert isinstance(event.timestamp, datetime)
    
    def test_event_auto_id_unique(self):
        """Test de que los IDs son únicos"""
        events = [Event("test_event", {}) for _ in range(10)]
        ids = [event.event_id for event in events]
        
        assert len(ids) == len(set(ids))  # Todos únicos


class TestEventBusRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para EventBus"""
    
    @pytest.fixture
    def event_bus(self):
        """Fixture para EventBus"""
        return EventBus()
    
    def test_event_bus_init(self, event_bus):
        """Test de inicialización"""
        assert event_bus.subscribers == {}
    
    def test_subscribe(self, event_bus):
        """Test de suscripción"""
        handler = Mock()
        
        event_bus.subscribe("test_event", handler)
        
        assert "test_event" in event_bus.subscribers
        assert handler in event_bus.subscribers["test_event"]
    
    def test_subscribe_multiple_handlers(self, event_bus):
        """Test de múltiples handlers para el mismo evento"""
        handler1 = Mock()
        handler2 = Mock()
        
        event_bus.subscribe("test_event", handler1)
        event_bus.subscribe("test_event", handler2)
        
        assert len(event_bus.subscribers["test_event"]) == 2
    
    def test_publish(self, event_bus):
        """Test de publicación"""
        handler = Mock()
        event_bus.subscribe("test_event", handler)
        
        event_id = event_bus.publish("test_event", {"key": "value"})
        
        assert event_id is not None
        handler.assert_called_once()
    
    def test_publish_no_subscribers(self, event_bus):
        """Test de publicación sin subscribers"""
        event_id = event_bus.publish("test_event", {"key": "value"})
        
        assert event_id is not None
    
    @pytest.mark.parametrize("source", [None, "test_source", "generator", "processor"])
    def test_publish_with_source(self, event_bus, source):
        """Test de publicación con diferentes fuentes"""
        handler = Mock()
        event_bus.subscribe("test_event", handler)
        
        event_id = event_bus.publish("test_event", {"key": "value"}, source=source)
        
        assert event_id is not None
        handler.assert_called_once()
        if source:
            call_args = handler.call_args[0][0]
            assert call_args.source == source
    
    def test_unsubscribe(self, event_bus):
        """Test de desuscripción"""
        handler = Mock()
        
        event_bus.subscribe("test_event", handler)
        event_bus.unsubscribe("test_event", handler)
        
        assert handler not in event_bus.subscribers["test_event"]
    
    def test_unsubscribe_not_subscribed(self, event_bus):
        """Test de desuscripción cuando no está suscrito"""
        handler = Mock()
        
        # No debería lanzar error
        event_bus.unsubscribe("test_event", handler)
    
    @pytest.mark.parametrize("event_type", [
        "test_event",
        "music_generated",
        "audio_processed",
        "user_created"
    ])
    def test_publish_different_event_types(self, event_bus, event_type):
        """Test de publicación de diferentes tipos de eventos"""
        handler = Mock()
        event_bus.subscribe(event_type, handler)
        
        event_id = event_bus.publish(event_type, {"data": "test"})
        
        assert event_id is not None
        handler.assert_called_once()



