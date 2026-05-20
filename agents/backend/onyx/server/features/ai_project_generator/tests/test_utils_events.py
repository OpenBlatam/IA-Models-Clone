"""
Tests for EventSystem utility
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from ..utils.event_system import EventSystem


class TestEventSystem:
    """Test suite for EventSystem"""

    def test_init(self):
        """Test EventSystem initialization"""
        event_system = EventSystem()
        assert event_system.subscribers == {}
        assert event_system.event_history == []

    def test_subscribe(self):
        """Test subscribing to events"""
        event_system = EventSystem()
        handler = Mock()
        
        event_system.subscribe("project.created", handler)
        
        assert "project.created" in event_system.subscribers
        assert handler in event_system.subscribers["project.created"]

    def test_subscribe_multiple_handlers(self):
        """Test subscribing multiple handlers to same event"""
        event_system = EventSystem()
        handler1 = Mock()
        handler2 = Mock()
        
        event_system.subscribe("project.created", handler1)
        event_system.subscribe("project.created", handler2)
        
        assert len(event_system.subscribers["project.created"]) == 2

    @pytest.mark.asyncio
    async def test_emit_sync_handler(self):
        """Test emitting event to sync handler"""
        event_system = EventSystem()
        handler = Mock()
        
        event_system.subscribe("test.event", handler)
        
        await event_system.emit("test.event", {"data": "test"})
        
        handler.assert_called_once()
        assert handler.call_args[0][0]["type"] == "test.event"

    @pytest.mark.asyncio
    async def test_emit_async_handler(self):
        """Test emitting event to async handler"""
        event_system = EventSystem()
        handler = AsyncMock()
        
        event_system.subscribe("test.event", handler)
        
        await event_system.emit("test.event", {"data": "test"})
        
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_no_subscribers(self):
        """Test emitting event with no subscribers"""
        event_system = EventSystem()
        
        # Should not crash
        await event_system.emit("test.event", {"data": "test"})
        
        # Should still be in history
        assert len(event_system.event_history) == 1

    @pytest.mark.asyncio
    async def test_emit_adds_to_history(self):
        """Test that emitted events are added to history"""
        event_system = EventSystem()
        
        await event_system.emit("test.event", {"data": "test1"})
        await event_system.emit("test.event", {"data": "test2"})
        
        assert len(event_system.event_history) == 2
        assert event_system.event_history[0]["data"]["data"] == "test1"
        assert event_system.event_history[1]["data"]["data"] == "test2"

    @pytest.mark.asyncio
    async def test_emit_handler_error(self):
        """Test that handler errors don't crash the system"""
        event_system = EventSystem()
        
        def failing_handler(event):
            raise Exception("Handler error")
        
        event_system.subscribe("test.event", failing_handler)
        
        # Should not crash
        await event_system.emit("test.event", {"data": "test"})
        
        # Event should still be in history
        assert len(event_system.event_history) == 1

    def test_get_event_history(self):
        """Test getting event history"""
        event_system = EventSystem()
        
        # Add events manually
        event_system.event_history = [
            {"type": "event1", "data": {}, "timestamp": "2024-01-01"},
            {"type": "event2", "data": {}, "timestamp": "2024-01-02"},
            {"type": "event1", "data": {}, "timestamp": "2024-01-03"},
        ]
        
        all_events = event_system.get_event_history()
        assert len(all_events) == 3
        
        filtered = event_system.get_event_history(event_type="event1")
        assert len(filtered) == 2

    def test_get_event_history_limit(self):
        """Test event history limit"""
        event_system = EventSystem()
        
        # Add many events
        for i in range(150):
            event_system.event_history.append({
                "type": "test",
                "data": {},
                "timestamp": f"2024-01-{i:02d}"
            })
        
        limited = event_system.get_event_history(limit=50)
        assert len(limited) == 50

    @pytest.mark.asyncio
    async def test_event_history_limit(self):
        """Test that event history is limited to 1000"""
        event_system = EventSystem()
        
        # Emit more than 1000 events
        for i in range(1100):
            await event_system.emit("test.event", {"index": i})
        
        # Should be limited to 1000
        assert len(event_system.event_history) == 1000
        # Should have the latest events
        assert event_system.event_history[-1]["data"]["index"] == 1099

    @pytest.mark.asyncio
    async def test_emit_multiple_handlers(self):
        """Test emitting to multiple handlers"""
        event_system = EventSystem()
        handler1 = Mock()
        handler2 = Mock()
        handler3 = AsyncMock()
        
        event_system.subscribe("test.event", handler1)
        event_system.subscribe("test.event", handler2)
        event_system.subscribe("test.event", handler3)
        
        await event_system.emit("test.event", {"data": "test"})
        
        handler1.assert_called_once()
        handler2.assert_called_once()
        handler3.assert_called_once()

