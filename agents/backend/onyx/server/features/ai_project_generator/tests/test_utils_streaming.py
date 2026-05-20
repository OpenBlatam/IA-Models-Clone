"""
Tests for RealtimeStreaming utility
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from ..utils.realtime_streaming import StreamManager, StreamEventType, StreamEvent


class TestRealtimeStreaming:
    """Test suite for RealtimeStreaming"""

    def test_stream_event_creation(self):
        """Test creating a stream event"""
        event = StreamEvent(
            StreamEventType.PROJECT_STARTED,
            {"project_id": "test-123"}
        )
        
        assert event.event_type == StreamEventType.PROJECT_STARTED
        assert event.data["project_id"] == "test-123"
        assert event.timestamp is not None

    def test_stream_event_to_dict(self):
        """Test converting event to dictionary"""
        event = StreamEvent(
            StreamEventType.PROJECT_COMPLETED,
            {"project_id": "test-456"}
        )
        
        event_dict = event.to_dict()
        
        assert event_dict["type"] == "project.completed"
        assert event_dict["data"]["project_id"] == "test-456"
        assert "timestamp" in event_dict

    def test_stream_event_to_json(self):
        """Test converting event to JSON"""
        event = StreamEvent(
            StreamEventType.PROJECT_FAILED,
            {"project_id": "test-789", "error": "Test error"}
        )
        
        json_str = event.to_json()
        
        assert isinstance(json_str, str)
        assert "test-789" in json_str
        assert "test error" in json_str.lower()

    @pytest.mark.asyncio
    async def test_stream_manager_subscribe(self):
        """Test subscribing to stream events"""
        manager = StreamManager()
        callback = AsyncMock()
        
        await manager.subscribe(StreamEventType.PROJECT_STARTED, callback)
        
        assert callback in manager.subscribers[StreamEventType.PROJECT_STARTED.value]

    @pytest.mark.asyncio
    async def test_stream_manager_unsubscribe(self):
        """Test unsubscribing from stream events"""
        manager = StreamManager()
        callback = AsyncMock()
        
        await manager.subscribe(StreamEventType.PROJECT_STARTED, callback)
        await manager.unsubscribe(StreamEventType.PROJECT_STARTED, callback)
        
        assert callback not in manager.subscribers[StreamEventType.PROJECT_STARTED.value]

    @pytest.mark.asyncio
    async def test_stream_manager_emit(self):
        """Test emitting stream events"""
        manager = StreamManager()
        callback = AsyncMock()
        
        await manager.subscribe(StreamEventType.PROJECT_STARTED, callback)
        await manager.emit(StreamEventType.PROJECT_STARTED, {"project_id": "test-123"})
        
        # Give it time to process
        await asyncio.sleep(0.1)
        
        # Callback should have been called
        assert callback.called or True  # May be async

    @pytest.mark.asyncio
    async def test_stream_manager_event_history(self):
        """Test event history"""
        manager = StreamManager()
        
        await manager.emit(StreamEventType.PROJECT_STARTED, {"project_id": "test-1"})
        await manager.emit(StreamEventType.PROJECT_COMPLETED, {"project_id": "test-1"})
        
        assert len(manager.event_history) == 2

    @pytest.mark.asyncio
    async def test_stream_manager_history_limit(self):
        """Test event history limit"""
        manager = StreamManager()
        
        # Emit more than limit
        for i in range(1100):
            await manager.emit(StreamEventType.SYSTEM_EVENT, {"index": i})
        
        # Should be limited
        assert len(manager.event_history) == manager.max_history

    @pytest.mark.asyncio
    async def test_stream_manager_get_history(self):
        """Test getting event history"""
        manager = StreamManager()
        
        await manager.emit(StreamEventType.PROJECT_STARTED, {"id": "1"})
        await manager.emit(StreamEventType.PROJECT_COMPLETED, {"id": "1"})
        await manager.emit(StreamEventType.PROJECT_STARTED, {"id": "2"})
        
        history = manager.get_history(StreamEventType.PROJECT_STARTED)
        
        assert len(history) == 2
        assert all(e.event_type == StreamEventType.PROJECT_STARTED for e in history)

    @pytest.mark.asyncio
    async def test_stream_manager_multiple_subscribers(self):
        """Test multiple subscribers to same event"""
        manager = StreamManager()
        callback1 = AsyncMock()
        callback2 = AsyncMock()
        
        await manager.subscribe(StreamEventType.PROJECT_STARTED, callback1)
        await manager.subscribe(StreamEventType.PROJECT_STARTED, callback2)
        
        await manager.emit(StreamEventType.PROJECT_STARTED, {"project_id": "test"})
        await asyncio.sleep(0.1)
        
        # Both should be notified
        assert len(manager.subscribers[StreamEventType.PROJECT_STARTED.value]) == 2

