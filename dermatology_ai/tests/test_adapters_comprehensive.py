"""
Tests for Adapters (Comprehensive)
Tests for all adapter implementations
"""

import pytest
from unittest.mock import Mock, AsyncMock

from core.infrastructure.adapters.database_adapter import IDatabaseAdapter
from core.infrastructure.adapters.event_publisher_adapter import EventPublisherAdapter
from core.infrastructure.adapters.fallback_adapters import (
    FallbackDatabaseAdapter,
    NoOpCacheAdapter
)
from core.domain.interfaces import IEventPublisher, ICacheService


class TestEventPublisherAdapter:
    """Tests for EventPublisherAdapter"""
    
    @pytest.fixture
    def event_publisher(self):
        """Create event publisher adapter"""
        return EventPublisherAdapter()
    
    @pytest.mark.asyncio
    async def test_publish_event(self, event_publisher):
        """Test publishing an event"""
        result = await event_publisher.publish(
            "analysis.completed",
            {"analysis_id": "test-123", "user_id": "user-123"}
        )
        
        # Should publish successfully (implementation dependent)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_publish_multiple_events(self, event_publisher):
        """Test publishing multiple events"""
        events = [
            ("analysis.started", {"id": "1"}),
            ("analysis.completed", {"id": "2"}),
            ("analysis.failed", {"id": "3"})
        ]
        
        results = []
        for event_type, data in events:
            result = await event_publisher.publish(event_type, data)
            results.append(result)
        
        assert len(results) == 3


class TestFallbackDatabaseAdapter:
    """Tests for FallbackDatabaseAdapter"""
    
    @pytest.fixture
    def fallback_adapter(self):
        """Create fallback database adapter"""
        return FallbackDatabaseAdapter()
    
    @pytest.mark.asyncio
    async def test_connect(self, fallback_adapter):
        """Test connecting with fallback adapter"""
        result = await fallback_adapter.connect()
        
        # Should always succeed (fallback)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_insert(self, fallback_adapter):
        """Test insert with fallback adapter"""
        result = await fallback_adapter.insert(
            "analyses",
            {"id": "test-123", "user_id": "user-123"}
        )
        
        # Should return ID (fallback implementation)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_get(self, fallback_adapter):
        """Test get with fallback adapter"""
        result = await fallback_adapter.get(
            "analyses",
            {"id": "test-123"}
        )
        
        # May return None or mock data (fallback)
        assert result is None or isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_query(self, fallback_adapter):
        """Test query with fallback adapter"""
        result = await fallback_adapter.query(
            "analyses",
            filter_conditions={"user_id": "user-123"},
            limit=10
        )
        
        # Should return list (may be empty in fallback)
        assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_update(self, fallback_adapter):
        """Test update with fallback adapter"""
        result = await fallback_adapter.update(
            "analyses",
            {"id": "test-123"},
            {"status": "completed"}
        )
        
        # Should return success (fallback)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete(self, fallback_adapter):
        """Test delete with fallback adapter"""
        result = await fallback_adapter.delete(
            "analyses",
            {"id": "test-123"}
        )
        
        # Should return success (fallback)
        assert result is True


class TestNoOpCacheAdapter:
    """Tests for NoOpCacheAdapter (already tested, but comprehensive)"""
    
    @pytest.fixture
    def noop_cache(self):
        """Create NoOp cache adapter"""
        return NoOpCacheAdapter()
    
    @pytest.mark.asyncio
    async def test_get_always_none(self, noop_cache):
        """Test that get always returns None"""
        result = await noop_cache.get("any-key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_set_always_succeeds(self, noop_cache):
        """Test that set always succeeds"""
        result = await noop_cache.set("key", "value", ttl=3600)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_always_succeeds(self, noop_cache):
        """Test that delete always succeeds"""
        result = await noop_cache.delete("key")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_noop_cache_operations(self, noop_cache):
        """Test all NoOp cache operations"""
        # Set multiple values
        for i in range(10):
            await noop_cache.set(f"key-{i}", f"value-{i}")
        
        # All gets should return None
        for i in range(10):
            result = await noop_cache.get(f"key-{i}")
            assert result is None
        
        # All deletes should succeed
        for i in range(10):
            result = await noop_cache.delete(f"key-{i}")
            assert result is True


class TestAdapterFactory:
    """Tests for AdapterFactory"""
    
    @pytest.mark.asyncio
    async def test_create_database_adapter(self):
        """Test creating database adapter"""
        from core.adapter_factory import AdapterFactory
        
        factory = AdapterFactory()
        
        adapter = factory.create_database_adapter("fallback")
        
        assert adapter is not None
        assert isinstance(adapter, IDatabaseAdapter)
    
    @pytest.mark.asyncio
    async def test_create_cache_adapter(self):
        """Test creating cache adapter"""
        from core.adapter_factory import AdapterFactory
        
        factory = AdapterFactory()
        
        adapter = factory.create_cache_adapter("noop")
        
        assert adapter is not None
        assert isinstance(adapter, ICacheService)
    
    @pytest.mark.asyncio
    async def test_create_event_publisher_adapter(self):
        """Test creating event publisher adapter"""
        from core.adapter_factory import AdapterFactory
        
        factory = AdapterFactory()
        
        adapter = factory.create_event_publisher_adapter()
        
        assert adapter is not None
        assert isinstance(adapter, IEventPublisher)



