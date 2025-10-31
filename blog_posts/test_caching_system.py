from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import pytest
import asyncio
import time
import json
from typing import Dict, List, Tuple, Any
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import tempfile
import os
from caching_system import (
from typing import Any, List, Dict, Optional
import logging
"""
🧪 COMPREHENSIVE CACHING SYSTEM TESTS
====================================

Test suite for the comprehensive caching system covering:
- In-memory cache functionality
- Redis cache operations
- Multi-tier caching
- Cache decorators
- Cache invalidation
- Cache warming
- Cache monitoring
- Performance testing
- Error handling
- Integration scenarios

Features:
- Unit tests for each cache component
- Integration tests for cache interactions
- Performance benchmarking
- Error scenario testing
- Mock Redis testing
- Async/await testing
- Configuration testing
"""


    CacheConfig, CacheStrategy, CacheTier, CacheKeyGenerator,
    CacheSerializer, InMemoryCache, RedisCache, MultiTierCache,
    CacheWarmer, CacheMonitor, CacheManager, cached, cache_invalidate,
    create_cache_config, create_cache_manager
)

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def cache_config():
    """Create cache configuration for testing."""
    return CacheConfig(
        redis_url="redis://localhost:6379",
        memory_cache_size=100,
        memory_cache_ttl=60,
        enable_multi_tier=True,
        enable_monitoring=False,
        enable_cache_warming=False
    )

@pytest.fixture
def memory_cache_config():
    """Create memory-only cache configuration."""
    return CacheConfig(
        redis_url=None,
        memory_cache_size=50,
        memory_cache_ttl=30,
        enable_multi_tier=False,
        enable_monitoring=False
    )

@pytest.fixture
def test_data():
    """Create test data for caching."""
    return {
        "string": "test_string",
        "number": 42,
        "list": [1, 2, 3, 4, 5],
        "dict": {"key": "value", "nested": {"data": "test"}},
        "complex": {
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"}
            ],
            "metadata": {
                "created": "2024-01-15T10:30:00Z",
                "version": "1.0.0"
            }
        }
    }

# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestCacheConfig:
    """Test cache configuration."""
    
    def test_default_config(self) -> Any:
        """Test default configuration values."""
        config = CacheConfig()
        
        assert config.redis_url is None
        assert config.redis_ttl == 3600
        assert config.memory_cache_size == 1000
        assert config.memory_cache_ttl == 300
        assert config.memory_cache_strategy == CacheStrategy.TTL
        assert config.enable_multi_tier is True
        assert config.cache_tier == CacheTier.BOTH
        assert config.key_prefix == "cache"
        assert config.enable_compression is True
        assert config.enable_monitoring is True
    
    def test_custom_config(self) -> Any:
        """Test custom configuration values."""
        config = CacheConfig(
            redis_url="redis://test:6379",
            memory_cache_size=500,
            memory_cache_strategy=CacheStrategy.LRU,
            enable_multi_tier=False,
            cache_tier=CacheTier.L1
        )
        
        assert config.redis_url == "redis://test:6379"
        assert config.memory_cache_size == 500
        assert config.memory_cache_strategy == CacheStrategy.LRU
        assert config.enable_multi_tier is False
        assert config.cache_tier == CacheTier.L1
    
    def test_config_validation(self) -> Any:
        """Test configuration validation."""
        # Should not raise validation error
        config = CacheConfig()
        config.memory_cache_size = 200
        assert config.memory_cache_size == 200

# ============================================================================
# CACHE KEY GENERATOR TESTS
# ============================================================================

class TestCacheKeyGenerator:
    """Test cache key generation."""
    
    def test_generate_key(self) -> Any:
        """Test basic key generation."""
        generator = CacheKeyGenerator(prefix="test", separator=":")
        
        key = generator.generate_key("user", 123)
        assert key == "test:user:123"
        
        key = generator.generate_key("user", 123, active=True)
        assert key == "test:user:123:active=True"
    
    def test_generate_key_with_kwargs(self) -> Any:
        """Test key generation with keyword arguments."""
        generator = CacheKeyGenerator()
        
        key = generator.generate_key("user", user_id=123, active=True)
        assert key == "cache:user:active=True:user_id=123"
        
        # Should be consistent regardless of order
        key2 = generator.generate_key("user", active=True, user_id=123)
        assert key == key2
    
    def test_generate_hash_key(self) -> Any:
        """Test hash-based key generation."""
        generator = CacheKeyGenerator()
        
        key1 = generator.generate_hash_key("user", 123)
        key2 = generator.generate_hash_key("user", 123)
        
        assert key1 == key2
        assert len(key1) == 32  # MD5 hash length
        assert key1.isalnum()  # Should be alphanumeric
    
    def test_generate_pattern_key(self) -> Any:
        """Test pattern-based key generation."""
        generator = CacheKeyGenerator()
        
        key = generator.generate_pattern_key("user:*", user_id=123)
        assert key == "cache:user:*:user_id=123"

# ============================================================================
# SERIALIZATION TESTS
# ============================================================================

class TestCacheSerializer:
    """Test cache serialization."""
    
    def test_serialize_deserialize(self, test_data) -> Any:
        """Test basic serialization and deserialization."""
        serializer = CacheSerializer()
        
        for key, value in test_data.items():
            serialized = serializer.serialize(value)
            deserialized = serializer.deserialize(serialized)
            
            assert deserialized == value
    
    def test_serialize_none(self) -> Any:
        """Test serialization of None values."""
        serializer = CacheSerializer()
        
        serialized = serializer.serialize(None)
        deserialized = serializer.deserialize(serialized)
        
        assert deserialized is None
    
    def test_compression(self) -> Any:
        """Test data compression."""
        serializer = CacheSerializer(enable_compression=True, compression_threshold=10)
        
        # Small data should not be compressed
        small_data = "small"
        serialized = serializer.serialize(small_data)
        assert not serialized.startswith(b"gzip:")
        
        # Large data should be compressed
        large_data = "x" * 1000
        serialized = serializer.serialize(large_data)
        assert serialized.startswith(b"gzip:")
        
        # Should deserialize correctly
        deserialized = serializer.deserialize(serialized)
        assert deserialized == large_data
    
    def test_compression_disabled(self) -> Any:
        """Test with compression disabled."""
        serializer = CacheSerializer(enable_compression=False)
        
        large_data = "x" * 1000
        serialized = serializer.serialize(large_data)
        assert not serialized.startswith(b"gzip:")
        
        deserialized = serializer.deserialize(serialized)
        assert deserialized == large_data

# ============================================================================
# IN-MEMORY CACHE TESTS
# ============================================================================

class TestInMemoryCache:
    """Test in-memory cache functionality."""
    
    def test_initialization(self, memory_cache_config) -> Any:
        """Test cache initialization."""
        cache = InMemoryCache(memory_cache_config)
        
        assert cache.config == memory_cache_config
        assert cache.cache is not None
        assert cache.serializer is not None
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0
    
    def test_get_set(self, memory_cache_config, test_data) -> Optional[Dict[str, Any]]:
        """Test basic get and set operations."""
        cache = InMemoryCache(memory_cache_config)
        
        # Test setting and getting data
        for key, value in test_data.items():
            success = cache.set(key, value)
            assert success is True
            
            retrieved = cache.get(key)
            assert retrieved == value
        
        # Test cache miss
        assert cache.get("nonexistent") is None
    
    def test_delete(self, memory_cache_config, test_data) -> Any:
        """Test delete operations."""
        cache = InMemoryCache(memory_cache_config)
        
        # Set some data
        cache.set("test_key", test_data["string"])
        assert cache.get("test_key") == test_data["string"]
        
        # Delete the data
        success = cache.delete("test_key")
        assert success is True
        
        # Should not exist anymore
        assert cache.get("test_key") is None
        
        # Delete non-existent key
        success = cache.delete("nonexistent")
        assert success is False
    
    def test_clear(self, memory_cache_config, test_data) -> Any:
        """Test cache clearing."""
        cache = InMemoryCache(memory_cache_config)
        
        # Add some data
        for key, value in test_data.items():
            cache.set(key, value)
        
        # Clear cache
        success = cache.clear()
        assert success is True
        
        # All data should be gone
        for key in test_data.keys():
            assert cache.get(key) is None
    
    def test_stats(self, memory_cache_config) -> Any:
        """Test cache statistics."""
        cache = InMemoryCache(memory_cache_config)
        
        # Initial stats
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["sets"] == 0
        assert stats["hit_rate"] == 0
        
        # Add some operations
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        cache.delete("key1")
        
        # Check updated stats
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["sets"] == 1
        assert stats["deletes"] == 1
        assert stats["hit_rate"] == 0.5
    
    def test_ttl_expiration(self) -> Any:
        """Test TTL-based expiration."""
        config = CacheConfig(
            memory_cache_strategy=CacheStrategy.TTL,
            memory_cache_ttl=1  # 1 second TTL
        )
        cache = InMemoryCache(config)
        
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        
        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("test_key") is None

# ============================================================================
# REDIS CACHE TESTS
# ============================================================================

class TestRedisCache:
    """Test Redis cache functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, cache_config) -> Any:
        """Test Redis cache initialization."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            cache = RedisCache(cache_config)
            
            assert cache.config == cache_config
            assert cache.redis_client is not None
            assert cache.serializer is not None
    
    @pytest.mark.asyncio
    async def test_initialization_no_redis(self) -> Any:
        """Test initialization without Redis."""
        config = CacheConfig(redis_url=None)
        cache = RedisCache(config)
        
        assert cache.redis_client is None
    
    @pytest.mark.asyncio
    async def test_get_set(self, cache_config, test_data) -> Optional[Dict[str, Any]]:
        """Test Redis get and set operations."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            
            cache = RedisCache(cache_config)
            
            # Mock Redis responses
            mock_client.get.return_value = cache.serializer.serialize(test_data["string"])
            mock_client.setex.return_value = True
            
            # Test set
            success = await cache.set("test_key", test_data["string"])
            assert success is True
            mock_client.setex.assert_called_once()
            
            # Test get
            result = await cache.get("test_key")
            assert result == test_data["string"]
            mock_client.get.assert_called_once_with("test_key")
    
    @pytest.mark.asyncio
    async def test_get_miss(self, cache_config) -> Optional[Dict[str, Any]]:
        """Test Redis cache miss."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            
            cache = RedisCache(cache_config)
            
            # Mock cache miss
            mock_client.get.return_value = None
            
            result = await cache.get("nonexistent")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_delete(self, cache_config) -> Any:
        """Test Redis delete operations."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            
            cache = RedisCache(cache_config)
            
            # Mock successful delete
            mock_client.delete.return_value = 1
            
            success = await cache.delete("test_key")
            assert success is True
            mock_client.delete.assert_called_once_with("test_key")
            
            # Mock unsuccessful delete
            mock_client.delete.return_value = 0
            
            success = await cache.delete("nonexistent")
            assert success is False
    
    @pytest.mark.asyncio
    async def test_clear_pattern(self, cache_config) -> Any:
        """Test Redis pattern clearing."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            
            cache = RedisCache(cache_config)
            
            # Mock keys and delete
            mock_client.keys.return_value = [b"key1", b"key2", b"key3"]
            mock_client.delete.return_value = 3
            
            deleted = await cache.clear_pattern("test:*")
            assert deleted == 3
            mock_client.keys.assert_called_once_with("test:*")
            mock_client.delete.assert_called_once_with(b"key1", b"key2", b"key3")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, cache_config) -> Any:
        """Test Redis error handling."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            
            cache = RedisCache(cache_config)
            
            # Mock Redis error
            mock_client.get.side_effect = Exception("Redis error")
            
            result = await cache.get("test_key")
            assert result is None
            assert cache.stats["errors"] == 1

# ============================================================================
# MULTI-TIER CACHE TESTS
# ============================================================================

class TestMultiTierCache:
    """Test multi-tier cache functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, cache_config) -> Any:
        """Test multi-tier cache initialization."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            cache = MultiTierCache(cache_config)
            
            assert cache.config == cache_config
            assert cache.l1_cache is not None
            assert cache.l2_cache is not None
            assert cache.key_generator is not None
    
    @pytest.mark.asyncio
    async def test_get_l1_hit(self, cache_config, test_data) -> Optional[Dict[str, Any]]:
        """Test L1 cache hit."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            cache = MultiTierCache(cache_config)
            
            # Set in L1 cache
            cache.l1_cache.set("test_key", test_data["string"])
            
            # Get from cache
            result = await cache.get("test_key", CacheTier.BOTH)
            assert result == test_data["string"]
    
    @pytest.mark.asyncio
    async def test_get_l2_hit(self, cache_config, test_data) -> Optional[Dict[str, Any]]:
        """Test L2 cache hit with L1 population."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            
            cache = MultiTierCache(cache_config)
            
            # Mock L2 cache hit
            mock_client.get.return_value = cache.l2_cache.serializer.serialize(test_data["string"])
            mock_client.setex.return_value = True
            
            # Get from cache (should hit L2 and populate L1)
            result = await cache.get("test_key", CacheTier.BOTH)
            assert result == test_data["string"]
            
            # Should now be in L1 cache
            l1_result = cache.l1_cache.get("test_key")
            assert l1_result == test_data["string"]
    
    @pytest.mark.asyncio
    async def test_get_l2_only(self, cache_config, test_data) -> Optional[Dict[str, Any]]:
        """Test L2-only cache access."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            
            cache = MultiTierCache(cache_config)
            
            # Mock L2 cache hit
            mock_client.get.return_value = cache.l2_cache.serializer.serialize(test_data["string"])
            
            # Get from L2 only
            result = await cache.get("test_key", CacheTier.L2)
            assert result == test_data["string"]
            
            # Should not be in L1 cache
            l1_result = cache.l1_cache.get("test_key")
            assert l1_result is None
    
    @pytest.mark.asyncio
    async def test_set_both_tiers(self, cache_config, test_data) -> Any:
        """Test setting in both cache tiers."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            
            cache = MultiTierCache(cache_config)
            
            # Mock L2 cache set
            mock_client.setex.return_value = True
            
            # Set in both tiers
            success = await cache.set("test_key", test_data["string"], tier=CacheTier.BOTH)
            assert success is True
            
            # Should be in L1 cache
            l1_result = cache.l1_cache.get("test_key")
            assert l1_result == test_data["string"]
            
            # Should be in L2 cache
            mock_client.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_both_tiers(self, cache_config) -> Any:
        """Test deleting from both cache tiers."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            
            cache = MultiTierCache(cache_config)
            
            # Add to L1 cache
            cache.l1_cache.set("test_key", "test_value")
            
            # Mock L2 cache delete
            mock_client.delete.return_value = 1
            
            # Delete from both tiers
            success = await cache.delete("test_key", tier=CacheTier.BOTH)
            assert success is True
            
            # Should not be in L1 cache
            l1_result = cache.l1_cache.get("test_key")
            assert l1_result is None
            
            # Should be deleted from L2 cache
            mock_client.delete.assert_called_once_with("test_key")
    
    @pytest.mark.asyncio
    async def test_clear_pattern(self, cache_config) -> Any:
        """Test clearing cache patterns."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            
            cache = MultiTierCache(cache_config)
            
            # Add some data to L1 cache
            cache.l1_cache.set("user:1", "data1")
            cache.l1_cache.set("user:2", "data2")
            
            # Mock L2 cache pattern clear
            mock_client.keys.return_value = [b"user:1", b"user:2"]
            mock_client.delete.return_value = 2
            
            # Clear pattern
            deleted = await cache.clear_pattern("user:*")
            assert deleted == 3  # 2 from L2 + 1 from L1 clear

# ============================================================================
# CACHE DECORATOR TESTS
# ============================================================================

class TestCacheDecorators:
    """Test cache decorators."""
    
    @pytest.mark.asyncio
    async def test_cached_decorator(self, cache_config) -> Any:
        """Test cached decorator functionality."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            cache = MultiTierCache(cache_config)
            
            call_count = 0
            
            @cached(ttl=3600, key_prefix="test", cache_instance=cache)
            async def expensive_function(user_id: int, active: bool = True):
                
    """expensive_function function."""
nonlocal call_count
                call_count += 1
                return {"user_id": user_id, "active": active, "data": "expensive_result"}
            
            # First call should execute function
            result1 = await expensive_function(123, active=True)
            assert result1["user_id"] == 123
            assert call_count == 1
            
            # Second call should use cache
            result2 = await expensive_function(123, active=True)
            assert result2["user_id"] == 123
            assert call_count == 1  # Should not increment
            
            # Different parameters should execute function
            result3 = await expensive_function(123, active=False)
            assert result3["active"] is False
            assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_cache_invalidate_decorator(self, cache_config) -> bool:
        """Test cache invalidation decorator."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            cache = MultiTierCache(cache_config)
            
            # Add some data to cache
            await cache.set("user:123", "user_data")
            await cache.set("user:456", "user_data")
            
            @cache_invalidate("user:*", cache_instance=cache)
            async def update_user(user_id: int, data: dict):
                
    """update_user function."""
return {"user_id": user_id, **data}
            
            # Execute function
            result = await update_user(123, {"name": "Updated"})
            assert result["user_id"] == 123
            
            # Cache should be cleared
            user_data = await cache.get("user:123")
            assert user_data is None

# ============================================================================
# CACHE WARMER TESTS
# ============================================================================

class TestCacheWarmer:
    """Test cache warming functionality."""
    
    @pytest.mark.asyncio
    async def test_warm_cache(self, cache_config) -> Any:
        """Test cache warming."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            cache = MultiTierCache(cache_config)
            warmer = CacheWarmer(cache)
            
            # Mock data source
            async def data_source():
                
    """data_source function."""
return [
                    ("user:1", {"id": 1, "name": "Alice"}),
                    ("user:2", {"id": 2, "name": "Bob"}),
                    ("user:3", {"id": 3, "name": "Charlie"})
                ]
            
            # Warm cache
            await warmer.warm_cache(data_source, batch_size=2)
            
            # Check that data is in cache
            user1 = await cache.get("user:1")
            assert user1["name"] == "Alice"
            
            user2 = await cache.get("user:2")
            assert user2["name"] == "Bob"
            
            user3 = await cache.get("user:3")
            assert user3["name"] == "Charlie"

# ============================================================================
# CACHE MONITOR TESTS
# ============================================================================

class TestCacheMonitor:
    """Test cache monitoring functionality."""
    
    @pytest.mark.asyncio
    async def test_monitor_start_stop(self, cache_config) -> Any:
        """Test monitor start and stop."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            cache = MultiTierCache(cache_config)
            monitor = CacheMonitor(cache, interval=1)
            
            # Start monitoring
            await monitor.start_monitoring()
            assert monitor.monitoring_task is not None
            
            # Stop monitoring
            await monitor.stop_monitoring()
            assert monitor.monitoring_task is None
    
    @pytest.mark.asyncio
    async def test_monitor_loop(self, cache_config) -> Any:
        """Test monitoring loop."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            cache = MultiTierCache(cache_config)
            monitor = CacheMonitor(cache, interval=0.1)  # Short interval for testing
            
            # Start monitoring
            await monitor.start_monitoring()
            
            # Wait a bit for monitoring to run
            await asyncio.sleep(0.2)
            
            # Stop monitoring
            await monitor.stop_monitoring()

# ============================================================================
# CACHE MANAGER TESTS
# ============================================================================

class TestCacheManager:
    """Test cache manager functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, cache_config) -> Any:
        """Test cache manager initialization."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            manager = CacheManager(cache_config)
            
            assert manager.config == cache_config
            assert manager.cache is not None
            assert manager.key_generator is not None
    
    @pytest.mark.asyncio
    async def test_start_stop(self, cache_config) -> Any:
        """Test cache manager start and stop."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            manager = CacheManager(cache_config)
            
            # Start manager
            await manager.start()
            
            # Stop manager
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_basic_operations(self, cache_config, test_data) -> Any:
        """Test basic cache operations through manager."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            manager = CacheManager(cache_config)
            
            # Set data
            success = await manager.set("test_key", test_data["string"])
            assert success is True
            
            # Get data
            result = await manager.get("test_key")
            assert result == test_data["string"]
            
            # Delete data
            success = await manager.delete("test_key")
            assert success is True
            
            # Should not exist
            result = await manager.get("test_key")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_manager_decorators(self, cache_config) -> Any:
        """Test cache manager decorators."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            manager = CacheManager(cache_config)
            
            call_count = 0
            
            @manager.cached(ttl=3600, key_prefix="manager_test")
            async def test_function(param: str):
                
    """test_function function."""
nonlocal call_count
                call_count += 1
                return f"result_{param}"
            
            # First call
            result1 = await test_function("test")
            assert result1 == "result_test"
            assert call_count == 1
            
            # Second call (cached)
            result2 = await test_function("test")
            assert result2 == "result_test"
            assert call_count == 1  # Should not increment

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCacheIntegration:
    """Test cache integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, cache_config) -> Any:
        """Test complete caching workflow."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            
            # Create cache manager
            manager = create_cache_manager(cache_config)
            await manager.start()
            
            # Simulate expensive operation
            call_count = 0
            
            @manager.cached(ttl=3600, key_prefix="workflow")
            async def expensive_operation(user_id: int, operation: str):
                
    """expensive_operation function."""
nonlocal call_count
                call_count += 1
                await asyncio.sleep(0.1)  # Simulate work
                return {"user_id": user_id, "operation": operation, "result": "success"}
            
            # First call
            result1 = await expensive_operation(123, "read")
            assert result1["user_id"] == 123
            assert call_count == 1
            
            # Second call (should use cache)
            result2 = await expensive_operation(123, "read")
            assert result2["user_id"] == 123
            assert call_count == 1  # Should not increment
            
            # Invalidate cache
            @manager.cache_invalidate("workflow:*")
            async def update_user(user_id: int):
                
    """update_user function."""
return {"user_id": user_id, "updated": True}
            
            await update_user(123)
            
            # Third call (should execute again)
            result3 = await expensive_operation(123, "read")
            assert result3["user_id"] == 123
            assert call_count == 2  # Should increment
            
            # Get cache statistics
            stats = await manager.get_stats()
            assert "l1_cache" in stats
            assert "l2_cache" in stats
            
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_multi_user_scenario(self, cache_config) -> Any:
        """Test multi-user caching scenario."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            
            manager = create_cache_manager(cache_config)
            await manager.start()
            
            # Simulate user data operations
            @manager.cached(ttl=1800, key_prefix="user_profile")
            async def get_user_profile(user_id: int):
                
    """get_user_profile function."""
await asyncio.sleep(0.05)  # Simulate DB call
                return {"user_id": user_id, "profile": f"profile_{user_id}"}
            
            @manager.cache_invalidate("user_profile:*")
            async def update_user_profile(user_id: int, profile_data: dict):
                
    """update_user_profile function."""
return {"user_id": user_id, "updated": True}
            
            # Multiple users accessing profiles
            users = [1, 2, 3, 4, 5]
            
            # First access for all users
            for user_id in users:
                profile = await get_user_profile(user_id)
                assert profile["user_id"] == user_id
            
            # Second access (should use cache)
            for user_id in users:
                profile = await get_user_profile(user_id)
                assert profile["user_id"] == user_id
            
            # Update one user
            await update_user_profile(1, {"name": "Updated"})
            
            # Access updated user (should not use cache)
            profile = await get_user_profile(1)
            assert profile["user_id"] == 1
            
            await manager.stop()

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestCachePerformance:
    """Test cache performance."""
    
    @pytest.mark.asyncio
    async def test_cache_performance(self, cache_config) -> Any:
        """Test cache performance with multiple operations."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            
            manager = create_cache_manager(cache_config)
            await manager.start()
            
            # Performance test with many operations
            start_time = time.time()
            
            for i in range(100):
                await manager.set(f"key_{i}", f"value_{i}")
            
            for i in range(100):
                await manager.get(f"key_{i}")
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should complete quickly
            assert duration < 5.0  # Should complete in under 5 seconds
            
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_decorator_performance(self, cache_config) -> Any:
        """Test decorator performance."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            
            manager = create_cache_manager(cache_config)
            await manager.start()
            
            call_count = 0
            
            @manager.cached(ttl=3600, key_prefix="perf_test")
            async def performance_test_function(param: int):
                
    """performance_test_function function."""
nonlocal call_count
                call_count += 1
                await asyncio.sleep(0.01)  # Simulate work
                return param * 2
            
            # Multiple calls
            start_time = time.time()
            
            for i in range(50):
                result = await performance_test_function(i)
                assert result == i * 2
            
            # Second round (should use cache)
            for i in range(50):
                result = await performance_test_function(i)
                assert result == i * 2
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should be much faster on second round
            assert call_count == 50  # Only 50 actual calls
            assert duration < 2.0  # Should complete quickly
            
            await manager.stop()

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestCacheErrorHandling:
    """Test cache error handling."""
    
    @pytest.mark.asyncio
    async def test_redis_connection_error(self) -> Any:
        """Test handling of Redis connection errors."""
        config = CacheConfig(redis_url="redis://invalid:6379")
        
        # Should not raise exception
        manager = create_cache_manager(config)
        await manager.start()
        
        # Operations should still work (fallback to memory cache)
        success = await manager.set("test_key", "test_value")
        assert success is True
        
        result = await manager.get("test_key")
        assert result == "test_value"
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_serialization_error(self, cache_config) -> Any:
        """Test handling of serialization errors."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            
            manager = create_cache_manager(cache_config)
            await manager.start()
            
            # Try to cache non-serializable object
            class NonSerializable:
                pass
            
            # Should handle gracefully
            success = await manager.set("test_key", NonSerializable())
            # This might fail, but shouldn't crash
            
            await manager.stop()

# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_cache_config(self) -> Any:
        """Test cache config creation."""
        config = create_cache_config(
            redis_url="redis://test:6379",
            memory_cache_size=500
        )
        
        assert config.redis_url == "redis://test:6379"
        assert config.memory_cache_size == 500
    
    @pytest.mark.asyncio
    async def test_create_cache_manager(self) -> Any:
        """Test cache manager creation."""
        with patch('caching_system.redis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            
            manager = create_cache_manager()
            assert manager is not None
            
            await manager.start()
            await manager.stop()

match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 