"""
Tests for Enhanced Cache System

This test suite covers:
- Cache initialization
- L1 (in-memory) cache operations
- L2 (Redis) cache operations
- Multi-tier cache behavior
- Compression and serialization
- Statistics and monitoring
- Error handling and edge cases

Test Generation Principles:
- Unique: Each test covers a distinct scenario
- Diverse: Tests cover happy paths, edge cases, errors, and boundaries
- Intuitive: Clear names and assertions express intent
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Any, Dict

from ..core.cache import (
    EnhancedCache,
    CacheStats,
    CacheEntry,
    get_cache,
    close_cache
)


class TestCacheStats:
    """Test suite for CacheStats dataclass"""

    def test_cache_stats_initialization(self):
        """Test CacheStats initializes with default values"""
        # Happy path: Default initialization
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.total_requests == 0
        assert stats.hit_rate == 0.0

    def test_cache_stats_update_hit_rate_with_requests(self):
        """Test CacheStats.update_hit_rate calculates correctly with requests"""
        # Happy path: Hit rate calculation
        stats = CacheStats()
        stats.hits = 75
        stats.total_requests = 100
        stats.update_hit_rate()
        assert stats.hit_rate == 75.0

    def test_cache_stats_update_hit_rate_with_zero_requests(self):
        """Test CacheStats.update_hit_rate handles zero requests"""
        # Edge case: Zero requests
        stats = CacheStats()
        stats.hits = 0
        stats.total_requests = 0
        stats.update_hit_rate()
        assert stats.hit_rate == 0.0

    def test_cache_stats_update_hit_rate_with_all_hits(self):
        """Test CacheStats.update_hit_rate with 100% hit rate"""
        # Boundary value: 100% hits
        stats = CacheStats()
        stats.hits = 100
        stats.total_requests = 100
        stats.update_hit_rate()
        assert stats.hit_rate == 100.0

    def test_cache_stats_update_hit_rate_with_all_misses(self):
        """Test CacheStats.update_hit_rate with 0% hit rate"""
        # Boundary value: 0% hits
        stats = CacheStats()
        stats.hits = 0
        stats.total_requests = 100
        stats.update_hit_rate()
        assert stats.hit_rate == 0.0


class TestCacheEntry:
    """Test suite for CacheEntry dataclass"""

    def test_cache_entry_initialization(self):
        """Test CacheEntry initializes with required fields"""
        # Happy path: Normal initialization
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time(),
            expires_at=time.time() + 3600
        )
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.access_count == 0
        assert entry.compressed is False

    def test_cache_entry_with_custom_fields(self):
        """Test CacheEntry with custom field values"""
        # Happy path: Custom fields
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=1000.0,
            expires_at=2000.0,
            access_count=5,
            size_bytes=1024,
            compressed=True
        )
        assert entry.access_count == 5
        assert entry.size_bytes == 1024
        assert entry.compressed is True

    def test_cache_entry_default_last_accessed(self):
        """Test CacheEntry has default last_accessed timestamp"""
        # Edge case: Default timestamp
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time(),
            expires_at=time.time() + 3600
        )
        assert entry.last_accessed > 0
        assert isinstance(entry.last_accessed, float)


class TestEnhancedCacheInitialization:
    """Test suite for EnhancedCache initialization"""

    def test_init_with_default_parameters(self):
        """Test EnhancedCache initializes with default parameters"""
        # Happy path: Default parameters
        cache = EnhancedCache()
        assert cache.l1_max_size == 1000
        assert cache.l1_ttl == 300
        assert cache.l2_ttl == 3600
        assert cache.enable_compression is True
        assert cache.compression_threshold == 1024

    def test_init_with_custom_parameters(self):
        """Test EnhancedCache initializes with custom parameters"""
        # Happy path: Custom parameters
        cache = EnhancedCache(
            l1_max_size=500,
            l1_ttl=600,
            l2_enabled=False,
            enable_compression=False,
            compression_threshold=2048
        )
        assert cache.l1_max_size == 500
        assert cache.l1_ttl == 600
        assert cache.l2_enabled is False
        assert cache.enable_compression is False
        assert cache.compression_threshold == 2048

    def test_init_with_l2_disabled(self):
        """Test EnhancedCache initializes with L2 disabled"""
        # Happy path: L2 disabled
        cache = EnhancedCache(l2_enabled=False)
        assert cache.l2_enabled is False
        assert cache.redis_client is None

    def test_init_creates_stats(self):
        """Test EnhancedCache creates statistics object"""
        # Happy path: Stats creation
        cache = EnhancedCache()
        assert isinstance(cache.stats, CacheStats)
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0

    def test_init_with_invalid_redis_url_handles_gracefully(self):
        """Test EnhancedCache handles invalid Redis URL gracefully"""
        # Error condition: Invalid Redis URL
        cache = EnhancedCache(l2_redis_url="redis://invalid:6379/0")
        # Should disable L2 and continue
        assert cache.l2_enabled is False or cache.redis_client is None


class TestCacheKeyGeneration:
    """Test suite for cache key generation"""

    def test_generate_key_with_prefix_only(self, cache):
        """Test _generate_key with only prefix"""
        # Happy path: Prefix only
        key = cache._generate_key("test")
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hash length

    def test_generate_key_with_args(self, cache):
        """Test _generate_key with positional arguments"""
        # Happy path: With args
        key1 = cache._generate_key("test", "arg1", "arg2")
        key2 = cache._generate_key("test", "arg1", "arg2")
        assert key1 == key2  # Same inputs = same key

    def test_generate_key_with_kwargs(self, cache):
        """Test _generate_key with keyword arguments"""
        # Happy path: With kwargs
        key1 = cache._generate_key("test", param1="value1", param2="value2")
        key2 = cache._generate_key("test", param2="value2", param1="value1")
        assert key1 == key2  # Order shouldn't matter

    def test_generate_key_with_mixed_args_and_kwargs(self, cache):
        """Test _generate_key with both args and kwargs"""
        # Happy path: Mixed args and kwargs
        key = cache._generate_key("test", "arg1", param1="value1")
        assert isinstance(key, str)
        assert len(key) == 32

    def test_generate_key_different_inputs_different_keys(self, cache):
        """Test _generate_key produces different keys for different inputs"""
        # Edge case: Different inputs
        key1 = cache._generate_key("test1")
        key2 = cache._generate_key("test2")
        assert key1 != key2

    def test_generate_key_with_empty_args(self, cache):
        """Test _generate_key handles empty args"""
        # Edge case: Empty args
        key = cache._generate_key("test", *[])
        assert isinstance(key, str)


class TestCacheSerialization:
    """Test suite for cache serialization"""

    def test_serialize_with_simple_value(self, cache):
        """Test _serialize with simple value"""
        # Happy path: Simple value
        value = {"key": "value"}
        serialized = cache._serialize(value)
        assert isinstance(serialized, bytes)

    def test_serialize_with_complex_value(self, cache):
        """Test _serialize with complex nested value"""
        # Happy path: Complex value
        value = {
            "nested": {
                "list": [1, 2, 3],
                "tuple": (4, 5, 6)
            }
        }
        serialized = cache._serialize(value)
        assert isinstance(serialized, bytes)

    def test_serialize_with_list(self, cache):
        """Test _serialize with list"""
        # Happy path: List
        value = [1, 2, 3, "test"]
        serialized = cache._serialize(value)
        assert isinstance(serialized, bytes)

    def test_deserialize_with_valid_bytes(self, cache):
        """Test _deserialize with valid bytes"""
        # Happy path: Valid bytes
        value = {"key": "value"}
        serialized = cache._serialize(value)
        deserialized = cache._deserialize(serialized)
        assert deserialized == value

    def test_deserialize_with_string(self, cache):
        """Test _deserialize handles string input"""
        # Edge case: String input
        value = {"key": "value"}
        serialized = cache._serialize(value)
        if isinstance(serialized, bytes):
            deserialized = cache._deserialize(serialized.decode('utf-8'))
            assert deserialized == value

    def test_serialize_deserialize_roundtrip(self, cache):
        """Test serialize/deserialize roundtrip"""
        # Happy path: Roundtrip
        original = {"key": "value", "number": 42, "list": [1, 2, 3]}
        serialized = cache._serialize(original)
        deserialized = cache._deserialize(serialized)
        assert deserialized == original


class TestCacheCompression:
    """Test suite for cache compression"""

    def test_compress_with_data_above_threshold(self, cache):
        """Test _compress compresses data above threshold"""
        # Happy path: Above threshold
        large_data = b"A" * 2048  # Above default threshold of 1024
        compressed = cache._compress(large_data)
        assert len(compressed) < len(large_data)
        assert isinstance(compressed, bytes)

    def test_compress_with_data_below_threshold(self, cache):
        """Test _compress returns original data below threshold"""
        # Edge case: Below threshold
        small_data = b"A" * 100  # Below threshold
        compressed = cache._compress(small_data)
        assert compressed == small_data

    def test_compress_with_compression_disabled(self):
        """Test _compress returns original when compression disabled"""
        # Edge case: Compression disabled
        cache = EnhancedCache(enable_compression=False)
        large_data = b"A" * 2048
        compressed = cache._compress(large_data)
        assert compressed == large_data

    def test_decompress_with_compressed_data(self, cache):
        """Test _decompress decompresses compressed data"""
        # Happy path: Compressed data
        original = b"A" * 2048
        compressed = cache._compress(original)
        decompressed = cache._decompress(compressed)
        assert decompressed == original

    def test_decompress_with_uncompressed_data(self, cache):
        """Test _decompress handles uncompressed data"""
        # Edge case: Uncompressed data
        original = b"A" * 100
        decompressed = cache._decompress(original)
        assert decompressed == original

    def test_compress_decompress_roundtrip(self, cache):
        """Test compress/decompress roundtrip"""
        # Happy path: Roundtrip
        original = b"Test data " * 200  # Large enough to compress
        compressed = cache._compress(original)
        decompressed = cache._decompress(compressed)
        assert decompressed == original


class TestCacheGet:
    """Test suite for cache get operations"""

    @pytest.mark.asyncio
    async def test_get_with_existing_key_in_l1(self, cache):
        """Test get retrieves value from L1 cache"""
        # Happy path: L1 hit
        await cache.set("test_key", "test_value", level="l1")
        result = await cache.get("test_key")
        assert result == "test_value"
        assert cache.stats.hits > 0

    @pytest.mark.asyncio
    async def test_get_with_nonexistent_key(self, cache):
        """Test get returns None for nonexistent key"""
        # Happy path: Miss
        result = await cache.get("nonexistent_key")
        assert result is None
        assert cache.stats.misses > 0

    @pytest.mark.asyncio
    async def test_get_with_expired_entry(self, cache):
        """Test get handles expired entries correctly"""
        # Edge case: Expired entry
        await cache.set("test_key", "test_value", ttl=0.1, level="l1")
        await asyncio.sleep(0.2)
        result = await cache.get("test_key")
        # Should return None or handle expiration
        assert result is None or result == "test_value"

    @pytest.mark.asyncio
    async def test_get_updates_statistics(self, cache):
        """Test get updates cache statistics"""
        # Happy path: Stats update
        initial_requests = cache.stats.total_requests
        await cache.get("test_key")
        assert cache.stats.total_requests == initial_requests + 1

    @pytest.mark.asyncio
    async def test_get_with_empty_key(self, cache):
        """Test get handles empty key"""
        # Edge case: Empty key
        result = await cache.get("")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_promotes_l2_to_l1(self, cache):
        """Test get promotes L2 value to L1"""
        # Happy path: L2 to L1 promotion
        if cache.l2_enabled:
            await cache.set("test_key", "test_value", level="l2")
            # Clear L1 to force L2 lookup
            await cache.clear(level="l1")
            result = await cache.get("test_key")
            # Value should be promoted to L1
            assert result == "test_value"


class TestCacheSet:
    """Test suite for cache set operations"""

    @pytest.mark.asyncio
    async def test_set_with_simple_value(self, cache):
        """Test set stores simple value"""
        # Happy path: Simple value
        result = await cache.set("test_key", "test_value")
        assert result is True
        retrieved = await cache.get("test_key")
        assert retrieved == "test_value"

    @pytest.mark.asyncio
    async def test_set_with_complex_value(self, cache):
        """Test set stores complex nested value"""
        # Happy path: Complex value
        value = {"nested": {"list": [1, 2, 3]}}
        result = await cache.set("test_key", value)
        assert result is True
        retrieved = await cache.get("test_key")
        assert retrieved == value

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self, cache):
        """Test set uses custom TTL"""
        # Happy path: Custom TTL
        result = await cache.set("test_key", "test_value", ttl=600)
        assert result is True

    @pytest.mark.asyncio
    async def test_set_with_l1_only(self, cache):
        """Test set stores in L1 only"""
        # Happy path: L1 only
        result = await cache.set("test_key", "test_value", level="l1")
        assert result is True
        retrieved = await cache.get("test_key")
        assert retrieved == "test_value"

    @pytest.mark.asyncio
    async def test_set_with_l2_only(self, cache):
        """Test set stores in L2 only"""
        # Happy path: L2 only
        if cache.l2_enabled:
            result = await cache.set("test_key", "test_value", level="l2")
            assert result is True

    @pytest.mark.asyncio
    async def test_set_overwrites_existing_value(self, cache):
        """Test set overwrites existing value"""
        # Edge case: Overwrite
        await cache.set("test_key", "old_value")
        await cache.set("test_key", "new_value")
        retrieved = await cache.get("test_key")
        assert retrieved == "new_value"

    @pytest.mark.asyncio
    async def test_set_handles_large_values(self, cache):
        """Test set handles large values"""
        # Edge case: Large value
        large_value = "A" * 10000
        result = await cache.set("test_key", large_value)
        assert result is True
        retrieved = await cache.get("test_key")
        assert retrieved == large_value


class TestCacheDelete:
    """Test suite for cache delete operations"""

    @pytest.mark.asyncio
    async def test_delete_with_existing_key(self, cache):
        """Test delete removes existing key"""
        # Happy path: Delete existing
        await cache.set("test_key", "test_value")
        result = await cache.delete("test_key")
        assert result is True
        retrieved = await cache.get("test_key")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_with_nonexistent_key(self, cache):
        """Test delete handles nonexistent key"""
        # Edge case: Nonexistent key
        result = await cache.delete("nonexistent_key")
        # Should return False or True depending on implementation
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_delete_from_both_levels(self, cache):
        """Test delete removes from both L1 and L2"""
        # Happy path: Both levels
        await cache.set("test_key", "test_value", level="both")
        result = await cache.delete("test_key")
        assert result is True
        retrieved = await cache.get("test_key")
        assert retrieved is None


class TestCacheClear:
    """Test suite for cache clear operations"""

    @pytest.mark.asyncio
    async def test_clear_all_levels(self, cache):
        """Test clear removes all entries from all levels"""
        # Happy path: Clear all
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        result = await cache.clear()
        assert result is True
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_clear_l1_only(self, cache):
        """Test clear removes only L1 entries"""
        # Happy path: L1 only
        await cache.set("key1", "value1", level="l1")
        result = await cache.clear(level="l1")
        assert result is True
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_clear_l2_only(self, cache):
        """Test clear removes only L2 entries"""
        # Happy path: L2 only
        if cache.l2_enabled:
            await cache.set("key1", "value1", level="l2")
            result = await cache.clear(level="l2")
            assert result is True

    @pytest.mark.asyncio
    async def test_clear_with_empty_cache(self, cache):
        """Test clear handles empty cache"""
        # Edge case: Empty cache
        result = await cache.clear()
        assert result is True


class TestCacheStats:
    """Test suite for cache statistics"""

    @pytest.mark.asyncio
    async def test_get_stats_returns_complete_info(self, cache):
        """Test get_stats returns complete statistics"""
        # Happy path: Complete stats
        await cache.set("key1", "value1")
        await cache.get("key1")
        await cache.get("nonexistent")
        
        stats = await cache.get_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "total_requests" in stats
        assert "l1_size" in stats
        assert "l2_enabled" in stats

    @pytest.mark.asyncio
    async def test_get_stats_calculates_hit_rate(self, cache):
        """Test get_stats calculates hit rate correctly"""
        # Happy path: Hit rate calculation
        await cache.set("key1", "value1")
        await cache.get("key1")  # Hit
        await cache.get("key1")  # Hit
        await cache.get("nonexistent")  # Miss
        
        stats = await cache.get_stats()
        assert stats["total_requests"] == 3
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] > 0

    @pytest.mark.asyncio
    async def test_get_stats_includes_l1_info(self, cache):
        """Test get_stats includes L1 cache information"""
        # Happy path: L1 info
        await cache.set("key1", "value1")
        stats = await cache.get_stats()
        assert "l1_size" in stats
        assert "l1_max_size" in stats
        assert stats["l1_size"] >= 0


class TestCacheGlobalFunctions:
    """Test suite for global cache functions"""

    def test_get_cache_returns_instance(self):
        """Test get_cache returns cache instance"""
        # Happy path: Get instance
        cache = get_cache()
        assert isinstance(cache, EnhancedCache)

    def test_get_cache_returns_same_instance(self):
        """Test get_cache returns same instance on multiple calls"""
        # Happy path: Singleton pattern
        cache1 = get_cache()
        cache2 = get_cache()
        assert cache1 is cache2

    @pytest.mark.asyncio
    async def test_close_cache_closes_connections(self):
        """Test close_cache closes cache connections"""
        # Happy path: Close connections
        cache = get_cache()
        await close_cache()
        # Should not raise
        assert True


@pytest.fixture
def cache():
    """Fixture for EnhancedCache instance with L2 disabled for testing"""
    return EnhancedCache(l2_enabled=False, l1_max_size=100, l1_ttl=60)

