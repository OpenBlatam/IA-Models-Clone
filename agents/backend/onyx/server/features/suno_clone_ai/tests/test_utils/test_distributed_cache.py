"""
Comprehensive Unit Tests for Distributed Cache

Tests cover distributed cache functionality with diverse test cases
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from utils.distributed_cache import DistributedCache


class TestDistributedCache:
    """Test cases for DistributedCache class"""
    
    def test_distributed_cache_init_without_redis(self):
        """Test initializing cache without Redis"""
        with patch('utils.distributed_cache.REDIS_AVAILABLE', False):
            cache = DistributedCache()
            assert cache.redis_client is None
            assert cache.default_ttl == 3600
    
    def test_distributed_cache_init_with_redis_url(self):
        """Test initializing cache with Redis URL"""
        with patch('utils.distributed_cache.REDIS_AVAILABLE', True):
            mock_redis = Mock()
            mock_redis.ping = Mock()
            with patch('utils.distributed_cache.redis.from_url', return_value=mock_redis):
                cache = DistributedCache(redis_url="redis://localhost:6379")
                assert cache.redis_client == mock_redis
    
    def test_distributed_cache_init_redis_connection_failure(self):
        """Test fallback when Redis connection fails"""
        with patch('utils.distributed_cache.REDIS_AVAILABLE', True):
            with patch('utils.distributed_cache.redis.from_url', side_effect=Exception("Connection failed")):
                cache = DistributedCache(redis_url="redis://localhost:6379")
                assert cache.redis_client is None
    
    def test_make_key(self):
        """Test key generation with prefix"""
        cache = DistributedCache(key_prefix="test:")
        key = cache._make_key("mykey")
        assert key == "test:mykey"
    
    def test_make_key_default_prefix(self):
        """Test key generation with default prefix"""
        cache = DistributedCache()
        key = cache._make_key("mykey")
        assert key.startswith("suno_clone:")
    
    def test_serialize_and_deserialize(self):
        """Test serialization and deserialization"""
        cache = DistributedCache()
        data = {"key": "value", "number": 123}
        
        serialized = cache._serialize(data)
        assert isinstance(serialized, bytes)
        
        deserialized = cache._deserialize(serialized)
        assert deserialized == data
    
    def test_get_from_redis(self):
        """Test getting value from Redis"""
        with patch('utils.distributed_cache.REDIS_AVAILABLE', True):
            mock_redis = Mock()
            mock_data = b'pickled_data'
            mock_redis.get.return_value = mock_data
            mock_redis.ping = Mock()
            
            with patch('utils.distributed_cache.redis.from_url', return_value=mock_redis):
                cache = DistributedCache(redis_url="redis://localhost:6379")
                
                with patch.object(cache, '_deserialize', return_value="value"):
                    result = cache.get("key")
                    assert result == "value"
                    mock_redis.get.assert_called_once()
    
    def test_get_from_fallback(self):
        """Test getting value from fallback cache"""
        cache = DistributedCache()
        cache.redis_client = None
        cache._fallback_cache["key"] = "value"
        
        result = cache.get("key")
        assert result == "value"
    
    def test_get_not_found(self):
        """Test getting non-existent key"""
        cache = DistributedCache()
        cache.redis_client = None
        
        result = cache.get("nonexistent")
        assert result is None
    
    def test_set_to_redis(self):
        """Test setting value to Redis"""
        with patch('utils.distributed_cache.REDIS_AVAILABLE', True):
            mock_redis = Mock()
            mock_redis.ping = Mock()
            mock_redis.set = Mock()
            
            with patch('utils.distributed_cache.redis.from_url', return_value=mock_redis):
                cache = DistributedCache(redis_url="redis://localhost:6379")
                
                with patch.object(cache, '_serialize', return_value=b'data'):
                    cache.set("key", "value")
                    
                    mock_redis.set.assert_called_once()
                    call_args = mock_redis.set.call_args
                    assert call_args[0][0].endswith(":key")
    
    def test_set_to_fallback(self):
        """Test setting value to fallback cache"""
        cache = DistributedCache()
        cache.redis_client = None
        
        cache.set("key", "value")
        
        assert cache._fallback_cache["key"] == "value"
    
    def test_set_with_ttl(self):
        """Test setting value with TTL"""
        with patch('utils.distributed_cache.REDIS_AVAILABLE', True):
            mock_redis = Mock()
            mock_redis.ping = Mock()
            mock_redis.set = Mock()
            
            with patch('utils.distributed_cache.redis.from_url', return_value=mock_redis):
                cache = DistributedCache(redis_url="redis://localhost:6379")
                
                with patch.object(cache, '_serialize', return_value=b'data'):
                    cache.set("key", "value", ttl=1800)
                    
                    call_args = mock_redis.set.call_args
                    assert call_args[1]["ex"] == 1800
    
    def test_delete_from_redis(self):
        """Test deleting from Redis"""
        with patch('utils.distributed_cache.REDIS_AVAILABLE', True):
            mock_redis = Mock()
            mock_redis.ping = Mock()
            mock_redis.delete = Mock()
            
            with patch('utils.distributed_cache.redis.from_url', return_value=mock_redis):
                cache = DistributedCache(redis_url="redis://localhost:6379")
                cache.delete("key")
                
                mock_redis.delete.assert_called_once()
    
    def test_delete_from_fallback(self):
        """Test deleting from fallback cache"""
        cache = DistributedCache()
        cache.redis_client = None
        cache._fallback_cache["key"] = "value"
        
        cache.delete("key")
        
        assert "key" not in cache._fallback_cache
    
    def test_clear_cache(self):
        """Test clearing cache"""
        with patch('utils.distributed_cache.REDIS_AVAILABLE', True):
            mock_redis = Mock()
            mock_redis.ping = Mock()
            mock_redis.flushdb = Mock()
            
            with patch('utils.distributed_cache.redis.from_url', return_value=mock_redis):
                cache = DistributedCache(redis_url="redis://localhost:6379")
                cache.clear()
                
                mock_redis.flushdb.assert_called_once()
                assert len(cache._fallback_cache) == 0
    
    def test_clear_fallback_cache(self):
        """Test clearing fallback cache"""
        cache = DistributedCache()
        cache.redis_client = None
        cache._fallback_cache["key1"] = "value1"
        cache._fallback_cache["key2"] = "value2"
        
        cache.clear()
        
        assert len(cache._fallback_cache) == 0















