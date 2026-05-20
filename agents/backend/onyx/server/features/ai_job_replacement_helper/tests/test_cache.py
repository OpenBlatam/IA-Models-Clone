"""
Tests for Cache Service
"""

import pytest
import time
from core.cache import CacheService, cached


@pytest.fixture
def cache_service():
    return CacheService(ttl_seconds=1)


def test_cache_set_get(cache_service):
    """Test basic cache operations"""
    cache_service.set("key1", "value1")
    assert cache_service.get("key1") == "value1"


def test_cache_expiration(cache_service):
    """Test cache expiration"""
    cache_service.set("key1", "value1", ttl=1)
    assert cache_service.get("key1") == "value1"
    
    time.sleep(1.1)
    assert cache_service.get("key1") is None


def test_cache_delete(cache_service):
    """Test cache deletion"""
    cache_service.set("key1", "value1")
    cache_service.delete("key1")
    assert cache_service.get("key1") is None


def test_cache_decorator():
    """Test cache decorator"""
    call_count = [0]
    
    @cached(ttl=60)
    def expensive_function(x):
        call_count[0] += 1
        return x * 2
    
    # Primera llamada
    result1 = expensive_function(5)
    assert result1 == 10
    assert call_count[0] == 1
    
    # Segunda llamada (debería usar caché)
    result2 = expensive_function(5)
    assert result2 == 10
    assert call_count[0] == 1  # No debería incrementar




