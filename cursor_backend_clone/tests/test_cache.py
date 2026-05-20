"""
Tests for Cache
===============

Tests para el sistema de caché.
"""

import pytest
import asyncio
from ..core.cache import Cache, CommandCache


@pytest.mark.asyncio
async def test_cache_basic():
    """Test básico de caché"""
    cache = Cache(max_size=10, default_ttl=60.0)
    
    await cache.set("key1", "value1")
    value = await cache.get("key1")
    assert value == "value1"
    
    await cache.delete("key1")
    value = await cache.get("key1")
    assert value is None


@pytest.mark.asyncio
async def test_cache_expiration():
    """Test expiración de caché"""
    cache = Cache(max_size=10, default_ttl=0.1)  # 100ms
    
    await cache.set("key1", "value1")
    value = await cache.get("key1")
    assert value == "value1"
    
    await asyncio.sleep(0.2)  # Esperar expiración
    
    value = await cache.get("key1")
    assert value is None


@pytest.mark.asyncio
async def test_cache_eviction():
    """Test evicción de caché"""
    cache = Cache(max_size=3, eviction_policy="lru")
    
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    await cache.set("key3", "value3")
    await cache.set("key4", "value4")  # Debería evictar key1
    
    assert await cache.get("key1") is None
    assert await cache.get("key4") == "value4"


@pytest.mark.asyncio
async def test_command_cache():
    """Test caché de comandos"""
    cache = CommandCache(max_size=10, ttl=60.0)
    
    await cache.set_result("print('test')", "test output")
    result = await cache.get_result("print('test')")
    assert result == "test output"
    
    # Comando diferente no debería estar en caché
    result = await cache.get_result("print('other')")
    assert result is None


