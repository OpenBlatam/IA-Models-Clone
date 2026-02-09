"""
Tests para cache de respuestas
"""

import pytest
import asyncio
import time
from api.utils.response_cache import (
    cache_response,
    clear_response_cache,
    get_cache_stats
)


@pytest.mark.unit
@pytest.mark.api
class TestCacheResponse:
    """Tests para cache_response decorator"""
    
    @pytest.mark.asyncio
    async def test_cache_response_first_call(self):
        """Test de primera llamada (sin cache)"""
        call_count = 0
        
        @cache_response(ttl=60)
        async def test_func(param):
            nonlocal call_count
            call_count += 1
            return {"result": param, "count": call_count}
        
        result = await test_func("test")
        
        assert result["result"] == "test"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_cache_response_cached_call(self):
        """Test de llamada cacheada"""
        call_count = 0
        
        @cache_response(ttl=60)
        async def test_func(param):
            nonlocal call_count
            call_count += 1
            return {"result": param, "count": call_count}
        
        result1 = await test_func("test")
        result2 = await test_func("test")
        
        # Debería usar cache, no incrementar count
        assert result1["count"] == 1
        assert result2["count"] == 1
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_cache_response_different_params(self):
        """Test con diferentes parámetros"""
        call_count = 0
        
        @cache_response(ttl=60)
        async def test_func(param):
            nonlocal call_count
            call_count += 1
            return {"result": param}
        
        await test_func("param1")
        await test_func("param2")
        
        # Debería llamar dos veces con diferentes parámetros
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_cache_response_ttl_expires(self):
        """Test de expiración de TTL"""
        call_count = 0
        
        @cache_response(ttl=1)  # 1 segundo
        async def test_func():
            nonlocal call_count
            call_count += 1
            return {"count": call_count}
        
        await test_func()
        await test_func()  # Debería usar cache
        
        assert call_count == 1
        
        # Esperar a que expire
        time.sleep(1.1)
        
        await test_func()  # Debería llamar de nuevo
        
        assert call_count == 2


@pytest.mark.unit
@pytest.mark.api
class TestClearResponseCache:
    """Tests para clear_response_cache"""
    
    @pytest.mark.asyncio
    async def test_clear_response_cache(self):
        """Test de limpieza de cache"""
        call_count = 0
        
        @cache_response(ttl=60)
        async def test_func():
            nonlocal call_count
            call_count += 1
            return {"count": call_count}
        
        await test_func()
        await test_func()  # Usa cache
        
        assert call_count == 1
        
        clear_response_cache()
        
        await test_func()  # Debería llamar de nuevo
        
        assert call_count == 2


@pytest.mark.unit
@pytest.mark.api
class TestGetCacheStats:
    """Tests para get_cache_stats"""
    
    @pytest.mark.asyncio
    async def test_get_cache_stats(self):
        """Test de obtención de stats de cache"""
        @cache_response(ttl=60)
        async def test_func():
            return {"result": "test"}
        
        await test_func()
        
        stats = get_cache_stats()
        
        assert isinstance(stats, dict)
        assert "size" in stats
        assert "keys" in stats
        assert stats["size"] > 0
    
    @pytest.mark.asyncio
    async def test_get_cache_stats_empty(self):
        """Test de stats con cache vacío"""
        clear_response_cache()
        
        stats = get_cache_stats()
        
        assert stats["size"] == 0
        assert len(stats["keys"]) == 0



