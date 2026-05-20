"""
Tests de caché y optimización
"""

import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime, timedelta


class TestCaching:
    """Tests de sistema de caché"""
    
    def test_cache_basic_operations(self):
        """Test de operaciones básicas de caché"""
        cache = {}
        
        def get_cached(key, compute_func, ttl=3600):
            if key in cache:
                entry = cache[key]
                if time.time() - entry["timestamp"] < ttl:
                    return entry["data"]
            
            data = compute_func()
            cache[key] = {
                "data": data,
                "timestamp": time.time()
            }
            return data
        
        call_count = [0]
        
        def expensive_operation():
            call_count[0] += 1
            return "result"
        
        # Primera llamada
        result1 = get_cached("key1", expensive_operation)
        assert result1 == "result"
        assert call_count[0] == 1
        
        # Segunda llamada (debe usar caché)
        result2 = get_cached("key1", expensive_operation)
        assert result2 == "result"
        assert call_count[0] == 1  # No debe incrementar
    
    def test_cache_expiration(self):
        """Test de expiración de caché"""
        cache = {}
        
        def get_cached(key, compute_func, ttl=0.1):
            if key in cache:
                entry = cache[key]
                if time.time() - entry["timestamp"] < ttl:
                    return entry["data"]
                else:
                    del cache[key]
            
            data = compute_func()
            cache[key] = {
                "data": data,
                "timestamp": time.time()
            }
            return data
        
        call_count = [0]
        
        def operation():
            call_count[0] += 1
            return "result"
        
        # Primera llamada
        get_cached("key1", operation, ttl=0.1)
        assert call_count[0] == 1
        
        # Esperar expiración
        time.sleep(0.2)
        
        # Segunda llamada (debe recalcular)
        get_cached("key1", operation, ttl=0.1)
        assert call_count[0] == 2
    
    def test_cache_invalidation(self):
        """Test de invalidación de caché"""
        cache = {}
        
        def invalidate_cache(key):
            if key in cache:
                del cache[key]
        
        cache["key1"] = {"data": "value1", "timestamp": time.time()}
        
        assert "key1" in cache
        
        invalidate_cache("key1")
        
        assert "key1" not in cache
    
    def test_cache_size_limit(self):
        """Test de límite de tamaño de caché"""
        cache = {}
        max_size = 3
        
        def add_to_cache(key, value):
            if len(cache) >= max_size:
                # Eliminar el más antiguo
                oldest_key = min(cache.keys(), key=lambda k: cache[k]["timestamp"])
                del cache[oldest_key]
            
            cache[key] = {
                "data": value,
                "timestamp": time.time()
            }
        
        # Agregar hasta el límite
        for i in range(max_size):
            add_to_cache(f"key{i}", f"value{i}")
        
        assert len(cache) == max_size
        
        # Agregar uno más (debe eliminar el más antiguo)
        add_to_cache("key_new", "value_new")
        
        assert len(cache) == max_size
        assert "key_new" in cache


class TestCachePerformance:
    """Tests de performance de caché"""
    
    def test_cache_hit_performance(self):
        """Test de performance con cache hit"""
        cache = {"key1": {"data": "cached", "timestamp": time.time()}}
        
        def get_cached(key):
            if key in cache:
                return cache[key]["data"]
            return None
        
        start = time.time()
        for _ in range(1000):
            get_cached("key1")
        elapsed = time.time() - start
        
        # Cache hit debe ser muy rápido
        assert elapsed < 0.01
    
    def test_cache_miss_performance(self):
        """Test de performance con cache miss"""
        cache = {}
        
        def get_cached(key, compute_func):
            if key in cache:
                return cache[key]["data"]
            
            data = compute_func()
            cache[key] = {"data": data, "timestamp": time.time()}
            return data
        
        def expensive():
            time.sleep(0.001)  # Simular operación costosa
            return "result"
        
        start = time.time()
        result = get_cached("key1", expensive)
        elapsed = time.time() - start
        
        assert result == "result"
        # Cache miss debe tomar tiempo de la operación
        assert elapsed >= 0.001


class TestCacheStrategies:
    """Tests de estrategias de caché"""
    
    def test_lru_cache(self):
        """Test de caché LRU (Least Recently Used)"""
        cache = {}
        access_order = []
        
        def get_lru(key, compute_func, max_size=3):
            if key in cache:
                access_order.remove(key)
                access_order.append(key)
                return cache[key]
            
            if len(cache) >= max_size:
                # Eliminar el menos recientemente usado
                lru_key = access_order.pop(0)
                del cache[lru_key]
            
            data = compute_func()
            cache[key] = data
            access_order.append(key)
            return data
        
        get_lru("key1", lambda: "value1")
        get_lru("key2", lambda: "value2")
        get_lru("key3", lambda: "value3")
        
        assert len(cache) == 3
        
        # Acceder a key1 (mover al final)
        get_lru("key1", lambda: "value1")
        
        # Agregar nuevo (key2 debe ser eliminado)
        get_lru("key4", lambda: "value4")
        
        assert "key4" in cache
        assert "key1" in cache
        assert "key3" in cache
    
    def test_ttl_cache(self):
        """Test de caché con TTL (Time To Live)"""
        cache = {}
        
        def get_with_ttl(key, compute_func, ttl=1.0):
            if key in cache:
                entry = cache[key]
                if time.time() - entry["created"] < ttl:
                    return entry["data"]
                else:
                    del cache[key]
            
            data = compute_func()
            cache[key] = {
                "data": data,
                "created": time.time()
            }
            return data
        
        call_count = [0]
        
        def operation():
            call_count[0] += 1
            return "result"
        
        get_with_ttl("key1", operation, ttl=0.1)
        assert call_count[0] == 1
        
        time.sleep(0.15)
        
        get_with_ttl("key1", operation, ttl=0.1)
        assert call_count[0] == 2  # Debe recalcular


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

