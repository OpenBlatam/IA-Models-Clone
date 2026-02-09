"""
Tests para cache strategies
"""

import pytest
import time

from core.cache.cache_strategies import (
    CacheStrategy,
    LRUCache,
    FIFOCache,
    TTLCache
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestLRUCache(BaseServiceTestCase, StandardTestMixin):
    """Tests para LRUCache strategy"""
    
    @pytest.fixture
    def lru_cache(self):
        """Fixture para LRUCache"""
        return LRUCache()
    
    def test_lru_cache_init(self, lru_cache):
        """Test de inicialización"""
        assert len(lru_cache.access_order) == 0
    
    def test_access(self, lru_cache):
        """Test de registro de acceso"""
        lru_cache.access("key1")
        lru_cache.access("key2")
        
        assert "key1" in lru_cache.access_order
        assert "key2" in lru_cache.access_order
    
    def test_access_updates_order(self, lru_cache):
        """Test de que access actualiza el orden"""
        lru_cache.access("key1")
        lru_cache.access("key2")
        lru_cache.access("key1")  # key1 debería moverse al final
        
        # El último accedido debería estar al final
        assert list(lru_cache.access_order.keys())[-1] == "key1"
    
    def test_should_evict(self, lru_cache):
        """Test de evicción LRU"""
        cache = {"key1": "value1", "key2": "value2"}
        
        lru_cache.access("key1")
        lru_cache.access("key2")
        lru_cache.access("key1")  # key1 es más reciente
        
        key_to_evict = lru_cache.should_evict(cache)
        
        assert key_to_evict == "key2"  # key2 es el menos reciente
    
    def test_should_evict_empty(self, lru_cache):
        """Test de evicción con cache vacío"""
        cache = {}
        
        key_to_evict = lru_cache.should_evict(cache)
        
        assert key_to_evict is None


class TestFIFOCache(BaseServiceTestCase, StandardTestMixin):
    """Tests para FIFOCache strategy"""
    
    @pytest.fixture
    def fifo_cache(self):
        """Fixture para FIFOCache"""
        return FIFOCache()
    
    def test_fifo_cache_init(self, fifo_cache):
        """Test de inicialización"""
        assert len(fifo_cache.insertion_order) == 0
    
    def test_insert(self, fifo_cache):
        """Test de inserción"""
        fifo_cache.insert("key1")
        fifo_cache.insert("key2")
        
        assert "key1" in fifo_cache.insertion_order
        assert "key2" in fifo_cache.insertion_order
    
    def test_insert_duplicate(self, fifo_cache):
        """Test de inserción duplicada"""
        fifo_cache.insert("key1")
        fifo_cache.insert("key1")
        
        assert fifo_cache.insertion_order.count("key1") == 1
    
    def test_should_evict(self, fifo_cache):
        """Test de evicción FIFO"""
        cache = {"key1": "value1", "key2": "value2", "key3": "value3"}
        
        fifo_cache.insert("key1")
        fifo_cache.insert("key2")
        fifo_cache.insert("key3")
        
        key_to_evict = fifo_cache.should_evict(cache)
        
        assert key_to_evict == "key1"  # El primero insertado
    
    def test_should_evict_empty(self, fifo_cache):
        """Test de evicción con cache vacío"""
        cache = {}
        
        key_to_evict = fifo_cache.should_evict(cache)
        
        assert key_to_evict is None


class TestTTLCache(BaseServiceTestCase, StandardTestMixin):
    """Tests para TTLCache strategy"""
    
    @pytest.fixture
    def ttl_cache(self):
        """Fixture para TTLCache"""
        return TTLCache(ttl=1.0)  # 1 segundo TTL
    
    def test_ttl_cache_init(self, ttl_cache):
        """Test de inicialización"""
        assert ttl_cache.ttl == 1.0
        assert len(ttl_cache.expiry_times) == 0
    
    def test_set_expiry(self, ttl_cache):
        """Test de establecer expiración"""
        ttl_cache.set_expiry("key1")
        
        assert "key1" in ttl_cache.expiry_times
        assert ttl_cache.expiry_times["key1"] > time.time()
    
    def test_should_evict_expired(self, ttl_cache):
        """Test de evicción de claves expiradas"""
        cache = {"key1": "value1"}
        
        # Establecer expiración en el pasado
        ttl_cache.expiry_times["key1"] = time.time() - 1
        
        key_to_evict = ttl_cache.should_evict(cache)
        
        assert key_to_evict == "key1"
        assert "key1" not in ttl_cache.expiry_times
    
    def test_should_evict_not_expired(self, ttl_cache):
        """Test de que no se evictan claves no expiradas"""
        cache = {"key1": "value1"}
        
        ttl_cache.set_expiry("key1")
        
        key_to_evict = ttl_cache.should_evict(cache)
        
        assert key_to_evict is None
    
    def test_should_evict_empty(self, ttl_cache):
        """Test de evicción con cache vacío"""
        cache = {}
        
        key_to_evict = ttl_cache.should_evict(cache)
        
        assert key_to_evict is None



