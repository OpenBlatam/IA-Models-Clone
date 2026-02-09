"""
Tests para cache backend
"""

import pytest
import tempfile
import os
from pathlib import Path

from core.cache.cache_backend import (
    CacheBackend,
    MemoryCache,
    FileCache
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestMemoryCache(BaseServiceTestCase, StandardTestMixin):
    """Tests para MemoryCache"""
    
    @pytest.fixture
    def memory_cache(self):
        """Fixture para MemoryCache"""
        return MemoryCache(max_size=100)
    
    def test_memory_cache_init(self, memory_cache):
        """Test de inicialización"""
        assert memory_cache.max_size == 100
        assert len(memory_cache.cache) == 0
    
    def test_set_and_get(self, memory_cache):
        """Test de set y get"""
        memory_cache.set("key1", "value1")
        
        result = memory_cache.get("key1")
        
        assert result == "value1"
    
    @pytest.mark.parametrize("key,value", [
        ("key1", "value1"),
        ("key2", 123),
        ("key3", {"nested": "data"}),
        ("key4", [1, 2, 3])
    ])
    def test_set_get_different_types(self, memory_cache, key, value):
        """Test de set y get con diferentes tipos de datos"""
        memory_cache.set(key, value)
        
        result = memory_cache.get(key)
        
        assert result == value
    
    def test_get_nonexistent(self, memory_cache):
        """Test de get de clave inexistente"""
        result = memory_cache.get("nonexistent")
        
        assert result is None
    
    def test_delete(self, memory_cache):
        """Test de delete"""
        memory_cache.set("key1", "value1")
        memory_cache.delete("key1")
        
        result = memory_cache.get("key1")
        
        assert result is None
    
    def test_clear(self, memory_cache):
        """Test de clear"""
        memory_cache.set("key1", "value1")
        memory_cache.set("key2", "value2")
        memory_cache.clear()
        
        assert len(memory_cache.cache) == 0
    
    def test_max_size_eviction(self, memory_cache):
        """Test de evicción cuando se alcanza max_size"""
        # Llenar cache hasta el límite
        for i in range(100):
            memory_cache.set(f"key{i}", f"value{i}")
        
        # Agregar uno más debería evictar el más viejo
        memory_cache.set("key100", "value100")
        
        # El primer key debería estar evictado
        assert memory_cache.get("key0") is None
        assert memory_cache.get("key100") == "value100"


class TestFileCache(BaseServiceTestCase, StandardTestMixin):
    """Tests para FileCache"""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Directorio temporal para cache"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def file_cache(self, temp_cache_dir):
        """Fixture para FileCache"""
        return FileCache(cache_dir=temp_cache_dir)
    
    def test_file_cache_init(self, file_cache, temp_cache_dir):
        """Test de inicialización"""
        assert file_cache.cache_dir == Path(temp_cache_dir)
        assert os.path.exists(temp_cache_dir)
    
    def test_set_and_get(self, file_cache):
        """Test de set y get"""
        file_cache.set("key1", "value1")
        
        result = file_cache.get("key1")
        
        assert result == "value1"
    
    @pytest.mark.parametrize("key,value", [
        ("key1", "value1"),
        ("key2", 123),
        ("key3", {"nested": "data"})
    ])
    def test_set_get_different_types(self, file_cache, key, value):
        """Test de set y get con diferentes tipos"""
        file_cache.set(key, value)
        
        result = file_cache.get(key)
        
        assert result == value
    
    def test_get_nonexistent(self, file_cache):
        """Test de get de clave inexistente"""
        result = file_cache.get("nonexistent")
        
        assert result is None
    
    def test_delete(self, file_cache):
        """Test de delete"""
        file_cache.set("key1", "value1")
        file_cache.delete("key1")
        
        result = file_cache.get("key1")
        
        assert result is None
    
    def test_clear(self, file_cache):
        """Test de clear"""
        file_cache.set("key1", "value1")
        file_cache.set("key2", "value2")
        file_cache.clear()
        
        result1 = file_cache.get("key1")
        result2 = file_cache.get("key2")
        
        assert result1 is None
        assert result2 is None



