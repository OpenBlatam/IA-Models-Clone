"""
Comprehensive Unit Tests for Factories

Tests cover factory pattern implementations with diverse test cases
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from core.factories import ServiceFactory, MusicGeneratorFactory, CacheFactory


class TestServiceFactory:
    """Test cases for ServiceFactory class"""
    
    def test_register_service(self):
        """Test registering a service"""
        class TestService:
            pass
        
        ServiceFactory.register("test_service", TestService)
        
        assert "test_service" in ServiceFactory._registry
        assert ServiceFactory._registry["test_service"] == TestService
    
    def test_create_service_singleton(self):
        """Test creating service returns singleton"""
        class TestService:
            def __init__(self):
                self.value = "test"
        
        ServiceFactory.register("test_service", TestService)
        
        instance1 = ServiceFactory.create("test_service")
        instance2 = ServiceFactory.create("test_service")
        
        assert instance1 is instance2
        assert isinstance(instance1, TestService)
    
    def test_create_service_with_kwargs(self):
        """Test creating service with kwargs"""
        class TestService:
            def __init__(self, param1, param2=None):
                self.param1 = param1
                self.param2 = param2
        
        ServiceFactory.register("test_service", TestService)
        
        instance = ServiceFactory.create("test_service", param1="value1", param2="value2")
        
        assert instance.param1 == "value1"
        assert instance.param2 == "value2"
    
    def test_create_service_not_registered(self):
        """Test creating non-registered service raises error"""
        with pytest.raises(ValueError, match="not registered"):
            ServiceFactory.create("nonexistent")
    
    def test_get_service_exists(self):
        """Test getting existing service"""
        class TestService:
            pass
        
        ServiceFactory.register("test_service", TestService)
        ServiceFactory.create("test_service")
        
        instance = ServiceFactory.get("test_service")
        assert instance is not None
        assert isinstance(instance, TestService)
    
    def test_get_service_not_exists(self):
        """Test getting non-existent service"""
        instance = ServiceFactory.get("nonexistent")
        assert instance is None
    
    def test_reset_factory(self):
        """Test resetting factory"""
        class TestService:
            pass
        
        ServiceFactory.register("test_service", TestService)
        instance1 = ServiceFactory.create("test_service")
        
        ServiceFactory.reset()
        
        # Should create new instance after reset
        instance2 = ServiceFactory.create("test_service")
        assert instance1 is not instance2


class TestMusicGeneratorFactory:
    """Test cases for MusicGeneratorFactory class"""
    
    @patch('core.factories.MusicGenerator')
    def test_create_generator_default(self, mock_generator_class):
        """Test creating default generator"""
        mock_instance = Mock()
        mock_generator_class.return_value = mock_instance
        
        generator = MusicGeneratorFactory.create_generator()
        
        assert generator == mock_instance
        mock_generator_class.assert_called_once()
    
    @patch('core.factories.FastMusicGenerator')
    def test_create_generator_fast(self, mock_fast_class):
        """Test creating fast generator"""
        mock_instance = Mock()
        mock_fast_class.return_value = mock_instance
        
        generator = MusicGeneratorFactory.create_generator("fast")
        
        assert generator == mock_instance
        mock_fast_class.assert_called_once()
    
    @patch('core.factories.DiffusionMusicGenerator')
    def test_create_generator_diffusion(self, mock_diffusion_class):
        """Test creating diffusion generator"""
        mock_instance = Mock()
        mock_diffusion_class.return_value = mock_instance
        
        generator = MusicGeneratorFactory.create_generator("diffusion")
        
        assert generator == mock_instance
        mock_diffusion_class.assert_called_once()
    
    @patch('core.factories.OptimizedMusicGenerator')
    def test_create_generator_optimized(self, mock_optimized_class):
        """Test creating optimized generator"""
        mock_instance = Mock()
        mock_optimized_class.return_value = mock_instance
        
        generator = MusicGeneratorFactory.create_generator("optimized")
        
        assert generator == mock_instance
        mock_optimized_class.assert_called_once()
    
    def test_create_generator_with_kwargs(self):
        """Test creating generator with kwargs"""
        with patch('core.factories.MusicGenerator') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            generator = MusicGeneratorFactory.create_generator(param1="value1")
            
            mock_class.assert_called_once_with(param1="value1")


class TestCacheFactory:
    """Test cases for CacheFactory class"""
    
    @patch('core.factories.LRUCache')
    def test_create_cache_memory(self, mock_lru):
        """Test creating memory cache"""
        mock_instance = Mock()
        mock_lru.return_value = mock_instance
        
        cache = CacheFactory.create_cache("memory")
        
        assert cache == mock_instance
        mock_lru.assert_called_once()
    
    @patch('core.factories.DiskCache')
    def test_create_cache_disk(self, mock_disk):
        """Test creating disk cache"""
        mock_instance = Mock()
        mock_disk.return_value = mock_instance
        
        cache = CacheFactory.create_cache("disk", cache_dir="/tmp")
        
        assert cache == mock_instance
        mock_disk.assert_called_once()
    
    @patch('core.factories.DistributedCache')
    def test_create_cache_distributed(self, mock_distributed):
        """Test creating distributed cache"""
        mock_instance = Mock()
        mock_distributed.return_value = mock_instance
        
        cache = CacheFactory.create_cache("distributed", redis_url="redis://localhost")
        
        assert cache == mock_instance
        mock_distributed.assert_called_once()
    
    @patch('core.factories.SmartCache')
    def test_create_cache_smart(self, mock_smart):
        """Test creating smart cache"""
        mock_instance = Mock()
        mock_smart.return_value = mock_instance
        
        cache = CacheFactory.create_cache("smart")
        
        assert cache == mock_instance
        mock_smart.assert_called_once()
    
    def test_create_cache_default(self):
        """Test creating cache with default type"""
        with patch('core.factories.LRUCache') as mock_lru:
            mock_instance = Mock()
            mock_lru.return_value = mock_instance
            
            cache = CacheFactory.create_cache()
            
            assert cache == mock_instance















