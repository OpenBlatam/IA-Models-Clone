"""
Comprehensive Unit Tests for Service Locator

Tests cover service locator pattern with diverse test cases
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from core.service_locator import ServiceLocator, get_service, resolve_service


class TestServiceLocator:
    """Test cases for ServiceLocator class"""
    
    def test_service_locator_init(self):
        """Test initializing service locator"""
        with patch('core.service_locator.get_container') as mock_get_container:
            mock_container = Mock()
            mock_get_container.return_value = mock_container
            
            locator = ServiceLocator()
            
            assert locator._container == mock_container
    
    def test_get_service_exists(self):
        """Test getting existing service"""
        mock_container = Mock()
        mock_service = Mock()
        mock_container.get.return_value = mock_service
        
        with patch('core.service_locator.get_container', return_value=mock_container):
            locator = ServiceLocator()
            result = locator.get("test_service")
            
            assert result == mock_service
            mock_container.get.assert_called_once_with("test_service")
    
    def test_get_service_not_exists(self):
        """Test getting non-existent service"""
        mock_container = Mock()
        mock_container.get.return_value = None
        
        with patch('core.service_locator.get_container', return_value=mock_container):
            locator = ServiceLocator()
            result = locator.get("nonexistent")
            
            assert result is None
    
    def test_resolve_service_by_type(self):
        """Test resolving service by type"""
        class TestService:
            pass
        
        mock_container = Mock()
        mock_service = TestService()
        mock_container.resolve.return_value = mock_service
        
        with patch('core.service_locator.get_container', return_value=mock_container):
            locator = ServiceLocator()
            result = locator.resolve(TestService)
            
            assert result == mock_service
            mock_container.resolve.assert_called_once_with(TestService)
    
    def test_has_service_exists(self):
        """Test checking if service exists"""
        mock_container = Mock()
        mock_container.has.return_value = True
        
        with patch('core.service_locator.get_container', return_value=mock_container):
            locator = ServiceLocator()
            result = locator.has("test_service")
            
            assert result is True
            mock_container.has.assert_called_once_with("test_service")
    
    def test_has_service_not_exists(self):
        """Test checking if service doesn't exist"""
        mock_container = Mock()
        mock_container.has.return_value = False
        
        with patch('core.service_locator.get_container', return_value=mock_container):
            locator = ServiceLocator()
            result = locator.has("nonexistent")
            
            assert result is False


class TestGetService:
    """Test cases for get_service function"""
    
    def test_get_service_singleton(self):
        """Test that get_service uses singleton locator"""
        mock_container = Mock()
        mock_service = Mock()
        mock_container.get.return_value = mock_service
        
        with patch('core.service_locator.get_container', return_value=mock_container):
            result1 = get_service("test_service")
            result2 = get_service("test_service")
            
            # Should use same locator instance
            assert result1 == mock_service
            assert result2 == mock_service


class TestResolveService:
    """Test cases for resolve_service function"""
    
    def test_resolve_service_by_type(self):
        """Test resolving service by type"""
        class TestService:
            pass
        
        mock_container = Mock()
        mock_service = TestService()
        mock_container.resolve.return_value = mock_service
        
        with patch('core.service_locator.get_container', return_value=mock_container):
            result = resolve_service(TestService)
            
            assert result == mock_service
            assert isinstance(result, TestService)















