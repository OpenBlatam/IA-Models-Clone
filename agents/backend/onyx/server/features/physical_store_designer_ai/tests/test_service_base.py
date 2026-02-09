"""
Tests for BaseService
"""

import pytest
from datetime import datetime

from ..core.service_base import BaseService, TimestampedService


class TestBaseService:
    """Test cases for BaseService"""
    
    def test_generate_id(self):
        """Test ID generation"""
        service = BaseService("TestService")
        id1 = service.generate_id("test")
        id2 = service.generate_id("test")
        
        assert id1.startswith("test_")
        assert id2.startswith("test_")
        assert id1 != id2  # Should be unique
    
    def test_generate_timestamp_id(self):
        """Test timestamp ID generation"""
        service = BaseService("TestService")
        id1 = service.generate_timestamp_id("test")
        
        assert id1.startswith("test_")
        assert len(id1) > len("test_") + 10  # Should have timestamp
    
    def test_create_response(self):
        """Test response creation"""
        service = BaseService("TestService")
        response = service.create_response(
            data={"key": "value"},
            resource_id="test_123",
            note="Test note"
        )
        
        assert "created_at" in response
        assert "test_id" in response
        assert response["test_id"] == "test_123"
        assert "key" in response
        assert response["note"] == "Test note"
    
    def test_logging_methods(self):
        """Test logging methods"""
        service = BaseService("TestService")
        
        # Should not raise exceptions
        service.log_info("Test info")
        service.log_warning("Test warning")
        service.log_error("Test error")


class TestTimestampedService:
    """Test cases for TimestampedService"""
    
    def test_store_resource(self):
        """Test storing a resource"""
        service = TimestampedService("TestService")
        resource_id = "test_123"
        resource = {"key": "value"}
        
        service.store_resource(resource_id, resource)
        
        retrieved = service.get_resource(resource_id)
        assert retrieved == resource
    
    def test_get_resource_not_found(self):
        """Test getting non-existent resource"""
        service = TimestampedService("TestService")
        result = service.get_resource("non_existent")
        assert result is None
    
    def test_list_resources(self):
        """Test listing all resources"""
        service = TimestampedService("TestService")
        
        service.store_resource("test_1", {"key": "value1"})
        service.store_resource("test_2", {"key": "value2"})
        
        resources = service.list_resources()
        assert len(resources) == 2
    
    def test_delete_resource(self):
        """Test deleting a resource"""
        service = TimestampedService("TestService")
        resource_id = "test_123"
        
        service.store_resource(resource_id, {"key": "value"})
        result = service.delete_resource(resource_id)
        
        assert result is True
        assert service.get_resource(resource_id) is None
        
        # Delete non-existent
        result = service.delete_resource("non_existent")
        assert result is False








