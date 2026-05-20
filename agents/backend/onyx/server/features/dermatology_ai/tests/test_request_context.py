"""
Tests for Request Context
Tests for request context management
"""

import pytest
from unittest.mock import Mock
from fastapi import Request

from core.infrastructure.request_context import RequestContext


class TestRequestContext:
    """Tests for RequestContext"""
    
    @pytest.fixture
    def request_context(self):
        """Create request context"""
        return RequestContext()
    
    def test_set_request(self, request_context):
        """Test setting request in context"""
        request = Mock(spec=Request)
        request.url = Mock(path="/test")
        
        request_context.set_request(request)
        
        assert request_context.get_request() == request
    
    def test_get_request_id(self, request_context):
        """Test getting request ID"""
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.request_id = "req-123"
        
        request_context.set_request(request)
        
        request_id = request_context.get_request_id()
        
        assert request_id == "req-123"
    
    def test_set_user_id(self, request_context):
        """Test setting user ID in context"""
        request_context.set_user_id("user-123")
        
        user_id = request_context.get_user_id()
        
        assert user_id == "user-123"
    
    def test_get_user_id(self, request_context):
        """Test getting user ID from context"""
        request_context.set_user_id("user-456")
        
        user_id = request_context.get_user_id()
        
        assert user_id == "user-456"
    
    def test_set_metadata(self, request_context):
        """Test setting metadata in context"""
        metadata = {"key": "value", "operation": "analyze"}
        
        request_context.set_metadata(metadata)
        
        context_metadata = request_context.get_metadata()
        
        assert context_metadata["key"] == "value"
        assert context_metadata["operation"] == "analyze"
    
    def test_get_metadata(self, request_context):
        """Test getting metadata from context"""
        request_context.set_metadata({"test": "data"})
        
        metadata = request_context.get_metadata()
        
        assert metadata["test"] == "data"
    
    def test_clear_context(self, request_context):
        """Test clearing request context"""
        request_context.set_user_id("user-123")
        request_context.set_metadata({"key": "value"})
        
        request_context.clear()
        
        assert request_context.get_user_id() is None
        assert request_context.get_metadata() == {}


class TestRequestContextMiddleware:
    """Tests for request context in middleware"""
    
    def test_context_in_middleware(self):
        """Test request context in middleware chain"""
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.request_id = "req-123"
        
        context = RequestContext()
        context.set_request(request)
        
        # Context should be available throughout request
        assert context.get_request() == request
        assert context.get_request_id() == "req-123"
    
    def test_context_isolation(self):
        """Test that contexts are isolated"""
        context1 = RequestContext()
        context2 = RequestContext()
        
        context1.set_user_id("user-1")
        context2.set_user_id("user-2")
        
        assert context1.get_user_id() == "user-1"
        assert context2.get_user_id() == "user-2"
        assert context1.get_user_id() != context2.get_user_id()



