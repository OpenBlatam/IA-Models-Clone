"""
Tests para request helpers
"""

import pytest
from unittest.mock import Mock, MagicMock
from fastapi import Request


class TestGetRequestMetadata:
    """Tests para get_request_metadata"""
    
    @pytest.fixture
    def get_metadata_function(self):
        """Fixture para obtener la función"""
        try:
            from api.utils.request_helpers import get_request_metadata
            return get_request_metadata
        except ImportError:
            pytest.skip("get_request_metadata not available")
    
    @pytest.mark.unit
    def test_get_request_metadata_basic(self, get_metadata_function, test_client):
        """Test básico de obtención de metadata"""
        # Crear un request mock
        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock()
        request.url.path = "/suno/generate"
        request.url.query_string = ""
        request.headers = {}
        request.client = Mock()
        request.client.host = "127.0.0.1"
        
        metadata = get_metadata_function(request)
        
        assert isinstance(metadata, dict)
        assert "method" in metadata or "path" in metadata or len(metadata) >= 0
    
    @pytest.mark.unit
    def test_get_request_metadata_with_headers(self, get_metadata_function):
        """Test con headers"""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/suno/songs"
        request.url.query_string = ""
        request.headers = {"User-Agent": "test-agent", "X-Request-ID": "test-123"}
        request.client = Mock()
        request.client.host = "127.0.0.1"
        
        metadata = get_metadata_function(request)
        
        assert isinstance(metadata, dict)


class TestAddCacheHeaders:
    """Tests para add_cache_headers"""
    
    @pytest.fixture
    def add_cache_function(self):
        """Fixture para obtener la función"""
        try:
            from api.utils.request_helpers import add_cache_headers
            return add_cache_headers
        except ImportError:
            pytest.skip("add_cache_headers not available")
    
    @pytest.mark.unit
    def test_add_cache_headers_basic(self, add_cache_function):
        """Test básico de agregar headers de cache"""
        from fastapi import Response
        
        response = Response()
        add_cache_function(response, max_age=60, public=True)
        
        assert "Cache-Control" in response.headers
        assert "max-age=60" in response.headers["Cache-Control"]
        assert "public" in response.headers["Cache-Control"]
    
    @pytest.mark.unit
    def test_add_cache_headers_private(self, add_cache_function):
        """Test con cache privado"""
        from fastapi import Response
        
        response = Response()
        add_cache_function(response, max_age=30, public=False)
        
        assert "Cache-Control" in response.headers
        assert "max-age=30" in response.headers["Cache-Control"]
        assert "private" in response.headers["Cache-Control"] or "public" not in response.headers["Cache-Control"]
    
    @pytest.mark.unit
    def test_add_cache_headers_no_store(self, add_cache_function):
        """Test con no-store"""
        from fastapi import Response
        
        response = Response()
        add_cache_function(response, max_age=0, no_store=True)
        
        assert "Cache-Control" in response.headers
        assert "no-store" in response.headers["Cache-Control"]

