"""
Tests para helpers de requests
"""

import pytest
from unittest.mock import Mock, MagicMock
from fastapi import Request, Response

from api.utils.request_helpers import (
    get_client_ip,
    get_user_agent,
    get_accept_encoding,
    add_cache_headers,
    add_cors_headers
)


@pytest.mark.unit
@pytest.mark.api
class TestGetClientIP:
    """Tests para get_client_ip"""
    
    def test_get_client_ip_from_forwarded_for(self):
        """Test de obtención de IP desde X-Forwarded-For"""
        request = Mock(spec=Request)
        request.headers = {"X-Forwarded-For": "192.168.1.1, 10.0.0.1"}
        request.client = None
        
        ip = get_client_ip(request)
        
        assert ip == "192.168.1.1"
    
    def test_get_client_ip_from_real_ip(self):
        """Test de obtención de IP desde X-Real-IP"""
        request = Mock(spec=Request)
        request.headers = {"X-Real-IP": "192.168.1.2"}
        request.client = None
        
        ip = get_client_ip(request)
        
        assert ip == "192.168.1.2"
    
    def test_get_client_ip_from_client(self):
        """Test de obtención de IP desde client"""
        request = Mock(spec=Request)
        request.headers = {}
        request.client = Mock()
        request.client.host = "192.168.1.3"
        
        ip = get_client_ip(request)
        
        assert ip == "192.168.1.3"
    
    def test_get_client_ip_unknown(self):
        """Test de IP desconocida"""
        request = Mock(spec=Request)
        request.headers = {}
        request.client = None
        
        ip = get_client_ip(request)
        
        assert ip == "unknown"
    
    def test_get_client_ip_priority(self):
        """Test de prioridad de headers"""
        request = Mock(spec=Request)
        request.headers = {
            "X-Forwarded-For": "192.168.1.1",
            "X-Real-IP": "192.168.1.2"
        }
        request.client = Mock()
        request.client.host = "192.168.1.3"
        
        ip = get_client_ip(request)
        
        # X-Forwarded-For tiene prioridad
        assert ip == "192.168.1.1"


@pytest.mark.unit
@pytest.mark.api
class TestGetUserAgent:
    """Tests para get_user_agent"""
    
    def test_get_user_agent_from_header(self):
        """Test de obtención de User-Agent desde header"""
        request = Mock(spec=Request)
        request.headers = {"User-Agent": "Mozilla/5.0"}
        
        ua = get_user_agent(request)
        
        assert ua == "Mozilla/5.0"
    
    def test_get_user_agent_unknown(self):
        """Test de User-Agent desconocido"""
        request = Mock(spec=Request)
        request.headers = {}
        
        ua = get_user_agent(request)
        
        assert ua == "unknown"


@pytest.mark.unit
@pytest.mark.api
class TestGetAcceptEncoding:
    """Tests para get_accept_encoding"""
    
    def test_get_accept_encoding_from_header(self):
        """Test de obtención de Accept-Encoding desde header"""
        request = Mock(spec=Request)
        request.headers = {"Accept-Encoding": "gzip, br"}
        
        encoding = get_accept_encoding(request)
        
        assert encoding == "gzip, br"
    
    def test_get_accept_encoding_none(self):
        """Test de Accept-Encoding None"""
        request = Mock(spec=Request)
        request.headers = {}
        
        encoding = get_accept_encoding(request)
        
        assert encoding is None


@pytest.mark.unit
@pytest.mark.api
class TestAddCacheHeaders:
    """Tests para add_cache_headers"""
    
    def test_add_cache_headers_public(self):
        """Test de agregar headers de cache públicos"""
        response = Response()
        add_cache_headers(response, max_age=60, public=True)
        
        cache_control = response.headers.get("Cache-Control")
        assert "public" in cache_control
        assert "max-age=60" in cache_control
    
    def test_add_cache_headers_private(self):
        """Test de agregar headers de cache privados"""
        response = Response()
        add_cache_headers(response, max_age=120, public=False)
        
        cache_control = response.headers.get("Cache-Control")
        assert "private" in cache_control
        assert "max-age=120" in cache_control
    
    def test_add_cache_headers_must_revalidate(self):
        """Test de agregar must-revalidate"""
        response = Response()
        add_cache_headers(response, max_age=60, must_revalidate=True)
        
        cache_control = response.headers.get("Cache-Control")
        assert "must-revalidate" in cache_control
    
    def test_add_cache_headers_custom_max_age(self):
        """Test de max_age personalizado"""
        response = Response()
        add_cache_headers(response, max_age=300)
        
        cache_control = response.headers.get("Cache-Control")
        assert "max-age=300" in cache_control


@pytest.mark.unit
@pytest.mark.api
class TestAddCorsHeaders:
    """Tests para add_cors_headers"""
    
    def test_add_cors_headers_default(self):
        """Test de agregar headers CORS por defecto"""
        response = Response()
        add_cors_headers(response)
        
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
    
    def test_add_cors_headers_custom_origin(self):
        """Test de origin personalizado"""
        response = Response()
        add_cors_headers(response, origin="https://example.com")
        
        assert response.headers["Access-Control-Allow-Origin"] == "https://example.com"
    
    def test_add_cors_headers_custom_methods(self):
        """Test de métodos personalizados"""
        response = Response()
        add_cors_headers(response, allow_methods="GET, POST")
        
        assert response.headers["Access-Control-Allow-Methods"] == "GET, POST"
    
    def test_add_cors_headers_custom_headers(self):
        """Test de headers personalizados"""
        response = Response()
        add_cors_headers(response, allow_headers="Content-Type, Authorization")
        
        assert response.headers["Access-Control-Allow-Headers"] == "Content-Type, Authorization"



