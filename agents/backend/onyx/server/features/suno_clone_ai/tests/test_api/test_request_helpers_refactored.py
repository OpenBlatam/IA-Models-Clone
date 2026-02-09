"""
Tests refactorizados para helpers de requests
Usando clases base y helpers para eliminar duplicación
"""

import pytest
from unittest.mock import Mock
from fastapi import Request, Response

from api.utils.request_helpers import (
    get_client_ip,
    get_user_agent,
    get_accept_encoding,
    add_cache_headers,
    add_cors_headers
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestGetClientIPRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para get_client_ip"""
    
    @pytest.mark.parametrize("headers,client_host,expected", [
        ({"X-Forwarded-For": "192.168.1.1, 10.0.0.1"}, None, "192.168.1.1"),
        ({"X-Real-IP": "192.168.1.2"}, None, "192.168.1.2"),
        ({}, "192.168.1.3", "192.168.1.3"),
        ({}, None, "unknown")
    ])
    def test_get_client_ip(self, headers, client_host, expected):
        """Test de obtención de IP con diferentes escenarios"""
        request = Mock(spec=Request)
        request.headers = headers
        if client_host:
            request.client = Mock()
            request.client.host = client_host
        else:
            request.client = None
        
        ip = get_client_ip(request)
        
        assert ip == expected
    
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


class TestGetUserAgentRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para get_user_agent"""
    
    @pytest.mark.parametrize("headers,expected", [
        ({"User-Agent": "Mozilla/5.0"}, "Mozilla/5.0"),
        ({}, "unknown")
    ])
    def test_get_user_agent(self, headers, expected):
        """Test de obtención de User-Agent"""
        request = Mock(spec=Request)
        request.headers = headers
        
        ua = get_user_agent(request)
        
        assert ua == expected


class TestGetAcceptEncodingRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para get_accept_encoding"""
    
    @pytest.mark.parametrize("headers,expected", [
        ({"Accept-Encoding": "gzip, br"}, "gzip, br"),
        ({}, None)
    ])
    def test_get_accept_encoding(self, headers, expected):
        """Test de obtención de Accept-Encoding"""
        request = Mock(spec=Request)
        request.headers = headers
        
        encoding = get_accept_encoding(request)
        
        assert encoding == expected


class TestAddCacheHeadersRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para add_cache_headers"""
    
    @pytest.mark.parametrize("public,must_revalidate,expected_parts", [
        (True, False, ["public", "max-age=60"]),
        (False, False, ["private", "max-age=60"]),
        (True, True, ["public", "max-age=60", "must-revalidate"])
    ])
    def test_add_cache_headers(self, public, must_revalidate, expected_parts):
        """Test de agregar headers de cache"""
        response = Response()
        add_cache_headers(response, max_age=60, public=public, must_revalidate=must_revalidate)
        
        cache_control = response.headers.get("Cache-Control")
        for part in expected_parts:
            assert part in cache_control
    
    def test_add_cache_headers_custom_max_age(self):
        """Test de max_age personalizado"""
        response = Response()
        add_cache_headers(response, max_age=300)
        
        cache_control = response.headers.get("Cache-Control")
        assert "max-age=300" in cache_control


class TestAddCorsHeadersRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para add_cors_headers"""
    
    def test_add_cors_headers_default(self):
        """Test de agregar headers CORS por defecto"""
        response = Response()
        add_cors_headers(response)
        
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
    
    @pytest.mark.parametrize("origin,methods,headers", [
        ("https://example.com", None, None),
        (None, "GET, POST", None),
        (None, None, "Content-Type, Authorization")
    ])
    def test_add_cors_headers_custom(self, origin, methods, headers):
        """Test de headers CORS personalizados"""
        response = Response()
        add_cors_headers(response, origin=origin, allow_methods=methods, allow_headers=headers)
        
        if origin:
            assert response.headers["Access-Control-Allow-Origin"] == origin
        if methods:
            assert response.headers["Access-Control-Allow-Methods"] == methods
        if headers:
            assert response.headers["Access-Control-Allow-Headers"] == headers



