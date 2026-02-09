"""
Tests para API Gateway middleware
"""

import pytest
from unittest.mock import Mock, AsyncMock
from fastapi import Request
from starlette.responses import Response

from middleware.api_gateway_middleware import APIGatewayMiddleware


@pytest.fixture
def mock_app():
    """Mock de aplicación FastAPI"""
    app = Mock()
    return app


@pytest.fixture
def aws_gateway_middleware(mock_app):
    """Fixture para APIGatewayMiddleware con AWS"""
    return APIGatewayMiddleware(mock_app, gateway_type="aws")


@pytest.fixture
def kong_gateway_middleware(mock_app):
    """Fixture para APIGatewayMiddleware con Kong"""
    return APIGatewayMiddleware(mock_app, gateway_type="kong")


@pytest.mark.unit
@pytest.mark.middleware
class TestAPIGatewayMiddleware:
    """Tests para APIGatewayMiddleware"""
    
    @pytest.mark.asyncio
    async def test_aws_gateway_extracts_info(self, aws_gateway_middleware):
        """Test de extracción de info de AWS Gateway"""
        request = Mock(spec=Request)
        request.headers = {
            "x-amzn-requestid": "req-123",
            "x-amzn-stage": "prod",
            "x-api-key-id": "key-123"
        }
        request.state = Mock()
        
        response = Response(content=b"test")
        call_next = AsyncMock(return_value=response)
        
        result = await aws_gateway_middleware.dispatch(request, call_next)
        
        assert result.headers.get("X-Gateway-Type") == "aws"
        assert result.headers.get("X-Request-ID") == "req-123"
        assert result.headers.get("X-API-Stage") == "prod"
        assert hasattr(request.state, "gateway_info")
    
    @pytest.mark.asyncio
    async def test_kong_gateway_extracts_info(self, kong_gateway_middleware):
        """Test de extracción de info de Kong Gateway"""
        request = Mock(spec=Request)
        request.headers = {
            "x-kong-request-id": "kong-req-123",
            "x-consumer-id": "consumer-123"
        }
        request.state = Mock()
        
        response = Response(content=b"test")
        call_next = AsyncMock(return_value=response)
        
        result = await kong_gateway_middleware.dispatch(request, call_next)
        
        assert result.headers.get("X-Gateway-Type") == "kong"
        assert hasattr(request.state, "gateway_info")
    
    @pytest.mark.asyncio
    async def test_gateway_middleware_api_key(self, aws_gateway_middleware):
        """Test de manejo de API key"""
        request = Mock(spec=Request)
        request.headers = {"x-api-key": "test-api-key"}
        request.state = Mock()
        
        response = Response(content=b"test")
        call_next = AsyncMock(return_value=response)
        
        result = await aws_gateway_middleware.dispatch(request, call_next)
        
        assert hasattr(request.state, "api_key")
        assert request.state.api_key == "test-api-key"
    
    @pytest.mark.asyncio
    async def test_gateway_middleware_client_id(self, aws_gateway_middleware):
        """Test de manejo de client ID"""
        request = Mock(spec=Request)
        request.headers = {"x-client-id": "client-123"}
        request.state = Mock()
        
        response = Response(content=b"test")
        call_next = AsyncMock(return_value=response)
        
        result = await aws_gateway_middleware.dispatch(request, call_next)
        
        assert hasattr(request.state, "client_id")
        assert request.state.client_id == "client-123"
    
    @pytest.mark.asyncio
    async def test_gateway_middleware_traefik(self, mock_app):
        """Test de Traefik Gateway"""
        middleware = APIGatewayMiddleware(mock_app, gateway_type="traefik")
        
        request = Mock(spec=Request)
        request.headers = {
            "x-request-id": "traefik-req-123",
            "x-forwarded-for": "192.168.1.1"
        }
        request.state = Mock()
        
        response = Response(content=b"test")
        call_next = AsyncMock(return_value=response)
        
        result = await middleware.dispatch(request, call_next)
        
        assert result.headers.get("X-Gateway-Type") == "traefik"
        assert hasattr(request.state, "gateway_info")



