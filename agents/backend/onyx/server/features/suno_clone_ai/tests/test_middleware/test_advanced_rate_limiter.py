"""
Tests para advanced rate limiter
"""

import pytest
import time
from unittest.mock import Mock, AsyncMock
from fastapi import Request, HTTPException, status

from middleware.advanced_rate_limiter import (
    RateLimitConfig,
    AdvancedRateLimiter,
    AdvancedRateLimiterMiddleware
)


@pytest.mark.unit
@pytest.mark.middleware
class TestRateLimitConfig:
    """Tests para RateLimitConfig"""
    
    def test_rate_limit_config_default(self):
        """Test de configuración por defecto"""
        config = RateLimitConfig()
        
        assert config.requests_per_minute == 60
        assert config.requests_per_hour == 1000
        assert config.requests_per_day == 10000
        assert config.burst_size == 10
    
    def test_rate_limit_config_custom(self):
        """Test de configuración personalizada"""
        config = RateLimitConfig(
            requests_per_minute=120,
            requests_per_hour=5000,
            requests_per_day=50000,
            burst_size=20
        )
        
        assert config.requests_per_minute == 120
        assert config.requests_per_hour == 5000
        assert config.requests_per_day == 50000
        assert config.burst_size == 20


@pytest.mark.unit
@pytest.mark.middleware
class TestAdvancedRateLimiter:
    """Tests para AdvancedRateLimiter"""
    
    @pytest.fixture
    def rate_limiter(self):
        """Fixture para AdvancedRateLimiter"""
        return AdvancedRateLimiter()
    
    def test_get_client_id_from_user_id(self, rate_limiter):
        """Test de obtención de client_id desde user_id"""
        request = Mock(spec=Request)
        request.headers = {"X-User-ID": "user-123", "X-User-Type": "premium"}
        request.client = None
        
        client_id, user_type = rate_limiter.get_client_id(request)
        
        assert client_id == "user:user-123"
        assert user_type == "premium"
    
    def test_get_client_id_from_ip(self, rate_limiter):
        """Test de obtención de client_id desde IP"""
        request = Mock(spec=Request)
        request.headers = {}
        request.client = Mock()
        request.client.host = "192.168.1.1"
        
        client_id, user_type = rate_limiter.get_client_id(request)
        
        assert client_id == "ip:192.168.1.1"
        assert user_type == "default"
    
    def test_is_rate_limited_not_limited(self, rate_limiter):
        """Test de no estar limitado"""
        client_id = "test-client"
        is_limited, reason = rate_limiter.is_rate_limited(client_id)
        
        assert is_limited is False
        assert reason is None
    
    def test_is_rate_limited_minute_exceeded(self, rate_limiter):
        """Test de exceder límite por minuto"""
        client_id = "test-client-2"
        config = rate_limiter.configs["default"]
        
        # Registrar requests hasta el límite
        current_time = time.time()
        for _ in range(config.requests_per_minute):
            rate_limiter.record_request(client_id, current_time)
        
        # El siguiente debería estar limitado
        is_limited, reason = rate_limiter.is_rate_limited(client_id)
        
        assert is_limited is True
        assert "minute" in reason.lower()
    
    def test_is_rate_limited_premium_user(self, rate_limiter):
        """Test de límites para usuario premium"""
        client_id = "premium-user"
        premium_config = rate_limiter.configs["premium"]
        
        # Registrar requests hasta el límite premium
        current_time = time.time()
        for _ in range(premium_config.requests_per_minute):
            rate_limiter.record_request(client_id, current_time)
        
        is_limited, reason = rate_limiter.is_rate_limited(client_id, user_type="premium")
        
        assert is_limited is True
    
    def test_get_remaining_requests(self, rate_limiter):
        """Test de obtención de requests restantes"""
        client_id = "test-client-3"
        config = rate_limiter.configs["default"]
        
        # Registrar algunos requests
        current_time = time.time()
        for _ in range(10):
            rate_limiter.record_request(client_id, current_time)
        
        remaining = rate_limiter.get_remaining_requests(client_id)
        
        assert remaining["minute"] == config.requests_per_minute - 10
        assert remaining["hour"] == config.requests_per_hour - 10
        assert remaining["day"] == config.requests_per_day - 10


@pytest.mark.unit
@pytest.mark.middleware
class TestAdvancedRateLimiterMiddleware:
    """Tests para AdvancedRateLimiterMiddleware"""
    
    @pytest.fixture
    def mock_app(self):
        """Mock de aplicación"""
        return Mock()
    
    @pytest.fixture
    def rate_limiter_middleware(self, mock_app):
        """Fixture para AdvancedRateLimiterMiddleware"""
        return AdvancedRateLimiterMiddleware(mock_app)
    
    @pytest.mark.asyncio
    async def test_middleware_allows_excluded_paths(self, rate_limiter_middleware):
        """Test de permitir paths excluidos"""
        request = Mock(spec=Request)
        request.url.path = "/health"
        request.headers = {}
        
        response = Mock()
        call_next = AsyncMock(return_value=response)
        
        result = await rate_limiter_middleware.dispatch(request, call_next)
        
        assert result == response
        call_next.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_middleware_allows_normal_request(self, rate_limiter_middleware):
        """Test de permitir request normal"""
        request = Mock(spec=Request)
        request.url.path = "/api/test"
        request.headers = {}
        request.client = Mock()
        request.client.host = "192.168.1.1"
        
        response = Mock()
        call_next = AsyncMock(return_value=response)
        
        result = await rate_limiter_middleware.dispatch(request, call_next)
        
        assert result == response
    
    @pytest.mark.asyncio
    async def test_middleware_blocks_rate_limited(self, rate_limiter_middleware):
        """Test de bloquear request con rate limit excedido"""
        request = Mock(spec=Request)
        request.url.path = "/api/test"
        request.headers = {}
        request.client = Mock()
        request.client.host = "192.168.1.1"
        
        client_id = "ip:192.168.1.1"
        config = rate_limiter_middleware.rate_limiter.configs["default"]
        
        # Llenar rate limit
        current_time = time.time()
        for _ in range(config.requests_per_minute):
            rate_limiter_middleware.rate_limiter.record_request(client_id, current_time)
        
        call_next = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await rate_limiter_middleware.dispatch(request, call_next)
        
        assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS



