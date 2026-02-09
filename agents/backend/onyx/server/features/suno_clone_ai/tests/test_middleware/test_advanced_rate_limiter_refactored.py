"""
Tests refactorizados para advanced rate limiter
Usando clases base y helpers para eliminar duplicación
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
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestRateLimitConfigRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para RateLimitConfig"""
    
    @pytest.mark.parametrize("rpm,rph,rpd,burst", [
        (60, 1000, 10000, 10),
        (120, 5000, 50000, 20),
        (1000, 100000, 1000000, 100)
    ])
    def test_rate_limit_config(self, rpm, rph, rpd, burst):
        """Test de configuración con diferentes valores"""
        config = RateLimitConfig(
            requests_per_minute=rpm,
            requests_per_hour=rph,
            requests_per_day=rpd,
            burst_size=burst
        )
        
        assert config.requests_per_minute == rpm
        assert config.requests_per_hour == rph
        assert config.requests_per_day == rpd
        assert config.burst_size == burst


class TestAdvancedRateLimiterRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para AdvancedRateLimiter"""
    
    @pytest.fixture
    def rate_limiter(self):
        """Fixture para AdvancedRateLimiter"""
        return AdvancedRateLimiter()
    
    @pytest.mark.parametrize("user_id,user_type,expected_id,expected_type", [
        ("user-123", "premium", "user:user-123", "premium"),
        ("user-456", "default", "user:user-456", "default"),
        (None, "default", "ip:192.168.1.1", "default")
    ])
    def test_get_client_id(self, rate_limiter, user_id, user_type, expected_id, expected_type):
        """Test de obtención de client_id con diferentes escenarios"""
        request = Mock(spec=Request)
        request.headers = {}
        if user_id:
            request.headers["X-User-ID"] = user_id
        request.headers["X-User-Type"] = user_type
        
        if not user_id:
            request.client = Mock()
            request.client.host = "192.168.1.1"
        else:
            request.client = None
        
        client_id, result_type = rate_limiter.get_client_id(request)
        
        assert client_id == expected_id
        assert result_type == expected_type
    
    @pytest.mark.parametrize("user_type,requests_per_minute", [
        ("default", 60),
        ("premium", 120),
        ("admin", 1000)
    ])
    def test_is_rate_limited_by_user_type(self, rate_limiter, user_type, requests_per_minute):
        """Test de rate limiting por tipo de usuario"""
        client_id = f"test-{user_type}"
        config = rate_limiter.configs[user_type]
        
        # Registrar requests hasta el límite
        current_time = time.time()
        for _ in range(config.requests_per_minute):
            rate_limiter.record_request(client_id, current_time)
        
        is_limited, reason = rate_limiter.is_rate_limited(client_id, user_type=user_type)
        
        assert is_limited is True
        assert "minute" in reason.lower()


class TestAdvancedRateLimiterMiddlewareRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para AdvancedRateLimiterMiddleware"""
    
    @pytest.fixture
    def mock_app(self):
        """Mock de aplicación"""
        return Mock()
    
    @pytest.fixture
    def rate_limiter_middleware(self, mock_app):
        """Fixture para AdvancedRateLimiterMiddleware"""
        return AdvancedRateLimiterMiddleware(mock_app)
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("path,should_exclude", [
        ("/health", True),
        ("/metrics", True),
        ("/docs", True),
        ("/api/test", False),
        ("/api/songs", False)
    ])
    async def test_middleware_excluded_paths(self, rate_limiter_middleware, path, should_exclude):
        """Test de paths excluidos"""
        request = Mock(spec=Request)
        request.url.path = path
        request.headers = {}
        
        response = Mock()
        call_next = AsyncMock(return_value=response)
        
        result = await rate_limiter_middleware.dispatch(request, call_next)
        
        assert result == response
        if should_exclude:
            # Paths excluidos no deberían verificar rate limit
            call_next.assert_called_once()



