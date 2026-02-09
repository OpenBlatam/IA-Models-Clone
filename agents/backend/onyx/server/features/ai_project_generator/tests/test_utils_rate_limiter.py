"""
Tests for RateLimiter utility
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch

from ..utils.rate_limiter import RateLimiter


class TestRateLimiter:
    """Test suite for RateLimiter"""

    def test_init(self):
        """Test RateLimiter initialization"""
        limiter = RateLimiter()
        assert limiter is not None
        assert "default" in limiter.limits
        assert "generate" in limiter.limits
        assert "search" in limiter.limits

    def test_is_allowed_default(self):
        """Test rate limit check when allowed (default endpoint)"""
        limiter = RateLimiter()
        
        # Default limit is 100 requests per hour
        for i in range(10):
            allowed, info = limiter.is_allowed("test_client")
            assert allowed is True
            assert info["remaining"] >= 0

    def test_is_allowed_exceeded(self):
        """Test rate limit check when exceeded"""
        limiter = RateLimiter()
        # Modify limit for testing
        limiter.limits["default"] = {"requests": 5, "window": 3600}
        
        # Make requests up to limit
        for i in range(5):
            allowed, info = limiter.is_allowed("test_client")
            assert allowed is True
        
        # Next request should be blocked
        allowed, info = limiter.is_allowed("test_client")
        assert allowed is False
        assert info["remaining"] == 0

    def test_is_allowed_per_client(self):
        """Test that rate limits are per client"""
        limiter = RateLimiter()
        limiter.limits["default"] = {"requests": 5, "window": 3600}
        
        # Client 1 uses all requests
        for i in range(5):
            limiter.is_allowed("client1")
        
        # Client 2 should still have requests
        allowed, info = limiter.is_allowed("client2")
        assert allowed is True
        assert info["remaining"] == 4

    def test_is_allowed_per_endpoint(self):
        """Test that rate limits are per endpoint"""
        limiter = RateLimiter()
        limiter.limits["default"] = {"requests": 5, "window": 3600}
        limiter.limits["generate"] = {"requests": 2, "window": 3600}
        
        # Use all generate requests
        for i in range(2):
            allowed, _ = limiter.is_allowed("test_client", "generate")
            assert allowed is True
        
        # Generate should be blocked
        allowed, _ = limiter.is_allowed("test_client", "generate")
        assert allowed is False
        
        # But default should still work
        allowed, info = limiter.is_allowed("test_client", "default")
        assert allowed is True

    def test_get_rate_limit_info(self):
        """Test getting rate limit information"""
        limiter = RateLimiter()
        
        # Make some requests
        for i in range(3):
            limiter.is_allowed("test_client")
        
        info = limiter.get_rate_limit_info("test_client")
        assert info["limit"] == 100  # Default limit
        assert info["remaining"] == 97
        assert info["used"] == 3
        assert "window_seconds" in info

    def test_get_rate_limit_info_endpoint(self):
        """Test getting rate limit info for specific endpoint"""
        limiter = RateLimiter()
        
        # Make requests to generate endpoint
        limiter.is_allowed("test_client", "generate")
        limiter.is_allowed("test_client", "generate")
        
        info = limiter.get_rate_limit_info("test_client", "generate")
        assert info["limit"] == 10  # Generate limit
        assert info["used"] == 2
        assert info["remaining"] == 8

    def test_rate_limit_cleanup_old_requests(self):
        """Test that old requests are cleaned up"""
        limiter = RateLimiter()
        limiter.limits["default"] = {"requests": 5, "window": 1}  # 1 second window
        
        # Make requests
        for i in range(3):
            limiter.is_allowed("test_client")
        
        # Wait for window to expire
        time.sleep(1.1)
        
        # Old requests should be cleaned up
        info = limiter.get_rate_limit_info("test_client")
        assert info["used"] == 0
        assert info["remaining"] == 5

