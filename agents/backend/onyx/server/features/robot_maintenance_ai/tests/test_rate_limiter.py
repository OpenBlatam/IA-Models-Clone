"""
Tests for rate limiter.
"""

import time
import pytest
from ..utils.rate_limiter import RateLimiter


class TestRateLimiter:
    """Test cases for RateLimiter."""
    
    def test_rate_limit_allowed(self):
        """Test that requests within limit are allowed."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        identifier = "test_client"
        
        for i in range(5):
            allowed, _ = limiter.is_allowed(identifier)
            assert allowed is True
    
    def test_rate_limit_exceeded(self):
        """Test that exceeding limit is blocked."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        identifier = "test_client"
        
        # Make 3 requests (should all be allowed)
        for i in range(3):
            allowed, _ = limiter.is_allowed(identifier)
            assert allowed is True
        
        # 4th request should be blocked
        allowed, error_msg = limiter.is_allowed(identifier)
        assert allowed is False
        assert error_msg is not None
        assert "Rate limit exceeded" in error_msg
    
    def test_rate_limit_reset(self):
        """Test resetting rate limit."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        identifier = "test_client"
        
        # Use up limit
        limiter.is_allowed(identifier)
        limiter.is_allowed(identifier)
        
        # Should be blocked
        allowed, _ = limiter.is_allowed(identifier)
        assert allowed is False
        
        # Reset
        limiter.reset(identifier)
        
        # Should be allowed again
        allowed, _ = limiter.is_allowed(identifier)
        assert allowed is True
    
    def test_get_remaining(self):
        """Test getting remaining requests."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        identifier = "test_client"
        
        assert limiter.get_remaining(identifier) == 5
        
        limiter.is_allowed(identifier)
        assert limiter.get_remaining(identifier) == 4
        
        limiter.is_allowed(identifier)
        assert limiter.get_remaining(identifier) == 3
    
    def test_window_expiration(self):
        """Test that old requests expire after window."""
        limiter = RateLimiter(max_requests=2, window_seconds=1)  # 1 second window
        identifier = "test_client"
        
        # Use up limit
        limiter.is_allowed(identifier)
        limiter.is_allowed(identifier)
        
        # Should be blocked
        allowed, _ = limiter.is_allowed(identifier)
        assert allowed is False
        
        # Wait for window to expire
        time.sleep(1.1)
        
        # Should be allowed again
        allowed, _ = limiter.is_allowed(identifier)
        assert allowed is True






