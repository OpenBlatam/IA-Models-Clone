"""
Tests for rate limiter.
"""

import pytest
import asyncio
from piel_mejorador_ai_sam3.core.rate_limiter import RateLimiter, RateLimitConfig


@pytest.mark.asyncio
class TestRateLimiter:
    """Tests for RateLimiter."""
    
    async def test_rate_limiter_allows_requests(self):
        """Test that rate limiter allows requests within limit."""
        limiter = RateLimiter(
            default_config=RateLimitConfig(
                requests_per_second=10.0,
                burst_size=20
            )
        )
        
        # Should allow multiple requests
        for _ in range(10):
            allowed = await limiter.is_allowed("test_client")
            assert allowed is True
    
    async def test_rate_limiter_blocks_excessive_requests(self):
        """Test that rate limiter blocks excessive requests."""
        limiter = RateLimiter(
            default_config=RateLimitConfig(
                requests_per_second=1.0,
                burst_size=2
            )
        )
        
        # First 2 should be allowed (burst)
        assert await limiter.is_allowed("test_client") is True
        assert await limiter.is_allowed("test_client") is True
        
        # Next should be blocked (rate limit)
        assert await limiter.is_allowed("test_client") is False
    
    async def test_get_wait_time(self):
        """Test getting wait time."""
        limiter = RateLimiter(
            default_config=RateLimitConfig(
                requests_per_second=1.0,
                burst_size=1
            )
        )
        
        # Consume token
        await limiter.is_allowed("test_client")
        
        # Should need to wait
        wait_time = await limiter.get_wait_time("test_client")
        assert wait_time > 0
    
    async def test_get_stats(self):
        """Test getting statistics."""
        limiter = RateLimiter()
        
        await limiter.is_allowed("client1")
        await limiter.is_allowed("client2")
        
        stats = limiter.get_stats()
        
        assert "total_requests" in stats
        assert "allowed_requests" in stats
        assert "rate_limited_requests" in stats
        assert stats["total_requests"] >= 2




