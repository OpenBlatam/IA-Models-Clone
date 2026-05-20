"""
Tests for Rate Limiter
======================
"""

import pytest
import asyncio
from ..core.rate_limiter import RateLimiter, RateLimitConfig


@pytest.fixture
def rate_limiter():
    """Create rate limiter for testing."""
    config = RateLimitConfig(
        max_requests=10,
        time_window=60.0
    )
    return RateLimiter(config)


@pytest.mark.asyncio
async def test_check_rate_limit_allowed(rate_limiter):
    """Test rate limit check when allowed."""
    identifier = "test_user"
    
    for i in range(5):
        allowed, info = await rate_limiter.check_rate_limit(identifier)
        assert allowed is True
        assert info["remaining"] >= 0


@pytest.mark.asyncio
async def test_check_rate_limit_exceeded(rate_limiter):
    """Test rate limit check when exceeded."""
    identifier = "test_user"
    
    # Exceed limit
    for i in range(15):
        await rate_limiter.check_rate_limit(identifier)
    
    # Should be blocked
    allowed, info = await rate_limiter.check_rate_limit(identifier)
    assert allowed is False
    assert info["remaining"] == 0


@pytest.mark.asyncio
async def test_rate_limit_reset(rate_limiter):
    """Test rate limit reset."""
    identifier = "test_user"
    
    # Exceed limit
    for i in range(15):
        await rate_limiter.check_rate_limit(identifier)
    
    # Reset
    await rate_limiter.reset_limit(identifier)
    
    # Should be allowed again
    allowed, info = await rate_limiter.check_rate_limit(identifier)
    assert allowed is True


@pytest.mark.asyncio
async def test_get_rate_limit_status(rate_limiter):
    """Test getting rate limit status."""
    identifier = "test_user"
    
    await rate_limiter.check_rate_limit(identifier)
    
    status = rate_limiter.get_status(identifier)
    
    assert status["identifier"] == identifier
    assert "remaining" in status
    assert "reset_at" in status


@pytest.mark.asyncio
async def test_rate_limit_stats(rate_limiter):
    """Test rate limit statistics."""
    await rate_limiter.check_rate_limit("user1")
    await rate_limiter.check_rate_limit("user2")
    
    stats = rate_limiter.get_stats()
    
    assert stats["total_checks"] >= 2
    assert "allowed" in stats
    assert "blocked" in stats


