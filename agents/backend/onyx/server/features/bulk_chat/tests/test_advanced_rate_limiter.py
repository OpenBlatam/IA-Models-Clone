"""
Tests for Advanced Rate Limiter
=================================
"""

import pytest
import asyncio
from ..core.advanced_rate_limiter import AdvancedRateLimiter, RateLimitStrategy


@pytest.fixture
def advanced_rate_limiter():
    """Create advanced rate limiter for testing."""
    return AdvancedRateLimiter()


def test_add_limit_rule(advanced_rate_limiter):
    """Test adding a rate limit rule."""
    rule_id = advanced_rate_limiter.add_limit(
        identifier="test_user",
        max_requests=100,
        window_seconds=60,
        strategy=RateLimitStrategy.FIXED_WINDOW
    )
    
    assert rule_id is not None
    assert rule_id in advanced_rate_limiter.limits


@pytest.mark.asyncio
async def test_check_rate_limit_allowed(advanced_rate_limiter):
    """Test rate limit check when allowed."""
    advanced_rate_limiter.add_limit("test_user", 100, 60, RateLimitStrategy.FIXED_WINDOW)
    
    allowed, info = await advanced_rate_limiter.check_limit("test_user")
    
    assert allowed is True
    assert info is not None


@pytest.mark.asyncio
async def test_check_rate_limit_blocked(advanced_rate_limiter):
    """Test rate limit check when blocked."""
    advanced_rate_limiter.add_limit("test_user", 5, 60, RateLimitStrategy.FIXED_WINDOW)
    
    # Exceed limit
    for _ in range(10):
        await advanced_rate_limiter.check_limit("test_user")
    
    allowed, info = await advanced_rate_limiter.check_limit("test_user")
    
    assert allowed is False or info["remaining"] == 0


@pytest.mark.asyncio
async def test_get_rate_limit_status(advanced_rate_limiter):
    """Test getting rate limit status."""
    advanced_rate_limiter.add_limit("test_user", 100, 60, RateLimitStrategy.FIXED_WINDOW)
    
    await advanced_rate_limiter.check_limit("test_user")
    
    status = advanced_rate_limiter.get_status("test_user")
    
    assert status is not None
    assert "identifier" in status or "remaining" in status


@pytest.mark.asyncio
async def test_reset_limit(advanced_rate_limiter):
    """Test resetting a rate limit."""
    advanced_rate_limiter.add_limit("test_user", 5, 60, RateLimitStrategy.FIXED_WINDOW)
    
    # Use up limit
    for _ in range(10):
        await advanced_rate_limiter.check_limit("test_user")
    
    # Reset
    advanced_rate_limiter.reset_limit("test_user")
    
    # Should be allowed again
    allowed, info = await advanced_rate_limiter.check_limit("test_user")
    assert allowed is True


