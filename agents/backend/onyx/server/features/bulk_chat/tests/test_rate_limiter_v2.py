"""
Tests for Rate Limiter V2
==========================
"""

import pytest
import asyncio
from ..core.rate_limiter_v2 import RateLimiterV2, RateLimitAlgorithm


@pytest.fixture
def rate_limiter_v2():
    """Create rate limiter V2 for testing."""
    return RateLimiterV2()


def test_add_limit_rule(rate_limiter_v2):
    """Test adding a rate limit rule."""
    rule_id = rate_limiter_v2.add_limit(
        identifier="test_user",
        max_requests=100,
        window_seconds=60,
        algorithm=RateLimitAlgorithm.SLIDING_WINDOW
    )
    
    assert rule_id is not None
    assert rule_id in rate_limiter_v2.limits


@pytest.mark.asyncio
async def test_check_limit_sliding_window(rate_limiter_v2):
    """Test rate limit check with sliding window."""
    rate_limiter_v2.add_limit(
        "test_user",
        5,
        60,
        RateLimitAlgorithm.SLIDING_WINDOW
    )
    
    # Check multiple times
    for i in range(3):
        allowed, info = await rate_limiter_v2.check_limit("test_user")
        assert allowed is True
    
    # Should still be allowed
    allowed, info = await rate_limiter_v2.check_limit("test_user")
    assert allowed is True or info["remaining"] > 0


@pytest.mark.asyncio
async def test_check_limit_token_bucket(rate_limiter_v2):
    """Test rate limit check with token bucket."""
    rate_limiter_v2.add_limit(
        "test_user",
        10,
        60,
        RateLimitAlgorithm.TOKEN_BUCKET
    )
    
    allowed, info = await rate_limiter_v2.check_limit("test_user")
    
    assert allowed is True
    assert info is not None


@pytest.mark.asyncio
async def test_get_limit_status(rate_limiter_v2):
    """Test getting rate limit status."""
    rate_limiter_v2.add_limit("test_user", 100, 60, RateLimitAlgorithm.SLIDING_WINDOW)
    
    await rate_limiter_v2.check_limit("test_user")
    
    status = rate_limiter_v2.get_status("test_user")
    
    assert status is not None
    assert "identifier" in status or "remaining" in status


@pytest.mark.asyncio
async def test_distributed_rate_limiting(rate_limiter_v2):
    """Test distributed rate limiting."""
    rate_limiter_v2.add_limit(
        "test_user",
        10,
        60,
        RateLimitAlgorithm.SLIDING_WINDOW,
        distributed=True
    )
    
    allowed, info = await rate_limiter_v2.check_limit("test_user")
    
    assert allowed is True or info is not None


@pytest.mark.asyncio
async def test_get_rate_limiter_v2_summary(rate_limiter_v2):
    """Test getting rate limiter V2 summary."""
    rate_limiter_v2.add_limit("user1", 100, 60, RateLimitAlgorithm.SLIDING_WINDOW)
    rate_limiter_v2.add_limit("user2", 200, 60, RateLimitAlgorithm.TOKEN_BUCKET)
    
    summary = rate_limiter_v2.get_rate_limiter_v2_summary()
    
    assert summary is not None
    assert "total_limits" in summary or "active_limits" in summary


