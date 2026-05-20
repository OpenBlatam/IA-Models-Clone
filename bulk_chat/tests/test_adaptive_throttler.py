"""
Tests for Adaptive Throttler
============================
"""

import pytest
import asyncio
from ..core.adaptive_throttler import AdaptiveThrottler, ThrottleStrategy


@pytest.fixture
def throttler():
    """Create adaptive throttler for testing."""
    return AdaptiveThrottler()


@pytest.mark.asyncio
async def test_add_rule(throttler):
    """Test adding a throttle rule."""
    rule_id = throttler.add_rule(
        rule_id="test_rule",
        identifier="test_user",
        base_limit=100,
        strategy=ThrottleStrategy.ADAPTIVE
    )
    
    assert rule_id == "test_rule"
    assert "test_rule" in throttler.rules


@pytest.mark.asyncio
async def test_record_metric(throttler):
    """Test recording metrics."""
    throttler.add_rule("test_rule", "test_user", 100)
    
    throttler.record_metric(
        rule_id="test_rule",
        success_rate=0.95,
        error_rate=0.05,
        response_time=0.5,
        system_load=0.7
    )
    
    # Wait for async adjustment
    await asyncio.sleep(0.1)
    
    metrics = throttler.metrics.get("test_rule")
    assert len(metrics) > 0


@pytest.mark.asyncio
async def test_check_throttle_allowed(throttler):
    """Test throttle check when allowed."""
    throttler.add_rule("test_rule", "test_user", 100)
    
    allowed, info = await throttler.check_throttle(
        rule_id="test_rule",
        identifier="test_user"
    )
    
    assert allowed is True
    assert info is not None


@pytest.mark.asyncio
async def test_get_throttle_status(throttler):
    """Test getting throttle status."""
    throttler.add_rule(
        "test_rule",
        "test_user",
        base_limit=100,
        min_limit=10,
        max_limit=1000
    )
    
    status = throttler.get_throttle_status("test_rule")
    
    assert status["rule_id"] == "test_rule"
    assert status["base_limit"] == 100
    assert status["min_limit"] == 10
    assert status["max_limit"] == 1000


@pytest.mark.asyncio
async def test_get_throttler_summary(throttler):
    """Test getting throttler summary."""
    throttler.add_rule("rule_1", "user_1", 100)
    throttler.add_rule("rule_2", "user_2", 200)
    
    summary = throttler.get_adaptive_throttler_summary()
    
    assert summary["total_rules"] >= 2

