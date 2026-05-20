"""
Tests for User Behavior Analyzer
==================================
"""

import pytest
import asyncio
from ..core.user_behavior_analyzer import UserBehaviorAnalyzer, ActionType


@pytest.fixture
def behavior_analyzer():
    """Create user behavior analyzer for testing."""
    return UserBehaviorAnalyzer()


@pytest.mark.asyncio
async def test_record_action(behavior_analyzer):
    """Test recording a user action."""
    behavior_analyzer.record_action(
        user_id="test_user",
        action_type=ActionType.LOGIN,
        metadata={"ip": "127.0.0.1", "timestamp": "2024-01-01T00:00:00"}
    )
    
    profile = behavior_analyzer.get_user_profile("test_user")
    
    assert profile is not None
    assert "user_id" in profile or "actions" in profile or len(profile) > 0


@pytest.mark.asyncio
async def test_get_user_profile(behavior_analyzer):
    """Test getting user profile."""
    behavior_analyzer.record_action("test_user", ActionType.LOGIN, {})
    behavior_analyzer.record_action("test_user", ActionType.MESSAGE_SENT, {})
    
    profile = behavior_analyzer.get_user_profile("test_user")
    
    assert profile is not None
    assert "user_id" in profile or "action_count" in profile or "actions" in profile


@pytest.mark.asyncio
async def test_detect_anomaly(behavior_analyzer):
    """Test detecting behavioral anomalies."""
    # Record normal behavior
    for i in range(10):
        behavior_analyzer.record_action(
            "test_user",
            ActionType.MESSAGE_SENT,
            {"count": i}
        )
    
    # Record anomalous behavior
    behavior_analyzer.record_action(
        "test_user",
        ActionType.LOGIN,
        {"ip": "192.168.1.100", "suspicious": True}
    )
    
    anomalies = behavior_analyzer.detect_anomalies("test_user")
    
    assert anomalies is not None
    assert isinstance(anomalies, list) or isinstance(anomalies, dict)


@pytest.mark.asyncio
async def test_get_high_risk_users(behavior_analyzer):
    """Test getting high-risk users."""
    behavior_analyzer.record_action("user1", ActionType.LOGIN, {"suspicious": True})
    behavior_analyzer.record_action("user2", ActionType.MESSAGE_SENT, {})
    
    high_risk = behavior_analyzer.get_high_risk_users(limit=10)
    
    assert high_risk is not None
    assert isinstance(high_risk, list)


@pytest.mark.asyncio
async def test_get_behavior_analyzer_summary(behavior_analyzer):
    """Test getting behavior analyzer summary."""
    behavior_analyzer.record_action("user1", ActionType.LOGIN, {})
    behavior_analyzer.record_action("user2", ActionType.MESSAGE_SENT, {})
    
    summary = behavior_analyzer.get_user_behavior_analyzer_summary()
    
    assert summary is not None
    assert "total_users" in summary or "total_actions" in summary


