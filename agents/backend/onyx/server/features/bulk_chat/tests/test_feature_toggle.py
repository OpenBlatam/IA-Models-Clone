"""
Tests for Feature Toggle Manager
==================================
"""

import pytest
from ..core.feature_toggle import FeatureToggleManager, ToggleStatus


@pytest.fixture
def feature_toggle_manager():
    """Create feature toggle manager for testing."""
    return FeatureToggleManager()


def test_create_toggle(feature_toggle_manager):
    """Test creating a feature toggle."""
    toggle_id = feature_toggle_manager.create_toggle(
        toggle_id="test_toggle",
        name="Test Toggle",
        enabled=True,
        rollout_percentage=100.0
    )
    
    assert toggle_id == "test_toggle"
    assert toggle_id in feature_toggle_manager.toggles


def test_check_toggle_enabled(feature_toggle_manager):
    """Test checking if toggle is enabled."""
    feature_toggle_manager.create_toggle("test_toggle", "Test", enabled=True)
    
    is_enabled = feature_toggle_manager.is_enabled("test_toggle", user_id="user1")
    
    assert is_enabled is True


def test_check_toggle_disabled(feature_toggle_manager):
    """Test checking if toggle is disabled."""
    feature_toggle_manager.create_toggle("test_toggle", "Test", enabled=False)
    
    is_enabled = feature_toggle_manager.is_enabled("test_toggle", user_id="user1")
    
    assert is_enabled is False


def test_gradual_rollout(feature_toggle_manager):
    """Test gradual rollout."""
    feature_toggle_manager.create_toggle(
        "test_toggle",
        "Test",
        enabled=True,
        rollout_percentage=50.0
    )
    
    # Should be enabled for 50% of users
    enabled_count = 0
    for i in range(100):
        if feature_toggle_manager.is_enabled("test_toggle", user_id=f"user{i}"):
            enabled_count += 1
    
    # Should be approximately 50% (with some variance)
    assert 30 <= enabled_count <= 70


def test_user_targeting(feature_toggle_manager):
    """Test user targeting."""
    feature_toggle_manager.create_toggle(
        "test_toggle",
        "Test",
        enabled=True,
        target_users=["user1", "user2"]
    )
    
    assert feature_toggle_manager.is_enabled("test_toggle", "user1") is True
    assert feature_toggle_manager.is_enabled("test_toggle", "user2") is True
    assert feature_toggle_manager.is_enabled("test_toggle", "user3") is False


def test_get_toggle_status(feature_toggle_manager):
    """Test getting toggle status."""
    feature_toggle_manager.create_toggle(
        "test_toggle",
        "Test Toggle",
        enabled=True,
        rollout_percentage=75.0
    )
    
    status = feature_toggle_manager.get_toggle_status("test_toggle")
    
    assert status is not None
    assert status["toggle_id"] == "test_toggle"
    assert status["enabled"] is True


