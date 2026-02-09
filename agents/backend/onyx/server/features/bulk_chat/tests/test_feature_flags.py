"""
Tests for Feature Flags
=======================
"""

import pytest
from ..core.feature_flags import FeatureFlagManager, FeatureFlag, FeatureStatus


@pytest.fixture
def feature_flag_manager():
    """Create feature flag manager for testing."""
    return FeatureFlagManager()


def test_create_feature_flag(feature_flag_manager):
    """Test creating a feature flag."""
    flag_id = feature_flag_manager.create_flag(
        flag_id="test_feature",
        name="Test Feature",
        enabled=True,
        description="Test feature flag"
    )
    
    assert flag_id == "test_feature"
    assert flag_id in feature_flag_manager.flags


def test_check_feature_enabled(feature_flag_manager):
    """Test checking if feature is enabled."""
    feature_flag_manager.create_flag("test_feature", "Test", enabled=True)
    
    is_enabled = feature_flag_manager.is_enabled("test_feature")
    
    assert is_enabled is True


def test_check_feature_disabled(feature_flag_manager):
    """Test checking if feature is disabled."""
    feature_flag_manager.create_flag("test_feature", "Test", enabled=False)
    
    is_enabled = feature_flag_manager.is_enabled("test_feature")
    
    assert is_enabled is False


def test_toggle_feature_flag(feature_flag_manager):
    """Test toggling a feature flag."""
    feature_flag_manager.create_flag("test_feature", "Test", enabled=True)
    
    feature_flag_manager.toggle_flag("test_feature")
    
    assert feature_flag_manager.is_enabled("test_feature") is False


def test_get_feature_flag(feature_flag_manager):
    """Test getting a feature flag."""
    feature_flag_manager.create_flag(
        "test_feature",
        "Test Feature",
        enabled=True,
        metadata={"version": "1.0"}
    )
    
    flag = feature_flag_manager.get_flag("test_feature")
    
    assert flag is not None
    assert flag.name == "Test Feature"
    assert flag.enabled is True


def test_list_feature_flags(feature_flag_manager):
    """Test listing feature flags."""
    feature_flag_manager.create_flag("feature1", "Feature 1")
    feature_flag_manager.create_flag("feature2", "Feature 2")
    
    flags = feature_flag_manager.list_flags()
    
    assert len(flags) >= 2
    assert any(f.flag_id == "feature1" for f in flags)
    assert any(f.flag_id == "feature2" for f in flags)


def test_delete_feature_flag(feature_flag_manager):
    """Test deleting a feature flag."""
    feature_flag_manager.create_flag("test_feature", "Test")
    
    assert "test_feature" in feature_flag_manager.flags
    
    feature_flag_manager.delete_flag("test_feature")
    
    assert "test_feature" not in feature_flag_manager.flags


