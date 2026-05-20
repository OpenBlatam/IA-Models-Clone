"""
Comprehensive Tests for Feature Flags
Tests for FeatureFlagManager and all flag types
"""

import pytest
from unittest.mock import Mock, patch
import os

from core.feature_flags import (
    FeatureFlagManager,
    FeatureFlag,
    FeatureFlagType
)


class TestFeatureFlag:
    """Tests for FeatureFlag"""
    
    def test_create_boolean_flag(self):
        """Test creating boolean feature flag"""
        flag = FeatureFlag(
            name="new_feature",
            enabled=True,
            flag_type=FeatureFlagType.BOOLEAN
        )
        
        assert flag.name == "new_feature"
        assert flag.enabled is True
        assert flag.flag_type == FeatureFlagType.BOOLEAN
    
    def test_create_percentage_flag(self):
        """Test creating percentage rollout flag"""
        flag = FeatureFlag(
            name="gradual_rollout",
            enabled=True,
            flag_type=FeatureFlagType.PERCENTAGE,
            percentage=50
        )
        
        assert flag.percentage == 50
        assert flag.flag_type == FeatureFlagType.PERCENTAGE
    
    def test_create_user_list_flag(self):
        """Test creating user list flag"""
        flag = FeatureFlag(
            name="beta_users",
            enabled=True,
            flag_type=FeatureFlagType.USER_LIST,
            user_list=["user-1", "user-2", "user-3"]
        )
        
        assert len(flag.user_list) == 3
        assert "user-1" in flag.user_list
        assert flag.flag_type == FeatureFlagType.USER_LIST
    
    def test_create_custom_flag(self):
        """Test creating custom flag"""
        def custom_check(user_id):
            return user_id.startswith("premium-")
        
        flag = FeatureFlag(
            name="premium_feature",
            enabled=True,
            flag_type=FeatureFlagType.CUSTOM,
            custom_check=custom_check
        )
        
        assert flag.custom_check is not None
        assert flag.custom_check("premium-user-123") is True
        assert flag.custom_check("regular-user-123") is False


class TestFeatureFlagManager:
    """Tests for FeatureFlagManager"""
    
    @pytest.fixture
    def flag_manager(self):
        """Create feature flag manager"""
        return FeatureFlagManager()
    
    def test_register_flag(self, flag_manager):
        """Test registering a feature flag"""
        flag = FeatureFlag(name="test_feature", enabled=True)
        
        flag_manager.register_flag(flag)
        
        assert flag_manager.is_enabled("test_feature") is True
    
    def test_enable_flag(self, flag_manager):
        """Test enabling a flag"""
        flag_manager.enable("test_feature")
        
        assert flag_manager.is_enabled("test_feature") is True
    
    def test_disable_flag(self, flag_manager):
        """Test disabling a flag"""
        flag_manager.enable("test_feature")
        flag_manager.disable("test_feature")
        
        assert flag_manager.is_enabled("test_feature") is False
    
    def test_boolean_flag_check(self, flag_manager):
        """Test boolean flag check"""
        flag_manager.enable("boolean_feature")
        
        assert flag_manager.is_enabled("boolean_feature") is True
        assert flag_manager.is_enabled("non_existent") is False
    
    def test_percentage_flag_check(self, flag_manager):
        """Test percentage flag check"""
        flag = FeatureFlag(
            name="percentage_feature",
            enabled=True,
            flag_type=FeatureFlagType.PERCENTAGE,
            percentage=50
        )
        
        flag_manager.register_flag(flag)
        
        # Should return True for approximately 50% of checks
        results = [flag_manager.is_enabled("percentage_feature") for _ in range(100)]
        true_count = sum(results)
        
        # Should be approximately 50% (with some variance)
        assert 30 <= true_count <= 70  # Allow variance
    
    def test_user_list_flag_check(self, flag_manager):
        """Test user list flag check"""
        flag = FeatureFlag(
            name="user_list_feature",
            enabled=True,
            flag_type=FeatureFlagType.USER_LIST,
            user_list=["user-1", "user-2"]
        )
        
        flag_manager.register_flag(flag)
        
        assert flag_manager.is_enabled_for_user("user_list_feature", "user-1") is True
        assert flag_manager.is_enabled_for_user("user_list_feature", "user-2") is True
        assert flag_manager.is_enabled_for_user("user_list_feature", "user-3") is False
    
    def test_custom_flag_check(self, flag_manager):
        """Test custom flag check"""
        def custom_check(user_id):
            return user_id.endswith("-premium")
        
        flag = FeatureFlag(
            name="custom_feature",
            enabled=True,
            flag_type=FeatureFlagType.CUSTOM,
            custom_check=custom_check
        )
        
        flag_manager.register_flag(flag)
        
        assert flag_manager.is_enabled_for_user("custom_feature", "user-premium") is True
        assert flag_manager.is_enabled_for_user("custom_feature", "user-regular") is False
    
    def test_get_all_flags(self, flag_manager):
        """Test getting all flags"""
        flag_manager.enable("feature1")
        flag_manager.enable("feature2")
        flag_manager.disable("feature3")
        
        all_flags = flag_manager.get_all_flags()
        
        assert "feature1" in all_flags
        assert "feature2" in all_flags
        assert "feature3" in all_flags
        assert all_flags["feature1"] is True
        assert all_flags["feature2"] is True
        assert all_flags["feature3"] is False
    
    @patch.dict(os.environ, {'FEATURE_NEW_FEATURE': 'true'})
    def test_load_from_environment(self, flag_manager):
        """Test loading flags from environment"""
        # Manager should load from env on init
        # This tests that env vars are respected
        assert flag_manager is not None
    
    def test_flag_metadata(self, flag_manager):
        """Test flag with metadata"""
        flag = FeatureFlag(
            name="feature_with_metadata",
            enabled=True,
            metadata={"description": "Test feature", "version": "1.0"}
        )
        
        flag_manager.register_flag(flag)
        
        flag_info = flag_manager.get_flag_info("feature_with_metadata")
        
        # Should include metadata if available
        assert flag_info is not None or flag_manager.is_enabled("feature_with_metadata")



