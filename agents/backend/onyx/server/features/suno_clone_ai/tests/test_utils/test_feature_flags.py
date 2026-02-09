"""
Comprehensive Unit Tests for Feature Flags

Tests cover feature flag system with diverse test cases
"""

import pytest
from unittest.mock import Mock, patch

from utils.feature_flags import (
    FeatureFlagService,
    FeatureFlag,
    FlagType
)


class TestFeatureFlag:
    """Test cases for FeatureFlag dataclass"""
    
    def test_feature_flag_creation(self):
        """Test creating a feature flag"""
        flag = FeatureFlag(
            name="test_flag",
            flag_type=FlagType.BOOLEAN,
            enabled=True
        )
        assert flag.name == "test_flag"
        assert flag.flag_type == FlagType.BOOLEAN
        assert flag.enabled is True
    
    def test_feature_flag_with_percentage(self):
        """Test flag with percentage"""
        flag = FeatureFlag(
            name="test_flag",
            flag_type=FlagType.PERCENTAGE,
            percentage=50
        )
        assert flag.percentage == 50


class TestFeatureFlagService:
    """Test cases for FeatureFlagService class"""
    
    def test_feature_flag_service_init(self):
        """Test initializing feature flag service"""
        service = FeatureFlagService()
        assert len(service.flags) > 0  # Should have default flags
    
    def test_create_flag_boolean(self):
        """Test creating boolean flag"""
        service = FeatureFlagService()
        flag = service.create_flag(
            name="new_feature",
            flag_type=FlagType.BOOLEAN,
            enabled=True
        )
        
        assert flag.name == "new_feature"
        assert flag.flag_type == FlagType.BOOLEAN
        assert flag.enabled is True
        assert "new_feature" in service.flags
    
    def test_create_flag_percentage(self):
        """Test creating percentage flag"""
        service = FeatureFlagService()
        flag = service.create_flag(
            name="gradual_rollout",
            flag_type=FlagType.PERCENTAGE,
            percentage=25
        )
        
        assert flag.flag_type == FlagType.PERCENTAGE
        assert flag.percentage == 25
    
    def test_create_flag_user_list(self):
        """Test creating user list flag"""
        service = FeatureFlagService()
        user_list = ["user1", "user2", "user3"]
        flag = service.create_flag(
            name="beta_users",
            flag_type=FlagType.USER_LIST,
            user_list=user_list
        )
        
        assert flag.flag_type == FlagType.USER_LIST
        assert flag.user_list == user_list
    
    def test_create_flag_attribute(self):
        """Test creating attribute-based flag"""
        service = FeatureFlagService()
        attributes = {"tier": "premium", "region": "US"}
        flag = service.create_flag(
            name="premium_feature",
            flag_type=FlagType.ATTRIBUTE,
            attributes=attributes
        )
        
        assert flag.flag_type == FlagType.ATTRIBUTE
        assert flag.attributes == attributes
    
    def test_get_flag_exists(self):
        """Test getting existing flag"""
        service = FeatureFlagService()
        service.create_flag("test_flag", FlagType.BOOLEAN)
        
        flag = service.get_flag("test_flag")
        assert flag is not None
        assert flag.name == "test_flag"
    
    def test_get_flag_not_exists(self):
        """Test getting non-existent flag"""
        service = FeatureFlagService()
        flag = service.get_flag("nonexistent")
        assert flag is None
    
    def test_is_enabled_boolean_true(self):
        """Test boolean flag enabled"""
        service = FeatureFlagService()
        service.create_flag("test", FlagType.BOOLEAN, enabled=True)
        
        assert service.is_enabled("test") is True
    
    def test_is_enabled_boolean_false(self):
        """Test boolean flag disabled"""
        service = FeatureFlagService()
        service.create_flag("test", FlagType.BOOLEAN, enabled=False)
        
        assert service.is_enabled("test") is False
    
    def test_is_enabled_percentage_100(self):
        """Test percentage flag at 100%"""
        service = FeatureFlagService()
        service.create_flag("test", FlagType.PERCENTAGE, percentage=100)
        
        # Should always be enabled at 100%
        assert service.is_enabled("test", user_id="any_user") is True
    
    def test_is_enabled_percentage_0(self):
        """Test percentage flag at 0%"""
        service = FeatureFlagService()
        service.create_flag("test", FlagType.PERCENTAGE, percentage=0)
        
        assert service.is_enabled("test", user_id="any_user") is False
    
    def test_is_enabled_percentage_deterministic(self):
        """Test percentage flag is deterministic for same user"""
        service = FeatureFlagService()
        service.create_flag("test", FlagType.PERCENTAGE, percentage=50)
        
        user_id = "test_user"
        # Should return same result for same user
        result1 = service.is_enabled("test", user_id=user_id)
        result2 = service.is_enabled("test", user_id=user_id)
        assert result1 == result2
    
    def test_is_enabled_user_list_included(self):
        """Test user list flag with user in list"""
        service = FeatureFlagService()
        service.create_flag(
            "test",
            FlagType.USER_LIST,
            user_list=["user1", "user2"]
        )
        
        assert service.is_enabled("test", user_id="user1") is True
        assert service.is_enabled("test", user_id="user2") is True
    
    def test_is_enabled_user_list_not_included(self):
        """Test user list flag with user not in list"""
        service = FeatureFlagService()
        service.create_flag(
            "test",
            FlagType.USER_LIST,
            user_list=["user1", "user2"]
        )
        
        assert service.is_enabled("test", user_id="user3") is False
    
    def test_is_enabled_attribute_match(self):
        """Test attribute flag with matching attributes"""
        service = FeatureFlagService()
        service.create_flag(
            "test",
            FlagType.ATTRIBUTE,
            attributes={"tier": "premium"}
        )
        
        user_attrs = {"tier": "premium", "region": "US"}
        assert service.is_enabled("test", user_attributes=user_attrs) is True
    
    def test_is_enabled_attribute_no_match(self):
        """Test attribute flag with non-matching attributes"""
        service = FeatureFlagService()
        service.create_flag(
            "test",
            FlagType.ATTRIBUTE,
            attributes={"tier": "premium"}
        )
        
        user_attrs = {"tier": "basic", "region": "US"}
        assert service.is_enabled("test", user_attributes=user_attrs) is False
    
    def test_update_flag(self):
        """Test updating a flag"""
        service = FeatureFlagService()
        service.create_flag("test", FlagType.BOOLEAN, enabled=True)
        
        service.update_flag("test", enabled=False)
        flag = service.get_flag("test")
        assert flag.enabled is False
    
    def test_update_flag_not_exists(self):
        """Test updating non-existent flag"""
        service = FeatureFlagService()
        result = service.update_flag("nonexistent", enabled=True)
        assert result is False
    
    def test_delete_flag(self):
        """Test deleting a flag"""
        service = FeatureFlagService()
        service.create_flag("test", FlagType.BOOLEAN)
        
        result = service.delete_flag("test")
        assert result is True
        assert service.get_flag("test") is None
    
    def test_delete_flag_not_exists(self):
        """Test deleting non-existent flag"""
        service = FeatureFlagService()
        result = service.delete_flag("nonexistent")
        assert result is False
    
    def test_list_flags(self):
        """Test listing all flags"""
        service = FeatureFlagService()
        service.create_flag("flag1", FlagType.BOOLEAN)
        service.create_flag("flag2", FlagType.BOOLEAN)
        
        flags = service.list_flags()
        assert len(flags) >= 2
        assert any(f.name == "flag1" for f in flags)
        assert any(f.name == "flag2" for f in flags)
    
    def test_get_flag_stats(self):
        """Test getting flag statistics"""
        service = FeatureFlagService()
        service.create_flag("flag1", FlagType.BOOLEAN, enabled=True)
        service.create_flag("flag2", FlagType.BOOLEAN, enabled=False)
        
        stats = service.get_flag_stats()
        assert stats["total_flags"] >= 2
        assert stats["enabled_flags"] >= 1















