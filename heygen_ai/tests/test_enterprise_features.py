"""
Tests for Enterprise Features Module
===================================

Comprehensive tests for the EnterpriseFeatures class including:
- User management
- Role-based access control
- SSO configuration
- Audit logging
- Compliance features
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from core.enterprise_features import (
    EnterpriseFeatures, User, Role, Permission, AuditLog, 
    SSOConfig, ComplianceConfig
)
from core.base_service import ServiceStatus, HealthCheckResult


class TestEnterpriseFeatures:
    """Test suite for EnterpriseFeatures class"""
    
    @pytest.fixture
    def enterprise_features(self):
        """Create an EnterpriseFeatures instance for testing"""
        features = EnterpriseFeatures()
        return features
    
    @pytest.fixture
    def sample_user_data(self):
        """Sample user data for testing"""
        return {
            "username": "testuser",
            "email": "test@example.com",
            "full_name": "Test User",
            "role": "user"
        }
    
    @pytest.fixture
    def sample_role_data(self):
        """Sample role data for testing"""
        return {
            "name": "test_role",
            "description": "Test role for testing",
            "permissions": ["perm_video_read", "perm_video_write"]
        }

    def test_enterprise_features_initialization(self):
        """Test EnterpriseFeatures initialization"""
        features = EnterpriseFeatures()
        assert features is not None
        assert features.name == "EnterpriseFeatures"
        assert features.enterprise_stats["total_users"] == 0
        assert features.enterprise_stats["total_roles"] == 0
        assert features.enterprise_stats["total_permissions"] == 0

    @pytest.mark.asyncio
    async def test_enterprise_features_initialization_async(self, enterprise_features):
        """Test async initialization of EnterpriseFeatures"""
        await enterprise_features.initialize()
        assert enterprise_features.status == ServiceStatus.RUNNING
        assert len(enterprise_features.roles) > 0  # Should have default roles
        assert len(enterprise_features.permissions) > 0  # Should have default permissions
        assert len(enterprise_features.sso_configs) > 0  # Should have default SSO configs

    @pytest.mark.asyncio
    async def test_create_user(self, enterprise_features, sample_user_data):
        """Test user creation"""
        user_id = await enterprise_features.create_user(**sample_user_data)
        
        assert user_id is not None
        assert user_id in enterprise_features.users
        assert enterprise_features.users[user_id].username == sample_user_data["username"]
        assert enterprise_features.users[user_id].email == sample_user_data["email"]
        assert enterprise_features.users[user_id].role == sample_user_data["role"]
        assert enterprise_features.enterprise_stats["total_users"] == 1
        assert enterprise_features.enterprise_stats["active_users"] == 1

    @pytest.mark.asyncio
    async def test_create_user_duplicate_username(self, enterprise_features, sample_user_data):
        """Test creating user with duplicate username"""
        # Create first user
        await enterprise_features.create_user(**sample_user_data)
        
        # Try to create user with same username
        with pytest.raises(ValueError, match="Username already exists"):
            await enterprise_features.create_user(**sample_user_data)

    @pytest.mark.asyncio
    async def test_create_user_duplicate_email(self, enterprise_features, sample_user_data):
        """Test creating user with duplicate email"""
        # Create first user
        await enterprise_features.create_user(**sample_user_data)
        
        # Try to create user with same email
        sample_user_data["username"] = "different_username"
        with pytest.raises(ValueError, match="Email already exists"):
            await enterprise_features.create_user(**sample_user_data)

    @pytest.mark.asyncio
    async def test_create_user_invalid_role(self, enterprise_features, sample_user_data):
        """Test creating user with invalid role"""
        sample_user_data["role"] = "invalid_role"
        
        with pytest.raises(ValueError, match="Invalid role"):
            await enterprise_features.create_user(**sample_user_data)

    @pytest.mark.asyncio
    async def test_authenticate_user(self, enterprise_features, sample_user_data):
        """Test user authentication"""
        # Create user first
        user_id = await enterprise_features.create_user(**sample_user_data)
        
        # Authenticate user
        auth_user_id = await enterprise_features.authenticate_user(
            sample_user_data["username"], 
            "password123"
        )
        
        assert auth_user_id == user_id
        assert enterprise_features.users[user_id].last_login is not None

    @pytest.mark.asyncio
    async def test_authenticate_user_not_found(self, enterprise_features):
        """Test authentication with non-existent user"""
        auth_user_id = await enterprise_features.authenticate_user(
            "nonexistent", 
            "password123"
        )
        
        assert auth_user_id is None

    @pytest.mark.asyncio
    async def test_check_permission(self, enterprise_features, sample_user_data):
        """Test permission checking"""
        # Create user
        user_id = await enterprise_features.create_user(**sample_user_data)
        
        # Check permission (user role should have video read permission)
        has_permission = await enterprise_features.check_permission(
            user_id, "video", "read"
        )
        
        assert has_permission is True
        
        # Check permission user doesn't have
        has_permission = await enterprise_features.check_permission(
            user_id, "user", "delete"
        )
        
        assert has_permission is False

    @pytest.mark.asyncio
    async def test_check_permission_invalid_user(self, enterprise_features):
        """Test permission checking with invalid user"""
        has_permission = await enterprise_features.check_permission(
            "invalid_user_id", "video", "read"
        )
        
        assert has_permission is False

    @pytest.mark.asyncio
    async def test_create_role(self, enterprise_features, sample_role_data):
        """Test role creation"""
        role_id = await enterprise_features.create_role(**sample_role_data)
        
        assert role_id is not None
        assert role_id in enterprise_features.roles
        assert enterprise_features.roles[role_id].name == sample_role_data["name"]
        assert enterprise_features.roles[role_id].description == sample_role_data["description"]
        assert enterprise_features.roles[role_id].permissions == sample_role_data["permissions"]

    @pytest.mark.asyncio
    async def test_create_role_invalid_permissions(self, enterprise_features, sample_role_data):
        """Test creating role with invalid permissions"""
        sample_role_data["permissions"] = ["invalid_permission"]
        
        with pytest.raises(ValueError, match="Invalid permission"):
            await enterprise_features.create_role(**sample_role_data)

    @pytest.mark.asyncio
    async def test_create_role_duplicate_name(self, enterprise_features, sample_role_data):
        """Test creating role with duplicate name"""
        # Create first role
        await enterprise_features.create_role(**sample_role_data)
        
        # Try to create role with same name
        with pytest.raises(ValueError, match="Role name already exists"):
            await enterprise_features.create_role(**sample_role_data)

    @pytest.mark.asyncio
    async def test_assign_role_to_user(self, enterprise_features, sample_user_data, sample_role_data):
        """Test assigning role to user"""
        # Create user and role
        user_id = await enterprise_features.create_user(**sample_user_data)
        role_id = await enterprise_features.create_role(**sample_role_data)
        
        # Assign role to user
        result = await enterprise_features.assign_role_to_user(user_id, role_id)
        
        assert result is True
        assert enterprise_features.users[user_id].role == role_id
        assert enterprise_features.users[user_id].permissions == sample_role_data["permissions"]

    @pytest.mark.asyncio
    async def test_assign_role_invalid_user(self, enterprise_features, sample_role_data):
        """Test assigning role to invalid user"""
        role_id = await enterprise_features.create_role(**sample_role_data)
        
        with pytest.raises(ValueError, match="User not found"):
            await enterprise_features.assign_role_to_user("invalid_user_id", role_id)

    @pytest.mark.asyncio
    async def test_assign_role_invalid_role(self, enterprise_features, sample_user_data):
        """Test assigning invalid role to user"""
        user_id = await enterprise_features.create_user(**sample_user_data)
        
        with pytest.raises(ValueError, match="Role not found"):
            await enterprise_features.assign_role_to_user(user_id, "invalid_role_id")

    @pytest.mark.asyncio
    async def test_get_user_info(self, enterprise_features, sample_user_data):
        """Test getting user information"""
        # Create user
        user_id = await enterprise_features.create_user(**sample_user_data)
        
        # Get user info
        user_info = await enterprise_features.get_user_info(user_id)
        
        assert user_info is not None
        assert user_info["user_id"] == user_id
        assert user_info["username"] == sample_user_data["username"]
        assert user_info["email"] == sample_user_data["email"]
        assert user_info["full_name"] == sample_user_data["full_name"]
        assert user_info["role"] == sample_user_data["role"]

    @pytest.mark.asyncio
    async def test_get_user_info_invalid_user(self, enterprise_features):
        """Test getting info for invalid user"""
        user_info = await enterprise_features.get_user_info("invalid_user_id")
        
        assert user_info is None

    @pytest.mark.asyncio
    async def test_get_audit_logs(self, enterprise_features, sample_user_data):
        """Test getting audit logs"""
        # Create user to generate audit logs
        user_id = await enterprise_features.create_user(**sample_user_data)
        
        # Get audit logs
        logs = await enterprise_features.get_audit_logs()
        
        assert isinstance(logs, list)
        assert len(logs) > 0  # Should have at least the user creation log
        
        # Check log structure
        log = logs[0]
        assert "log_id" in log
        assert "user_id" in log
        assert "action" in log
        assert "resource" in log
        assert "timestamp" in log

    @pytest.mark.asyncio
    async def test_get_audit_logs_with_filters(self, enterprise_features, sample_user_data):
        """Test getting audit logs with filters"""
        # Create user to generate audit logs
        user_id = await enterprise_features.create_user(**sample_user_data)
        
        # Get audit logs with user filter
        logs = await enterprise_features.get_audit_logs(user_id=user_id)
        
        assert isinstance(logs, list)
        for log in logs:
            assert log["user_id"] == user_id

    @pytest.mark.asyncio
    async def test_health_check(self, enterprise_features):
        """Test health check functionality"""
        health_result = await enterprise_features.health_check()
        
        assert isinstance(health_result, HealthCheckResult)
        assert health_result.status == ServiceStatus.RUNNING
        assert "dependencies" in health_result.details
        assert "users" in health_result.details
        assert "sso" in health_result.details
        assert "compliance" in health_result.details
        assert "audit" in health_result.details

    @pytest.mark.asyncio
    async def test_cleanup_temp_files(self, enterprise_features):
        """Test cleanup of temporary files"""
        # This should not raise an exception
        await enterprise_features.cleanup_temp_files()

    @pytest.mark.asyncio
    async def test_shutdown(self, enterprise_features):
        """Test shutdown functionality"""
        # Add some data
        await enterprise_features.create_user(
            username="test", 
            email="test@test.com", 
            full_name="Test User"
        )
        
        # Verify data exists
        assert len(enterprise_features.users) > 0
        
        # Shutdown
        await enterprise_features.shutdown()
        
        # Verify data is cleared
        assert len(enterprise_features.users) == 0
        assert len(enterprise_features.roles) == 0
        assert len(enterprise_features.permissions) == 0
        assert len(enterprise_features.audit_logs) == 0
        assert enterprise_features.audit_encryption_key is None


class TestEnterpriseFeaturesDataStructures:
    """Test suite for enterprise features data structures"""
    
    def test_user_dataclass(self):
        """Test User dataclass"""
        user = User(
            user_id="test_id",
            username="testuser",
            email="test@example.com",
            full_name="Test User"
        )
        
        assert user.user_id == "test_id"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.role == "user"  # default
        assert user.is_active is True  # default
        assert isinstance(user.permissions, list)
        assert isinstance(user.metadata, dict)

    def test_role_dataclass(self):
        """Test Role dataclass"""
        role = Role(
            role_id="test_role_id",
            name="test_role",
            description="Test role",
            permissions=["perm1", "perm2"]
        )
        
        assert role.role_id == "test_role_id"
        assert role.name == "test_role"
        assert role.description == "Test role"
        assert role.permissions == ["perm1", "perm2"]
        assert role.is_system_role is False  # default
        assert isinstance(role.metadata, dict)

    def test_permission_dataclass(self):
        """Test Permission dataclass"""
        permission = Permission(
            permission_id="test_perm_id",
            name="test_permission",
            description="Test permission",
            resource="test_resource",
            action="test_action"
        )
        
        assert permission.permission_id == "test_perm_id"
        assert permission.name == "test_permission"
        assert permission.description == "Test permission"
        assert permission.resource == "test_resource"
        assert permission.action == "test_action"
        assert permission.scope == "global"  # default

    def test_audit_log_dataclass(self):
        """Test AuditLog dataclass"""
        audit_log = AuditLog(
            log_id="test_log_id",
            user_id="test_user_id",
            action="test_action",
            resource="test_resource",
            resource_id="test_resource_id",
            details={"key": "value"}
        )
        
        assert audit_log.log_id == "test_log_id"
        assert audit_log.user_id == "test_user_id"
        assert audit_log.action == "test_action"
        assert audit_log.resource == "test_resource"
        assert audit_log.resource_id == "test_resource_id"
        assert audit_log.details == {"key": "value"}
        assert audit_log.encrypted is True  # default
        assert isinstance(audit_log.timestamp, datetime)

    def test_sso_config_dataclass(self):
        """Test SSOConfig dataclass"""
        sso_config = SSOConfig(
            provider="saml",
            config={"key": "value"}
        )
        
        assert sso_config.provider == "saml"
        assert sso_config.enabled is True  # default
        assert sso_config.config == {"key": "value"}
        assert isinstance(sso_config.metadata, dict)

    def test_compliance_config_dataclass(self):
        """Test ComplianceConfig dataclass"""
        compliance_config = ComplianceConfig()
        
        assert compliance_config.gdpr_enabled is True  # default
        assert compliance_config.hipaa_enabled is False  # default
        assert compliance_config.sox_enabled is False  # default
        assert compliance_config.data_retention_days == 2555  # default
        assert compliance_config.audit_logging_enabled is True  # default
        assert compliance_config.encryption_enabled is True  # default
        assert isinstance(compliance_config.metadata, dict)


class TestEnterpriseFeaturesIntegration:
    """Integration tests for EnterpriseFeatures"""
    
    @pytest.mark.asyncio
    async def test_complete_user_workflow(self):
        """Test complete user management workflow"""
        features = EnterpriseFeatures()
        await features.initialize()
        
        try:
            # Create user
            user_id = await features.create_user(
                username="integration_test",
                email="integration@test.com",
                full_name="Integration Test User",
                role="user"
            )
            
            # Authenticate user
            auth_user_id = await features.authenticate_user("integration_test", "password")
            assert auth_user_id == user_id
            
            # Check permissions
            has_video_read = await features.check_permission(user_id, "video", "read")
            assert has_video_read is True
            
            # Create custom role
            role_id = await features.create_role(
                name="custom_role",
                description="Custom role for testing",
                permissions=["perm_video_read", "perm_analytics_read"]
            )
            
            # Assign role to user
            result = await features.assign_role_to_user(user_id, role_id)
            assert result is True
            
            # Check new permissions
            has_analytics_read = await features.check_permission(user_id, "analytics", "read")
            assert has_analytics_read is True
            
            # Get user info
            user_info = await features.get_user_info(user_id)
            assert user_info["role"] == role_id
            
            # Get audit logs
            logs = await features.get_audit_logs(user_id=user_id)
            assert len(logs) >= 3  # user creation, role creation, role assignment
            
        finally:
            await features.shutdown()

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        features = EnterpriseFeatures()
        await features.initialize()
        
        try:
            # Test invalid operations
            with pytest.raises(ValueError):
                await features.create_user(
                    username="test",
                    email="test@test.com",
                    full_name="Test",
                    role="invalid_role"
                )
            
            # Test that system is still functional after error
            user_id = await features.create_user(
                username="test",
                email="test@test.com",
                full_name="Test",
                role="user"
            )
            assert user_id is not None
            
            # Test health check after error
            health = await features.health_check()
            assert health.status == ServiceStatus.RUNNING
            
        finally:
            await features.shutdown()
