"""
Unit Tests for Security
=======================
Tests for security manager, access tokens, and audit logs.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any

# Try to import security classes
try:
    from security import (
        PermissionType,
        AccessToken,
        AuditLog,
        SecurityManager
    )
except ImportError:
    PermissionType = None
    AccessToken = None
    AuditLog = None
    SecurityManager = None


class TestPermissionType:
    """Tests for PermissionType enum."""
    
    def test_permission_type_values(self):
        """Test PermissionType enum values."""
        if PermissionType is None:
            pytest.skip("PermissionType not available")
        
        # Check expected permission types
        expected_permissions = ["read", "write", "delete", "admin"]
        actual_permissions = [e.value for e in PermissionType]
        
        for perm in expected_permissions:
            assert perm in actual_permissions or hasattr(PermissionType, perm.upper())


class TestAccessToken:
    """Tests for AccessToken class."""
    
    def test_access_token_creation(self):
        """Test creating AccessToken."""
        if AccessToken is None:
            pytest.skip("AccessToken not available")
        
        token = AccessToken(user_id="test_user", permissions=["read", "write"])
        assert token is not None
        assert token.user_id == "test_user"
    
    def test_access_token_expiration(self):
        """Test AccessToken expiration."""
        if AccessToken is None:
            pytest.skip("AccessToken not available")
        
        token = AccessToken(
            user_id="test_user",
            permissions=["read"],
            expires_at=datetime.now() + timedelta(hours=1)
        )
        
        assert hasattr(token, "expires_at")
        assert token.expires_at > datetime.now()
    
    def test_access_token_is_valid(self):
        """Test AccessToken validation."""
        if AccessToken is None:
            pytest.skip("AccessToken not available")
        
        # Valid token
        token = AccessToken(
            user_id="test_user",
            permissions=["read"],
            expires_at=datetime.now() + timedelta(hours=1)
        )
        
        if hasattr(token, "is_valid"):
            assert token.is_valid() is True
        
        # Expired token
        expired_token = AccessToken(
            user_id="test_user",
            permissions=["read"],
            expires_at=datetime.now() - timedelta(hours=1)
        )
        
        if hasattr(expired_token, "is_valid"):
            assert expired_token.is_valid() is False
    
    def test_access_token_has_permission(self):
        """Test checking permissions."""
        if AccessToken is None:
            pytest.skip("AccessToken not available")
        
        token = AccessToken(user_id="test_user", permissions=["read", "write"])
        
        if hasattr(token, "has_permission"):
            assert token.has_permission("read") is True
            assert token.has_permission("write") is True
            assert token.has_permission("delete") is False


class TestAuditLog:
    """Tests for AuditLog class."""
    
    def test_audit_log_creation(self):
        """Test creating AuditLog."""
        if AuditLog is None:
            pytest.skip("AuditLog not available")
        
        log = AuditLog(
            user_id="test_user",
            action="pdf_upload",
            resource_id="file_123"
        )
        assert log is not None
        assert log.user_id == "test_user"
        assert log.action == "pdf_upload"
    
    def test_audit_log_timestamp(self):
        """Test AuditLog timestamp."""
        if AuditLog is None:
            pytest.skip("AuditLog not available")
        
        log = AuditLog(user_id="test_user", action="test")
        assert hasattr(log, "timestamp") or hasattr(log, "created_at")
    
    def test_audit_log_metadata(self):
        """Test AuditLog metadata."""
        if AuditLog is None:
            pytest.skip("AuditLog not available")
        
        metadata = {"file_id": "123", "file_size": 1024}
        log = AuditLog(
            user_id="test_user",
            action="pdf_upload",
            metadata=metadata
        )
        
        assert hasattr(log, "metadata")
        if log.metadata:
            assert log.metadata.get("file_id") == "123"


class TestSecurityManager:
    """Tests for SecurityManager class."""
    
    @pytest.fixture
    def security_manager(self):
        """Create SecurityManager instance."""
        if SecurityManager is None:
            pytest.skip("SecurityManager not available")
        return SecurityManager()
    
    def test_security_manager_initialization(self, security_manager):
        """Test SecurityManager initialization."""
        assert security_manager is not None
    
    @pytest.mark.asyncio
    async def test_generate_token(self, security_manager):
        """Test token generation."""
        if security_manager is None:
            pytest.skip("SecurityManager not available")
        
        if hasattr(security_manager, "generate_token"):
            token = await security_manager.generate_token(
                user_id="test_user",
                permissions=["read", "write"]
            )
            assert token is not None
            assert isinstance(token, str) or isinstance(token, AccessToken)
    
    @pytest.mark.asyncio
    async def test_validate_token(self, security_manager):
        """Test token validation."""
        if security_manager is None:
            pytest.skip("SecurityManager not available")
        
        if hasattr(security_manager, "validate_token"):
            # Generate token first
            if hasattr(security_manager, "generate_token"):
                token = await security_manager.generate_token("test_user", ["read"])
                token_str = token if isinstance(token, str) else str(token)
                
                # Validate token
                result = await security_manager.validate_token(token_str)
                assert result is not None
                assert hasattr(result, "user_id") or isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_check_permission(self, security_manager):
        """Test permission checking."""
        if security_manager is None:
            pytest.skip("SecurityManager not available")
        
        if hasattr(security_manager, "check_permission"):
            has_permission = await security_manager.check_permission(
                user_id="test_user",
                resource_id="file_123",
                permission="read"
            )
            assert isinstance(has_permission, bool)
    
    @pytest.mark.asyncio
    async def test_log_audit_event(self, security_manager):
        """Test logging audit events."""
        if security_manager is None:
            pytest.skip("SecurityManager not available")
        
        if hasattr(security_manager, "log_audit_event"):
            result = await security_manager.log_audit_event(
                user_id="test_user",
                action="pdf_upload",
                resource_id="file_123"
            )
            assert result is True or result is None
    
    @pytest.mark.asyncio
    async def test_get_audit_logs(self, security_manager):
        """Test getting audit logs."""
        if security_manager is None:
            pytest.skip("SecurityManager not available")
        
        if hasattr(security_manager, "get_audit_logs"):
            logs = await security_manager.get_audit_logs(
                user_id="test_user",
                limit=10
            )
            assert isinstance(logs, list)
    
    @pytest.mark.asyncio
    async def test_revoke_token(self, security_manager):
        """Test token revocation."""
        if security_manager is None:
            pytest.skip("SecurityManager not available")
        
        if hasattr(security_manager, "revoke_token"):
            # Generate token first
            if hasattr(security_manager, "generate_token"):
                token = await security_manager.generate_token("test_user", ["read"])
                token_str = token if isinstance(token, str) else str(token)
                
                # Revoke token
                result = await security_manager.revoke_token(token_str)
                assert result is True or result is None
                
                # Token should be invalid after revocation
                if hasattr(security_manager, "validate_token"):
                    validation = await security_manager.validate_token(token_str)
                    assert validation is None or validation is False
    
    @pytest.mark.asyncio
    async def test_encrypt_data(self, security_manager):
        """Test data encryption."""
        if security_manager is None:
            pytest.skip("SecurityManager not available")
        
        if hasattr(security_manager, "encrypt"):
            data = "sensitive data"
            encrypted = await security_manager.encrypt(data)
            assert encrypted != data
            assert isinstance(encrypted, str) or isinstance(encrypted, bytes)
    
    @pytest.mark.asyncio
    async def test_decrypt_data(self, security_manager):
        """Test data decryption."""
        if security_manager is None:
            pytest.skip("SecurityManager not available")
        
        if hasattr(security_manager, "encrypt") and hasattr(security_manager, "decrypt"):
            data = "sensitive data"
            encrypted = await security_manager.encrypt(data)
            decrypted = await security_manager.decrypt(encrypted)
            assert decrypted == data
    
    @pytest.mark.asyncio
    async def test_hash_password(self, security_manager):
        """Test password hashing."""
        if security_manager is None:
            pytest.skip("SecurityManager not available")
        
        if hasattr(security_manager, "hash_password"):
            password = "test_password"
            hashed = await security_manager.hash_password(password)
            assert hashed != password
            assert isinstance(hashed, str)
    
    @pytest.mark.asyncio
    async def test_verify_password(self, security_manager):
        """Test password verification."""
        if security_manager is None:
            pytest.skip("SecurityManager not available")
        
        if hasattr(security_manager, "hash_password") and hasattr(security_manager, "verify_password"):
            password = "test_password"
            hashed = await security_manager.hash_password(password)
            verified = await security_manager.verify_password(password, hashed)
            assert verified is True
            
            # Wrong password
            verified_wrong = await security_manager.verify_password("wrong", hashed)
            assert verified_wrong is False


class TestSecurityIntegration:
    """Integration tests for security."""
    
    @pytest.mark.asyncio
    async def test_token_lifecycle(self):
        """Test complete token lifecycle."""
        if SecurityManager is None:
            pytest.skip("SecurityManager not available")
        
        manager = SecurityManager()
        
        if hasattr(manager, "generate_token") and hasattr(manager, "validate_token"):
            # Generate token
            token = await manager.generate_token("test_user", ["read", "write"])
            token_str = token if isinstance(token, str) else str(token)
            
            # Validate token
            validated = await manager.validate_token(token_str)
            assert validated is not None
            
            # Revoke token
            if hasattr(manager, "revoke_token"):
                await manager.revoke_token(token_str)
                validated_after = await manager.validate_token(token_str)
                assert validated_after is None or validated_after is False



