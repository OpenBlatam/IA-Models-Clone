"""
Tests for AuthManager utility
"""

import pytest
import jwt
from datetime import datetime, timedelta

from ..utils.auth_manager import AuthManager


class TestAuthManager:
    """Test suite for AuthManager"""

    def test_init(self):
        """Test AuthManager initialization"""
        manager = AuthManager()
        assert manager.secret_key is not None
        assert manager.users == {}
        assert manager.api_keys == {}

    def test_init_with_secret(self):
        """Test AuthManager with custom secret"""
        secret = "test_secret_key"
        manager = AuthManager(secret_key=secret)
        assert manager.secret_key == secret

    def test_create_user(self):
        """Test creating a user"""
        manager = AuthManager()
        
        user_info = manager.create_user(
            username="testuser",
            password="testpass",
            email="test@example.com",
            role="user"
        )
        
        assert user_info["username"] == "testuser"
        assert user_info["role"] == "user"
        assert "testuser" in manager.users
        assert manager.users["testuser"]["email"] == "test@example.com"

    def test_create_user_admin(self):
        """Test creating admin user"""
        manager = AuthManager()
        
        user_info = manager.create_user(
            username="admin",
            password="adminpass",
            role="admin"
        )
        
        assert user_info["role"] == "admin"
        assert manager.users["admin"]["role"] == "admin"

    def test_authenticate_user_success(self):
        """Test successful user authentication"""
        manager = AuthManager()
        
        manager.create_user("testuser", "testpass")
        token = manager.authenticate_user("testuser", "testpass")
        
        assert token is not None
        # Verify token is valid JWT
        decoded = jwt.decode(token, manager.secret_key, algorithms=["HS256"])
        assert decoded["username"] == "testuser"

    def test_authenticate_user_wrong_password(self):
        """Test authentication with wrong password"""
        manager = AuthManager()
        
        manager.create_user("testuser", "testpass")
        token = manager.authenticate_user("testuser", "wrongpass")
        
        assert token is None

    def test_authenticate_user_not_found(self):
        """Test authentication with non-existent user"""
        manager = AuthManager()
        
        token = manager.authenticate_user("nonexistent", "password")
        
        assert token is None

    def test_authenticate_user_inactive(self):
        """Test authentication with inactive user"""
        manager = AuthManager()
        
        manager.create_user("testuser", "testpass")
        manager.users["testuser"]["active"] = False
        
        token = manager.authenticate_user("testuser", "testpass")
        
        assert token is None

    def test_verify_token_valid(self):
        """Test verifying valid token"""
        manager = AuthManager()
        
        manager.create_user("testuser", "testpass")
        token = manager.authenticate_user("testuser", "testpass")
        
        user_info = manager.verify_token(token)
        
        assert user_info is not None
        assert user_info["username"] == "testuser"

    def test_verify_token_invalid(self):
        """Test verifying invalid token"""
        manager = AuthManager()
        
        user_info = manager.verify_token("invalid_token")
        
        assert user_info is None

    def test_verify_token_expired(self):
        """Test verifying expired token"""
        manager = AuthManager()
        
        # Create expired token
        payload = {
            "user_id": "test",
            "username": "testuser",
            "exp": datetime.utcnow() - timedelta(hours=1),
            "iat": datetime.utcnow() - timedelta(hours=2),
        }
        expired_token = jwt.encode(payload, manager.secret_key, algorithm="HS256")
        
        user_info = manager.verify_token(expired_token)
        
        assert user_info is None

    def test_create_api_key(self):
        """Test creating API key"""
        manager = AuthManager()
        
        manager.create_user("testuser", "testpass")
        api_key_info = manager.create_api_key("testuser", "Test API Key")
        
        assert api_key_info["api_key"] is not None
        assert api_key_info["user_id"] is not None
        assert len(api_key_info["api_key"]) > 0

    def test_validate_api_key(self):
        """Test validating API key"""
        manager = AuthManager()
        
        manager.create_user("testuser", "testpass")
        api_key_info = manager.create_api_key("testuser", "Test Key")
        
        user_info = manager.validate_api_key(api_key_info["api_key"])
        
        assert user_info is not None
        assert user_info["username"] == "testuser"

    def test_validate_api_key_invalid(self):
        """Test validating invalid API key"""
        manager = AuthManager()
        
        user_info = manager.validate_api_key("invalid_key")
        
        assert user_info is None

    def test_password_hashing(self):
        """Test that passwords are hashed"""
        manager = AuthManager()
        
        manager.create_user("testuser", "testpass")
        
        # Password should be hashed, not plain text
        user = manager.users["testuser"]
        assert user["password_hash"] != "testpass"
        assert len(user["password_hash"]) == 64  # SHA256 hex length

