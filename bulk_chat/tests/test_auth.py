"""
Tests for Auth Manager
=======================
"""

import pytest
import asyncio
from ..core.auth import AuthManager, Role


@pytest.fixture
def auth_manager():
    """Create auth manager for testing."""
    return AuthManager(secret_key="test_secret_key_for_testing_only")


@pytest.mark.asyncio
async def test_register_user(auth_manager):
    """Test registering a user."""
    user_id = await auth_manager.register_user(
        username="testuser",
        email="test@example.com",
        password="testpassword123",
        role=Role.USER
    )
    
    assert user_id is not None
    assert user_id in auth_manager.users


@pytest.mark.asyncio
async def test_login_user(auth_manager):
    """Test user login."""
    await auth_manager.register_user(
        username="testuser",
        email="test@example.com",
        password="testpassword123"
    )
    
    token = await auth_manager.login("testuser", "testpassword123")
    
    assert token is not None
    assert isinstance(token, str)


@pytest.mark.asyncio
async def test_login_invalid_credentials(auth_manager):
    """Test login with invalid credentials."""
    await auth_manager.register_user(
        username="testuser",
        email="test@example.com",
        password="testpassword123"
    )
    
    token = await auth_manager.login("testuser", "wrongpassword")
    
    assert token is None


@pytest.mark.asyncio
async def test_verify_token(auth_manager):
    """Test verifying a token."""
    await auth_manager.register_user(
        username="testuser",
        email="test@example.com",
        password="testpassword123"
    )
    
    token = await auth_manager.login("testuser", "testpassword123")
    user_data = auth_manager.verify_token(token)
    
    assert user_data is not None
    assert user_data["username"] == "testuser"


@pytest.mark.asyncio
async def test_verify_invalid_token(auth_manager):
    """Test verifying an invalid token."""
    user_data = auth_manager.verify_token("invalid_token")
    
    assert user_data is None


@pytest.mark.asyncio
async def test_check_permission(auth_manager):
    """Test checking user permissions."""
    await auth_manager.register_user(
        username="admin",
        email="admin@example.com",
        password="admin123",
        role=Role.ADMIN
    )
    
    token = await auth_manager.login("admin", "admin123")
    user_data = auth_manager.verify_token(token)
    
    has_permission = auth_manager.check_permission(
        user_data["role"],
        "admin_action"
    )
    
    assert has_permission is True


@pytest.mark.asyncio
async def test_get_user_by_id(auth_manager):
    """Test getting user by ID."""
    user_id = await auth_manager.register_user(
        username="testuser",
        email="test@example.com",
        password="test123"
    )
    
    user = auth_manager.get_user_by_id(user_id)
    
    assert user is not None
    assert user["username"] == "testuser"
    assert user["email"] == "test@example.com"


