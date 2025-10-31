"""
Tests for Blaze AI Security Module

This test suite covers all security functionality including authentication,
authorization, user management, role management, and security auditing.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from ..modules.security import (
    SecurityModule,
    SecurityConfig,
    User,
    Role,
    Permission,
    SecurityEvent,
    SecurityEventType,
    SecurityLevel,
    PermissionLevel,
    AuthenticationMethod,
    PasswordAuthenticationProvider,
    APIKeyAuthenticationProvider,
    JWTAuthenticationProvider,
    AuthorizationManager,
    SecurityAuditor
)

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def security_config():
    """Create a test security configuration."""
    return SecurityConfig(
        enable_password_auth=True,
        enable_api_key_auth=True,
        enable_jwt_auth=True,
        min_password_length=6,
        max_login_attempts=3,
        lockout_duration_minutes=15,
        session_timeout_minutes=30,
        user_storage_path="./test_security/users",
        audit_log_path="./test_security/audit"
    )

@pytest.fixture
def security_module(security_config):
    """Create a test security module."""
    return SecurityModule(security_config)

@pytest.fixture
def test_user():
    """Create a test user."""
    return User(
        user_id="test_user_001",
        username="testuser",
        email="test@example.com",
        password_hash="hashed_password",
        salt="test_salt",
        roles=["user"],
        permissions={"read": PermissionLevel.READ}
    )

@pytest.fixture
def admin_user():
    """Create an admin user."""
    return User(
        user_id="admin_001",
        username="admin",
        email="admin@example.com",
        password_hash="hashed_password",
        salt="admin_salt",
        roles=["admin"],
        permissions={"*": PermissionLevel.SUPER_ADMIN}
    )

@pytest.fixture
def test_role():
    """Create a test role."""
    return Role(
        role_id="test_role",
        name="Test Role",
        description="A test role",
        permissions={"test:read": PermissionLevel.READ}
    )

# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

def test_security_config_defaults():
    """Test security configuration default values."""
    config = SecurityConfig()
    
    assert config.enable_password_auth is True
    assert config.enable_api_key_auth is True
    assert config.enable_jwt_auth is True
    assert config.min_password_length == 8
    assert config.max_login_attempts == 5
    assert config.lockout_duration_minutes == 30
    assert config.session_timeout_minutes == 60

def test_security_config_custom():
    """Test security configuration with custom values."""
    config = SecurityConfig(
        enable_password_auth=False,
        min_password_length=12,
        max_login_attempts=10
    )
    
    assert config.enable_password_auth is False
    assert config.min_password_length == 12
    assert config.max_login_attempts == 10

# ============================================================================
# USER ENTITY TESTS
# ============================================================================

def test_user_creation():
    """Test user entity creation."""
    user = User(
        user_id="test_001",
        username="testuser",
        email="test@example.com",
        password_hash="hash",
        salt="salt"
    )
    
    assert user.user_id == "test_001"
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert user.is_active is True
    assert user.is_verified is False
    assert len(user.roles) == 0
    assert len(user.permissions) == 0

def test_user_with_roles_and_permissions():
    """Test user with roles and permissions."""
    user = User(
        user_id="test_002",
        username="poweruser",
        email="power@example.com",
        password_hash="hash",
        salt="salt",
        roles=["user", "moderator"],
        permissions={"read": PermissionLevel.READ, "write": PermissionLevel.WRITE}
    )
    
    assert "user" in user.roles
    assert "moderator" in user.roles
    assert user.permissions["read"] == PermissionLevel.READ
    assert user.permissions["write"] == PermissionLevel.WRITE

# ============================================================================
# ROLE AND PERMISSION TESTS
# ============================================================================

def test_role_creation():
    """Test role entity creation."""
    role = Role(
        role_id="moderator",
        name="Moderator",
        description="Content moderator role",
        permissions={"content:moderate": PermissionLevel.WRITE}
    )
    
    assert role.role_id == "moderator"
    assert role.name == "Moderator"
    assert role.description == "Content moderator role"
    assert role.permissions["content:moderate"] == PermissionLevel.WRITE
    assert role.is_active is True

def test_permission_creation():
    """Test permission entity creation."""
    permission = Permission(
        permission_id="user:create",
        name="Create User",
        description="Permission to create new users",
        resource="user",
        action="create",
        level=PermissionLevel.ADMIN
    )
    
    assert permission.permission_id == "user:create"
    assert permission.resource == "user"
    assert permission.action == "create"
    assert permission.level == PermissionLevel.ADMIN

# ============================================================================
# SECURITY EVENT TESTS
# ============================================================================

def test_security_event_creation():
    """Test security event creation."""
    event = SecurityEvent(
        event_id="event_001",
        event_type=SecurityEventType.LOGIN_SUCCESS,
        user_id="user_001",
        username="testuser",
        ip_address="192.168.1.100",
        user_agent="Test Browser",
        details={"method": "password"},
        security_level=SecurityLevel.LOW
    )
    
    assert event.event_id == "event_001"
    assert event.event_type == SecurityEventType.LOGIN_SUCCESS
    assert event.user_id == "user_001"
    assert event.username == "testuser"
    assert event.ip_address == "192.168.1.100"
    assert event.security_level == SecurityLevel.LOW

# ============================================================================
# AUTHENTICATION PROVIDER TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_password_auth_provider_validation():
    """Test password authentication provider credential validation."""
    config = SecurityConfig(min_password_length=8)
    provider = PasswordAuthenticationProvider(config)
    
    # Valid credentials
    valid_creds = {"username": "testuser", "password": "password123"}
    assert await provider.validate_credentials(valid_creds) is True
    
    # Invalid credentials - missing username
    invalid_creds1 = {"password": "password123"}
    assert await provider.validate_credentials(invalid_creds1) is False
    
    # Invalid credentials - password too short
    invalid_creds2 = {"username": "testuser", "password": "123"}
    assert await provider.validate_credentials(invalid_creds2) is False

@pytest.mark.asyncio
async def test_api_key_auth_provider_validation():
    """Test API key authentication provider credential validation."""
    config = SecurityConfig(api_key_length=16)
    provider = APIKeyAuthenticationProvider(config)
    
    # Valid API key
    valid_creds = {"api_key": "1234567890123456"}
    assert await provider.validate_credentials(valid_creds) is True
    
    # Invalid API key - too short
    invalid_creds = {"api_key": "123"}
    assert await provider.validate_credentials(invalid_creds) is False
    
    # Invalid API key - missing
    invalid_creds2 = {}
    assert await provider.validate_credentials(invalid_creds2) is False

@pytest.mark.asyncio
async def test_jwt_auth_provider_validation():
    """Test JWT authentication provider credential validation."""
    config = SecurityConfig()
    provider = JWTAuthenticationProvider(config)
    
    # Valid JWT token
    valid_creds = {"token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"}
    assert await provider.validate_credentials(valid_creds) is True
    
    # Invalid JWT token - missing
    invalid_creds = {}
    assert await provider.validate_credentials(invalid_creds) is False

# ============================================================================
# AUTHORIZATION MANAGER TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_authorization_manager_permission_check():
    """Test authorization manager permission checking."""
    config = SecurityConfig()
    auth_manager = AuthorizationManager(config)
    
    # Add test role
    role = Role(
        role_id="tester",
        name="Tester",
        description="Test role",
        permissions={"test:read": PermissionLevel.READ, "test:write": PermissionLevel.WRITE}
    )
    await auth_manager.add_role(role)
    
    # Test user with specific permissions
    user = User(
        user_id="test_001",
        username="testuser",
        email="test@example.com",
        password_hash="hash",
        salt="salt",
        permissions={"admin:full": PermissionLevel.ADMIN}
    )
    
    # Check specific permissions
    assert await auth_manager.check_permission(user, "admin", "full") is True
    assert await auth_manager.check_permission(user, "test", "read") is False
    
    # Test user with roles
    role_user = User(
        user_id="role_user_001",
        username="roleuser",
        email="role@example.com",
        password_hash="hash",
        salt="salt",
        roles=["tester"]
    )
    
    assert await auth_manager.check_permission(role_user, "test", "read") is True
    assert await auth_manager.check_permission(role_user, "test", "write") is True
    assert await auth_manager.check_permission(role_user, "test", "delete") is False

@pytest.mark.asyncio
async def test_authorization_manager_super_admin():
    """Test authorization manager super admin permissions."""
    config = SecurityConfig()
    auth_manager = AuthorizationManager(config)
    
    # Super admin user
    super_admin = User(
        user_id="super_001",
        username="superadmin",
        email="super@example.com",
        password_hash="hash",
        salt="salt",
        permissions={"*": PermissionLevel.SUPER_ADMIN}
    )
    
    # Super admin should have access to everything
    assert await auth_manager.check_permission(super_admin, "any", "resource") is True
    assert await auth_manager.check_permission(super_admin, "system", "shutdown") is True

# ============================================================================
# SECURITY AUDITOR TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_security_auditor_event_logging():
    """Test security auditor event logging."""
    config = SecurityConfig()
    auditor = SecurityAuditor(config)
    
    # Create test event
    event = SecurityEvent(
        event_id="test_event_001",
        event_type=SecurityEventType.LOGIN_SUCCESS,
        user_id="user_001",
        username="testuser",
        ip_address="192.168.1.100"
    )
    
    # Log event
    success = await auditor.log_event(event)
    assert success is True
    
    # Get events
    events = await auditor.get_events()
    assert len(events) == 1
    assert events[0].event_id == "test_event_001"

@pytest.mark.asyncio
async def test_security_auditor_event_filtering():
    """Test security auditor event filtering."""
    config = SecurityConfig()
    auditor = SecurityAuditor(config)
    
    # Create multiple events
    events = [
        SecurityEvent(
            event_id=f"event_{i}",
            event_type=SecurityEventType.LOGIN_SUCCESS if i % 2 == 0 else SecurityEventType.LOGIN_FAILURE,
            user_id=f"user_{i}",
            username=f"user{i}",
            timestamp=datetime.now() + timedelta(minutes=i)
        )
        for i in range(5)
    ]
    
    # Log all events
    for event in events:
        await auditor.log_event(event)
    
    # Test filtering by event type
    login_success_events = await auditor.get_events(event_type=SecurityEventType.LOGIN_SUCCESS)
    assert len(login_success_events) == 3
    
    login_failure_events = await auditor.get_events(event_type=SecurityEventType.LOGIN_FAILURE)
    assert len(login_failure_events) == 2
    
    # Test filtering by user
    user_events = await auditor.get_events(user_id="user_1")
    assert len(user_events) == 1

# ============================================================================
# SECURITY MODULE INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_security_module_initialization():
    """Test security module initialization."""
    config = SecurityConfig()
    module = SecurityModule(config)
    
    # Initialize module
    success = await module.initialize()
    assert success is True
    assert module.status.value == "ACTIVE"
    
    # Check that providers were set up
    assert AuthenticationMethod.PASSWORD in module.auth_providers
    assert AuthenticationMethod.API_KEY in module.auth_providers
    assert AuthenticationMethod.JWT in module.auth_providers
    
    # Cleanup
    await module.shutdown()

@pytest.mark.asyncio
async def test_security_module_user_creation():
    """Test security module user creation."""
    config = SecurityConfig()
    module = SecurityModule(config)
    await module.initialize()
    
    # Create user
    user = await module.create_user(
        username="newuser",
        email="new@example.com",
        password="password123",
        roles=["user"]
    )
    
    assert user is not None
    assert user.username == "newuser"
    assert user.email == "new@example.com"
    assert "user" in user.roles
    
    # Check metrics
    metrics = await module.get_metrics()
    assert metrics.total_users > 0
    assert metrics.active_users > 0
    
    # Cleanup
    await module.shutdown()

@pytest.mark.asyncio
async def test_security_module_authentication():
    """Test security module authentication."""
    config = SecurityConfig()
    module = SecurityModule(config)
    await module.initialize()
    
    # Create user first
    user = await module.create_user(
        username="authuser",
        email="auth@example.com",
        password="password123"
    )
    
    # Test authentication
    auth_user = await module.authenticate_user(
        AuthenticationMethod.PASSWORD,
        {
            "username": "authuser",
            "password": "password123",
            "ip_address": "192.168.1.100"
        }
    )
    
    assert auth_user is not None
    assert auth_user.username == "authuser"
    
    # Test failed authentication
    failed_user = await module.authenticate_user(
        AuthenticationMethod.PASSWORD,
        {
            "username": "authuser",
            "password": "wrongpassword",
            "ip_address": "192.168.1.100"
        }
    )
    
    assert failed_user is None
    
    # Cleanup
    await module.shutdown()

@pytest.mark.asyncio
async def test_security_module_permission_checking():
    """Test security module permission checking."""
    config = SecurityConfig()
    module = SecurityModule(config)
    await module.initialize()
    
    # Create user with specific permissions
    user = await module.create_user(
        username="permuser",
        email="perm@example.com",
        password="password123",
        roles=["user"]
    )
    
    # Check permissions
    has_read = await module.check_permission(user, "public", "read")
    has_write = await module.check_permission(user, "private", "write")
    
    # User should have read access but not write access to private resources
    assert has_read is True  # Based on default user role
    assert has_write is False
    
    # Cleanup
    await module.shutdown()

@pytest.mark.asyncio
async def test_security_module_role_management():
    """Test security module role management."""
    config = SecurityConfig()
    module = SecurityModule(config)
    await module.initialize()
    
    # Create user
    user = await module.create_user(
        username="roleuser",
        email="role@example.com",
        password="password123"
    )
    
    # Assign role
    success = await module.assign_role(user.user_id, "tester")
    assert success is True
    
    # Check that role was assigned
    updated_user = None
    for u in module.users.values():
        if u.user_id == user.user_id:
            updated_user = u
            break
    
    assert "tester" in updated_user.roles
    
    # Revoke role
    success = await module.revoke_role(user.user_id, "tester")
    assert success is True
    
    # Check that role was revoked
    for u in module.users.values():
        if u.user_id == user.user_id:
            updated_user = u
            break
    
    assert "tester" not in updated_user.roles
    
    # Cleanup
    await module.shutdown()

@pytest.mark.asyncio
async def test_security_module_audit_logging():
    """Test security module audit logging."""
    config = SecurityConfig()
    module = SecurityModule(config)
    await module.initialize()
    
    # Create user and authenticate to generate events
    user = await module.create_user(
        username="audituser",
        email="audit@example.com",
        password="password123"
    )
    
    await module.authenticate_user(
        AuthenticationMethod.PASSWORD,
        {
            "username": "audituser",
            "password": "password123",
            "ip_address": "192.168.1.100"
        }
    )
    
    # Get security events
    events = await module.get_security_events()
    assert len(events) > 0
    
    # Should have user creation and login success events
    event_types = [event.event_type for event in events]
    assert SecurityEventType.USER_CREATED in event_types
    assert SecurityEventType.LOGIN_SUCCESS in event_types
    
    # Cleanup
    await module.shutdown()

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_security_module_performance():
    """Test security module performance under load."""
    config = SecurityConfig()
    module = SecurityModule(config)
    await module.initialize()
    
    # Create multiple users
    start_time = time.time()
    
    for i in range(100):
        await module.create_user(
            username=f"perfuser{i}",
            email=f"perf{i}@example.com",
            password="password123"
        )
    
    creation_time = time.time() - start_time
    
    # Performance should be reasonable (less than 1 second for 100 users)
    assert creation_time < 1.0
    
    # Test authentication performance
    start_time = time.time()
    
    for i in range(50):
        await module.authenticate_user(
            AuthenticationMethod.PASSWORD,
            {
                "username": f"perfuser{i}",
                "password": "password123"
            }
        )
    
    auth_time = time.time() - start_time
    
    # Authentication should be fast (less than 0.5 seconds for 50 auths)
    assert auth_time < 0.5
    
    # Cleanup
    await module.shutdown()

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_security_module_error_handling():
    """Test security module error handling."""
    config = SecurityConfig()
    module = SecurityModule(config)
    await module.initialize()
    
    # Test invalid user creation
    invalid_user = await module.create_user("", "", "")
    assert invalid_user is None
    
    # Test invalid authentication
    invalid_auth = await module.authenticate_user(
        AuthenticationMethod.PASSWORD,
        {}
    )
    assert invalid_auth is None
    
    # Test permission check with invalid user
    has_perm = await module.check_permission(None, "resource", "action")
    assert has_perm is False
    
    # Cleanup
    await module.shutdown()

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
