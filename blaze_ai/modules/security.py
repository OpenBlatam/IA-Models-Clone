"""
Blaze AI Security Module v7.4.0

Advanced security system providing authentication, authorization,
user management, role-based access control, and security auditing.
"""

import asyncio
import hashlib
import hmac
import logging
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Set
from datetime import datetime, timedelta
import jwt
from pathlib import Path
import json

from .base import BaseModule, ModuleConfig, ModuleStatus

logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND CONFIGURATION
# ============================================================================

class AuthenticationMethod(Enum):
    """Authentication methods."""
    PASSWORD = "password"
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    BIOMETRIC = "biometric"
    MULTI_FACTOR = "multi_factor"

class PermissionLevel(Enum):
    """Permission levels."""
    NONE = 0
    READ = 1
    WRITE = 2
    ADMIN = 3
    SUPER_ADMIN = 4

class SecurityEventType(Enum):
    """Security event types."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PERMISSION_DENIED = "permission_denied"
    USER_CREATED = "user_created"
    USER_MODIFIED = "user_modified"
    USER_DELETED = "user_deleted"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"
    API_ACCESS = "api_access"
    SECURITY_ALERT = "security_alert"

class SecurityLevel(Enum):
    """Security levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityConfig(ModuleConfig):
    """Configuration for Security module."""
    # Authentication settings
    enable_password_auth: bool = True
    enable_api_key_auth: bool = True
    enable_jwt_auth: bool = True
    enable_oauth2_auth: bool = False
    enable_biometric_auth: bool = False
    enable_multi_factor: bool = False
    
    # Password settings
    min_password_length: int = 8
    require_special_chars: bool = True
    require_numbers: bool = True
    require_uppercase: bool = True
    password_expiry_days: int = 90
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    
    # JWT settings
    jwt_secret_key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    jwt_refresh_expiry_days: int = 7
    
    # API Key settings
    api_key_length: int = 32
    api_key_expiry_days: int = 365
    
    # Security settings
    enable_audit_logging: bool = True
    enable_rate_limiting: bool = True
    enable_ip_whitelist: bool = False
    ip_whitelist: List[str] = field(default_factory=list)
    session_timeout_minutes: int = 60
    
    # Storage settings
    user_storage_path: str = "./security/users"
    audit_log_path: str = "./security/audit"
    backup_enabled: bool = True
    backup_interval_hours: int = 24

@dataclass
class User:
    """User entity."""
    user_id: str
    username: str
    email: str
    password_hash: str
    salt: str
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    password_changed_at: datetime = field(default_factory=datetime.now)
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    roles: List[str] = field(default_factory=list)
    permissions: Dict[str, PermissionLevel] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Role:
    """Role entity."""
    role_id: str
    name: str
    description: str
    permissions: Dict[str, PermissionLevel] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Permission:
    """Permission entity."""
    permission_id: str
    name: str
    description: str
    resource: str
    action: str
    level: PermissionLevel
    is_active: bool = True

@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str
    event_type: SecurityEventType
    user_id: Optional[str]
    username: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    security_level: SecurityLevel = SecurityLevel.MEDIUM

@dataclass
class SecurityMetrics:
    """Security metrics."""
    total_users: int = 0
    active_users: int = 0
    total_roles: int = 0
    total_permissions: int = 0
    successful_logins: int = 0
    failed_logins: int = 0
    security_events: int = 0
    active_sessions: int = 0
    last_security_scan: Optional[datetime] = None

# ============================================================================
# AUTHENTICATION PROVIDERS
# ============================================================================

class AuthenticationProvider(ABC):
    """Base class for authentication providers."""
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[User]:
        """Authenticate user with credentials."""
        pass
    
    @abstractmethod
    async def validate_credentials(self, credentials: Dict[str, Any]) -> bool:
        """Validate credentials format."""
        pass

class PasswordAuthenticationProvider(AuthenticationProvider):
    """Password-based authentication provider."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[User]:
        """Authenticate user with username and password."""
        username = credentials.get("username")
        password = credentials.get("password")
        
        if not username or not password:
            return None
        
        # This would typically query a user database
        # For now, we'll simulate user lookup
        user = await self._get_user_by_username(username)
        if not user:
            return None
        
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.now():
            return None
        
        # Verify password
        if await self._verify_password(password, user.password_hash, user.salt):
            # Reset failed attempts on successful login
            user.failed_login_attempts = 0
            user.last_login = datetime.now()
            return user
        else:
            # Increment failed attempts
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= self.config.max_login_attempts:
                user.locked_until = datetime.now() + timedelta(minutes=self.config.lockout_duration_minutes)
            return None
    
    async def validate_credentials(self, credentials: Dict[str, Any]) -> bool:
        """Validate password credentials format."""
        username = credentials.get("username")
        password = credentials.get("password")
        
        if not username or not password:
            return False
        
        if len(password) < self.config.min_password_length:
            return False
        
        return True
    
    async def _get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username (simulated)."""
        # This would typically query a database
        # For demonstration, we'll create a simulated user
        if username == "admin":
            return User(
                user_id="admin_001",
                username="admin",
                email="admin@blaze.ai",
                password_hash="hashed_password",
                salt="salt_value",
                roles=["admin"],
                permissions={"*": PermissionLevel.SUPER_ADMIN}
            )
        elif username == "user":
            return User(
                user_id="user_001",
                username="user",
                email="user@blaze.ai",
                password_hash="hashed_password",
                salt="salt_value",
                roles=["user"],
                permissions={"read": PermissionLevel.READ}
            )
        return None
    
    async def _verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash."""
        # This would use proper password hashing (e.g., bcrypt)
        # For demonstration, we'll use a simple hash
        expected_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return hmac.compare_digest(expected_hash, stored_hash)

class APIKeyAuthenticationProvider(AuthenticationProvider):
    """API key-based authentication provider."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.api_keys: Dict[str, Dict[str, Any]] = {}
    
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[User]:
        """Authenticate user with API key."""
        api_key = credentials.get("api_key")
        
        if not api_key:
            return None
        
        # Look up API key
        key_info = self.api_keys.get(api_key)
        if not key_info:
            return None
        
        # Check if key is expired
        if key_info.get("expires_at") and key_info["expires_at"] < datetime.now():
            return None
        
        # Get user associated with this API key
        user = await self._get_user_by_id(key_info["user_id"])
        return user
    
    async def validate_credentials(self, credentials: Dict[str, Any]) -> bool:
        """Validate API key credentials format."""
        api_key = credentials.get("api_key")
        return bool(api_key and len(api_key) >= self.config.api_key_length)
    
    async def _get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID (simulated)."""
        # This would typically query a database
        if user_id == "admin_001":
            return User(
                user_id="admin_001",
                username="admin",
                email="admin@blaze.ai",
                password_hash="",
                salt="",
                roles=["admin"],
                permissions={"*": PermissionLevel.SUPER_ADMIN}
            )
        return None

class JWTAuthenticationProvider(AuthenticationProvider):
    """JWT-based authentication provider."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[User]:
        """Authenticate user with JWT token."""
        token = credentials.get("token")
        
        if not token:
            return None
        
        try:
            # Decode JWT token
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Check if token is expired
            if payload.get("exp") and payload["exp"] < time.time():
                return None
            
            # Get user from payload
            user_id = payload.get("user_id")
            if not user_id:
                return None
            
            user = await self._get_user_by_id(user_id)
            return user
            
        except jwt.InvalidTokenError:
            return None
    
    async def validate_credentials(self, credentials: Dict[str, Any]) -> bool:
        """Validate JWT credentials format."""
        token = credentials.get("token")
        return bool(token)
    
    async def _get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID (simulated)."""
        # This would typically query a database
        if user_id == "admin_001":
            return User(
                user_id="admin_001",
                username="admin",
                email="admin@blaze.ai",
                password_hash="",
                salt="",
                roles=["admin"],
                permissions={"*": PermissionLevel.SUPER_ADMIN}
            )
        return None

# ============================================================================
# AUTHORIZATION SYSTEM
# ============================================================================

class AuthorizationManager:
    """Manages user authorization and permissions."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.roles: Dict[str, Role] = {}
        self.permissions: Dict[str, Permission] = {}
    
    async def check_permission(self, user: User, resource: str, action: str) -> bool:
        """Check if user has permission for resource and action."""
        if not user or not user.is_active:
            return False
        
        # Check super admin permissions
        if user.permissions.get("*") == PermissionLevel.SUPER_ADMIN:
            return True
        
        # Check specific permissions
        permission_key = f"{resource}:{action}"
        user_permission = user.permissions.get(permission_key)
        
        if user_permission and user_permission.value > 0:
            return True
        
        # Check role-based permissions
        for role_name in user.roles:
            role = self.roles.get(role_name)
            if role and role.is_active:
                role_permission = role.permissions.get(permission_key)
                if role_permission and role_permission.value > 0:
                    return True
        
        return False
    
    async def get_user_permissions(self, user: User) -> Dict[str, PermissionLevel]:
        """Get all permissions for a user."""
        if not user or not user.is_active:
            return {}
        
        permissions = user.permissions.copy()
        
        # Add role-based permissions
        for role_name in user.roles:
            role = self.roles.get(role_name)
            if role and role.is_active:
                for perm_key, perm_level in role.permissions.items():
                    if perm_key not in permissions or permissions[perm_key].value < perm_level.value:
                        permissions[perm_key] = perm_level
        
        return permissions
    
    async def add_role(self, role: Role) -> bool:
        """Add a new role."""
        try:
            self.roles[role.role_id] = role
            return True
        except Exception as e:
            logger.error(f"Failed to add role: {e}")
            return False
    
    async def remove_role(self, role_id: str) -> bool:
        """Remove a role."""
        try:
            if role_id in self.roles:
                del self.roles[role_id]
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove role: {e}")
            return False

# ============================================================================
# AUDIT SYSTEM
# ============================================================================

class SecurityAuditor:
    """Handles security event logging and auditing."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.audit_log_path = Path(config.audit_log_path)
        self.audit_log_path.mkdir(parents=True, exist_ok=True)
        self.events: List[SecurityEvent] = []
    
    async def log_event(self, event: SecurityEvent) -> bool:
        """Log a security event."""
        try:
            # Add to memory
            self.events.append(event)
            
            # Write to file
            await self._write_event_to_file(event)
            
            # Check for security alerts
            await self._check_security_alerts(event)
            
            return True
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
            return False
    
    async def get_events(self, 
                        event_type: Optional[SecurityEventType] = None,
                        user_id: Optional[str] = None,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> List[SecurityEvent]:
        """Get security events with optional filtering."""
        filtered_events = self.events
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        
        return sorted(filtered_events, key=lambda x: x.timestamp, reverse=True)
    
    async def _write_event_to_file(self, event: SecurityEvent):
        """Write event to audit log file."""
        try:
            log_file = self.audit_log_path / f"security_{datetime.now().strftime('%Y-%m-%d')}.log"
            
            event_data = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "user_id": event.user_id,
                "username": event.username,
                "ip_address": event.ip_address,
                "user_agent": event.user_agent,
                "timestamp": event.timestamp.isoformat(),
                "details": event.details,
                "security_level": event.security_level.value
            }
            
            with open(log_file, "a") as f:
                f.write(json.dumps(event_data) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to write event to file: {e}")
    
    async def _check_security_alerts(self, event: SecurityEvent):
        """Check if event triggers security alerts."""
        if event.event_type == SecurityEventType.LOGIN_FAILURE:
            # Check for multiple failed login attempts
            recent_failures = await self.get_events(
                event_type=SecurityEventType.LOGIN_FAILURE,
                user_id=event.user_id,
                start_time=datetime.now() - timedelta(minutes=5)
            )
            
            if len(recent_failures) >= 3:
                alert_event = SecurityEvent(
                    event_id=f"alert_{int(time.time())}",
                    event_type=SecurityEventType.SECURITY_ALERT,
                    user_id=event.user_id,
                    username=event.username,
                    ip_address=event.ip_address,
                    details={
                        "alert_type": "multiple_login_failures",
                        "failure_count": len(recent_failures),
                        "time_window": "5 minutes"
                    },
                    security_level=SecurityLevel.HIGH
                )
                
                await self.log_event(alert_event)

# ============================================================================
# MAIN SECURITY MODULE
# ============================================================================

class SecurityModule(BaseModule):
    """Comprehensive security module for Blaze AI system."""
    
    def __init__(self, config: SecurityConfig):
        super().__init__(config)
        self.config = config
        self.metrics = SecurityMetrics()
        
        # Initialize components
        self.auth_providers: Dict[AuthenticationMethod, AuthenticationProvider] = {}
        self.authorization_manager = AuthorizationManager(config)
        self.auditor = SecurityAuditor(config)
        
        # User and session management
        self.users: Dict[str, User] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.rate_limiters: Dict[str, List[float]] = {}
        
        # Initialize authentication providers
        self._setup_auth_providers()
        self._setup_default_roles()
    
    def _setup_auth_providers(self):
        """Setup authentication providers based on configuration."""
        if self.config.enable_password_auth:
            self.auth_providers[AuthenticationMethod.PASSWORD] = PasswordAuthenticationProvider(self.config)
        
        if self.config.enable_api_key_auth:
            self.auth_providers[AuthenticationMethod.API_KEY] = APIKeyAuthenticationProvider(self.config)
        
        if self.config.enable_jwt_auth:
            self.auth_providers[AuthenticationMethod.JWT] = JWTAuthenticationProvider(self.config)
    
    def _setup_default_roles(self):
        """Setup default roles and permissions."""
        # Admin role
        admin_role = Role(
            role_id="admin",
            name="Administrator",
            description="System administrator with full access",
            permissions={
                "*": PermissionLevel.SUPER_ADMIN
            }
        )
        
        # User role
        user_role = Role(
            role_id="user",
            name="User",
            description="Standard user with limited access",
            permissions={
                "read": PermissionLevel.READ,
                "write": PermissionLevel.WRITE
            }
        )
        
        # Guest role
        guest_role = Role(
            role_id="guest",
            name="Guest",
            description="Guest user with read-only access",
            permissions={
                "read": PermissionLevel.READ
            }
        )
        
        asyncio.create_task(self.authorization_manager.add_role(admin_role))
        asyncio.create_task(self.authorization_manager.add_role(user_role))
        asyncio.create_task(self.authorization_manager.add_role(guest_role))
    
    async def initialize(self) -> bool:
        """Initialize the Security module."""
        try:
            await super().initialize()
            
            # Load existing users and data
            await self._load_users()
            
            # Start background tasks
            asyncio.create_task(self._cleanup_expired_sessions())
            asyncio.create_task(self._backup_security_data())
            
            self.status = ModuleStatus.ACTIVE
            logger.info("Security module initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Security module: {e}")
            self.status = ModuleStatus.ERROR
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the Security module."""
        try:
            # Save current state
            await self._save_users()
            
            await super().shutdown()
            logger.info("Security module shutdown successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during Security module shutdown: {e}")
            return False
    
    async def authenticate_user(self, method: AuthenticationMethod, credentials: Dict[str, Any]) -> Optional[User]:
        """Authenticate user using specified method."""
        try:
            provider = self.auth_providers.get(method)
            if not provider:
                logger.warning(f"Authentication method {method} not supported")
                return None
            
            # Validate credentials format
            if not await provider.validate_credentials(credentials):
                logger.warning(f"Invalid credentials format for {method}")
                return None
            
            # Authenticate user
            user = await provider.authenticate(credentials)
            
            if user:
                # Log successful authentication
                event = SecurityEvent(
                    event_id=f"auth_{int(time.time())}",
                    event_type=SecurityEventType.LOGIN_SUCCESS,
                    user_id=user.user_id,
                    username=user.username,
                    ip_address=credentials.get("ip_address"),
                    user_agent=credentials.get("user_agent"),
                    details={"method": method.value}
                )
                await self.auditor.log_event(event)
                
                # Update metrics
                self.metrics.successful_logins += 1
                self.metrics.active_users = len([u for u in self.users.values() if u.is_active])
                
                # Create session
                await self._create_session(user)
                
            else:
                # Log failed authentication
                event = SecurityEvent(
                    event_id=f"auth_fail_{int(time.time())}",
                    event_type=SecurityEventType.LOGIN_FAILURE,
                    username=credentials.get("username"),
                    ip_address=credentials.get("ip_address"),
                    user_agent=credentials.get("user_agent"),
                    details={"method": method.value, "reason": "invalid_credentials"}
                )
                await self.auditor.log_event(event)
                
                # Update metrics
                self.metrics.failed_logins += 1
            
            return user
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return None
    
    async def check_permission(self, user: User, resource: str, action: str) -> bool:
        """Check if user has permission for resource and action."""
        try:
            has_permission = await self.authorization_manager.check_permission(user, resource, action)
            
            if not has_permission:
                # Log permission denied event
                event = SecurityEvent(
                    event_id=f"perm_denied_{int(time.time())}",
                    event_type=SecurityEventType.PERMISSION_DENIED,
                    user_id=user.user_id,
                    username=user.username,
                    details={
                        "resource": resource,
                        "action": action,
                        "user_roles": user.roles,
                        "user_permissions": {k: v.value for k, v in user.permissions.items()}
                    },
                    security_level=SecurityLevel.MEDIUM
                )
                await self.auditor.log_event(event)
            
            return has_permission
            
        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False
    
    async def create_user(self, username: str, email: str, password: str, roles: List[str] = None) -> Optional[User]:
        """Create a new user."""
        try:
            # Validate input
            if not username or not email or not password:
                return None
            
            if len(password) < self.config.min_password_length:
                return None
            
            # Check if user already exists
            if any(u.username == username for u in self.users.values()):
                return None
            
            # Generate salt and hash password
            salt = secrets.token_hex(16)
            password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            
            # Create user
            user = User(
                user_id=f"user_{int(time.time())}",
                username=username,
                email=email,
                password_hash=password_hash,
                salt=salt,
                roles=roles or ["user"]
            )
            
            # Add user
            self.users[user.user_id] = user
            
            # Log event
            event = SecurityEvent(
                event_id=f"user_created_{int(time.time())}",
                event_type=SecurityEventType.USER_CREATED,
                user_id=user.user_id,
                username=username,
                details={"email": email, "roles": roles}
            )
            await self.auditor.log_event(event)
            
            # Update metrics
            self.metrics.total_users += 1
            self.metrics.active_users += 1
            
            return user
            
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            return None
    
    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user information."""
        try:
            user = self.users.get(user_id)
            if not user:
                return False
            
            # Update fields
            for key, value in updates.items():
                if hasattr(user, key) and key not in ["user_id", "created_at"]:
                    setattr(user, key, value)
            
            # Log event
            event = SecurityEvent(
                event_id=f"user_modified_{int(time.time())}",
                event_type=SecurityEventType.USER_MODIFIED,
                user_id=user_id,
                username=user.username,
                details={"updates": updates}
            )
            await self.auditor.log_event(event)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user: {e}")
            return False
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        try:
            user = self.users.get(user_id)
            if not user:
                return False
            
            # Log event before deletion
            event = SecurityEvent(
                event_id=f"user_deleted_{int(time.time())}",
                event_type=SecurityEventType.USER_DELETED,
                user_id=user_id,
                username=user.username,
                details={"deleted_at": datetime.now().isoformat()}
            )
            await self.auditor.log_event(event)
            
            # Remove user
            del self.users[user_id]
            
            # Update metrics
            self.metrics.total_users -= 1
            if user.is_active:
                self.metrics.active_users -= 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete user: {e}")
            return False
    
    async def assign_role(self, user_id: str, role_name: str) -> bool:
        """Assign a role to a user."""
        try:
            user = self.users.get(user_id)
            if not user:
                return False
            
            if role_name not in user.roles:
                user.roles.append(role_name)
                
                # Log event
                event = SecurityEvent(
                    event_id=f"role_assigned_{int(time.time())}",
                    event_type=SecurityEventType.ROLE_ASSIGNED,
                    user_id=user_id,
                    username=user.username,
                    details={"role": role_name}
                )
                await self.auditor.log_event(event)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to assign role: {e}")
            return False
    
    async def revoke_role(self, user_id: str, role_name: str) -> bool:
        """Revoke a role from a user."""
        try:
            user = self.users.get(user_id)
            if not user:
                return False
            
            if role_name in user.roles:
                user.roles.remove(role_name)
                
                # Log event
                event = SecurityEvent(
                    event_id=f"role_revoked_{int(time.time())}",
                    event_type=SecurityEventType.ROLE_REVOKED,
                    user_id=user_id,
                    username=user.username,
                    details={"role": role_name}
                )
                await self.auditor.log_event(event)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke role: {e}")
            return False
    
    async def get_security_events(self, **filters) -> List[SecurityEvent]:
        """Get security events with optional filtering."""
        return await self.auditor.get_events(**filters)
    
    async def _create_session(self, user: User):
        """Create a new session for user."""
        try:
            session_id = secrets.token_urlsafe(32)
            session_data = {
                "user_id": user.user_id,
                "username": user.username,
                "created_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(minutes=self.config.session_timeout_minutes)
            }
            
            self.active_sessions[session_id] = session_data
            self.metrics.active_sessions = len(self.active_sessions)
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        while self.status == ModuleStatus.ACTIVE:
            try:
                current_time = datetime.now()
                expired_sessions = [
                    session_id for session_id, session_data in self.active_sessions.items()
                    if session_data["expires_at"] < current_time
                ]
                
                for session_id in expired_sessions:
                    del self.active_sessions[session_id]
                
                if expired_sessions:
                    self.metrics.active_sessions = len(self.active_sessions)
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Session cleanup failed: {e}")
                await asyncio.sleep(60)
    
    async def _backup_security_data(self):
        """Backup security data periodically."""
        while self.status == ModuleStatus.ACTIVE:
            try:
                if self.config.backup_enabled:
                    await self._save_users()
                    logger.info("Security data backup completed")
                
                await asyncio.sleep(self.config.backup_interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Security data backup failed: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    async def _load_users(self):
        """Load users from storage."""
        try:
            user_file = Path(self.config.user_storage_path) / "users.json"
            if user_file.exists():
                with open(user_file, "r") as f:
                    user_data = json.load(f)
                
                for user_dict in user_data:
                    # Convert datetime strings back to datetime objects
                    for field in ["created_at", "last_login", "password_changed_at", "locked_until"]:
                        if field in user_dict and user_dict[field]:
                            user_dict[field] = datetime.fromisoformat(user_dict[field])
                    
                    user = User(**user_dict)
                    self.users[user.user_id] = user
                
                self.metrics.total_users = len(self.users)
                self.metrics.active_users = len([u for u in self.users.values() if u.is_active])
                
                logger.info(f"Loaded {len(self.users)} users from storage")
                
        except Exception as e:
            logger.error(f"Failed to load users: {e}")
    
    async def _save_users(self):
        """Save users to storage."""
        try:
            user_dir = Path(self.config.user_storage_path)
            user_dir.mkdir(parents=True, exist_ok=True)
            
            user_file = user_dir / "users.json"
            
            # Convert users to serializable format
            user_data = []
            for user in self.users.values():
                user_dict = {
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "password_hash": user.password_hash,
                    "salt": user.salt,
                    "is_active": user.is_active,
                    "is_verified": user.is_verified,
                    "created_at": user.created_at.isoformat(),
                    "last_login": user.last_login.isoformat() if user.last_login else None,
                    "password_changed_at": user.password_changed_at.isoformat(),
                    "failed_login_attempts": user.failed_login_attempts,
                    "locked_until": user.locked_until.isoformat() if user.locked_until else None,
                    "roles": user.roles,
                    "permissions": {k: v.value for k, v in user.permissions.items()},
                    "metadata": user.metadata
                }
                user_data.append(user_dict)
            
            with open(user_file, "w") as f:
                json.dump(user_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save users: {e}")
    
    async def get_metrics(self) -> SecurityMetrics:
        """Get security metrics."""
        return self.metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Check module health."""
        health = await super().health_check()
        health["total_users"] = self.metrics.total_users
        health["active_sessions"] = self.metrics.active_sessions
        health["auth_providers"] = len(self.auth_providers)
        health["roles"] = len(self.authorization_manager.roles)
        return health

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_security_module(config: Optional[SecurityConfig] = None) -> SecurityModule:
    """Create Security module."""
    if config is None:
        config = SecurityConfig()
    return SecurityModule(config)

def create_security_module_with_defaults(**kwargs) -> SecurityModule:
    """Create Security module with default configuration."""
    config = SecurityConfig(**kwargs)
    return SecurityModule(config)

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "AuthenticationMethod",
    "PermissionLevel",
    "SecurityEventType",
    "SecurityLevel",
    
    # Configuration and Data Classes
    "SecurityConfig",
    "User",
    "Role",
    "Permission",
    "SecurityEvent",
    "SecurityMetrics",
    
    # Authentication Providers
    "AuthenticationProvider",
    "PasswordAuthenticationProvider",
    "APIKeyAuthenticationProvider",
    "JWTAuthenticationProvider",
    
    # Authorization and Audit
    "AuthorizationManager",
    "SecurityAuditor",
    
    # Main Module
    "SecurityModule",
    
    # Factory Functions
    "create_security_module",
    "create_security_module_with_defaults"
]
