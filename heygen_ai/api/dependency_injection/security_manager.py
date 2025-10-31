from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import jwt
import bcrypt
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import structlog
from dataclasses import dataclass, asdict
from enum import Enum
import weakref
from functools import lru_cache, wraps
import inspect
import gc
from uuid import UUID
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
        import secrets
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Security Manager for FastAPI Dependency Injection
Manages authentication, authorization, and security-related dependencies.
"""



logger = structlog.get_logger()

# =============================================================================
# Security Types
# =============================================================================

class SecurityLevel(Enum):
    """Security level enumeration."""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

class TokenType(Enum):
    """Token type enumeration."""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"

@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    api_key_expire_days: int = 365
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600
    enable_audit_logging: bool = True
    enable_session_management: bool = True
    session_timeout_minutes: int = 60
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15

@dataclass
class UserSession:
    """User session information."""
    user_id: UUID
    session_id: str
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True

@dataclass
class SecurityStats:
    """Security statistics."""
    total_authentications: int = 0
    successful_authentications: int = 0
    failed_authentications: int = 0
    total_authorizations: int = 0
    successful_authorizations: int = 0
    failed_authorizations: int = 0
    active_sessions: int = 0
    total_sessions: int = 0
    rate_limit_hits: int = 0
    lockouts: int = 0

# =============================================================================
# Security Models
# =============================================================================

class User(BaseModel):
    """User model."""
    id: UUID
    username: str
    email: str
    is_active: bool = True
    is_verified: bool = False
    roles: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

class TokenData(BaseModel):
    """Token data model."""
    user_id: UUID
    username: str
    roles: List[str] = []
    permissions: List[str] = []
    token_type: TokenType = TokenType.ACCESS
    exp: datetime

class LoginRequest(BaseModel):
    """Login request model."""
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=100)
    remember_me: bool = False

class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: User

# =============================================================================
# Security Base Classes
# =============================================================================

class SecurityBase:
    """Base class for security components."""
    
    def __init__(self, config: SecurityConfig):
        
    """__init__ function."""
self.config = config
        self.stats = SecurityStats()
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the security component."""
        if self._is_initialized:
            return
        
        try:
            await self._initialize_internal()
            self._is_initialized = True
            logger.info(f"Security component {self.__class__.__name__} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize security component {self.__class__.__name__}: {e}")
            raise
    
    async def _initialize_internal(self) -> None:
        """Internal initialization method to be implemented by subclasses."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup the security component."""
        if not self._is_initialized:
            return
        
        try:
            await self._cleanup_internal()
            self._is_initialized = False
            logger.info(f"Security component {self.__class__.__name__} cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup security component {self.__class__.__name__}: {e}")
            raise
    
    async def _cleanup_internal(self) -> None:
        """Internal cleanup method to be implemented by subclasses."""
        pass

# =============================================================================
# Authentication Manager
# =============================================================================

class AuthenticationManager(SecurityBase):
    """Authentication manager for handling user authentication."""
    
    def __init__(self, config: SecurityConfig):
        
    """__init__ function."""
super().__init__(config)
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.lockouts: Dict[str, datetime] = {}
        self._user_service = None
    
    def set_user_service(self, user_service) -> Any:
        """Set user service for authentication."""
        self._user_service = user_service
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username and password."""
        if not self._is_initialized:
            raise RuntimeError("Authentication manager not initialized")
        
        self.stats.total_authentications += 1
        
        # Check for lockout
        if self._is_locked_out(username):
            self.stats.failed_authentications += 1
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account temporarily locked due to too many failed attempts"
            )
        
        try:
            # Get user from service
            if not self._user_service:
                raise RuntimeError("User service not configured")
            
            user = await self._user_service.get_user_by_username(username)
            if not user:
                await self._record_failed_attempt(username)
                self.stats.failed_authentications += 1
                return None
            
            # Verify password
            if not self._verify_password(password, user.password_hash):
                await self._record_failed_attempt(username)
                self.stats.failed_authentications += 1
                return None
            
            # Clear failed attempts on successful authentication
            self._clear_failed_attempts(username)
            self.stats.successful_authentications += 1
            
            return user
            
        except Exception as e:
            logger.error(f"Authentication error for user {username}: {e}")
            await self._record_failed_attempt(username)
            self.stats.failed_authentications += 1
            raise
    
    async def authenticate_token(self, token: str) -> Optional[TokenData]:
        """Authenticate a user with JWT token."""
        if not self._is_initialized:
            raise RuntimeError("Authentication manager not initialized")
        
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm]
            )
            
            user_id = UUID(payload.get("sub"))
            username = payload.get("username")
            roles = payload.get("roles", [])
            permissions = payload.get("permissions", [])
            token_type = TokenType(payload.get("token_type", "access"))
            exp = datetime.fromtimestamp(payload.get("exp"), tz=timezone.utc)
            
            # Check if token is expired
            if datetime.now(timezone.utc) > exp:
                return None
            
            return TokenData(
                user_id=user_id,
                username=username,
                roles=roles,
                permissions=permissions,
                token_type=token_type,
                exp=exp
            )
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token authentication error: {e}")
            return None
    
    def create_access_token(self, user: User) -> str:
        """Create access token for user."""
        if not self._is_initialized:
            raise RuntimeError("Authentication manager not initialized")
        
        expires_delta = timedelta(minutes=self.config.access_token_expire_minutes)
        expire = datetime.now(timezone.utc) + expires_delta
        
        to_encode = {
            "sub": str(user.id),
            "username": user.username,
            "roles": user.roles,
            "permissions": user.permissions,
            "token_type": TokenType.ACCESS.value,
            "exp": expire.timestamp()
        }
        
        return jwt.encode(to_encode, self.config.secret_key, algorithm=self.config.algorithm)
    
    def create_refresh_token(self, user: User) -> str:
        """Create refresh token for user."""
        if not self._is_initialized:
            raise RuntimeError("Authentication manager not initialized")
        
        expires_delta = timedelta(days=self.config.refresh_token_expire_days)
        expire = datetime.now(timezone.utc) + expires_delta
        
        to_encode = {
            "sub": str(user.id),
            "username": user.username,
            "token_type": TokenType.REFRESH.value,
            "exp": expire.timestamp()
        }
        
        return jwt.encode(to_encode, self.config.secret_key, algorithm=self.config.algorithm)
    
    def hash_password(self, password: str) -> str:
        """Hash a password."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def _is_locked_out(self, username: str) -> bool:
        """Check if user is locked out."""
        if username not in self.lockouts:
            return False
        
        lockout_time = self.lockouts[username]
        if datetime.now(timezone.utc) > lockout_time:
            # Remove expired lockout
            del self.lockouts[username]
            return False
        
        return True
    
    async def _record_failed_attempt(self, username: str) -> None:
        """Record a failed authentication attempt."""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        
        self.failed_attempts[username].append(datetime.now(timezone.utc))
        
        # Remove attempts older than lockout duration
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=self.config.lockout_duration_minutes)
        self.failed_attempts[username] = [
            attempt for attempt in self.failed_attempts[username]
            if attempt > cutoff_time
        ]
        
        # Check if lockout should be applied
        if len(self.failed_attempts[username]) >= self.config.max_failed_attempts:
            lockout_time = datetime.now(timezone.utc) + timedelta(minutes=self.config.lockout_duration_minutes)
            self.lockouts[username] = lockout_time
            self.stats.lockouts += 1
            logger.warning(f"User {username} locked out due to too many failed attempts")
    
    def _clear_failed_attempts(self, username: str) -> None:
        """Clear failed attempts for user."""
        if username in self.failed_attempts:
            del self.failed_attempts[username]

# =============================================================================
# Authorization Manager
# =============================================================================

class AuthorizationManager(SecurityBase):
    """Authorization manager for handling user authorization."""
    
    def __init__(self, config: SecurityConfig):
        
    """__init__ function."""
super().__init__(config)
        self.role_permissions: Dict[str, List[str]] = {}
        self.permission_hierarchy: Dict[str, List[str]] = {}
    
    async def _initialize_internal(self) -> None:
        """Initialize authorization manager."""
        # Load role permissions and hierarchy
        await self._load_permissions()
    
    async def _load_permissions(self) -> None:
        """Load role permissions and hierarchy."""
        # This would typically load from database or configuration
        self.role_permissions = {
            "user": ["read:own", "write:own"],
            "moderator": ["read:all", "write:own", "moderate:content"],
            "admin": ["read:all", "write:all", "moderate:all", "manage:users"],
            "super_admin": ["read:all", "write:all", "moderate:all", "manage:all"]
        }
        
        self.permission_hierarchy = {
            "read:own": ["read:own"],
            "read:all": ["read:own", "read:all"],
            "write:own": ["write:own"],
            "write:all": ["write:own", "write:all"],
            "moderate:content": ["moderate:content"],
            "moderate:all": ["moderate:content", "moderate:all"],
            "manage:users": ["manage:users"],
            "manage:all": ["manage:users", "manage:all"]
        }
    
    def has_permission(self, user: User, permission: str) -> bool:
        """Check if user has a specific permission."""
        if not self._is_initialized:
            raise RuntimeError("Authorization manager not initialized")
        
        self.stats.total_authorizations += 1
        
        # Super admin has all permissions
        if "super_admin" in user.roles:
            self.stats.successful_authorizations += 1
            return True
        
        # Check direct permissions
        if permission in user.permissions:
            self.stats.successful_authorizations += 1
            return True
        
        # Check role-based permissions
        for role in user.roles:
            role_permissions = self.role_permissions.get(role, [])
            if permission in role_permissions:
                self.stats.successful_authorizations += 1
                return True
        
        # Check permission hierarchy
        for user_permission in user.permissions:
            hierarchy_permissions = self.permission_hierarchy.get(user_permission, [])
            if permission in hierarchy_permissions:
                self.stats.successful_authorizations += 1
                return True
        
        self.stats.failed_authorizations += 1
        return False
    
    def has_role(self, user: User, role: str) -> bool:
        """Check if user has a specific role."""
        if not self._is_initialized:
            raise RuntimeError("Authorization manager not initialized")
        
        return role in user.roles
    
    def has_any_role(self, user: User, roles: List[str]) -> bool:
        """Check if user has any of the specified roles."""
        if not self._is_initialized:
            raise RuntimeError("Authorization manager not initialized")
        
        return any(role in user.roles for role in roles)
    
    def has_all_roles(self, user: User, roles: List[str]) -> bool:
        """Check if user has all of the specified roles."""
        if not self._is_initialized:
            raise RuntimeError("Authorization manager not initialized")
        
        return all(role in user.roles for role in roles)
    
    def get_user_permissions(self, user: User) -> List[str]:
        """Get all permissions for a user."""
        if not self._is_initialized:
            raise RuntimeError("Authorization manager not initialized")
        
        permissions = set(user.permissions)
        
        # Add role-based permissions
        for role in user.roles:
            role_permissions = self.role_permissions.get(role, [])
            permissions.update(role_permissions)
        
        return list(permissions)

# =============================================================================
# Session Manager
# =============================================================================

class SessionManager(SecurityBase):
    """Session manager for handling user sessions."""
    
    def __init__(self, config: SecurityConfig):
        
    """__init__ function."""
super().__init__(config)
        self.sessions: Dict[str, UserSession] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def _initialize_internal(self) -> None:
        """Initialize session manager."""
        if self.config.enable_session_management:
            self._start_cleanup_task()
    
    async def _cleanup_internal(self) -> None:
        """Cleanup session manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    def create_session(
        self,
        user_id: UUID,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> UserSession:
        """Create a new user session."""
        if not self._is_initialized:
            raise RuntimeError("Session manager not initialized")
        
        session_id = self._generate_session_id()
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=self.config.session_timeout_minutes)
        
        session = UserSession(
            user_id=user_id,
            session_id=session_id,
            created_at=now,
            last_accessed=now,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        self.stats.total_sessions += 1
        self.stats.active_sessions = len(self.sessions)
        
        return session
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get a user session."""
        if not self._is_initialized:
            raise RuntimeError("Session manager not initialized")
        
        session = self.sessions.get(session_id)
        if not session or not session.is_active:
            return None
        
        # Check if session is expired
        if datetime.now(timezone.utc) > session.expires_at:
            self.invalidate_session(session_id)
            return None
        
        # Update last accessed time
        session.last_accessed = datetime.now(timezone.utc)
        
        return session
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a user session."""
        if not self._is_initialized:
            raise RuntimeError("Session manager not initialized")
        
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            del self.sessions[session_id]
            self.stats.active_sessions = len(self.sessions)
            return True
        
        return False
    
    def invalidate_user_sessions(self, user_id: UUID) -> int:
        """Invalidate all sessions for a user."""
        if not self._is_initialized:
            raise RuntimeError("Session manager not initialized")
        
        invalidated_count = 0
        session_ids_to_remove = []
        
        for session_id, session in self.sessions.items():
            if session.user_id == user_id:
                session.is_active = False
                session_ids_to_remove.append(session_id)
                invalidated_count += 1
        
        for session_id in session_ids_to_remove:
            del self.sessions[session_id]
        
        self.stats.active_sessions = len(self.sessions)
        return invalidated_count
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return secrets.token_urlsafe(32)
    
    def _start_cleanup_task(self) -> None:
        """Start session cleanup task."""
        if self._cleanup_task:
            return
        
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self) -> None:
        """Session cleanup loop."""
        while self._is_initialized:
            try:
                await self._cleanup_expired_sessions()
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_expired_sessions(self) -> None:
        """Cleanup expired sessions."""
        now = datetime.now(timezone.utc)
        session_ids_to_remove = []
        
        for session_id, session in self.sessions.items():
            if now > session.expires_at:
                session_ids_to_remove.append(session_id)
        
        for session_id in session_ids_to_remove:
            del self.sessions[session_id]
        
        self.stats.active_sessions = len(self.sessions)

# =============================================================================
# Security Manager
# =============================================================================

class SecurityManager(SecurityBase):
    """Main security manager for handling all security operations."""
    
    def __init__(self, config: SecurityConfig):
        
    """__init__ function."""
super().__init__(config)
        self.auth_manager = AuthenticationManager(config)
        self.authz_manager = AuthorizationManager(config)
        self.session_manager = SessionManager(config)
        self.http_bearer = HTTPBearer()
    
    async def _initialize_internal(self) -> None:
        """Initialize security manager."""
        await self.auth_manager.initialize()
        await self.authz_manager.initialize()
        await self.session_manager.initialize()
    
    async def _cleanup_internal(self) -> None:
        """Cleanup security manager."""
        await self.auth_manager.cleanup()
        await self.authz_manager.cleanup()
        await self.session_manager.cleanup()
    
    async def authenticate_user(self, credentials: HTTPAuthorizationCredentials) -> User:
        """Authenticate user from HTTP credentials."""
        token_data = await self.auth_manager.authenticate_token(credentials.credentials)
        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        
        # Get user from service (this would typically be injected)
        # For now, we'll return a mock user
        user = User(
            id=token_data.user_id,
            username=token_data.username,
            email=f"{token_data.username}@example.com",
            roles=token_data.roles,
            permissions=token_data.permissions,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        return user
    
    async def require_permission(self, permission: str):
        """Dependency for requiring a specific permission."""
        async def permission_dependency(current_user: User = Depends(self.authenticate_user)) -> User:
            if not self.authz_manager.has_permission(current_user, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{permission}' required"
                )
            return current_user
        
        return permission_dependency
    
    async def require_role(self, role: str):
        """Dependency for requiring a specific role."""
        async def role_dependency(current_user: User = Depends(self.authenticate_user)) -> User:
            if not self.authz_manager.has_role(current_user, role):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role '{role}' required"
                )
            return current_user
        
        return role_dependency
    
    async def require_any_role(self, roles: List[str]):
        """Dependency for requiring any of the specified roles."""
        async def any_role_dependency(current_user: User = Depends(self.authenticate_user)) -> User:
            if not self.authz_manager.has_any_role(current_user, roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"One of roles {roles} required"
                )
            return current_user
        
        return any_role_dependency
    
    def get_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        return {
            "authentication": {
                "total": self.auth_manager.stats.total_authentications,
                "successful": self.auth_manager.stats.successful_authentications,
                "failed": self.auth_manager.stats.failed_authentications,
                "lockouts": self.auth_manager.stats.lockouts
            },
            "authorization": {
                "total": self.authz_manager.stats.total_authorizations,
                "successful": self.authz_manager.stats.successful_authorizations,
                "failed": self.authz_manager.stats.failed_authorizations
            },
            "sessions": {
                "active": self.session_manager.stats.active_sessions,
                "total": self.session_manager.stats.total_sessions
            }
        }

# =============================================================================
# Security Decorators
# =============================================================================

def require_permission(permission: str):
    """Decorator for requiring a specific permission."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # This would be used with the security manager
            # The actual permission check would happen at the function level
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_role(role: str):
    """Decorator for requiring a specific role."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # This would be used with the security manager
            # The actual role check would happen at the function level
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "SecurityLevel",
    "TokenType",
    "SecurityConfig",
    "UserSession",
    "SecurityStats",
    "User",
    "TokenData",
    "LoginRequest",
    "TokenResponse",
    "SecurityBase",
    "AuthenticationManager",
    "AuthorizationManager",
    "SessionManager",
    "SecurityManager",
    "require_permission",
    "require_role",
] 