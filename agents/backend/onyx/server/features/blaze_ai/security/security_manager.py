"""
Advanced security management system for Blaze AI.

This module provides comprehensive security features including JWT authentication,
role-based access control (RBAC), rate limiting, security monitoring, and audit logging.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from collections import defaultdict, deque
import secrets
import threading

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

from ..core.interfaces import CoreConfig
from ..utils.logging import get_logger

# =============================================================================
# Core Security Types and Enums
# =============================================================================

class SecurityLevel(Enum):
    """Security levels for operations."""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    ADMIN = "admin"
    SYSTEM = "system"

class Permission(Enum):
    """System permissions."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    MONITOR = "monitor"

class RateLimitType(Enum):
    """Rate limiting types."""
    USER = "user"
    IP = "ip"
    ENDPOINT = "endpoint"
    GLOBAL = "global"

@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_secret: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600  # 1 hour
    jwt_refresh_expiration: int = 86400  # 24 hours
    bcrypt_rounds: int = 12
    rate_limit_enabled: bool = True
    rate_limit_window: int = 60  # 1 minute
    rate_limit_max_requests: int = 100
    session_timeout: int = 1800  # 30 minutes
    max_failed_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    enable_audit_logging: bool = True
    enable_security_monitoring: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_csrf_protection: bool = True
    csrf_token_expiration: int = 3600

# =============================================================================
# User and Role Management
# =============================================================================

@dataclass
class User:
    """User entity with security information."""
    id: str
    username: str
    email: str
    password_hash: str
    roles: List[str] = field(default_factory=list)
    permissions: List[Permission] = field(default_factory=list)
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Role:
    """Role entity with permissions."""
    name: str
    description: str
    permissions: List[Permission] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.AUTHENTICATED
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True

@dataclass
class Session:
    """User session information."""
    id: str
    user_id: str
    token: str
    refresh_token: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    last_activity: datetime = field(default_factory=datetime.utcnow)

# =============================================================================
# Authentication and Authorization
# =============================================================================

class AuthenticationManager:
    """Manages user authentication and session management."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = get_logger("auth_manager")
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.failed_attempts: Dict[str, int] = defaultdict(int)
        self.lockouts: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
        
        self._setup_default_users()
    
    def _setup_default_users(self):
        """Setup default system users."""
        default_users = [
            User(
                id="system",
                username="system",
                email="system@blaze.ai",
                password_hash=self._hash_password("system_password"),
                roles=["system"],
                permissions=[Permission.ADMIN, Permission.MONITOR],
                security_level=SecurityLevel.SYSTEM
            ),
            User(
                id="admin",
                username="admin",
                email="admin@blaze.ai",
                password_hash=self._hash_password("admin_password"),
                roles=["admin"],
                permissions=[Permission.ADMIN],
                security_level=SecurityLevel.ADMIN
            )
        ]
        
        for user in default_users:
            self.users[user.id] = user
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        if BCRYPT_AVAILABLE:
            return bcrypt.hashpw(password.encode(), bcrypt.gensalt(self.config.bcrypt_rounds)).decode()
        else:
            # Fallback to SHA256 if bcrypt not available
            return hashlib.sha256(password.encode()).hexdigest()
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        if BCRYPT_AVAILABLE:
            return bcrypt.checkpw(password.encode(), password_hash.encode())
        else:
            # Fallback to SHA256 if bcrypt not available
            return hashlib.sha256(password.encode()).hexdigest() == password_hash
    
    async def authenticate(self, username: str, password: str, ip_address: str, user_agent: str) -> Optional[Session]:
        """Authenticate user and create session."""
        async with self._lock:
            # Find user
            user = None
            for u in self.users.values():
                if u.username == username:
                    user = u
                    break
            
            if not user or not user.is_active:
                return None
            
            # Check if user is locked
            if user.locked_until and datetime.utcnow() < user.locked_until:
                remaining = user.locked_until - datetime.utcnow()
                self.logger.warning(f"User {username} is locked for {remaining}")
                return None
            
            # Verify password
            if not self._verify_password(password, user.password_hash):
                await self._handle_failed_attempt(username)
                return None
            
            # Reset failed attempts on successful login
            user.failed_attempts = 0
            user.locked_until = None
            user.last_login = datetime.utcnow()
            
            # Create session
            session = await self._create_session(user, ip_address, user_agent)
            
            self.logger.info(f"User {username} authenticated successfully")
            return session
    
    async def _handle_failed_attempt(self, username: str):
        """Handle failed authentication attempt."""
        self.failed_attempts[username] += 1
        
        if self.failed_attempts[username] >= self.config.max_failed_attempts:
            # Lock user account
            lockout_until = datetime.utcnow() + timedelta(seconds=self.config.lockout_duration)
            self.lockouts[username] = lockout_until
            
            # Find and lock user
            for user in self.users.values():
                if user.username == username:
                    user.locked_until = lockout_until
                    break
            
            self.logger.warning(f"User {username} locked due to multiple failed attempts")
    
    async def _create_session(self, user: User, ip_address: str, user_agent: str) -> Session:
        """Create new user session."""
        session_id = str(uuid.uuid4())
        token = self._generate_jwt_token(user, session_id)
        refresh_token = self._generate_refresh_token(user, session_id)
        
        expires_at = datetime.utcnow() + timedelta(seconds=self.config.jwt_expiration)
        
        session = Session(
            id=session_id,
            user_id=user.id,
            token=token,
            refresh_token=refresh_token,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        return session
    
    def _generate_jwt_token(self, user: User, session_id: str) -> str:
        """Generate JWT token for user."""
        if not JWT_AVAILABLE:
            # Fallback to simple token if JWT not available
            return f"{user.id}:{session_id}:{int(time.time())}"
        
        payload = {
            "user_id": user.id,
            "username": user.username,
            "roles": user.roles,
            "permissions": [p.value for p in user.permissions],
            "session_id": session_id,
            "exp": datetime.utcnow() + timedelta(seconds=self.config.jwt_expiration),
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
    
    def _generate_refresh_token(self, user: User, session_id: str) -> str:
        """Generate refresh token for user."""
        if not JWT_AVAILABLE:
            return f"refresh:{user.id}:{session_id}:{int(time.time())}"
        
        payload = {
            "user_id": user.id,
            "session_id": session_id,
            "type": "refresh",
            "exp": datetime.utcnow() + timedelta(seconds=self.config.jwt_refresh_expiration),
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
    
    async def validate_token(self, token: str) -> Optional[User]:
        """Validate JWT token and return user."""
        try:
            if not JWT_AVAILABLE:
                # Simple token validation fallback
                parts = token.split(":")
                if len(parts) != 3:
                    return None
                
                user_id, session_id, timestamp = parts
                if int(timestamp) + self.config.jwt_expiration < time.time():
                    return None
                
                # Find session
                session = self.sessions.get(session_id)
                if not session or not session.is_active:
                    return None
                
                return self.users.get(user_id)
            
            # JWT validation
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm])
            
            session_id = payload.get("session_id")
            if not session_id:
                return None
            
            # Find session
            session = self.sessions.get(session_id)
            if not session or not session.is_active:
                return None
            
            # Update last activity
            session.last_activity = datetime.utcnow()
            
            return self.users.get(payload["user_id"])
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Token validation error: {e}")
            return None
    
    async def refresh_session(self, refresh_token: str) -> Optional[Session]:
        """Refresh user session using refresh token."""
        try:
            if not JWT_AVAILABLE:
                # Simple refresh token validation
                parts = refresh_token.split(":")
                if len(parts) != 4 or parts[0] != "refresh":
                    return None
                
                user_id, session_id, timestamp = parts[1:]
                if int(timestamp) + self.config.jwt_refresh_expiration < time.time():
                    return None
                
                user = self.users.get(user_id)
                if not user:
                    return None
                
                # Create new session
                return await self._create_session(user, "", "")
            
            # JWT refresh token validation
            payload = jwt.decode(refresh_token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm])
            
            if payload.get("type") != "refresh":
                return None
            
            user_id = payload.get("user_id")
            session_id = payload.get("session_id")
            
            user = self.users.get(user_id)
            if not user:
                return None
            
            # Invalidate old session
            if session_id in self.sessions:
                self.sessions[session_id].is_active = False
            
            # Create new session
            return await self._create_session(user, "", "")
            
        except Exception as e:
            self.logger.error(f"Session refresh error: {e}")
            return None
    
    async def logout(self, session_id: str) -> bool:
        """Logout user by invalidating session."""
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            return True
        return False

class AuthorizationManager:
    """Manages user authorization and permission checking."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = get_logger("authz_manager")
        self.roles: Dict[str, Role] = {}
        self.permission_cache: Dict[str, Dict[str, bool]] = {}
        
        self._setup_default_roles()
    
    def _setup_default_roles(self):
        """Setup default system roles."""
        default_roles = [
            Role(
                name="system",
                description="System role with full access",
                permissions=[p for p in Permission],
                security_level=SecurityLevel.SYSTEM
            ),
            Role(
                name="admin",
                description="Administrator role",
                permissions=[Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN],
                security_level=SecurityLevel.ADMIN
            ),
            Role(
                name="user",
                description="Standard user role",
                permissions=[Permission.READ, Permission.WRITE],
                security_level=SecurityLevel.AUTHENTICATED
            ),
            Role(
                name="guest",
                description="Guest role with limited access",
                permissions=[Permission.READ],
                security_level=SecurityLevel.PUBLIC
            )
        ]
        
        for role in default_roles:
            self.roles[role.name] = role
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission."""
        cache_key = f"{user.id}:{permission.value}"
        
        if cache_key in self.permission_cache:
            return self.permission_cache[cache_key]
        
        # Check user permissions directly
        if permission in user.permissions:
            self.permission_cache[cache_key] = True
            return True
        
        # Check role permissions
        for role_name in user.roles:
            role = self.roles.get(role_name)
            if role and role.is_active and permission in role.permissions:
                self.permission_cache[cache_key] = True
                return True
        
        self.permission_cache[cache_key] = False
        return False
    
    def check_security_level(self, user: User, required_level: SecurityLevel) -> bool:
        """Check if user meets required security level."""
        if required_level == SecurityLevel.PUBLIC:
            return True
        
        if required_level == SecurityLevel.AUTHENTICATED:
            return user.is_active and user.is_verified
        
        if required_level == SecurityLevel.AUTHORIZED:
            return user.is_active and user.is_verified and len(user.permissions) > 0
        
        if required_level == SecurityLevel.ADMIN:
            return user.is_active and user.is_verified and Permission.ADMIN in user.permissions
        
        if required_level == SecurityLevel.SYSTEM:
            return user.is_active and user.is_verified and "system" in user.roles
        
        return False
    
    def get_user_permissions(self, user: User) -> List[Permission]:
        """Get all permissions for user (direct + role-based)."""
        permissions = set(user.permissions)
        
        for role_name in user.roles:
            role = self.roles.get(role_name)
            if role and role.is_active:
                permissions.update(role.permissions)
        
        return list(permissions)
    
    def clear_permission_cache(self, user_id: Optional[str] = None):
        """Clear permission cache."""
        if user_id:
            # Clear specific user's cache
            keys_to_remove = [k for k in self.permission_cache.keys() if k.startswith(f"{user_id}:")]
            for key in keys_to_remove:
                del self.permission_cache[key]
        else:
            # Clear all cache
            self.permission_cache.clear()

# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    """Advanced rate limiting with multiple strategies."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = get_logger("rate_limiter")
        self.limits: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque()))
        self._lock = asyncio.Lock()
    
    async def check_rate_limit(self, identifier: str, limit_type: RateLimitType, max_requests: Optional[int] = None) -> bool:
        """Check if request is within rate limit."""
        if not self.config.rate_limit_enabled:
            return True
        
        async with self._lock:
            current_time = time.time()
            window_start = current_time - self.config.rate_limit_window
            
            # Get or create limit tracker
            limit_key = f"{limit_type.value}:{identifier}"
            requests = self.limits[limit_type.value][identifier]
            
            # Remove expired requests
            while requests and requests[0] < window_start:
                requests.popleft()
            
            # Check if limit exceeded
            max_req = max_requests or self.config.rate_limit_max_requests
            if len(requests) >= max_req:
                self.logger.warning(f"Rate limit exceeded for {limit_key}: {len(requests)} requests")
                return False
            
            # Add current request
            requests.append(current_time)
            return True
    
    async def get_rate_limit_info(self, identifier: str, limit_type: RateLimitType) -> Dict[str, Any]:
        """Get rate limit information for identifier."""
        async with self._lock:
            current_time = time.time()
            window_start = current_time - self.config.rate_limit_window
            
            requests = self.limits[limit_type.value][identifier]
            
            # Remove expired requests
            while requests and requests[0] < window_start:
                requests.popleft()
            
            max_req = self.config.rate_limit_max_requests
            remaining = max(0, max_req - len(requests))
            reset_time = window_start + self.config.rate_limit_window
            
            return {
                "limit": max_req,
                "remaining": remaining,
                "reset_time": reset_time,
                "current_requests": len(requests)
            }

# =============================================================================
# Security Monitoring and Audit
# =============================================================================

@dataclass
class SecurityEvent:
    """Security event for monitoring and audit."""
    id: str
    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    details: Dict[str, Any]
    severity: str = "info"
    resolved: bool = False

class SecurityMonitor:
    """Monitors security events and provides alerts."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = get_logger("security_monitor")
        self.events: List[SecurityEvent] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default security alert rules."""
        self.alert_rules = {
            "failed_login": {
                "threshold": 5,
                "window": 300,  # 5 minutes
                "severity": "warning"
            },
            "rate_limit_exceeded": {
                "threshold": 10,
                "window": 60,  # 1 minute
                "severity": "warning"
            },
            "unauthorized_access": {
                "threshold": 3,
                "window": 600,  # 10 minutes
                "severity": "high"
            }
        }
    
    async def log_event(self, event_type: str, user_id: Optional[str], ip_address: str, 
                       user_agent: str, details: Dict[str, Any], severity: str = "info"):
        """Log security event."""
        async with self._lock:
            event = SecurityEvent(
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                event_type=event_type,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                details=details,
                severity=severity
            )
            
            self.events.append(event)
            
            # Check for alerts
            await self._check_alerts(event)
            
            # Keep only recent events (last 1000)
            if len(self.events) > 1000:
                self.events = self.events[-1000:]
    
    async def _check_alerts(self, event: SecurityEvent):
        """Check if event triggers any alerts."""
        if event.event_type in self.alert_rules:
            rule = self.alert_rules[event.event_type]
            window_start = datetime.utcnow() - timedelta(seconds=rule["window"])
            
            # Count events in window
            count = sum(1 for e in self.events 
                       if e.event_type == event.event_type and e.timestamp > window_start)
            
            if count >= rule["threshold"]:
                self.logger.warning(f"Security alert: {event.event_type} threshold exceeded "
                                  f"({count}/{rule['threshold']}) in {rule['window']}s")
    
    async def get_security_summary(self) -> Dict[str, Any]:
        """Get security monitoring summary."""
        async with self._lock:
            current_time = datetime.utcnow()
            last_hour = current_time - timedelta(hours=1)
            last_day = current_time - timedelta(days=1)
            
            recent_events = [e for e in self.events if e.timestamp > last_hour]
            daily_events = [e for e in self.events if e.timestamp > last_day]
            
            return {
                "total_events": len(self.events),
                "recent_events": len(recent_events),
                "daily_events": len(daily_events),
                "events_by_type": self._count_events_by_type(recent_events),
                "events_by_severity": self._count_events_by_severity(recent_events),
                "top_ips": self._get_top_ips(recent_events),
                "unresolved_alerts": len([e for e in self.events if not e.resolved])
            }
    
    def _count_events_by_type(self, events: List[SecurityEvent]) -> Dict[str, int]:
        """Count events by type."""
        counts = defaultdict(int)
        for event in events:
            counts[event.event_type] += 1
        return dict(counts)
    
    def _count_events_by_severity(self, events: List[SecurityEvent]) -> Dict[str, int]:
        """Count events by severity."""
        counts = defaultdict(int)
        for event in events:
            counts[event.severity] += 1
        return dict(counts)
    
    def _get_top_ips(self, events: List[SecurityEvent], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top IP addresses by event count."""
        ip_counts = defaultdict(int)
        for event in events:
            ip_counts[event.ip_address] += 1
        
        sorted_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"ip": ip, "count": count} for ip, count in sorted_ips[:limit]]

# =============================================================================
# Main Security Manager
# =============================================================================

class SecurityManager:
    """Main security manager coordinating all security features."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.logger = get_logger("security_manager")
        
        # Initialize security components
        self.auth_manager = AuthenticationManager(self.config)
        self.authz_manager = AuthorizationManager(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.security_monitor = SecurityMonitor(self.config)
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background security tasks."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def authenticate_user(self, username: str, password: str, ip_address: str, user_agent: str) -> Optional[Session]:
        """Authenticate user and create session."""
        try:
            session = await self.auth_manager.authenticate(username, password, ip_address, user_agent)
            
            if session:
                await self.security_monitor.log_event(
                    "login_success", session.user_id, ip_address, user_agent,
                    {"username": username}, "info"
                )
            else:
                await self.security_monitor.log_event(
                    "login_failed", None, ip_address, user_agent,
                    {"username": username}, "warning"
                )
            
            return session
            
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            await self.security_monitor.log_event(
                "authentication_error", None, ip_address, user_agent,
                {"error": str(e)}, "error"
            )
            return None
    
    async def validate_request(self, token: str, required_permission: Optional[Permission] = None,
                             required_level: SecurityLevel = SecurityLevel.AUTHENTICATED,
                             ip_address: str = "", user_agent: str = "") -> Optional[User]:
        """Validate request token and permissions."""
        try:
            # Validate token
            user = await self.auth_manager.validate_token(token)
            if not user:
                await self.security_monitor.log_event(
                    "invalid_token", None, ip_address, user_agent,
                    {"token": token[:20] + "..."}, "warning"
                )
                return None
            
            # Check security level
            if not self.authz_manager.check_security_level(user, required_level):
                await self.security_monitor.log_event(
                    "insufficient_security_level", user.id, ip_address, user_agent,
                    {"required_level": required_level.value, "user_level": user.roles}, "warning"
                )
                return None
            
            # Check specific permission if required
            if required_permission and not self.authz_manager.has_permission(user, required_permission):
                await self.security_monitor.log_event(
                    "insufficient_permissions", user.id, ip_address, user_agent,
                    {"required_permission": required_permission.value}, "warning"
                )
                return None
            
            return user
            
        except Exception as e:
            self.logger.error(f"Request validation error: {e}")
            await self.security_monitor.log_event(
                "validation_error", None, ip_address, user_agent,
                {"error": str(e)}, "error"
            )
            return None
    
    async def check_rate_limit(self, identifier: str, limit_type: RateLimitType, max_requests: Optional[int] = None) -> bool:
        """Check rate limit for identifier."""
        try:
            allowed = await self.rate_limiter.check_rate_limit(identifier, limit_type, max_requests)
            
            if not allowed:
                await self.security_monitor.log_event(
                    "rate_limit_exceeded", None, identifier, "",
                    {"limit_type": limit_type.value, "identifier": identifier}, "warning"
                )
            
            return allowed
            
        except Exception as e:
            self.logger.error(f"Rate limit check error: {e}")
            return True  # Allow on error
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            "authentication": {
                "active_sessions": len([s for s in self.auth_manager.sessions.values() if s.is_active]),
                "total_users": len(self.auth_manager.users),
                "locked_users": len([u for u in self.auth_manager.users.values() if u.locked_until])
            },
            "authorization": {
                "total_roles": len(self.authz_manager.roles),
                "permission_cache_size": len(self.authz_manager.permission_cache)
            },
            "rate_limiting": {
                "enabled": self.config.rate_limit_enabled,
                "limits": dict(self.rate_limiter.limits)
            },
            "security_monitoring": await self.security_monitor.get_security_summary()
        }
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while not self._shutdown_event.is_set():
            try:
                # Clean expired sessions
                current_time = datetime.utcnow()
                expired_sessions = [
                    sid for sid, session in self.auth_manager.sessions.items()
                    if session.expires_at < current_time
                ]
                
                for session_id in expired_sessions:
                    self.auth_manager.sessions[session_id].is_active = False
                
                if expired_sessions:
                    self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
                # Clear old permission cache entries
                if len(self.authz_manager.permission_cache) > 1000:
                    self.authz_manager.clear_permission_cache()
                    self.logger.info("Cleared permission cache")
                
                await asyncio.sleep(300)  # Clean every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(600)
    
    async def shutdown(self):
        """Shutdown security manager."""
        self.logger.info("Shutting down security manager...")
        self._shutdown_event.set()
        
        # Cancel background tasks
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Security manager shutdown complete")

# =============================================================================
# Global Security Instance
# =============================================================================

_default_security_manager: Optional[SecurityManager] = None

def get_security_manager(config: Optional[SecurityConfig] = None) -> SecurityManager:
    """Get the global security manager instance."""
    global _default_security_manager
    if _default_security_manager is None:
        _default_security_manager = SecurityManager(config)
    return _default_security_manager

async def shutdown_security_manager():
    """Shutdown the global security manager."""
    global _default_security_manager
    if _default_security_manager:
        await _default_security_manager.shutdown()
        _default_security_manager = None

# Export main classes
__all__ = [
    "SecurityManager",
    "SecurityConfig",
    "SecurityLevel",
    "Permission",
    "RateLimitType",
    "User",
    "Role",
    "Session",
    "SecurityEvent",
    "get_security_manager",
    "shutdown_security_manager"
]


