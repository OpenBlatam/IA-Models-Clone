"""
Authentication & Authorization System V2
=========================================

Advanced authentication and authorization system.
"""

import time
import hashlib
import secrets
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Permission types."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"


class TokenType(Enum):
    """Token type."""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"


@dataclass
class User:
    """User entity."""
    id: str
    username: str
    email: str
    password_hash: str
    roles: List[str] = None
    permissions: Set[Permission] = None
    created_at: float = 0.0
    last_login: Optional[float] = None
    active: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.roles is None:
            self.roles = []
        if self.permissions is None:
            self.permissions = set()
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Token:
    """Authentication token."""
    token: str
    user_id: str
    token_type: TokenType
    expires_at: float
    created_at: float = 0.0
    scopes: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.scopes is None:
            self.scopes = []
        if self.metadata is None:
            self.metadata = {}
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return time.time() > self.expires_at


class AuthSystemV2:
    """Advanced authentication and authorization system."""
    
    def __init__(self):
        """Initialize auth system."""
        self.users: Dict[str, User] = {}
        self.tokens: Dict[str, Token] = {}
        self.roles_permissions: Dict[str, Set[Permission]] = {}
        self._setup_default_roles()
    
    def _setup_default_roles(self) -> None:
        """Setup default roles."""
        self.roles_permissions = {
            "admin": {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN, Permission.EXECUTE},
            "user": {Permission.READ, Permission.WRITE},
            "viewer": {Permission.READ},
        }
    
    def _hash_password(self, password: str) -> str:
        """Hash password."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _generate_token(self) -> str:
        """Generate secure token."""
        return secrets.token_urlsafe(32)
    
    def register_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: Optional[List[str]] = None,
    ) -> User:
        """
        Register a new user.
        
        Args:
            username: Username
            email: Email
            password: Password
            roles: Optional roles
            
        Returns:
            Created user
        """
        user_id = hashlib.md5(f"{username}:{email}".encode()).hexdigest()
        
        if user_id in self.users:
            raise ValueError(f"User already exists: {username}")
        
        password_hash = self._hash_password(password)
        
        # Get permissions from roles
        permissions = set()
        for role in (roles or ["user"]):
            if role in self.roles_permissions:
                permissions.update(self.roles_permissions[role])
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles or ["user"],
            permissions=permissions,
        )
        
        self.users[user_id] = user
        logger.info(f"User registered: {username}")
        
        return user
    
    def authenticate(
        self,
        username: str,
        password: str,
    ) -> Optional[Token]:
        """
        Authenticate user.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Access token or None
        """
        # Find user
        user = None
        for u in self.users.values():
            if u.username == username or u.email == username:
                user = u
                break
        
        if not user or not user.active:
            return None
        
        # Verify password
        password_hash = self._hash_password(password)
        if password_hash != user.password_hash:
            return None
        
        # Update last login
        user.last_login = time.time()
        
        # Generate token
        token_str = self._generate_token()
        token = Token(
            token=token_str,
            user_id=user.id,
            token_type=TokenType.ACCESS,
            expires_at=time.time() + 3600,  # 1 hour
        )
        
        self.tokens[token_str] = token
        logger.info(f"User authenticated: {username}")
        
        return token
    
    def validate_token(self, token: str) -> Optional[User]:
        """
        Validate token and return user.
        
        Args:
            token: Token string
            
        Returns:
            User if valid, None otherwise
        """
        if token not in self.tokens:
            return None
        
        token_obj = self.tokens[token]
        
        if token_obj.is_expired():
            del self.tokens[token]
            return None
        
        user = self.users.get(token_obj.user_id)
        
        if not user or not user.active:
            return None
        
        return user
    
    def check_permission(
        self,
        user: User,
        permission: Permission,
    ) -> bool:
        """
        Check if user has permission.
        
        Args:
            user: User object
            permission: Permission to check
            
        Returns:
            True if user has permission
        """
        return permission in user.permissions
    
    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token.
        
        Args:
            token: Token string
            
        Returns:
            True if revoked
        """
        if token in self.tokens:
            del self.tokens[token]
            logger.info("Token revoked")
            return True
        return False
    
    def create_api_key(
        self,
        user_id: str,
        scopes: Optional[List[str]] = None,
    ) -> Token:
        """
        Create API key for user.
        
        Args:
            user_id: User ID
            scopes: Optional scopes
            
        Returns:
            API key token
        """
        if user_id not in self.users:
            raise ValueError(f"User not found: {user_id}")
        
        token_str = f"api_{self._generate_token()}"
        token = Token(
            token=token_str,
            user_id=user_id,
            token_type=TokenType.API_KEY,
            expires_at=time.time() + 31536000,  # 1 year
            scopes=scopes or [],
        )
        
        self.tokens[token_str] = token
        logger.info(f"API key created for user: {user_id}")
        
        return token
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get auth system statistics."""
        active_tokens = [t for t in self.tokens.values() if not t.is_expired()]
        
        return {
            "total_users": len(self.users),
            "active_users": len([u for u in self.users.values() if u.active]),
            "total_tokens": len(self.tokens),
            "active_tokens": len(active_tokens),
            "roles": list(self.roles_permissions.keys()),
        }

