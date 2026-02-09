"""
Authentication Module - Authentication and authorization.

Provides:
- JWT token management
- Role-based access control (RBAC)
- API key management
- User management
"""

import logging
import jwt
import hashlib
import secrets
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """User roles."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    API = "api"


@dataclass
class User:
    """User model."""
    id: str
    username: str
    email: str
    role: UserRole
    api_key: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_login: Optional[str] = None
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "active": self.active,
        }


@dataclass
class Token:
    """JWT token."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600


class AuthManager:
    """Authentication and authorization manager."""
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize auth manager.
        
        Args:
            secret_key: JWT secret key (defaults to random)
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id
        self._create_default_admin()
    
    def _create_default_admin(self) -> None:
        """Create default admin user."""
        admin = User(
            id="admin",
            username="admin",
            email="admin@example.com",
            role=UserRole.ADMIN,
        )
        self.users["admin"] = admin
    
    def create_user(
        self,
        username: str,
        email: str,
        role: UserRole = UserRole.USER,
        generate_api_key: bool = False,
    ) -> User:
        """
        Create a new user.
        
        Args:
            username: Username
            email: Email address
            role: User role
            generate_api_key: Generate API key
            
        Returns:
            Created user
        """
        user_id = f"user_{len(self.users)}"
        api_key = None
        
        if generate_api_key:
            api_key = secrets.token_urlsafe(32)
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            role=role,
            api_key=api_key,
        )
        
        self.users[user_id] = user
        
        if api_key:
            self.api_keys[api_key] = user_id
        
        logger.info(f"Created user: {username} ({role.value})")
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[Token]:
        """
        Authenticate user and return token.
        
        Args:
            username: Username
            password: Password (in production, hash and verify)
            
        Returns:
            JWT token or None
        """
        user = next((u for u in self.users.values() if u.username == username), None)
        
        if not user or not user.active:
            return None
        
        # In production, verify password hash here
        # For now, accept any password for demo
        
        user.last_login = datetime.now().isoformat()
        
        token = self.generate_token(user)
        return token
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """
        Authenticate using API key.
        
        Args:
            api_key: API key
            
        Returns:
            User or None
        """
        user_id = self.api_keys.get(api_key)
        if not user_id:
            return None
        
        user = self.users.get(user_id)
        if not user or not user.active:
            return None
        
        return user
    
    def generate_token(self, user: User, expires_in: int = 3600) -> Token:
        """
        Generate JWT token.
        
        Args:
            user: User object
            expires_in: Token expiration in seconds
            
        Returns:
            Token object
        """
        payload = {
            "sub": user.id,
            "username": user.username,
            "role": user.role.value,
            "exp": datetime.utcnow() + timedelta(seconds=expires_in),
            "iat": datetime.utcnow(),
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        
        return Token(
            access_token=token,
            expires_in=expires_in,
        )
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify JWT token.
        
        Args:
            token: JWT token
            
        Returns:
            Decoded payload or None
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def has_permission(self, user: User, resource: str, action: str) -> bool:
        """
        Check if user has permission.
        
        Args:
            user: User object
            resource: Resource name
            action: Action (read, write, delete, admin)
            
        Returns:
            True if user has permission
        """
        if user.role == UserRole.ADMIN:
            return True
        
        if user.role == UserRole.VIEWER:
            return action == "read"
        
        if user.role == UserRole.USER:
            return action in ["read", "write"]
        
        return False
    
    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke API key.
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if revoked
        """
        if api_key in self.api_keys:
            user_id = self.api_keys.pop(api_key)
            user = self.users.get(user_id)
            if user:
                user.api_key = None
            return True
        return False












