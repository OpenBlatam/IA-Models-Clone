"""
Identity and Access Management (IAM) for Flux2 Clothing Changer
==============================================================

Identity and access management system.
"""

import time
import hashlib
import secrets
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Permission types."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"


class Role(Enum):
    """User roles."""
    GUEST = "guest"
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


@dataclass
class User:
    """User information."""
    user_id: str
    username: str
    email: str
    role: Role
    permissions: Set[Permission]
    created_at: float
    last_login: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AccessToken:
    """Access token."""
    token: str
    user_id: str
    expires_at: float
    permissions: Set[Permission]
    created_at: float = time.time()


class IAMSystem:
    """Identity and Access Management system."""
    
    def __init__(
        self,
        token_expiry: float = 3600.0,  # 1 hour
    ):
        """
        Initialize IAM system.
        
        Args:
            token_expiry: Token expiry time in seconds
        """
        self.token_expiry = token_expiry
        
        self.users: Dict[str, User] = {}
        self.tokens: Dict[str, AccessToken] = {}
        self.role_permissions: Dict[Role, Set[Permission]] = {
            Role.GUEST: {Permission.READ},
            Role.USER: {Permission.READ, Permission.WRITE},
            Role.PREMIUM: {Permission.READ, Permission.WRITE, Permission.EXECUTE},
            Role.ADMIN: {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN},
            Role.SUPER_ADMIN: set(Permission),
        }
        
        # Statistics
        self.stats = {
            "total_users": 0,
            "active_tokens": 0,
            "total_logins": 0,
        }
    
    def create_user(
        self,
        username: str,
        email: str,
        role: Role = Role.USER,
        password_hash: Optional[str] = None,
    ) -> User:
        """
        Create user.
        
        Args:
            username: Username
            email: Email address
            role: User role
            password_hash: Optional password hash
            
        Returns:
            Created user
        """
        user_id = self._generate_user_id(username, email)
        
        permissions = self.role_permissions.get(role, set())
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            permissions=permissions,
            created_at=time.time(),
        )
        
        if password_hash:
            user.metadata["password_hash"] = password_hash
        
        self.users[user_id] = user
        self.stats["total_users"] += 1
        
        logger.info(f"Created user: {user_id}")
        return user
    
    def authenticate(
        self,
        username: str,
        password_hash: str,
    ) -> Optional[AccessToken]:
        """
        Authenticate user.
        
        Args:
            username: Username
            password_hash: Password hash
            
        Returns:
            Access token or None
        """
        user = self._find_user_by_username(username)
        
        if not user:
            return None
        
        stored_hash = user.metadata.get("password_hash")
        if stored_hash != password_hash:
            return None
        
        # Update last login
        user.last_login = time.time()
        self.stats["total_logins"] += 1
        
        # Generate token
        token = self.generate_token(user.user_id, user.permissions)
        
        return token
    
    def generate_token(
        self,
        user_id: str,
        permissions: Optional[Set[Permission]] = None,
    ) -> AccessToken:
        """
        Generate access token.
        
        Args:
            user_id: User identifier
            permissions: Optional permissions override
            
        Returns:
            Access token
        """
        if user_id not in self.users:
            raise ValueError(f"User not found: {user_id}")
        
        user = self.users[user_id]
        token_permissions = permissions or user.permissions
        
        token_string = secrets.token_urlsafe(32)
        expires_at = time.time() + self.token_expiry
        
        token = AccessToken(
            token=token_string,
            user_id=user_id,
            expires_at=expires_at,
            permissions=token_permissions,
        )
        
        self.tokens[token_string] = token
        self.stats["active_tokens"] = len([t for t in self.tokens.values() if t.expires_at > time.time()])
        
        logger.info(f"Generated token for user: {user_id}")
        return token
    
    def validate_token(self, token: str) -> Optional[User]:
        """
        Validate access token.
        
        Args:
            token: Access token
            
        Returns:
            User if valid, None otherwise
        """
        if token not in self.tokens:
            return None
        
        access_token = self.tokens[token]
        
        if access_token.expires_at < time.time():
            del self.tokens[token]
            return None
        
        if access_token.user_id not in self.users:
            return None
        
        return self.users[access_token.user_id]
    
    def check_permission(
        self,
        token: str,
        permission: Permission,
    ) -> bool:
        """
        Check if token has permission.
        
        Args:
            token: Access token
            permission: Required permission
            
        Returns:
            True if has permission
        """
        access_token = self.validate_token(token)
        if not access_token:
            return False
        
        token_obj = self.tokens[token]
        return permission in token_obj.permissions
    
    def revoke_token(self, token: str) -> bool:
        """
        Revoke access token.
        
        Args:
            token: Access token
            
        Returns:
            True if revoked
        """
        if token in self.tokens:
            del self.tokens[token]
            self.stats["active_tokens"] = len([t for t in self.tokens.values() if t.expires_at > time.time()])
            logger.info(f"Revoked token: {token[:8]}...")
            return True
        return False
    
    def update_user_role(
        self,
        user_id: str,
        new_role: Role,
    ) -> bool:
        """
        Update user role.
        
        Args:
            user_id: User identifier
            new_role: New role
            
        Returns:
            True if updated
        """
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        user.role = new_role
        user.permissions = self.role_permissions.get(new_role, set())
        
        logger.info(f"Updated user {user_id} role to {new_role.value}")
        return True
    
    def _find_user_by_username(self, username: str) -> Optional[User]:
        """Find user by username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def _generate_user_id(self, username: str, email: str) -> str:
        """Generate user ID."""
        data = f"{username}:{email}:{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get IAM statistics."""
        return {
            **self.stats,
            "users_by_role": {
                role.value: len([u for u in self.users.values() if u.role == role])
                for role in Role
            },
        }


