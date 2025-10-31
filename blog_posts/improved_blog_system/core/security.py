"""
Security utilities for authentication and authorization
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..config.settings import get_settings
from .exceptions import AuthenticationError, AuthorizationError


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token security
security = HTTPBearer()


class SecurityService:
    """Security service for authentication and authorization."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        return pwd_context.hash(password)
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.settings.access_token_expire_minutes
            )
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(
            to_encode,
            self.settings.secret_key,
            algorithm=self.settings.algorithm
        )
        return encoded_jwt
    
    def create_refresh_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                days=self.settings.refresh_token_expire_days
            )
        
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(
            to_encode,
            self.settings.secret_key,
            algorithm=self.settings.algorithm
        )
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.settings.secret_key,
                algorithms=[self.settings.algorithm]
            )
            
            # Check token type
            if payload.get("type") != token_type:
                raise AuthenticationError("Invalid token type")
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
                raise AuthenticationError("Token has expired")
            
            return payload
            
        except JWTError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
    
    def extract_user_id(self, token: str) -> str:
        """Extract user ID from token."""
        payload = self.verify_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise AuthenticationError("Token missing user ID")
        return user_id


# Global security service instance
security_service = SecurityService()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    payload = security_service.verify_token(token)
    
    user_id = payload.get("sub")
    if not user_id:
        raise AuthenticationError("Token missing user ID")
    
    return {
        "user_id": user_id,
        "email": payload.get("email"),
        "username": payload.get("username"),
        "roles": payload.get("roles", []),
        "permissions": payload.get("permissions", [])
    }


async def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get current active user (not disabled)."""
    # In a real implementation, you would check if the user is active
    # For now, we'll just return the current user
    return current_user


async def require_permission(permission: str):
    """Dependency factory for requiring specific permission."""
    async def permission_checker(
        current_user: Dict[str, Any] = Depends(get_current_active_user)
    ) -> Dict[str, Any]:
        user_permissions = current_user.get("permissions", [])
        if permission not in user_permissions:
            raise AuthorizationError(f"Missing required permission: {permission}")
        return current_user
    
    return permission_checker


async def require_role(role: str):
    """Dependency factory for requiring specific role."""
    async def role_checker(
        current_user: Dict[str, Any] = Depends(get_current_active_user)
    ) -> Dict[str, Any]:
        user_roles = current_user.get("roles", [])
        if role not in user_roles:
            raise AuthorizationError(f"Missing required role: {role}")
        return current_user
    
    return role_checker


# Common permission dependencies
require_admin = require_role("admin")
require_editor = require_role("editor")
require_author = require_role("author")
require_reader = require_role("reader")































