"""
Authentication Middleware
========================

Advanced authentication middleware with JWT, OAuth, and session management.
"""

from __future__ import annotations
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import jwt
import hashlib
import secrets

from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext

from ...shared.config import get_settings


logger = logging.getLogger(__name__)


class AuthProvider(str, Enum):
    """Authentication providers"""
    JWT = "jwt"
    OAUTH2 = "oauth2"
    SESSION = "session"
    API_KEY = "api_key"


class UserRole(str, Enum):
    """User roles"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    MODERATOR = "moderator"


@dataclass
class User:
    """User data"""
    id: str
    username: str
    email: str
    roles: List[UserRole]
    permissions: List[str]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = None


@dataclass
class TokenData:
    """Token data"""
    user_id: str
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    exp: datetime
    iat: datetime
    jti: str


class AuthProvider(ABC):
    """Abstract authentication provider"""
    
    @abstractmethod
    async def authenticate(self, credentials: str) -> Optional[User]:
        """Authenticate user with credentials"""
        pass
    
    @abstractmethod
    async def create_token(self, user: User) -> str:
        """Create authentication token for user"""
        pass
    
    @abstractmethod
    async def validate_token(self, token: str) -> Optional[TokenData]:
        """Validate authentication token"""
        pass


class JWTAuthProvider(AuthProvider):
    """JWT authentication provider"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    async def authenticate(self, credentials: str) -> Optional[User]:
        """Authenticate user with JWT token"""
        try:
            token_data = await self.validate_token(credentials)
            if not token_data:
                return None
            
            # In a real implementation, you would fetch user from database
            # For now, we'll create a mock user
            return User(
                id=token_data.user_id,
                username=token_data.username,
                email=token_data.email,
                roles=[UserRole(role) for role in token_data.roles],
                permissions=token_data.permissions,
                is_active=True,
                created_at=datetime.utcnow(),
                last_login=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"JWT authentication failed: {e}")
            return None
    
    async def create_token(self, user: User) -> str:
        """Create JWT token for user"""
        try:
            now = datetime.utcnow()
            exp = now + timedelta(minutes=30)  # Token expires in 30 minutes
            
            payload = {
                "user_id": user.id,
                "username": user.username,
                "email": user.email,
                "roles": [role.value for role in user.roles],
                "permissions": user.permissions,
                "exp": exp,
                "iat": now,
                "jti": secrets.token_urlsafe(32)
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            return token
            
        except Exception as e:
            logger.error(f"JWT token creation failed: {e}")
            raise
    
    async def validate_token(self, token: str) -> Optional[TokenData]:
        """Validate JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            return TokenData(
                user_id=payload["user_id"],
                username=payload["username"],
                email=payload["email"],
                roles=payload["roles"],
                permissions=payload["permissions"],
                exp=datetime.fromtimestamp(payload["exp"]),
                iat=datetime.fromtimestamp(payload["iat"]),
                jti=payload["jti"]
            )
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            logger.error(f"JWT token validation failed: {e}")
            return None


class APIKeyAuthProvider(AuthProvider):
    """API Key authentication provider"""
    
    def __init__(self):
        self._api_keys: Dict[str, User] = {}
        self._initialize_default_keys()
    
    def _initialize_default_keys(self):
        """Initialize default API keys for testing"""
        # In production, these would be stored in database
        default_user = User(
            id="api_user_1",
            username="api_user",
            email="api@example.com",
            roles=[UserRole.USER],
            permissions=["read", "write"],
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        self._api_keys["test_api_key_123"] = default_user
    
    async def authenticate(self, credentials: str) -> Optional[User]:
        """Authenticate user with API key"""
        try:
            # Remove "Bearer " prefix if present
            api_key = credentials.replace("Bearer ", "").strip()
            
            user = self._api_keys.get(api_key)
            if user and user.is_active:
                return user
            
            return None
            
        except Exception as e:
            logger.error(f"API key authentication failed: {e}")
            return None
    
    async def create_token(self, user: User) -> str:
        """Create API key for user"""
        # In a real implementation, you would generate and store API keys
        api_key = secrets.token_urlsafe(32)
        self._api_keys[api_key] = user
        return api_key
    
    async def validate_token(self, token: str) -> Optional[TokenData]:
        """Validate API key"""
        try:
            user = self._api_keys.get(token)
            if not user:
                return None
            
            return TokenData(
                user_id=user.id,
                username=user.username,
                email=user.email,
                roles=[role.value for role in user.roles],
                permissions=user.permissions,
                exp=datetime.utcnow() + timedelta(days=365),  # API keys don't expire
                iat=datetime.utcnow(),
                jti=token
            )
            
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return None


class AuthMiddleware:
    """
    Advanced authentication middleware
    
    Provides multi-provider authentication with role-based access control,
    session management, and security features.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._providers: Dict[AuthProvider, AuthProvider] = {}
        self._security = HTTPBearer(auto_error=False)
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize authentication providers"""
        # JWT Provider
        jwt_provider = JWTAuthProvider(
            secret_key=self.settings.secret_key,
            algorithm=self.settings.jwt_algorithm
        )
        self._providers[AuthProvider.JWT] = jwt_provider
        
        # API Key Provider
        api_key_provider = APIKeyAuthProvider()
        self._providers[AuthProvider.API_KEY] = api_key_provider
    
    async def authenticate_token(self, token: str) -> Optional[str]:
        """Authenticate token and return user ID"""
        try:
            # Try JWT first
            jwt_provider = self._providers.get(AuthProvider.JWT)
            if jwt_provider:
                token_data = await jwt_provider.validate_token(token)
                if token_data:
                    return token_data.user_id
            
            # Try API Key
            api_key_provider = self._providers.get(AuthProvider.API_KEY)
            if api_key_provider:
                token_data = await api_key_provider.validate_token(token)
                if token_data:
                    return token_data.user_id
            
            return None
            
        except Exception as e:
            logger.error(f"Token authentication failed: {e}")
            return None
    
    async def get_current_user_id(self) -> str:
        """Get current user ID from request"""
        # This would be implemented to extract user ID from request context
        # For now, return a default user ID
        return "user_123"
    
    async def get_current_user(self, request: Request) -> Optional[User]:
        """Get current user from request"""
        try:
            # Extract token from request
            credentials = await self._security(request)
            if not credentials:
                return None
            
            # Try different providers
            for provider in self._providers.values():
                user = await provider.authenticate(credentials.credentials)
                if user:
                    return user
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get current user: {e}")
            return None
    
    async def require_auth(self, request: Request) -> User:
        """Require authentication for request"""
        user = await self.get_current_user(request)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"}
            )
        return user
    
    async def require_role(self, request: Request, required_roles: List[UserRole]) -> User:
        """Require specific roles for request"""
        user = await self.require_auth(request)
        
        if not any(role in user.roles for role in required_roles):
            raise HTTPException(
                status_code=403,
                detail=f"Required roles: {[role.value for role in required_roles]}"
            )
        
        return user
    
    async def require_permission(self, request: Request, required_permission: str) -> User:
        """Require specific permission for request"""
        user = await self.require_auth(request)
        
        if required_permission not in user.permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Required permission: {required_permission}"
            )
        
        return user
    
    async def create_user_token(self, user: User, provider: AuthProvider = AuthProvider.JWT) -> str:
        """Create authentication token for user"""
        try:
            auth_provider = self._providers.get(provider)
            if not auth_provider:
                raise ValueError(f"Provider {provider} not available")
            
            return await auth_provider.create_token(user)
            
        except Exception as e:
            logger.error(f"Failed to create user token: {e}")
            raise
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return self._providers[AuthProvider.JWT].pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self._providers[AuthProvider.JWT].pwd_context.verify(plain_password, hashed_password)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get authentication middleware statistics"""
        return {
            "providers": list(self._providers.keys()),
            "jwt_algorithm": self.settings.jwt_algorithm,
            "jwt_expire_minutes": self.settings.jwt_expire_minutes,
            "api_keys_count": len(self._providers[AuthProvider.API_KEY]._api_keys)
        }


# Global auth middleware instance
_auth_middleware: Optional[AuthMiddleware] = None


def get_auth_middleware() -> AuthMiddleware:
    """Get global auth middleware instance"""
    global _auth_middleware
    if _auth_middleware is None:
        _auth_middleware = AuthMiddleware()
    return _auth_middleware


# FastAPI dependencies
async def get_current_user(request: Request) -> Optional[User]:
    """FastAPI dependency to get current user"""
    auth_middleware = get_auth_middleware()
    return await auth_middleware.get_current_user(request)


async def require_auth(request: Request) -> User:
    """FastAPI dependency to require authentication"""
    auth_middleware = get_auth_middleware()
    return await auth_middleware.require_auth(request)


async def require_admin(request: Request) -> User:
    """FastAPI dependency to require admin role"""
    auth_middleware = get_auth_middleware()
    return await auth_middleware.require_role(request, [UserRole.ADMIN])


async def require_user_or_admin(request: Request) -> User:
    """FastAPI dependency to require user or admin role"""
    auth_middleware = get_auth_middleware()
    return await auth_middleware.require_role(request, [UserRole.USER, UserRole.ADMIN])




