"""
Authentication and Authorization for OpusClip Improved
====================================================

Advanced authentication system with JWT tokens and role-based access control.
"""

import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from uuid import UUID, uuid4
import hashlib
import secrets

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
import redis.asyncio as redis

from .schemas import get_settings
from .exceptions import AuthenticationError, AuthorizationError, create_authentication_error
from .database import get_database_session, DatabaseOperations, User

logger = logging.getLogger(__name__)

# Security configuration
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


class AuthenticationService:
    """Advanced authentication service"""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = None
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis client for token storage"""
        try:
            self.redis_client = redis.from_url(
                self.settings.redis_url,
                db=self.settings.redis_db,
                password=self.settings.redis_password,
                decode_responses=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize Redis for authentication: {e}")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        
        encoded_jwt = jwt.encode(to_encode, self.settings.secret_key, algorithm=ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode.update({"exp": expire, "type": "refresh"})
        
        encoded_jwt = jwt.encode(to_encode, self.settings.secret_key, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.settings.secret_key, algorithms=[ALGORITHM])
            return payload
        except JWTError as e:
            logger.error(f"Token verification failed: {e}")
            return None
    
    async def store_token(self, token: str, user_id: str, token_type: str = "access"):
        """Store token in Redis for session management"""
        try:
            if not self.redis_client:
                return
            
            key = f"token:{token_type}:{user_id}"
            await self.redis_client.setex(key, ACCESS_TOKEN_EXPIRE_MINUTES * 60, token)
        except Exception as e:
            logger.error(f"Failed to store token: {e}")
    
    async def revoke_token(self, user_id: str, token_type: str = "access"):
        """Revoke token by removing it from Redis"""
        try:
            if not self.redis_client:
                return
            
            key = f"token:{token_type}:{user_id}"
            await self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
    
    async def is_token_revoked(self, user_id: str, token: str, token_type: str = "access") -> bool:
        """Check if token is revoked"""
        try:
            if not self.redis_client:
                return False
            
            key = f"token:{token_type}:{user_id}"
            stored_token = await self.redis_client.get(key)
            
            return stored_token != token
        except Exception as e:
            logger.error(f"Failed to check token revocation: {e}")
            return False
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        try:
            async with get_database_session() as session:
                db_ops = DatabaseOperations(session)
                user = await db_ops.get_user_by_email(email)
                
                if not user:
                    return None
                
                if not self.verify_password(password, user.hashed_password):
                    return None
                
                if not user.is_active:
                    return None
                
                return user
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return None
    
    async def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create a new user"""
        try:
            # Hash password
            user_data["hashed_password"] = self.get_password_hash(user_data["password"])
            del user_data["password"]
            
            async with get_database_session() as session:
                db_ops = DatabaseOperations(session)
                user = await db_ops.create_user(user_data)
                return user
                
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            raise create_authentication_error("User creation failed", details={"error": str(e)})
    
    async def update_user_last_login(self, user_id: UUID):
        """Update user's last login timestamp"""
        try:
            async with get_database_session() as session:
                await session.execute(
                    "UPDATE users SET last_login = NOW() WHERE id = :user_id",
                    {"user_id": user_id}
                )
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to update last login: {e}")


class AuthorizationService:
    """Role-based authorization service"""
    
    def __init__(self):
        self.roles = {
            "admin": {
                "permissions": ["*"],  # All permissions
                "description": "Administrator with full access"
            },
            "moderator": {
                "permissions": [
                    "video:analyze",
                    "video:generate",
                    "video:export",
                    "project:read",
                    "project:write",
                    "analytics:read"
                ],
                "description": "Moderator with content management access"
            },
            "user": {
                "permissions": [
                    "video:analyze",
                    "video:generate",
                    "video:export",
                    "project:read",
                    "project:write"
                ],
                "description": "Regular user with basic access"
            },
            "viewer": {
                "permissions": [
                    "video:read",
                    "project:read",
                    "analytics:read"
                ],
                "description": "Viewer with read-only access"
            }
        }
    
    def has_permission(self, user_role: str, required_permission: str) -> bool:
        """Check if user role has required permission"""
        if user_role not in self.roles:
            return False
        
        role_permissions = self.roles[user_role]["permissions"]
        
        # Check for wildcard permission
        if "*" in role_permissions:
            return True
        
        # Check for specific permission
        return required_permission in role_permissions
    
    def get_user_permissions(self, user_role: str) -> List[str]:
        """Get all permissions for a user role"""
        if user_role not in self.roles:
            return []
        
        return self.roles[user_role]["permissions"]
    
    def can_access_resource(self, user_id: str, resource_owner_id: str, user_role: str) -> bool:
        """Check if user can access a resource"""
        # Admin can access everything
        if user_role == "admin":
            return True
        
        # Users can access their own resources
        if user_id == resource_owner_id:
            return True
        
        # Moderators can access user resources
        if user_role == "moderator":
            return True
        
        return False


# Global services
auth_service = AuthenticationService()
authz_service = AuthorizationService()


# Dependency functions
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None
) -> Dict[str, Any]:
    """Get current authenticated user"""
    try:
        token = credentials.credentials
        
        # Verify token
        payload = auth_service.verify_token(token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check token type
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user ID
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if token is revoked
        if await auth_service.is_token_revoked(user_id, token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from database
        async with get_database_session() as session:
            db_ops = DatabaseOperations(session)
            user = await db_ops.get_user_by_id(UUID(user_id))
            
            if not user or not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or inactive",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        
        # Update last login
        await auth_service.update_user_last_login(user.id)
        
        # Set user info in request state
        if request:
            request.state.user_id = str(user.id)
            request.state.user_role = user.role
        
        return {
            "user_id": str(user.id),
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "role": user.role,
            "is_active": user.is_active,
            "is_verified": user.is_verified
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_active_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Get current active user"""
    if not current_user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_current_verified_user(current_user: Dict[str, Any] = Depends(get_current_active_user)) -> Dict[str, Any]:
    """Get current verified user"""
    if not current_user.get("is_verified"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unverified user"
        )
    return current_user


def require_permission(permission: str):
    """Decorator to require specific permission"""
    def permission_checker(current_user: Dict[str, Any] = Depends(get_current_active_user)) -> Dict[str, Any]:
        user_role = current_user.get("role", "viewer")
        
        if not authz_service.has_permission(user_role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission}"
            )
        
        return current_user
    
    return permission_checker


def require_role(required_role: str):
    """Decorator to require specific role"""
    def role_checker(current_user: Dict[str, Any] = Depends(get_current_active_user)) -> Dict[str, Any]:
        user_role = current_user.get("role", "viewer")
        
        # Define role hierarchy
        role_hierarchy = {
            "admin": 4,
            "moderator": 3,
            "user": 2,
            "viewer": 1
        }
        
        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role. Required: {required_role}"
            )
        
        return current_user
    
    return role_checker


async def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify token and return payload"""
    return auth_service.verify_token(token)


# API Key authentication
class APIKeyAuth:
    """API Key authentication"""
    
    def __init__(self):
        self.api_keys = {}  # In production, store in database
    
    def generate_api_key(self, user_id: str, name: str) -> str:
        """Generate API key for user"""
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        self.api_keys[key_hash] = {
            "user_id": user_id,
            "name": name,
            "created_at": datetime.utcnow(),
            "last_used": None
        }
        
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash in self.api_keys:
            # Update last used
            self.api_keys[key_hash]["last_used"] = datetime.utcnow()
            return self.api_keys[key_hash]
        
        return None


# OAuth2 integration (placeholder)
class OAuth2Service:
    """OAuth2 integration service"""
    
    def __init__(self):
        self.providers = {
            "google": {
                "client_id": "your_google_client_id",
                "client_secret": "your_google_client_secret",
                "authorization_url": "https://accounts.google.com/o/oauth2/auth",
                "token_url": "https://oauth2.googleapis.com/token",
                "user_info_url": "https://www.googleapis.com/oauth2/v1/userinfo"
            },
            "github": {
                "client_id": "your_github_client_id",
                "client_secret": "your_github_client_secret",
                "authorization_url": "https://github.com/login/oauth/authorize",
                "token_url": "https://github.com/login/oauth/access_token",
                "user_info_url": "https://api.github.com/user"
            }
        }
    
    async def get_authorization_url(self, provider: str, state: str) -> str:
        """Get OAuth2 authorization URL"""
        if provider not in self.providers:
            raise ValueError(f"Unsupported OAuth2 provider: {provider}")
        
        config = self.providers[provider]
        params = {
            "client_id": config["client_id"],
            "redirect_uri": f"http://localhost:8000/auth/callback/{provider}",
            "scope": "openid email profile",
            "response_type": "code",
            "state": state
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{config['authorization_url']}?{query_string}"
    
    async def exchange_code_for_token(self, provider: str, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token"""
        # Implementation would make HTTP request to token endpoint
        # This is a placeholder
        return {"access_token": "placeholder_token", "token_type": "Bearer"}
    
    async def get_user_info(self, provider: str, access_token: str) -> Dict[str, Any]:
        """Get user info from OAuth2 provider"""
        # Implementation would make HTTP request to user info endpoint
        # This is a placeholder
        return {
            "id": "12345",
            "email": "user@example.com",
            "name": "User Name",
            "picture": "https://example.com/avatar.jpg"
        }


# Global OAuth2 service
oauth2_service = OAuth2Service()


# Session management
class SessionManager:
    """Session management service"""
    
    def __init__(self):
        self.redis_client = None
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis for session storage"""
        try:
            settings = get_settings()
            self.redis_client = redis.from_url(
                settings.redis_url,
                db=settings.redis_db,
                password=settings.redis_password,
                decode_responses=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize Redis for sessions: {e}")
    
    async def create_session(self, user_id: str, session_data: Dict[str, Any]) -> str:
        """Create new session"""
        try:
            session_id = str(uuid4())
            session_key = f"session:{session_id}"
            
            session_data.update({
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat()
            })
            
            await self.redis_client.setex(session_key, 3600, str(session_data))  # 1 hour TTL
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        try:
            session_key = f"session:{session_id}"
            session_data = await self.redis_client.get(session_key)
            
            if session_data:
                # Update last activity
                await self.redis_client.setex(session_key, 3600, session_data)
                return eval(session_data)  # In production, use proper JSON parsing
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    async def destroy_session(self, session_id: str):
        """Destroy session"""
        try:
            session_key = f"session:{session_id}"
            await self.redis_client.delete(session_key)
        except Exception as e:
            logger.error(f"Failed to destroy session: {e}")


# Global session manager
session_manager = SessionManager()





























