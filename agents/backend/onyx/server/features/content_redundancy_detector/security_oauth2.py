"""
OAuth2 Implementation for Secure API Access
Following best practices for microservices security
"""

import os
import time
import secrets
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import HTTPException, Security, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from config import settings

try:
    from python_jose import jwt as pyjwt
except ImportError:
    pyjwt = None

import logging

logger = logging.getLogger(__name__)


class Token(BaseModel):
    """Token response model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    """Token data model"""
    username: Optional[str] = None
    user_id: Optional[str] = None
    scopes: list = []


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/token",
    scopes={
        "read": "Read access",
        "write": "Write access",
        "admin": "Admin access"
    }
)

# API Key scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
    scopes: list = None
) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.access_token_expire_minutes
        )
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "scopes": scopes or ["read"]
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.algorithm
    )
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=30)  # Refresh tokens last 30 days
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.algorithm
    )
    return encoded_jwt


async def verify_token(token: str = Depends(oauth2_scheme)) -> TokenData:
    """
    Verify and decode JWT token
    Dependency for FastAPI route protection
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm]
        )
        
        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        scopes: list = payload.get("scopes", [])
        
        if username is None:
            raise credentials_exception
        
        token_data = TokenData(
            username=username,
            user_id=user_id,
            scopes=scopes
        )
        return token_data
        
    except JWTError:
        raise credentials_exception


async def verify_api_key(api_key: str = Security(api_key_header)) -> Optional[str]:
    """
    Verify API key
    For machine-to-machine authentication
    """
    if not api_key:
        return None
    
    # In production, validate against database
    # For now, simple validation
    try:
        # Store API keys in database and validate here
        # This is a placeholder
        from security_advanced import security_manager
        key_record = security_manager.validate_api_key(api_key)
        
        if key_record and key_record.is_active:
            return key_record.user_id
        return None
    except Exception as e:
        logger.error(f"API key validation error: {e}")
        return None


async def get_current_user(
    token_data: TokenData = Depends(verify_token)
) -> Dict[str, Any]:
    """
    Get current authenticated user
    Dependency for protected routes
    """
    # In production, fetch user from database
    return {
        "username": token_data.username,
        "user_id": token_data.user_id,
        "scopes": token_data.scopes
    }


async def require_scopes(*required_scopes: str):
    """
    Dependency factory for route scope requirements
    
    Usage:
        @router.get("/admin", dependencies=[Depends(require_scopes("admin"))])
    """
    async def scope_checker(current_user: Dict = Depends(get_current_user)):
        user_scopes = current_user.get("scopes", [])
        
        # Check if user has required scopes
        missing_scopes = set(required_scopes) - set(user_scopes)
        
        if missing_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required scopes: {', '.join(missing_scopes)}"
            )
        
        return current_user
    
    return scope_checker


def verify_api_key_or_token(
    api_key: Optional[str] = Security(api_key_header),
    token_data: Optional[TokenData] = Depends(verify_token)
) -> Dict[str, Any]:
    """
    Verify either API key or OAuth2 token
    For flexible authentication
    """
    if api_key:
        user_id = verify_api_key(api_key)
        if user_id:
            return {"user_id": user_id, "auth_type": "api_key"}
    
    if token_data:
        return {
            "username": token_data.username,
            "user_id": token_data.user_id,
            "scopes": token_data.scopes,
            "auth_type": "oauth2"
        }
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )

