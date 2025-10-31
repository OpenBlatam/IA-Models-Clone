"""
Gamma App - Authentication Utilities
JWT-based authentication and authorization
"""

import jwt
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from passlib.hash import bcrypt

from .config import get_settings

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    settings = get_settings()
    
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt

def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create JWT refresh token"""
    settings = get_settings()
    
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt

def verify_token(token: str) -> Dict[str, Any]:
    """Verify and decode JWT token"""
    settings = get_settings()
    
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.JWTError:
        raise ValueError("Invalid token")

def get_current_user_id(token: str) -> str:
    """Get current user ID from token"""
    payload = verify_token(token)
    user_id = payload.get("sub")
    if user_id is None:
        raise ValueError("Token does not contain user ID")
    return user_id



























