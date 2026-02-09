"""
OAuth2 Configuration for Secure API Access
===========================================

Implements OAuth2 with JWT tokens for secure API access.
"""

import os
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from pydantic import BaseModel
import redis.asyncio as aioredis
import logging

logger = logging.getLogger(__name__)

# OAuth2 configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/token",
    scopes={
        "read": "Read access",
        "write": "Write access",
        "admin": "Admin access",
    }
)

# Redis for token blacklist
redis_client: Optional[aioredis.Redis] = None


async def get_redis_client():
    """Get Redis client for token blacklist."""
    global redis_client
    if redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/2")
        redis_client = await aioredis.from_url(redis_url)
    return redis_client


class Token(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    scopes: list = []


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password."""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


async def verify_token(token: str, required_scopes: list = None) -> TokenData:
    """
    Verify JWT token.
    
    Args:
        token: JWT token
        required_scopes: Required OAuth2 scopes
    
    Returns:
        TokenData with user information
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Check if token is blacklisted
        redis = await get_redis_client()
        is_blacklisted = await redis.get(f"blacklist:{token}")
        if is_blacklisted:
            raise credentials_exception
        
        # Decode token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        scopes: list = payload.get("scopes", [])
        
        if username is None:
            raise credentials_exception
        
        # Check scopes
        if required_scopes:
            for scope in required_scopes:
                if scope not in scopes:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Not enough permissions. Required scope: {scope}",
                    )
        
        token_data = TokenData(username=username, scopes=scopes)
        return token_data
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise credentials_exception


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    required_scopes: list = None
) -> TokenData:
    """
    Get current user from token.
    
    Args:
        token: OAuth2 token
        required_scopes: Required scopes
    
    Returns:
        TokenData
    """
    return await verify_token(token, required_scopes)


async def get_current_active_user(
    current_user: TokenData = Depends(get_current_user)
) -> TokenData:
    """Get current active user."""
    # Add user status check here if needed
    return current_user


async def revoke_token(token: str):
    """Revoke (blacklist) a token."""
    try:
        redis = await get_redis_client()
        
        # Decode token to get expiration
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
        exp = payload.get("exp")
        
        if exp:
            # Set expiration time for blacklist entry
            ttl = exp - int(datetime.utcnow().timestamp())
            if ttl > 0:
                await redis.setex(f"blacklist:{token}", ttl, "1")
                logger.info(f"Token revoked: {token[:20]}...")
    except Exception as e:
        logger.error(f"Error revoking token: {e}")


# Rate limiting decorator for OAuth2
def rate_limit_oauth(max_attempts: int = 5, window_seconds: int = 300):
    """Rate limit OAuth2 login attempts."""
    async def decorator(username: str):
        redis = await get_redis_client()
        key = f"oauth_rate_limit:{username}"
        
        attempts = await redis.get(key)
        if attempts and int(attempts) >= max_attempts:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many login attempts. Please try again later.",
            )
        
        # Increment attempts
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, window_seconds)
        await pipe.execute()
        
        return True
    
    return decorator















