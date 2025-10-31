"""
Authentication API
==================

Simple and clear authentication API for the Document Workflow Chain system.
"""

from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta

from ..core.database import get_database
from ..core.config import settings
from ..models.user import User

# Create router
router = APIRouter()

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Request/Response models
class UserCreate(BaseModel):
    """User creation request"""
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    password: str


class UserLogin(BaseModel):
    """User login request"""
    username: str
    password: str


class Token(BaseModel):
    """Token response"""
    access_token: str
    token_type: str


class UserResponse(BaseModel):
    """User response"""
    id: int
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    created_at: str
    
    class Config:
        from_attributes = True


# Utility functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_database)
) -> User:
    """Get current user from token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    from sqlalchemy import select
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
    
    return user


# Auth endpoints
@router.post("/register", response_model=UserResponse)
async def register_user(
    user: UserCreate,
    db: AsyncSession = Depends(get_database)
):
    """Register new user - simple and clear"""
    try:
        from sqlalchemy import select
        
        # Check if user already exists
        result = await db.execute(select(User).where(User.username == user.username))
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        result = await db.execute(select(User).where(User.email == user.email))
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        hashed_password = get_password_hash(user.password)
        new_user = User(
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            hashed_password=hashed_password
        )
        
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        
        return UserResponse(
            id=new_user.id,
            username=new_user.username,
            email=new_user.email,
            full_name=new_user.full_name,
            is_active=new_user.is_active,
            created_at=new_user.created_at.isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register user: {str(e)}"
        )


@router.post("/login", response_model=Token)
async def login_user(
    user: UserLogin,
    db: AsyncSession = Depends(get_database)
):
    """Login user - simple and clear"""
    try:
        from sqlalchemy import select
        
        # Get user
        result = await db.execute(select(User).where(User.username == user.username))
        db_user = result.scalar_one_or_none()
        
        if not db_user or not verify_password(user.password, db_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": db_user.username}, expires_delta=access_token_expires
        )
        
        return Token(access_token=access_token, token_type="bearer")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to login user: {str(e)}"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user info - simple and clear"""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        created_at=current_user.created_at.isoformat()
    )


