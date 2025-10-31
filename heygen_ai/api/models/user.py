from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from pydantic import BaseModel, Field, validator, EmailStr, HttpUrl
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone, date
from enum import Enum
import re
from . import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
User Data Models for HeyGen AI FastAPI
FastAPI best practices for user data models with comprehensive validation and documentation.
"""


    TimestampedModel, IdentifiedModel, StatusEnum, RoleEnum,
    validate_email, validate_password
)

# =============================================================================
# User Enums
# =============================================================================

class UserStatusEnum(str, Enum):
    """User status enumeration following FastAPI best practices."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"
    DELETED = "deleted"

class UserSubscriptionEnum(str, Enum):
    """User subscription enumeration following FastAPI best practices."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

class UserLanguageEnum(str, Enum):
    """User language enumeration following FastAPI best practices."""
    EN = "en"
    ES = "es"
    FR = "fr"
    DE = "de"
    IT = "it"
    PT = "pt"
    RU = "ru"
    ZH = "zh"
    JA = "ja"
    KO = "ko"

# =============================================================================
# User Base Models
# =============================================================================

class UserBase(BaseModel):
    """Base user model following FastAPI best practices."""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="User's full name",
        example="John Doe"
    )
    
    email: EmailStr = Field(
        ...,
        description="User's email address",
        example="john.doe@example.com"
    )
    
    username: Optional[str] = Field(
        None,
        min_length=3,
        max_length=50,
        regex=r'^[a-zA-Z0-9_-]+$',
        description="User's unique username (alphanumeric, underscore, hyphen only)",
        example="john_doe_123"
    )
    
    bio: Optional[str] = Field(
        None,
        max_length=500,
        description="User's biography or description",
        example="AI enthusiast and video creator"
    )
    
    avatar_url: Optional[HttpUrl] = Field(
        None,
        description="URL to user's avatar image",
        example="https://example.com/avatars/john_doe.jpg"
    )
    
    language: UserLanguageEnum = Field(
        default=UserLanguageEnum.EN,
        description="User's preferred language",
        example=UserLanguageEnum.EN
    )
    
    timezone: Optional[str] = Field(
        None,
        description="User's timezone (IANA format)",
        example="America/New_York"
    )
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        schema_extra = {
            "example": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "username": "john_doe_123",
                "bio": "AI enthusiast and video creator",
                "avatar_url": "https://example.com/avatars/john_doe.jpg",
                "language": "en",
                "timezone": "America/New_York"
            }
        }

# =============================================================================
# User Request Models
# =============================================================================

class UserCreate(UserBase):
    """User creation model following FastAPI best practices."""
    
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="User's password (min 8 chars, must contain uppercase, lowercase, digit)",
        example="SecurePass123!"
    )
    
    confirm_password: str = Field(
        ...,
        description="Password confirmation",
        example="SecurePass123!"
    )
    
    accept_terms: bool = Field(
        ...,
        description="User must accept terms and conditions",
        example=True
    )
    
    marketing_emails: bool = Field(
        default=False,
        description="Opt-in for marketing emails",
        example=False
    )
    
    @validator('password')
    def validate_password(cls, v) -> bool:
        """Password validation following FastAPI best practices."""
        return validate_password(v)
    
    @validator('confirm_password')
    def validate_confirm_password(cls, v, values) -> bool:
        """Password confirmation validation."""
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v
    
    @validator('accept_terms')
    def validate_accept_terms(cls, v) -> bool:
        """Terms acceptance validation."""
        if not v:
            raise ValueError('Must accept terms and conditions')
        return v
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        schema_extra = {
            "example": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "username": "john_doe_123",
                "password": "SecurePass123!",
                "confirm_password": "SecurePass123!",
                "bio": "AI enthusiast and video creator",
                "accept_terms": True,
                "marketing_emails": False,
                "language": "en",
                "timezone": "America/New_York"
            }
        }

class UserUpdate(BaseModel):
    """User update model following FastAPI best practices."""
    
    name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=100,
        description="User's full name",
        example="John Doe"
    )
    
    username: Optional[str] = Field(
        None,
        min_length=3,
        max_length=50,
        regex=r'^[a-zA-Z0-9_-]+$',
        description="User's unique username",
        example="john_doe_123"
    )
    
    bio: Optional[str] = Field(
        None,
        max_length=500,
        description="User's biography or description",
        example="AI enthusiast and video creator"
    )
    
    avatar_url: Optional[HttpUrl] = Field(
        None,
        description="URL to user's avatar image",
        example="https://example.com/avatars/john_doe.jpg"
    )
    
    language: Optional[UserLanguageEnum] = Field(
        None,
        description="User's preferred language",
        example=UserLanguageEnum.EN
    )
    
    timezone: Optional[str] = Field(
        None,
        description="User's timezone (IANA format)",
        example="America/New_York"
    )
    
    marketing_emails: Optional[bool] = Field(
        None,
        description="Opt-in for marketing emails",
        example=False
    )
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        schema_extra = {
            "example": {
                "name": "John Doe Updated",
                "bio": "Updated bio: AI enthusiast and video creator",
                "language": "en",
                "marketing_emails": True
            }
        }

class UserLogin(BaseModel):
    """User login model following FastAPI best practices."""
    
    email: EmailStr = Field(
        ...,
        description="User's email address",
        example="john.doe@example.com"
    )
    
    password: str = Field(
        ...,
        min_length=1,
        description="User's password",
        example="SecurePass123!"
    )
    
    remember_me: bool = Field(
        default=False,
        description="Remember user session",
        example=False
    )
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        schema_extra = {
            "example": {
                "email": "john.doe@example.com",
                "password": "SecurePass123!",
                "remember_me": True
            }
        }

class UserPasswordChange(BaseModel):
    """User password change model following FastAPI best practices."""
    
    current_password: str = Field(
        ...,
        description="Current password",
        example="CurrentPass123!"
    )
    
    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="New password (min 8 chars, must contain uppercase, lowercase, digit)",
        example="NewSecurePass123!"
    )
    
    confirm_new_password: str = Field(
        ...,
        description="New password confirmation",
        example="NewSecurePass123!"
    )
    
    @validator('new_password')
    def validate_new_password(cls, v) -> bool:
        """New password validation."""
        return validate_password(v)
    
    @validator('confirm_new_password')
    def validate_confirm_new_password(cls, v, values) -> bool:
        """New password confirmation validation."""
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('New passwords do not match')
        return v
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        schema_extra = {
            "example": {
                "current_password": "CurrentPass123!",
                "new_password": "NewSecurePass123!",
                "confirm_new_password": "NewSecurePass123!"
            }
        }

class UserEmailChange(BaseModel):
    """User email change model following FastAPI best practices."""
    
    current_password: str = Field(
        ...,
        description="Current password for verification",
        example="CurrentPass123!"
    )
    
    new_email: EmailStr = Field(
        ...,
        description="New email address",
        example="john.doe.new@example.com"
    )
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        schema_extra = {
            "example": {
                "current_password": "CurrentPass123!",
                "new_email": "john.doe.new@example.com"
            }
        }

# =============================================================================
# User Response Models
# =============================================================================

class UserResponse(IdentifiedModel, UserBase):
    """User response model following FastAPI best practices."""
    
    id: int = Field(
        ...,
        description="User's unique identifier",
        example=1
    )
    
    status: UserStatusEnum = Field(
        ...,
        description="User's account status",
        example=UserStatusEnum.ACTIVE
    )
    
    role: RoleEnum = Field(
        ...,
        description="User's role in the system",
        example=RoleEnum.USER
    )
    
    subscription: UserSubscriptionEnum = Field(
        ...,
        description="User's subscription level",
        example=UserSubscriptionEnum.FREE
    )
    
    email_verified: bool = Field(
        ...,
        description="Whether user's email is verified",
        example=False
    )
    
    last_login: Optional[datetime] = Field(
        None,
        description="User's last login timestamp",
        example="2024-01-15T10:30:00Z"
    )
    
    login_count: int = Field(
        default=0,
        description="Number of times user has logged in",
        example=15
    )
    
    created_at: datetime = Field(
        ...,
        description="User account creation timestamp",
        example="2024-01-01T00:00:00Z"
    )
    
    updated_at: Optional[datetime] = Field(
        None,
        description="User account last update timestamp",
        example="2024-01-15T10:30:00Z"
    )
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        orm_mode = True
        schema_extra = {
            "example": {
                "id": 1,
                "name": "John Doe",
                "email": "john.doe@example.com",
                "username": "john_doe_123",
                "bio": "AI enthusiast and video creator",
                "avatar_url": "https://example.com/avatars/john_doe.jpg",
                "status": "active",
                "role": "user",
                "subscription": "premium",
                "email_verified": True,
                "language": "en",
                "timezone": "America/New_York",
                "last_login": "2024-01-15T10:30:00Z",
                "login_count": 15,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }

class UserProfileResponse(UserResponse):
    """User profile response model following FastAPI best practices."""
    
    # Extended profile information
    total_videos: int = Field(
        default=0,
        description="Total number of videos created by user",
        example=25
    )
    
    total_projects: int = Field(
        default=0,
        description="Total number of projects created by user",
        example=10
    )
    
    total_duration: float = Field(
        default=0.0,
        description="Total duration of all user videos in seconds",
        example=3600.5
    )
    
    storage_used: int = Field(
        default=0,
        description="Storage used by user in bytes",
        example=1073741824  # 1GB
    )
    
    storage_limit: int = Field(
        default=1073741824,
        description="Storage limit for user in bytes",
        example=5368709120  # 5GB
    )
    
    api_requests_used: int = Field(
        default=0,
        description="Number of API requests used by user",
        example=1500
    )
    
    api_requests_limit: int = Field(
        default=1000,
        description="API requests limit for user",
        example=5000
    )
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        orm_mode = True
        schema_extra = {
            "example": {
                "id": 1,
                "name": "John Doe",
                "email": "john.doe@example.com",
                "username": "john_doe_123",
                "bio": "AI enthusiast and video creator",
                "avatar_url": "https://example.com/avatars/john_doe.jpg",
                "status": "active",
                "role": "user",
                "subscription": "premium",
                "email_verified": True,
                "language": "en",
                "timezone": "America/New_York",
                "last_login": "2024-01-15T10:30:00Z",
                "login_count": 15,
                "total_videos": 25,
                "total_projects": 10,
                "total_duration": 3600.5,
                "storage_used": 1073741824,
                "storage_limit": 5368709120,
                "api_requests_used": 1500,
                "api_requests_limit": 5000,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }

class UserListResponse(BaseModel):
    """User list response model following FastAPI best practices."""
    
    users: List[UserResponse] = Field(
        ...,
        description="List of users"
    )
    
    total_count: int = Field(
        ...,
        description="Total number of users",
        example=100
    )
    
    page: int = Field(
        ...,
        description="Current page number",
        example=1
    )
    
    page_size: int = Field(
        ...,
        description="Number of users per page",
        example=20
    )
    
    total_pages: int = Field(
        ...,
        description="Total number of pages",
        example=5
    )
    
    has_next: bool = Field(
        ...,
        description="Whether there are more pages",
        example=True
    )
    
    has_prev: bool = Field(
        ...,
        description="Whether there are previous pages",
        example=False
    )
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        schema_extra = {
            "example": {
                "users": [
                    {
                        "id": 1,
                        "name": "John Doe",
                        "email": "john.doe@example.com",
                        "username": "john_doe_123",
                        "status": "active",
                        "role": "user",
                        "subscription": "premium",
                        "created_at": "2024-01-01T00:00:00Z"
                    }
                ],
                "total_count": 100,
                "page": 1,
                "page_size": 20,
                "total_pages": 5,
                "has_next": True,
                "has_prev": False
            }
        }

class UserAuthResponse(BaseModel):
    """User authentication response model following FastAPI best practices."""
    
    access_token: str = Field(
        ...,
        description="JWT access token",
        example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    )
    
    token_type: str = Field(
        default="bearer",
        description="Token type",
        example="bearer"
    )
    
    expires_in: int = Field(
        ...,
        description="Token expiration time in seconds",
        example=3600
    )
    
    refresh_token: Optional[str] = Field(
        None,
        description="JWT refresh token",
        example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    )
    
    user: UserResponse = Field(
        ...,
        description="User information"
    )
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600,
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "user": {
                    "id": 1,
                    "name": "John Doe",
                    "email": "john.doe@example.com",
                    "username": "john_doe_123",
                    "status": "active",
                    "role": "user",
                    "subscription": "premium",
                    "email_verified": True,
                    "created_at": "2024-01-01T00:00:00Z"
                }
            }
        }

# =============================================================================
# User Statistics Models
# =============================================================================

class UserStatistics(BaseModel):
    """User statistics model following FastAPI best practices."""
    
    user_id: int = Field(
        ...,
        description="User ID",
        example=1
    )
    
    period: str = Field(
        ...,
        description="Statistics period",
        example="30d"
    )
    
    start_date: datetime = Field(
        ...,
        description="Statistics start date",
        example="2024-01-01T00:00:00Z"
    )
    
    end_date: datetime = Field(
        ...,
        description="Statistics end date",
        example="2024-01-31T23:59:59Z"
    )
    
    videos: Dict[str, Any] = Field(
        default_factory=dict,
        description="Video statistics",
        example={
            "total_videos": 25,
            "total_duration": 3600.5,
            "completed_videos": 20,
            "processing_videos": 3,
            "failed_videos": 2
        }
    )
    
    projects: Dict[str, Any] = Field(
        default_factory=dict,
        description="Project statistics",
        example={
            "total_projects": 10,
            "active_projects": 5,
            "completed_projects": 5
        }
    )
    
    usage: Dict[str, Any] = Field(
        default_factory=dict,
        description="Usage statistics",
        example={
            "total_requests": 1500,
            "successful_requests": 1450,
            "failed_requests": 50,
            "avg_response_time": 250.5
        }
    )
    
    storage: Dict[str, Any] = Field(
        default_factory=dict,
        description="Storage statistics",
        example={
            "used_bytes": 1073741824,
            "limit_bytes": 5368709120,
            "usage_percentage": 20.0
        }
    )
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        schema_extra = {
            "example": {
                "user_id": 1,
                "period": "30d",
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-31T23:59:59Z",
                "videos": {
                    "total_videos": 25,
                    "total_duration": 3600.5,
                    "completed_videos": 20,
                    "processing_videos": 3,
                    "failed_videos": 2
                },
                "projects": {
                    "total_projects": 10,
                    "active_projects": 5,
                    "completed_projects": 5
                },
                "usage": {
                    "total_requests": 1500,
                    "successful_requests": 1450,
                    "failed_requests": 50,
                    "avg_response_time": 250.5
                },
                "storage": {
                    "used_bytes": 1073741824,
                    "limit_bytes": 5368709120,
                    "usage_percentage": 20.0
                }
            }
        }

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "UserStatusEnum",
    "UserSubscriptionEnum", 
    "UserLanguageEnum",
    "UserBase",
    "UserCreate",
    "UserUpdate",
    "UserLogin",
    "UserPasswordChange",
    "UserEmailChange",
    "UserResponse",
    "UserProfileResponse",
    "UserListResponse",
    "UserAuthResponse",
    "UserStatistics"
] 