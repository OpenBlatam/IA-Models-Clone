from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone, date
from enum import Enum
from pydantic import (
import re
from .base_schemas import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
User Schemas for HeyGen AI API
User management, authentication, and profile operations.
"""

    BaseModel, Field, validator, root_validator, 
    ConfigDict, computed_field, model_validator, EmailStr
)

    BaseRequest, BaseResponse, DataResponse, PaginatedDataResponse,
    IDField, TimestampFields, StatusFields, MetadataFields
)

# =============================================================================
# User Enums
# =============================================================================

class UserRole(str, Enum):
    """User role enumeration."""
    ADMIN = "admin"
    USER = "user"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    GUEST = "guest"

class UserStatus(str, Enum):
    """User status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"
    VERIFIED = "verified"

class SubscriptionTier(str, Enum):
    """Subscription tier enumeration."""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"

class AuthProvider(str, Enum):
    """Authentication provider enumeration."""
    EMAIL = "email"
    GOOGLE = "google"
    FACEBOOK = "facebook"
    APPLE = "apple"
    GITHUB = "github"
    LINKEDIN = "linkedin"

# =============================================================================
# User Base Models
# =============================================================================

class UserBase(BaseModel):
    """Base user model."""
    email: EmailStr = Field(
        description="User email address"
    )
    first_name: str = Field(
        min_length=1,
        max_length=50,
        description="User first name"
    )
    last_name: str = Field(
        min_length=1,
        max_length=50,
        description="User last name"
    )
    role: UserRole = Field(
        default=UserRole.USER,
        description="User role"
    )
    status: UserStatus = Field(
        default=UserStatus.PENDING,
        description="User status"
    )
    
    @validator('first_name', 'last_name')
    def validate_name(cls, v) -> bool:
        """Validate name fields."""
        if not v or not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()
    
    @validator('email')
    def validate_email(cls, v) -> bool:
        """Validate email format."""
        if not v or not v.strip():
            raise ValueError('Email cannot be empty')
        return v.lower().strip()

class UserProfile(BaseModel):
    """User profile model."""
    bio: Optional[str] = Field(
        default=None,
        max_length=500,
        description="User biography"
    )
    avatar_url: Optional[str] = Field(
        default=None,
        description="Avatar image URL"
    )
    website: Optional[str] = Field(
        default=None,
        description="Personal website URL"
    )
    location: Optional[str] = Field(
        default=None,
        max_length=100,
        description="User location"
    )
    company: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Company name"
    )
    job_title: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Job title"
    )
    phone: Optional[str] = Field(
        default=None,
        description="Phone number"
    )
    date_of_birth: Optional[date] = Field(
        default=None,
        description="Date of birth"
    )
    
    @validator('website')
    def validate_website(cls, v) -> bool:
        """Validate website URL."""
        if v and not v.startswith(('http://', 'https://')):
            v = 'https://' + v
        return v
    
    @validator('phone')
    def validate_phone(cls, v) -> bool:
        """Validate phone number."""
        if v:
            # Remove all non-digit characters
            digits_only = re.sub(r'\D', '', v)
            if len(digits_only) < 10:
                raise ValueError('Phone number must have at least 10 digits')
        return v

# =============================================================================
# User Request Models
# =============================================================================

class UserCreateRequest(BaseRequest):
    """User creation request model."""
    email: EmailStr = Field(
        description="User email address"
    )
    password: str = Field(
        min_length=8,
        max_length=128,
        description="User password"
    )
    first_name: str = Field(
        min_length=1,
        max_length=50,
        description="User first name"
    )
    last_name: str = Field(
        min_length=1,
        max_length=50,
        description="User last name"
    )
    role: UserRole = Field(
        default=UserRole.USER,
        description="User role"
    )
    profile: Optional[UserProfile] = Field(
        default=None,
        description="User profile information"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )
    
    @validator('password')
    def validate_password(cls, v) -> bool:
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        
        # Check for at least one uppercase letter
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        
        # Check for at least one lowercase letter
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        
        # Check for at least one digit
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        
        # Check for at least one special character
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        
        return v

class UserUpdateRequest(BaseRequest):
    """User update request model."""
    first_name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=50,
        description="User first name"
    )
    last_name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=50,
        description="User last name"
    )
    profile: Optional[UserProfile] = Field(
        default=None,
        description="User profile information"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )

class UserPasswordUpdateRequest(BaseRequest):
    """User password update request model."""
    current_password: str = Field(
        description="Current password"
    )
    new_password: str = Field(
        min_length=8,
        max_length=128,
        description="New password"
    )
    confirm_password: str = Field(
        description="Password confirmation"
    )
    
    @root_validator
    def validate_passwords(cls, values) -> bool:
        """Validate password update."""
        new_password = values.get('new_password')
        confirm_password = values.get('confirm_password')
        
        if new_password != confirm_password:
            raise ValueError('New password and confirmation do not match')
        
        # Validate new password strength
        if len(new_password) < 8:
            raise ValueError('Password must be at least 8 characters long')
        
        if not re.search(r'[A-Z]', new_password):
            raise ValueError('Password must contain at least one uppercase letter')
        
        if not re.search(r'[a-z]', new_password):
            raise ValueError('Password must contain at least one lowercase letter')
        
        if not re.search(r'\d', new_password):
            raise ValueError('Password must contain at least one digit')
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', new_password):
            raise ValueError('Password must contain at least one special character')
        
        return values

class UserSearchRequest(BaseRequest):
    """User search request model."""
    query: Optional[str] = Field(
        default=None,
        description="Search query"
    )
    role: Optional[UserRole] = Field(
        default=None,
        description="Filter by user role"
    )
    status: Optional[UserStatus] = Field(
        default=None,
        description="Filter by user status"
    )
    created_after: Optional[datetime] = Field(
        default=None,
        description="Filter users created after this date"
    )
    created_before: Optional[datetime] = Field(
        default=None,
        description="Filter users created before this date"
    )
    page: int = Field(
        default=1,
        ge=1,
        description="Page number"
    )
    per_page: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Items per page"
    )

# =============================================================================
# Authentication Models
# =============================================================================

class LoginRequest(BaseRequest):
    """User login request model."""
    email: EmailStr = Field(
        description="User email address"
    )
    password: str = Field(
        description="User password"
    )
    remember_me: bool = Field(
        default=False,
        description="Remember user session"
    )
    device_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Device information"
    )

class OAuthLoginRequest(BaseRequest):
    """OAuth login request model."""
    provider: AuthProvider = Field(
        description="OAuth provider"
    )
    code: str = Field(
        description="Authorization code"
    )
    redirect_uri: Optional[str] = Field(
        default=None,
        description="Redirect URI"
    )
    state: Optional[str] = Field(
        default=None,
        description="State parameter"
    )

class RefreshTokenRequest(BaseRequest):
    """Refresh token request model."""
    refresh_token: str = Field(
        description="Refresh token"
    )

class LogoutRequest(BaseRequest):
    """User logout request model."""
    refresh_token: Optional[str] = Field(
        default=None,
        description="Refresh token to invalidate"
    )
    all_sessions: bool = Field(
        default=False,
        description="Logout from all sessions"
    )

# =============================================================================
# User Response Models
# =============================================================================

class UserResponse(BaseModel):
    """User response model."""
    id: str = Field(
        description="User ID"
    )
    email: EmailStr = Field(
        description="User email address"
    )
    first_name: str = Field(
        description="User first name"
    )
    last_name: str = Field(
        description="User last name"
    )
    role: UserRole = Field(
        description="User role"
    )
    status: UserStatus = Field(
        description="User status"
    )
    profile: Optional[UserProfile] = Field(
        default=None,
        description="User profile"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )
    created_at: datetime = Field(
        description="Account creation timestamp"
    )
    updated_at: datetime = Field(
        description="Last update timestamp"
    )
    last_login_at: Optional[datetime] = Field(
        default=None,
        description="Last login timestamp"
    )
    email_verified_at: Optional[datetime] = Field(
        default=None,
        description="Email verification timestamp"
    )
    
    @computed_field
    @property
    def full_name(self) -> str:
        """Get user's full name."""
        return f"{self.first_name} {self.last_name}"
    
    @computed_field
    @property
    def is_verified(self) -> bool:
        """Check if user email is verified."""
        return self.email_verified_at is not None
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        use_enum_values=True
    )

class UserListResponse(PaginatedDataResponse[UserResponse]):
    """User list response model."""
    data: List[UserResponse] = Field(
        description="List of users"
    )

class UserDetailResponse(DataResponse[UserResponse]):
    """User detail response model."""
    data: UserResponse = Field(
        description="User details"
    )

# =============================================================================
# Authentication Response Models
# =============================================================================

class AuthToken(BaseModel):
    """Authentication token model."""
    access_token: str = Field(
        description="Access token"
    )
    refresh_token: str = Field(
        description="Refresh token"
    )
    token_type: str = Field(
        default="bearer",
        description="Token type"
    )
    expires_in: int = Field(
        description="Token expiration time in seconds"
    )
    expires_at: datetime = Field(
        description="Token expiration timestamp"
    )
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

class LoginResponse(DataResponse[Dict[str, Any]]):
    """Login response model."""
    data: Dict[str, Any] = Field(
        description="Login response data"
    )
    user: UserResponse = Field(
        description="User information"
    )
    tokens: AuthToken = Field(
        description="Authentication tokens"
    )

class TokenResponse(DataResponse[AuthToken]):
    """Token response model."""
    data: AuthToken = Field(
        description="Authentication tokens"
    )

# =============================================================================
# Subscription Models
# =============================================================================

class SubscriptionInfo(BaseModel):
    """Subscription information model."""
    tier: SubscriptionTier = Field(
        description="Subscription tier"
    )
    status: str = Field(
        description="Subscription status"
    )
    current_period_start: datetime = Field(
        description="Current period start"
    )
    current_period_end: datetime = Field(
        description="Current period end"
    )
    cancel_at_period_end: bool = Field(
        default=False,
        description="Cancel at period end"
    )
    features: List[str] = Field(
        default=[],
        description="Available features"
    )
    limits: Dict[str, int] = Field(
        default={},
        description="Usage limits"
    )
    
    @computed_field
    @property
    def is_active(self) -> bool:
        """Check if subscription is active."""
        now = datetime.now(timezone.utc)
        return (
            self.status == "active" and
            self.current_period_start <= now <= self.current_period_end
        )
    
    @computed_field
    @property
    def days_remaining(self) -> int:
        """Get days remaining in current period."""
        now = datetime.now(timezone.utc)
        if now > self.current_period_end:
            return 0
        delta = self.current_period_end - now
        return delta.days
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        use_enum_values=True
    )

class UserWithSubscriptionResponse(UserResponse):
    """User response with subscription information."""
    subscription: Optional[SubscriptionInfo] = Field(
        default=None,
        description="Subscription information"
    )

# =============================================================================
# Session Models
# =============================================================================

class UserSession(BaseModel):
    """User session model."""
    id: str = Field(
        description="Session ID"
    )
    user_id: str = Field(
        description="User ID"
    )
    device_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Device information"
    )
    ip_address: Optional[str] = Field(
        default=None,
        description="IP address"
    )
    user_agent: Optional[str] = Field(
        default=None,
        description="User agent string"
    )
    created_at: datetime = Field(
        description="Session creation timestamp"
    )
    last_activity_at: datetime = Field(
        description="Last activity timestamp"
    )
    expires_at: datetime = Field(
        description="Session expiration timestamp"
    )
    is_active: bool = Field(
        default=True,
        description="Session active status"
    )
    
    @computed_field
    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now(timezone.utc) > self.expires_at
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

class UserSessionsResponse(DataResponse[List[UserSession]]):
    """User sessions response model."""
    data: List[UserSession] = Field(
        description="List of user sessions"
    )

# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Enums
    "UserRole",
    "UserStatus",
    "SubscriptionTier",
    "AuthProvider",
    
    # Base Models
    "UserBase",
    "UserProfile",
    
    # Request Models
    "UserCreateRequest",
    "UserUpdateRequest",
    "UserPasswordUpdateRequest",
    "UserSearchRequest",
    
    # Authentication Models
    "LoginRequest",
    "OAuthLoginRequest",
    "RefreshTokenRequest",
    "LogoutRequest",
    
    # Response Models
    "UserResponse",
    "UserListResponse",
    "UserDetailResponse",
    
    # Authentication Response Models
    "AuthToken",
    "LoginResponse",
    "TokenResponse",
    
    # Subscription Models
    "SubscriptionInfo",
    "UserWithSubscriptionResponse",
    
    # Session Models
    "UserSession",
    "UserSessionsResponse",
] 