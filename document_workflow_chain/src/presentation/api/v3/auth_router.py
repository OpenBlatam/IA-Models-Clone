"""
Authentication API Router v3
============================

Advanced authentication and authorization endpoints.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import jwt
import hashlib
import secrets

from ....shared.container import Container
from ....shared.utils.decorators import rate_limit, log_execution
from ....shared.utils.helpers import DateTimeHelpers, StringHelpers, HashHelpers
from ....shared.config import get_settings


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3/auth", tags=["Authentication v3"])
security = HTTPBearer()


# Request/Response models
class LoginRequest(BaseModel):
    """Login request"""
    username: str = Field(..., min_length=3, max_length=100, description="Username or email")
    password: str = Field(..., min_length=8, description="Password")
    remember_me: bool = Field(False, description="Remember me option")
    device_info: Optional[Dict[str, Any]] = Field(None, description="Device information")


class RegisterRequest(BaseModel):
    """Registration request"""
    username: str = Field(..., min_length=3, max_length=100, description="Username")
    email: str = Field(..., description="Email address")
    password: str = Field(..., min_length=8, description="Password")
    full_name: Optional[str] = Field(None, max_length=255, description="Full name")
    terms_accepted: bool = Field(..., description="Terms and conditions acceptance")


class TokenResponse(BaseModel):
    """Token response"""
    access_token: str = Field(..., description="Access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user_info: Dict[str, Any] = Field(..., description="User information")


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""
    refresh_token: str = Field(..., description="Refresh token")


class ChangePasswordRequest(BaseModel):
    """Change password request"""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")
    confirm_password: str = Field(..., description="Confirm new password")


class ResetPasswordRequest(BaseModel):
    """Reset password request"""
    email: str = Field(..., description="Email address")


class ConfirmResetPasswordRequest(BaseModel):
    """Confirm reset password request"""
    token: str = Field(..., description="Reset token")
    new_password: str = Field(..., min_length=8, description="New password")
    confirm_password: str = Field(..., description="Confirm new password")


class UserProfileResponse(BaseModel):
    """User profile response"""
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    full_name: Optional[str] = Field(None, description="Full name")
    avatar_url: Optional[str] = Field(None, description="Avatar URL")
    bio: Optional[str] = Field(None, description="Bio")
    timezone: str = Field(..., description="Timezone")
    language: str = Field(..., description="Language")
    roles: List[str] = Field(..., description="User roles")
    permissions: List[str] = Field(..., description="User permissions")
    is_active: bool = Field(..., description="Active status")
    is_verified: bool = Field(..., description="Verification status")
    created_at: str = Field(..., description="Creation timestamp")
    last_login: Optional[str] = Field(None, description="Last login timestamp")


class APIKeyResponse(BaseModel):
    """API key response"""
    api_key: str = Field(..., description="API key")
    key_id: str = Field(..., description="Key ID")
    name: str = Field(..., description="Key name")
    permissions: List[str] = Field(..., description="Key permissions")
    expires_at: Optional[str] = Field(None, description="Expiration timestamp")
    created_at: str = Field(..., description="Creation timestamp")


# Authentication utilities
class AuthService:
    """Authentication service"""
    
    def __init__(self):
        self.settings = get_settings()
        self.secret_key = self.settings.security.secret_key
        self.algorithm = self.settings.security.algorithm
        self.access_token_expire_minutes = self.settings.security.access_token_expire_minutes
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = DateTimeHelpers.now_utc() + expires_delta
        else:
            expire = DateTimeHelpers.now_utc() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create refresh token"""
        to_encode = data.copy()
        expire = DateTimeHelpers.now_utc() + timedelta(days=30)  # Refresh tokens last 30 days
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def hash_password(self, password: str) -> str:
        """Hash password"""
        return HashHelpers.generate_hash(password, "sha256")
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password"""
        return HashHelpers.verify_hash(password, hashed_password, "sha256")


# Dependency injection
def get_auth_service() -> AuthService:
    """Get authentication service"""
    return AuthService()


# Authentication endpoints
@router.post(
    "/login",
    response_model=TokenResponse,
    summary="User Login",
    description="Authenticate user and return access tokens"
)
@rate_limit(max_calls=5, time_window=60)
@log_execution()
async def login(
    request: LoginRequest = Body(...),
    auth_service: AuthService = Depends(get_auth_service),
    http_request: Request = None
):
    """User login"""
    try:
        # Mock user authentication - in real implementation, this would check actual user database
        if request.username == "admin" and request.password == "admin123":
            user_data = {
                "user_id": "user_123",
                "username": "admin",
                "email": "admin@example.com",
                "full_name": "Administrator",
                "roles": ["admin", "user"],
                "permissions": ["read", "write", "admin"]
            }
        elif request.username == "user" and request.password == "user123":
            user_data = {
                "user_id": "user_456",
                "username": "user",
                "email": "user@example.com",
                "full_name": "Regular User",
                "roles": ["user"],
                "permissions": ["read", "write"]
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Create tokens
        access_token = auth_service.create_access_token(
            data={"sub": user_data["user_id"], "username": user_data["username"]}
        )
        refresh_token = auth_service.create_refresh_token(
            data={"sub": user_data["user_id"], "username": user_data["username"]}
        )
        
        # Log login
        logger.info(f"User {user_data['username']} logged in from {http_request.client.host if http_request else 'unknown'}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=auth_service.access_token_expire_minutes * 60,
            user_info=user_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="User Registration",
    description="Register new user account"
)
@rate_limit(max_calls=3, time_window=60)
@log_execution()
async def register(
    request: RegisterRequest = Body(...),
    auth_service: AuthService = Depends(get_auth_service)
):
    """User registration"""
    try:
        # Validate terms acceptance
        if not request.terms_accepted:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Terms and conditions must be accepted"
            )
        
        # Mock user registration - in real implementation, this would create actual user
        user_data = {
            "user_id": f"user_{DateTimeHelpers.now_utc().strftime('%Y%m%d_%H%M%S')}",
            "username": request.username,
            "email": request.email,
            "full_name": request.full_name,
            "roles": ["user"],
            "permissions": ["read", "write"]
        }
        
        # Create tokens
        access_token = auth_service.create_access_token(
            data={"sub": user_data["user_id"], "username": user_data["username"]}
        )
        refresh_token = auth_service.create_refresh_token(
            data={"sub": user_data["user_id"], "username": user_data["username"]}
        )
        
        # Log registration
        logger.info(f"New user registered: {user_data['username']} ({user_data['email']})")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=auth_service.access_token_expire_minutes * 60,
            user_info=user_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh Token",
    description="Refresh access token using refresh token"
)
@rate_limit(max_calls=10, time_window=60)
@log_execution()
async def refresh_token(
    request: RefreshTokenRequest = Body(...),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Refresh access token"""
    try:
        # Verify refresh token
        payload = auth_service.verify_token(request.refresh_token)
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        # Mock user data retrieval - in real implementation, this would get from database
        user_data = {
            "user_id": payload["sub"],
            "username": payload["username"],
            "email": "user@example.com",
            "full_name": "User",
            "roles": ["user"],
            "permissions": ["read", "write"]
        }
        
        # Create new access token
        access_token = auth_service.create_access_token(
            data={"sub": user_data["user_id"], "username": user_data["username"]}
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=request.refresh_token,  # Keep the same refresh token
            token_type="bearer",
            expires_in=auth_service.access_token_expire_minutes * 60,
            user_info=user_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="User Logout",
    description="Logout user and invalidate tokens"
)
@log_execution()
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
):
    """User logout"""
    try:
        # Verify token
        payload = auth_service.verify_token(credentials.credentials)
        
        # Log logout
        logger.info(f"User {payload.get('username')} logged out")
        
        # In real implementation, you would add the token to a blacklist
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get(
    "/profile",
    response_model=UserProfileResponse,
    summary="Get User Profile",
    description="Get current user profile information"
)
@log_execution()
async def get_profile(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Get user profile"""
    try:
        # Verify token
        payload = auth_service.verify_token(credentials.credentials)
        
        # Mock user profile - in real implementation, this would get from database
        profile = UserProfileResponse(
            user_id=payload["sub"],
            username=payload["username"],
            email="user@example.com",
            full_name="User Name",
            avatar_url=None,
            bio="User bio",
            timezone="UTC",
            language="en",
            roles=["user"],
            permissions=["read", "write"],
            is_active=True,
            is_verified=True,
            created_at="2024-01-01T00:00:00Z",
            last_login=DateTimeHelpers.now_utc().isoformat()
        )
        
        return profile
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.put(
    "/profile",
    response_model=UserProfileResponse,
    summary="Update User Profile",
    description="Update current user profile information"
)
@rate_limit(max_calls=10, time_window=60)
@log_execution()
async def update_profile(
    profile_data: Dict[str, Any] = Body(...),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Update user profile"""
    try:
        # Verify token
        payload = auth_service.verify_token(credentials.credentials)
        
        # Mock profile update - in real implementation, this would update database
        updated_profile = UserProfileResponse(
            user_id=payload["sub"],
            username=payload["username"],
            email=profile_data.get("email", "user@example.com"),
            full_name=profile_data.get("full_name", "User Name"),
            avatar_url=profile_data.get("avatar_url"),
            bio=profile_data.get("bio", "User bio"),
            timezone=profile_data.get("timezone", "UTC"),
            language=profile_data.get("language", "en"),
            roles=["user"],
            permissions=["read", "write"],
            is_active=True,
            is_verified=True,
            created_at="2024-01-01T00:00:00Z",
            last_login=DateTimeHelpers.now_utc().isoformat()
        )
        
        logger.info(f"User {payload['username']} updated profile")
        
        return updated_profile
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/change-password",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Change Password",
    description="Change user password"
)
@rate_limit(max_calls=3, time_window=60)
@log_execution()
async def change_password(
    request: ChangePasswordRequest = Body(...),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Change password"""
    try:
        # Verify token
        payload = auth_service.verify_token(credentials.credentials)
        
        # Validate new password confirmation
        if request.new_password != request.confirm_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password and confirmation do not match"
            )
        
        # Mock password change - in real implementation, this would update database
        logger.info(f"User {payload['username']} changed password")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to change password: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/reset-password",
    summary="Request Password Reset",
    description="Request password reset email"
)
@rate_limit(max_calls=3, time_window=60)
@log_execution()
async def request_password_reset(
    request: ResetPasswordRequest = Body(...)
):
    """Request password reset"""
    try:
        # Mock password reset request - in real implementation, this would send email
        reset_token = StringHelpers.generate_random_string(32)
        
        logger.info(f"Password reset requested for email: {request.email}")
        
        return {
            "message": "Password reset email sent",
            "email": request.email,
            "reset_token": reset_token  # In real implementation, this would be sent via email
        }
        
    except Exception as e:
        logger.error(f"Failed to request password reset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/confirm-reset-password",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Confirm Password Reset",
    description="Confirm password reset with token"
)
@rate_limit(max_calls=3, time_window=60)
@log_execution()
async def confirm_password_reset(
    request: ConfirmResetPasswordRequest = Body(...)
):
    """Confirm password reset"""
    try:
        # Validate password confirmation
        if request.new_password != request.confirm_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password and confirmation do not match"
            )
        
        # Mock password reset confirmation - in real implementation, this would update database
        logger.info(f"Password reset confirmed with token: {request.token}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to confirm password reset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/api-keys",
    response_model=APIKeyResponse,
    summary="Create API Key",
    description="Create new API key for user"
)
@rate_limit(max_calls=5, time_window=60)
@log_execution()
async def create_api_key(
    name: str = Body(..., description="API key name"),
    permissions: List[str] = Body(..., description="API key permissions"),
    expires_in_days: Optional[int] = Body(None, description="Expiration in days"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Create API key"""
    try:
        # Verify token
        payload = auth_service.verify_token(credentials.credentials)
        
        # Generate API key
        api_key = StringHelpers.generate_random_string(64)
        key_id = StringHelpers.generate_random_string(16)
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = (DateTimeHelpers.now_utc() + timedelta(days=expires_in_days)).isoformat()
        
        api_key_response = APIKeyResponse(
            api_key=api_key,
            key_id=key_id,
            name=name,
            permissions=permissions,
            expires_at=expires_at,
            created_at=DateTimeHelpers.now_utc().isoformat()
        )
        
        logger.info(f"API key created for user {payload['username']}: {key_id}")
        
        return api_key_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Health check endpoint
@router.get(
    "/health",
    summary="Auth Health Check",
    description="Check the health of the authentication service"
)
async def auth_health_check():
    """Auth health check"""
    return {
        "status": "healthy",
        "service": "auth-api-v3",
        "timestamp": DateTimeHelpers.now_utc().isoformat()
    }




