"""
Pydantic schemas for users and authentication endpoints
"""

from typing import Optional
from pydantic import BaseModel, Field, EmailStr, validator
from datetime import datetime


class CreateUserRequest(BaseModel):
    """Request schema for creating user"""
    user_id: str = Field(..., description="User ID")
    email: Optional[EmailStr] = Field(default=None, description="User email")
    name: Optional[str] = Field(default=None, description="User name")
    password: Optional[str] = Field(default=None, min_length=8, description="User password")


class UserResponse(BaseModel):
    """Response schema for user"""
    user_id: str = Field(..., description="User ID")
    email: Optional[str] = Field(default=None, description="User email")
    name: Optional[str] = Field(default=None, description="User name")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")


class RegisterRequest(BaseModel):
    """Request schema for user registration"""
    user_id: str = Field(..., description="User ID")
    email: Optional[EmailStr] = Field(default=None, description="User email")
    password: Optional[str] = Field(default=None, min_length=8, description="User password")
    name: Optional[str] = Field(default=None, description="User name")

    @validator('password')
    def validate_password(cls, v):
        """Validate password strength"""
        if v and len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class RegisterResponse(BaseModel):
    """Response schema for registration"""
    user_id: str = Field(..., description="User ID")
    email: Optional[str] = Field(default=None, description="User email")
    access_token: str = Field(..., description="Access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: Optional[int] = Field(default=3600, description="Token expiration in seconds")


class LoginRequest(BaseModel):
    """Request schema for login"""
    user_id: str = Field(..., description="User ID")
    password: Optional[str] = Field(default=None, description="User password")


class LoginResponse(BaseModel):
    """Response schema for login"""
    access_token: str = Field(..., description="Access token")
    token_type: str = Field(default="bearer", description="Token type")
    user_id: str = Field(..., description="User ID")
    expires_in: Optional[int] = Field(default=3600, description="Token expiration in seconds")

