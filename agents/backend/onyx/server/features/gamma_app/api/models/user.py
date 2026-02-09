"""
User Models
User-related Pydantic models
"""

from datetime import datetime
from typing import Dict, Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, EmailStr, field_validator

from .enums import UserRole
from .validators import sanitize_string_field

class User(BaseModel):
    """User model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    email: EmailStr = Field(..., description="User email address")
    name: str
    role: UserRole = UserRole.USER
    created_at: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    subscription_plan: str = "free"
    is_active: bool = True

class UserCreate(BaseModel):
    """User creation model"""
    email: EmailStr = Field(..., description="User email address")
    name: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=8, max_length=128)
    role: UserRole = UserRole.USER

    @field_validator('name')
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        """Sanitize name field"""
        return sanitize_string_field(v)

class UserUpdate(BaseModel):
    """User update model"""
    name: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    subscription_plan: Optional[str] = None







