from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID
from pydantic import Field, field_validator
from .base import AggregateRoot
        import re
        import re
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
User Entity
==========

User entity representing system users with permissions and preferences.
"""





class UserRole(str, Enum):
    """User roles."""
    USER = "user"
    PREMIUM = "premium"
    CREATOR = "creator"
    ADMIN = "admin"
    MODERATOR = "moderator"


class UserStatus(str, Enum):
    """User status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    BANNED = "banned"


class User(AggregateRoot):
    """
    User entity for the AI Video system.
    
    Users represent system users with their preferences, permissions,
    and usage statistics.
    """
    
    # Basic information
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: str = Field(..., description="User email address")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    
    # Authentication and security
    hashed_password: Optional[str] = Field(None, description="Hashed password")
    is_verified: bool = Field(default=False, description="Email verification status")
    two_factor_enabled: bool = Field(default=False, description="2FA enabled")
    
    # Role and status
    role: UserRole = Field(default=UserRole.USER, description="User role")
    status: UserStatus = Field(default=UserStatus.ACTIVE, description="User status")
    
    # Preferences
    language: str = Field(default="es", description="Preferred language")
    timezone: str = Field(default="UTC", description="User timezone")
    notification_preferences: Dict = Field(
        default={
            "email": True,
            "push": True,
            "sms": False,
        },
        description="Notification preferences"
    )
    
    # Usage statistics
    videos_created: int = Field(default=0, description="Number of videos created")
    total_video_duration: float = Field(default=0.0, description="Total video duration created")
    storage_used: int = Field(default=0, description="Storage used in bytes")
    storage_limit: int = Field(default=1073741824, description="Storage limit in bytes")  # 1GB
    
    # Subscription and billing
    subscription_plan: Optional[str] = Field(None, description="Subscription plan")
    subscription_expires: Optional[datetime] = Field(None, description="Subscription expiration")
    billing_info: Optional[Dict] = Field(None, description="Billing information")
    
    # Profile and settings
    avatar_url: Optional[str] = Field(None, description="Profile avatar URL")
    bio: Optional[str] = Field(None, max_length=500, description="User bio")
    website: Optional[str] = Field(None, description="Website URL")
    social_links: Dict = Field(
        default={},
        description="Social media links"
    )
    
    # Security and audit
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    login_attempts: int = Field(default=0, description="Failed login attempts")
    locked_until: Optional[datetime] = Field(None, description="Account lock until")
    
    # Permissions and access
    permissions: List[str] = Field(
        default=[],
        description="User permissions"
    )
    feature_flags: Dict = Field(
        default={},
        description="Feature flags for user"
    )
    
    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username."""
        if not v.strip():
            raise ValueError("Username cannot be empty")
        
        # Check for valid characters
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Username can only contain letters, numbers, underscores, and hyphens")
        
        return v.strip().lower()
    
    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate email address."""
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, v):
            raise ValueError("Invalid email address format")
        return v.lower().strip()
    
    def _validate_entity(self) -> None:
        """Validate user business rules."""
        if self.status == UserStatus.BANNED and self.role == UserRole.ADMIN:
            raise ValueError("Admin users cannot be banned")
        
        if self.storage_used > self.storage_limit:
            raise ValueError("Storage usage exceeds limit")
    
    def update_last_login(self) -> None:
        """Update last login timestamp."""
        self.last_login = datetime.utcnow()
        self.login_attempts = 0
        self.mark_as_dirty()
    
    def increment_login_attempts(self) -> None:
        """Increment failed login attempts."""
        self.login_attempts += 1
        
        # Lock account after 5 failed attempts
        if self.login_attempts >= 5:
            self.locked_until = datetime.utcnow() + datetime.timedelta(minutes=30)
        
        self.mark_as_dirty()
    
    def unlock_account(self) -> None:
        """Unlock user account."""
        self.login_attempts = 0
        self.locked_until = None
        self.mark_as_dirty()
    
    def is_account_locked(self) -> bool:
        """Check if account is locked."""
        if not self.locked_until:
            return False
        return datetime.utcnow() < self.locked_until
    
    def upgrade_role(self, new_role: UserRole) -> None:
        """Upgrade user role."""
        role_hierarchy = {
            UserRole.USER: 1,
            UserRole.PREMIUM: 2,
            UserRole.CREATOR: 3,
            UserRole.MODERATOR: 4,
            UserRole.ADMIN: 5,
        }
        
        current_level = role_hierarchy.get(self.role, 0)
        new_level = role_hierarchy.get(new_role, 0)
        
        if new_level <= current_level:
            raise ValueError("Cannot downgrade user role")
        
        self.role = new_role
        self.mark_as_dirty()
    
    def downgrade_role(self, new_role: UserRole) -> None:
        """Downgrade user role."""
        role_hierarchy = {
            UserRole.USER: 1,
            UserRole.PREMIUM: 2,
            UserRole.CREATOR: 3,
            UserRole.MODERATOR: 4,
            UserRole.ADMIN: 5,
        }
        
        current_level = role_hierarchy.get(self.role, 0)
        new_level = role_hierarchy.get(new_role, 0)
        
        if new_level >= current_level:
            raise ValueError("Cannot upgrade user role with downgrade method")
        
        self.role = new_role
        self.mark_as_dirty()
    
    def suspend_user(self, reason: str) -> None:
        """Suspend user account."""
        self.status = UserStatus.SUSPENDED
        self.feature_flags["suspension_reason"] = reason
        self.mark_as_dirty()
    
    def ban_user(self, reason: str) -> None:
        """Ban user account."""
        if self.role == UserRole.ADMIN:
            raise ValueError("Cannot ban admin users")
        
        self.status = UserStatus.BANNED
        self.feature_flags["ban_reason"] = reason
        self.mark_as_dirty()
    
    def activate_user(self) -> None:
        """Activate user account."""
        self.status = UserStatus.ACTIVE
        self.mark_as_dirty()
    
    def increment_video_creation(self, duration: float) -> None:
        """Increment video creation statistics."""
        self.videos_created += 1
        self.total_video_duration += duration
        self.mark_as_dirty()
    
    def update_storage_usage(self, bytes_used: int) -> None:
        """Update storage usage."""
        self.storage_used = max(0, self.storage_used + bytes_used)
        
        if self.storage_used > self.storage_limit:
            raise ValueError("Storage usage would exceed limit")
        
        self.mark_as_dirty()
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions
    
    def add_permission(self, permission: str) -> None:
        """Add permission to user."""
        if permission not in self.permissions:
            self.permissions.append(permission)
            self.mark_as_dirty()
    
    def remove_permission(self, permission: str) -> None:
        """Remove permission from user."""
        if permission in self.permissions:
            self.permissions.remove(permission)
            self.mark_as_dirty()
    
    def is_premium_user(self) -> bool:
        """Check if user has premium access."""
        return (
            self.role in [UserRole.PREMIUM, UserRole.CREATOR, UserRole.ADMIN, UserRole.MODERATOR] or
            (self.subscription_plan and self.subscription_expires and 
             datetime.utcnow() < self.subscription_expires)
        )
    
    def get_storage_usage_percentage(self) -> float:
        """Get storage usage as percentage."""
        if self.storage_limit == 0:
            return 0.0
        return (self.storage_used / self.storage_limit) * 100
    
    def can_create_video(self) -> bool:
        """Check if user can create videos."""
        if self.status != UserStatus.ACTIVE:
            return False
        
        if self.is_account_locked():
            return False
        
        return True
    
    def to_summary_dict(self) -> Dict:
        """Convert to summary dictionary for listings."""
        return {
            "id": str(self.id),
            "username": self.username,
            "full_name": self.full_name,
            "role": self.role.value,
            "status": self.status.value,
            "videos_created": self.videos_created,
            "is_premium": self.is_premium_user(),
            "avatar_url": self.avatar_url,
        }
    
    def get_usage_statistics(self) -> Dict:
        """Get user usage statistics."""
        return {
            "videos_created": self.videos_created,
            "total_video_duration": self.total_video_duration,
            "storage_used": self.storage_used,
            "storage_limit": self.storage_limit,
            "storage_usage_percentage": self.get_storage_usage_percentage(),
            "subscription_plan": self.subscription_plan,
            "subscription_expires": self.subscription_expires.isoformat() if self.subscription_expires else None,
        } 