from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
User domain entity for LinkedIn Posts system.
"""



class UserRole(str, Enum):
    """User role enumeration."""
    ADMIN = "admin"
    USER = "user"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class UserStatus(str, Enum):
    """User status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


@dataclass
class UserPreferences:
    """User preferences for post generation."""
    default_tone: str = "professional"
    preferred_hashtags: List[str] = field(default_factory=list)
    industry: Optional[str] = None
    target_audience: Optional[str] = None
    posting_frequency: Optional[str] = None
    ai_enhancement: bool = True
    auto_schedule: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_tone": self.default_tone,
            "preferred_hashtags": self.preferred_hashtags,
            "industry": self.industry,
            "target_audience": self.target_audience,
            "posting_frequency": self.posting_frequency,
            "ai_enhancement": self.ai_enhancement,
            "auto_schedule": self.auto_schedule
        }


@dataclass
class UserStats:
    """User statistics."""
    total_posts: int = 0
    published_posts: int = 0
    total_engagement: int = 0
    average_engagement_rate: float = 0.0
    followers_count: int = 0
    last_post_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_posts": self.total_posts,
            "published_posts": self.published_posts,
            "total_engagement": self.total_engagement,
            "average_engagement_rate": self.average_engagement_rate,
            "followers_count": self.followers_count,
            "last_post_date": self.last_post_date.isoformat() if self.last_post_date else None
        }


@dataclass
class User:
    """
    User domain entity.
    
    Features:
    - Role-based access control
    - User preferences
    - Statistics tracking
    - LinkedIn integration
    """
    
    # Core fields
    id: UUID = field(default_factory=uuid4)
    email: str = ""
    username: str = ""
    full_name: str = ""
    
    # Status and role
    role: UserRole = UserRole.USER
    status: UserStatus = UserStatus.ACTIVE
    
    # Preferences and stats
    preferences: UserPreferences = field(default_factory=UserPreferences)
    stats: UserStats = field(default_factory=UserStats)
    
    # LinkedIn integration
    linkedin_id: Optional[str] = None
    linkedin_access_token: Optional[str] = None
    linkedin_profile_url: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_login_at: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> Any:
        """Post-initialization processing."""
        if isinstance(self.preferences, dict):
            self.preferences = UserPreferences(**self.preferences)
        if isinstance(self.stats, dict):
            self.stats = UserStats(**self.stats)
    
    @property
    def is_active(self) -> bool:
        """Check if user is active."""
        return self.status == UserStatus.ACTIVE
    
    @property
    def is_premium(self) -> bool:
        """Check if user has premium access."""
        return self.role in [UserRole.PREMIUM, UserRole.ENTERPRISE]
    
    @property
    def is_admin(self) -> bool:
        """Check if user is admin."""
        return self.role == UserRole.ADMIN
    
    def update_stats(self, **stats) -> None:
        """Update user statistics."""
        for key, value in stats.items():
            if hasattr(self.stats, key):
                setattr(self.stats, key, value)
    
    def update_preferences(self, **prefs) -> None:
        """Update user preferences."""
        for key, value in prefs.items():
            if hasattr(self.preferences, key):
                setattr(self.preferences, key, value)
    
    def record_login(self) -> None:
        """Record user login."""
        self.last_login_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "email": self.email,
            "username": self.username,
            "full_name": self.full_name,
            "role": self.role.value,
            "status": self.status.value,
            "preferences": self.preferences.to_dict(),
            "stats": self.stats.to_dict(),
            "linkedin_id": self.linkedin_id,
            "linkedin_profile_url": self.linkedin_profile_url,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create from dictionary."""
        # Convert string ID to UUID
        if 'id' in data and isinstance(data['id'], str):
            data['id'] = UUID(data['id'])
        
        # Convert string dates to datetime
        for date_field in ['created_at', 'updated_at', 'last_login_at']:
            if date_field in data and data[date_field]:
                if isinstance(data[date_field], str):
                    data[date_field] = datetime.fromisoformat(data[date_field])
        
        # Convert enums
        if 'role' in data and isinstance(data['role'], str):
            data['role'] = UserRole(data['role'])
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = UserStatus(data['status'])
        
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation."""
        return f"User(id={self.id}, email='{self.email}', role={self.role.value})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"User(id={self.id}, email='{self.email}', role={self.role.value}, status={self.status.value})" 