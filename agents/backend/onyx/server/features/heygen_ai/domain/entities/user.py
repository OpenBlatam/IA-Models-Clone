from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
from .base import BaseEntity, EntityID
from ..value_objects.email import Email
from ..exceptions.domain_errors import UserValidationError, BusinessRuleViolationError
        from ..events.user_events import UserEmailChangedEvent
        from ..events.user_events import UserActivatedEvent
        from ..events.user_events import UserDeactivatedEvent
        from ..events.user_events import UserSuspendedEvent
        from ..events.user_events import UserUnsuspendedEvent
        from ..events.user_events import UserUpgradedToPremiumEvent
        from ..events.user_events import UserDowngradedFromPremiumEvent
        from ..events.user_events import VideoCreditsConsumedEvent
        from ..events.user_events import VideoCreditsAddedEvent
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
User Entity

Represents a user in the HeyGen AI system with all business rules and constraints.
"""




class UserID(EntityID):
    """Strongly-typed User ID."""
    pass


class User(BaseEntity[UserID]):
    """
    User entity representing a user in the system.
    
    Business Rules:
    - Username must be unique and 3-50 characters
    - Email must be valid and unique
    - Password must meet security requirements
    - Users can be active, inactive, or suspended
    - Premium users have additional privileges
    """
    
    def __init__(
        self,
        id: Optional[UserID] = None,
        username: Optional[str] = None,
        email: Optional[Email] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        is_active: bool = True,
        is_premium: bool = False,
        is_suspended: bool = False,
        video_credits: int = 10,
        max_video_duration: int = 60,  # seconds
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ):
        
    """__init__ function."""
super().__init__(id, created_at, updated_at)
        
        # Validate inputs
        self._validate_username(username)
        self._validate_names(first_name, last_name)
        self._validate_credits(video_credits)
        self._validate_duration(max_video_duration)
        
        # Set attributes
        self._username = username
        self._email = email
        self._first_name = first_name
        self._last_name = last_name
        self._is_active = is_active
        self._is_premium = is_premium
        self._is_suspended = is_suspended
        self._video_credits = video_credits
        self._max_video_duration = max_video_duration
        
        # Track usage
        self._videos_created_today = 0
        self._last_video_created: Optional[datetime] = None
    
    # Properties
    @property
    def username(self) -> Optional[str]:
        """Get username."""
        return self._username
    
    @property
    def email(self) -> Optional[Email]:
        """Get email."""
        return self._email
    
    @property
    def first_name(self) -> Optional[str]:
        """Get first name."""
        return self._first_name
    
    @property
    def last_name(self) -> Optional[str]:
        """Get last name."""
        return self._last_name
    
    @property
    def full_name(self) -> str:
        """Get full name."""
        if self._first_name and self._last_name:
            return f"{self._first_name} {self._last_name}"
        return self._first_name or self._last_name or self._username or "Unknown"
    
    @property
    def is_active(self) -> bool:
        """Check if user is active."""
        return self._is_active and not self._is_suspended and not self.is_deleted
    
    @property
    def is_premium(self) -> bool:
        """Check if user has premium account."""
        return self._is_premium
    
    @property
    def is_suspended(self) -> bool:
        """Check if user is suspended."""
        return self._is_suspended
    
    @property
    def video_credits(self) -> int:
        """Get available video credits."""
        return self._video_credits
    
    @property
    def max_video_duration(self) -> int:
        """Get maximum video duration in seconds."""
        return self._max_video_duration
    
    @property
    def videos_created_today(self) -> int:
        """Get number of videos created today."""
        return self._videos_created_today
    
    # Business Methods
    def update_profile(
        self,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None
    ) -> None:
        """Update user profile information."""
        if not self.is_active:
            raise BusinessRuleViolationError("Cannot update profile for inactive user")
        
        if first_name is not None:
            self._validate_name(first_name, "First name")
            self._first_name = first_name
        
        if last_name is not None:
            self._validate_name(last_name, "Last name")
            self._last_name = last_name
        
        self.mark_updated()
    
    def change_email(self, new_email: Email) -> None:
        """Change user email address."""
        if not self.is_active:
            raise BusinessRuleViolationError("Cannot change email for inactive user")
        
        if self._email == new_email:
            return  # No change needed
        
        old_email = self._email
        self._email = new_email
        self.mark_updated()
        
        # Domain event for email change
        self.add_domain_event(UserEmailChangedEvent(
            user_id=self.id,
            old_email=old_email.value if old_email else None,
            new_email=new_email.value
        ))
    
    def activate(self) -> None:
        """Activate user account."""
        if self._is_active:
            return  # Already active
        
        self._is_active = True
        self.mark_updated()
        
        self.add_domain_event(UserActivatedEvent(user_id=self.id))
    
    def deactivate(self) -> None:
        """Deactivate user account."""
        if not self._is_active:
            return  # Already inactive
        
        self._is_active = False
        self.mark_updated()
        
        self.add_domain_event(UserDeactivatedEvent(user_id=self.id))
    
    def suspend(self, reason: str) -> None:
        """Suspend user account."""
        if self._is_suspended:
            return  # Already suspended
        
        self._is_suspended = True
        self.mark_updated()
        
        self.add_domain_event(UserSuspendedEvent(
            user_id=self.id,
            reason=reason
        ))
    
    def unsuspend(self) -> None:
        """Remove suspension from user account."""
        if not self._is_suspended:
            return  # Not suspended
        
        self._is_suspended = False
        self.mark_updated()
        
        self.add_domain_event(UserUnsuspendedEvent(user_id=self.id))
    
    def upgrade_to_premium(self) -> None:
        """Upgrade user to premium account."""
        if self._is_premium:
            return  # Already premium
        
        self._is_premium = True
        self._video_credits = 100  # Premium users get more credits
        self._max_video_duration = 300  # 5 minutes for premium
        self.mark_updated()
        
        self.add_domain_event(UserUpgradedToPremiumEvent(user_id=self.id))
    
    def downgrade_from_premium(self) -> None:
        """Downgrade user from premium account."""
        if not self._is_premium:
            return  # Not premium
        
        self._is_premium = False
        self._video_credits = min(self._video_credits, 10)  # Limit to free tier
        self._max_video_duration = 60  # 1 minute for free tier
        self.mark_updated()
        
        self.add_domain_event(UserDowngradedFromPremiumEvent(user_id=self.id))
    
    def can_create_video(self, duration: int) -> bool:
        """Check if user can create a video with specified duration."""
        if not self.is_active:
            return False
        
        if self._video_credits <= 0:
            return False
        
        if duration > self._max_video_duration:
            return False
        
        # Free users have daily limits
        if not self._is_premium and self._videos_created_today >= 3:
            return False
        
        return True
    
    def consume_video_credit(self, duration: int) -> None:
        """Consume video credit for creating a video."""
        if not self.can_create_video(duration):
            raise BusinessRuleViolationError("User cannot create video with specified duration")
        
        self._video_credits -= 1
        self._videos_created_today += 1
        self._last_video_created = self._utc_now()
        self.mark_updated()
        
        self.add_domain_event(VideoCreditsConsumedEvent(
            user_id=self.id,
            credits_consumed=1,
            remaining_credits=self._video_credits
        ))
    
    def add_video_credits(self, credits: int) -> None:
        """Add video credits to user account."""
        if credits <= 0:
            raise UserValidationError("Credits must be positive")
        
        self._video_credits += credits
        self.mark_updated()
        
        self.add_domain_event(VideoCreditsAddedEvent(
            user_id=self.id,
            credits_added=credits,
            total_credits=self._video_credits
        ))
    
    def reset_daily_limits(self) -> None:
        """Reset daily video creation limits."""
        self._videos_created_today = 0
        self.mark_updated()
    
    # Validation Methods
    def _validate_username(self, username: Optional[str]) -> None:
        """Validate username."""
        if username is None:
            return
        
        if not isinstance(username, str):
            raise UserValidationError("Username must be a string")
        
        if len(username.strip()) < 3:
            raise UserValidationError("Username must be at least 3 characters")
        
        if len(username.strip()) > 50:
            raise UserValidationError("Username cannot exceed 50 characters")
        
        # Check for invalid characters
        if not username.replace("_", "").replace("-", "").replace(".", "").isalnum():
            raise UserValidationError("Username can only contain letters, numbers, underscore, hyphen, and period")
    
    def _validate_names(self, first_name: Optional[str], last_name: Optional[str]) -> None:
        """Validate first and last names."""
        if first_name is not None:
            self._validate_name(first_name, "First name")
        
        if last_name is not None:
            self._validate_name(last_name, "Last name")
    
    def _validate_name(self, name: str, field_name: str) -> None:
        """Validate a name field."""
        if not isinstance(name, str):
            raise UserValidationError(f"{field_name} must be a string")
        
        if len(name.strip()) == 0:
            raise UserValidationError(f"{field_name} cannot be empty")
        
        if len(name.strip()) > 100:
            raise UserValidationError(f"{field_name} cannot exceed 100 characters")
    
    def _validate_credits(self, credits: int) -> None:
        """Validate video credits."""
        if not isinstance(credits, int):
            raise UserValidationError("Video credits must be an integer")
        
        if credits < 0:
            raise UserValidationError("Video credits cannot be negative")
        
        if credits > 10000:
            raise UserValidationError("Video credits cannot exceed 10,000")
    
    def _validate_duration(self, duration: int) -> None:
        """Validate max video duration."""
        if not isinstance(duration, int):
            raise UserValidationError("Max video duration must be an integer")
        
        if duration <= 0:
            raise UserValidationError("Max video duration must be positive")
        
        if duration > 3600:  # 1 hour max
            raise UserValidationError("Max video duration cannot exceed 1 hour")
    
    def _generate_id(self) -> UserID:
        """Generate a new user ID."""
        return UserID(uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            "username": self._username,
            "email": self._email.value if self._email else None,
            "first_name": self._first_name,
            "last_name": self._last_name,
            "full_name": self.full_name,
            "is_active": self._is_active,
            "is_premium": self._is_premium,
            "is_suspended": self._is_suspended,
            "video_credits": self._video_credits,
            "max_video_duration": self._max_video_duration,
            "videos_created_today": self._videos_created_today,
            "last_video_created": self._last_video_created.isoformat() if self._last_video_created else None
        })
        return base_dict 