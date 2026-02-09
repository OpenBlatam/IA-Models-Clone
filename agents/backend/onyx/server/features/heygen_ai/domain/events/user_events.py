from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from typing import Optional
from uuid import UUID
from .base import DomainEvent
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
User Domain Events

Events related to user entity changes and actions.
"""




@dataclass(frozen=True)
class UserEmailChangedEvent(DomainEvent):
    """Event raised when user email is changed."""
    user_id: UUID
    old_email: Optional[str]
    new_email: str


@dataclass(frozen=True)
class UserActivatedEvent(DomainEvent):
    """Event raised when user account is activated."""
    user_id: UUID


@dataclass(frozen=True)
class UserDeactivatedEvent(DomainEvent):
    """Event raised when user account is deactivated."""
    user_id: UUID


@dataclass(frozen=True)
class UserSuspendedEvent(DomainEvent):
    """Event raised when user account is suspended."""
    user_id: UUID
    reason: str


@dataclass(frozen=True)
class UserUnsuspendedEvent(DomainEvent):
    """Event raised when user suspension is removed."""
    user_id: UUID


@dataclass(frozen=True)
class UserUpgradedToPremiumEvent(DomainEvent):
    """Event raised when user is upgraded to premium."""
    user_id: UUID


@dataclass(frozen=True)
class UserDowngradedFromPremiumEvent(DomainEvent):
    """Event raised when user is downgraded from premium."""
    user_id: UUID


@dataclass(frozen=True)
class VideoCreditsConsumedEvent(DomainEvent):
    """Event raised when user consumes video credits."""
    user_id: UUID
    credits_consumed: int
    remaining_credits: int


@dataclass(frozen=True)
class VideoCreditsAddedEvent(DomainEvent):
    """Event raised when video credits are added to user account."""
    user_id: UUID
    credits_added: int
    total_credits: int 