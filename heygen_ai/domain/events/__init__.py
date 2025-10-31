from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .base import DomainEvent
from .user_events import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Domain Events Package

Contains domain events that represent significant business occurrences.
"""

    UserEmailChangedEvent,
    UserActivatedEvent,
    UserDeactivatedEvent,
    UserSuspendedEvent,
    UserUnsuspendedEvent,
    UserUpgradedToPremiumEvent,
    UserDowngradedFromPremiumEvent,
    VideoCreditsConsumedEvent,
    VideoCreditsAddedEvent
)

__all__ = [
    "DomainEvent",
    "UserEmailChangedEvent",
    "UserActivatedEvent",
    "UserDeactivatedEvent", 
    "UserSuspendedEvent",
    "UserUnsuspendedEvent",
    "UserUpgradedToPremiumEvent",
    "UserDowngradedFromPremiumEvent",
    "VideoCreditsConsumedEvent",
    "VideoCreditsAddedEvent",
] 