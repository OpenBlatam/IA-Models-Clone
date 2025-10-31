from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .entities import *
from .value_objects import *
from .exceptions import *
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Domain Layer - Core Business Logic

This layer contains the core business logic and rules of the HeyGen AI application.
It is independent of any infrastructure concerns and defines the business entities,
value objects, and domain services.

Components:
- entities/: Business entities (User, Video, Avatar, etc.)
- value_objects/: Value objects (Email, VideoQuality, etc.)
- repositories/: Repository interfaces (contracts)
- services/: Domain services (business logic)
- exceptions/: Domain-specific exceptions
"""


__all__ = [
    # Will be populated as we add entities and value objects
] 