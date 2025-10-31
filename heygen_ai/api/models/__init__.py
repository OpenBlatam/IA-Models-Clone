from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from enum import Enum
import structlog
    import re
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
HeyGen AI FastAPI Data Models Package
FastAPI best practices for data models with Pydantic validation and serialization.
"""


logger = structlog.get_logger()

# =============================================================================
# Model Categories
# =============================================================================

class ModelCategory(Enum):
    """Model category enumeration following FastAPI best practices."""
    REQUEST = "request"
    RESPONSE = "response"
    DATABASE = "database"
    INTERNAL = "internal"
    VALIDATION = "validation"

# =============================================================================
# Base Models
# =============================================================================

class BaseModel:
    """Base model class following FastAPI best practices."""
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        # Use ORM mode for database models
        orm_mode = True
        
        # Allow extra fields (useful for API responses)
        extra = "allow"
        
        # Validate assignment
        validate_assignment = True
        
        # Use enum values
        use_enum_values = True
        
        # Allow population by field name
        allow_population_by_field_name = True
        
        # JSON encoders for custom types
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

class TimestampedModel(BaseModel):
    """Base model with timestamp fields following FastAPI best practices."""
    
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        """Pydantic configuration for timestamped models."""
        orm_mode = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

class IdentifiedModel(TimestampedModel):
    """Base model with ID field following FastAPI best practices."""
    
    id: Optional[int] = None
    
    class Config:
        """Pydantic configuration for identified models."""
        orm_mode = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

# =============================================================================
# Common Field Types
# =============================================================================

class StatusEnum(str, Enum):
    """Status enumeration following FastAPI best practices."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DELETED = "deleted"

class RoleEnum(str, Enum):
    """Role enumeration following FastAPI best practices."""
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"
    PREMIUM = "premium"

class VideoTypeEnum(str, Enum):
    """Video type enumeration following FastAPI best practices."""
    AI_GENERATED = "ai_generated"
    UPLOADED = "uploaded"
    EDITED = "edited"
    COMPOSITE = "composite"

class ProcessingStatusEnum(str, Enum):
    """Processing status enumeration following FastAPI best practices."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# =============================================================================
# Validation Helpers
# =============================================================================

def validate_email(email: str) -> str:
    """Email validation following FastAPI best practices."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValueError('Invalid email format')
    return email.lower()

def validate_password(password: str) -> str:
    """Password validation following FastAPI best practices."""
    if len(password) < 8:
        raise ValueError('Password must be at least 8 characters long')
    if not any(c.isupper() for c in password):
        raise ValueError('Password must contain at least one uppercase letter')
    if not any(c.islower() for c in password):
        raise ValueError('Password must contain at least one lowercase letter')
    if not any(c.isdigit() for c in password):
        raise ValueError('Password must contain at least one digit')
    return password

def validate_file_size(size_bytes: int, max_size_mb: int = 100) -> int:
    """File size validation following FastAPI best practices."""
    max_size_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_size_bytes:
        raise ValueError(f'File size must be less than {max_size_mb}MB')
    return size_bytes

def validate_video_duration(duration_seconds: float, max_duration: int = 3600) -> float:
    """Video duration validation following FastAPI best practices."""
    if duration_seconds > max_duration:
        raise ValueError(f'Video duration must be less than {max_duration} seconds')
    if duration_seconds < 0:
        raise ValueError('Video duration must be positive')
    return duration_seconds

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "ModelCategory",
    "BaseModel",
    "TimestampedModel", 
    "IdentifiedModel",
    "StatusEnum",
    "RoleEnum",
    "VideoTypeEnum",
    "ProcessingStatusEnum",
    "validate_email",
    "validate_password",
    "validate_file_size",
    "validate_video_duration"
] 