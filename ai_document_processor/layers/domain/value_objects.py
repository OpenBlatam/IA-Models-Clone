"""
Value Objects - Immutable Domain Values
=====================================

Value objects representing domain concepts that are defined by their attributes.
"""

import uuid
from datetime import datetime
from typing import Any, Optional
from dataclasses import dataclass
from enum import Enum


class DocumentType(str, Enum):
    """Document type enumeration."""
    MARKDOWN = "markdown"
    PDF = "pdf"
    WORD = "word"
    TEXT = "text"
    HTML = "html"
    XML = "xml"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class DocumentId:
    """Document ID value object."""
    value: str
    
    def __post_init__(self):
        """Validate document ID."""
        if not self.value or not self.value.strip():
            raise ValueError("Document ID cannot be empty")
    
    @classmethod
    def generate(cls) -> 'DocumentId':
        """Generate a new document ID."""
        return cls(str(uuid.uuid4()))
    
    def __str__(self) -> str:
        """String representation."""
        return self.value
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, DocumentId):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash(self.value)


@dataclass(frozen=True)
class UserId:
    """User ID value object."""
    value: str
    
    def __post_init__(self):
        """Validate user ID."""
        if not self.value or not self.value.strip():
            raise ValueError("User ID cannot be empty")
    
    @classmethod
    def generate(cls) -> 'UserId':
        """Generate a new user ID."""
        return cls(str(uuid.uuid4()))
    
    def __str__(self) -> str:
        """String representation."""
        return self.value
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, UserId):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash(self.value)


@dataclass(frozen=True)
class ProcessingId:
    """Processing ID value object."""
    value: str
    
    def __post_init__(self):
        """Validate processing ID."""
        if not self.value or not self.value.strip():
            raise ValueError("Processing ID cannot be empty")
    
    @classmethod
    def generate(cls) -> 'ProcessingId':
        """Generate a new processing ID."""
        return cls(str(uuid.uuid4()))
    
    def __str__(self) -> str:
        """String representation."""
        return self.value
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, ProcessingId):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash(self.value)


@dataclass(frozen=True)
class Email:
    """Email value object."""
    value: str
    
    def __post_init__(self):
        """Validate email format."""
        if not self.value or not self.value.strip():
            raise ValueError("Email cannot be empty")
        
        # Basic email validation
        if "@" not in self.value or "." not in self.value.split("@")[1]:
            raise ValueError("Invalid email format")
    
    def __str__(self) -> str:
        """String representation."""
        return self.value
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, Email):
            return False
        return self.value.lower() == other.value.lower()
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash(self.value.lower())


@dataclass(frozen=True)
class DocumentTitle:
    """Document title value object."""
    value: str
    
    def __post_init__(self):
        """Validate document title."""
        if not self.value or not self.value.strip():
            raise ValueError("Document title cannot be empty")
        
        if len(self.value) > 255:
            raise ValueError("Document title cannot exceed 255 characters")
    
    def __str__(self) -> str:
        """String representation."""
        return self.value
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, DocumentTitle):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash(self.value)


@dataclass(frozen=True)
class DocumentContent:
    """Document content value object."""
    value: str
    
    def __post_init__(self):
        """Validate document content."""
        if not self.value or not self.value.strip():
            raise ValueError("Document content cannot be empty")
        
        if len(self.value) > 10_000_000:  # 10MB limit
            raise ValueError("Document content cannot exceed 10MB")
    
    def get_word_count(self) -> int:
        """Get word count."""
        return len(self.value.split())
    
    def get_character_count(self) -> int:
        """Get character count."""
        return len(self.value)
    
    def get_line_count(self) -> int:
        """Get line count."""
        return len(self.value.splitlines())
    
    def __str__(self) -> str:
        """String representation."""
        return self.value
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, DocumentContent):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash(self.value)


@dataclass(frozen=True)
class ProcessingTime:
    """Processing time value object."""
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate processing time."""
        if self.completed_at and self.completed_at < self.started_at:
            raise ValueError("Completed time cannot be before started time")
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_completed(self) -> bool:
        """Check if processing is completed."""
        return self.completed_at is not None
    
    @property
    def is_in_progress(self) -> bool:
        """Check if processing is in progress."""
        return self.completed_at is None
    
    def mark_completed(self) -> 'ProcessingTime':
        """Mark processing as completed."""
        return ProcessingTime(
            started_at=self.started_at,
            completed_at=datetime.utcnow()
        )


@dataclass(frozen=True)
class DocumentMetadata:
    """Document metadata value object."""
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    language: Optional[str] = None
    encoding: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    character_count: Optional[int] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    custom_fields: dict = None
    
    def __post_init__(self):
        """Initialize custom fields."""
        if self.custom_fields is None:
            object.__setattr__(self, 'custom_fields', {})
    
    def get_custom_field(self, key: str, default: Any = None) -> Any:
        """Get custom field value."""
        return self.custom_fields.get(key, default)
    
    def has_custom_field(self, key: str) -> bool:
        """Check if custom field exists."""
        return key in self.custom_fields
    
    def get_all_custom_fields(self) -> dict:
        """Get all custom fields."""
        return self.custom_fields.copy()


@dataclass(frozen=True)
class ProcessingConfiguration:
    """Processing configuration value object."""
    enable_ai_classification: bool = True
    enable_ai_transformation: bool = True
    enable_validation: bool = True
    max_file_size_mb: int = 100
    timeout_seconds: int = 300
    retry_attempts: int = 3
    custom_settings: dict = None
    
    def __post_init__(self):
        """Initialize custom settings and validate."""
        if self.custom_settings is None:
            object.__setattr__(self, 'custom_settings', {})
        
        if self.max_file_size_mb <= 0:
            raise ValueError("Max file size must be positive")
        
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        
        if self.retry_attempts < 0:
            raise ValueError("Retry attempts cannot be negative")
    
    def get_custom_setting(self, key: str, default: Any = None) -> Any:
        """Get custom setting value."""
        return self.custom_settings.get(key, default)
    
    def has_custom_setting(self, key: str) -> bool:
        """Check if custom setting exists."""
        return key in self.custom_settings
    
    def get_all_custom_settings(self) -> dict:
        """Get all custom settings."""
        return self.custom_settings.copy()


@dataclass(frozen=True)
class DocumentTag:
    """Document tag value object."""
    value: str
    
    def __post_init__(self):
        """Validate document tag."""
        if not self.value or not self.value.strip():
            raise ValueError("Document tag cannot be empty")
        
        if len(self.value) > 50:
            raise ValueError("Document tag cannot exceed 50 characters")
        
        # Normalize tag (lowercase, no spaces)
        object.__setattr__(self, 'value', self.value.strip().lower().replace(' ', '-'))
    
    def __str__(self) -> str:
        """String representation."""
        return self.value
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, DocumentTag):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash(self.value)


@dataclass(frozen=True)
class ProcessingError:
    """Processing error value object."""
    code: str
    message: str
    details: Optional[dict] = None
    occurred_at: datetime = None
    
    def __post_init__(self):
        """Initialize occurred_at if not provided."""
        if self.occurred_at is None:
            object.__setattr__(self, 'occurred_at', datetime.utcnow())
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.code}: {self.message}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'code': self.code,
            'message': self.message,
            'details': self.details,
            'occurred_at': self.occurred_at.isoformat()
        }

















