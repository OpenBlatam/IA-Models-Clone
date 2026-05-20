from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from ..exceptions.domain_errors import ValueObjectValidationError
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Processing Status Value Object

Represents the status of video processing operations.
"""




class ProcessingStatusType(Enum):
    """Processing status types."""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ProcessingStatus:
    """
    Processing status value object.
    
    Tracks the status of video processing operations with progress and error information.
    """
    
    status: ProcessingStatusType
    progress_percentage: int = 0
    message: Optional[str] = None
    error_details: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self) -> None:
        """Validate processing status after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate processing status parameters."""
        # Validate status type
        if not isinstance(self.status, ProcessingStatusType):
            raise ValueObjectValidationError("Status must be a ProcessingStatusType enum")
        
        # Validate progress percentage
        if not isinstance(self.progress_percentage, int):
            raise ValueObjectValidationError("Progress percentage must be an integer")
        
        if self.progress_percentage < 0 or self.progress_percentage > 100:
            raise ValueObjectValidationError("Progress percentage must be between 0 and 100")
        
        # Validate message length
        if self.message is not None and len(self.message) > 500:
            raise ValueObjectValidationError("Message cannot exceed 500 characters")
        
        # Validate error details length
        if self.error_details is not None and len(self.error_details) > 2000:
            raise ValueObjectValidationError("Error details cannot exceed 2000 characters")
        
        # Validate status-specific constraints
        self._validate_status_constraints()
        
        # Validate timestamps
        self._validate_timestamps()
    
    def _validate_status_constraints(self) -> None:
        """Validate constraints based on status type."""
        if self.status == ProcessingStatusType.PENDING:
            if self.progress_percentage != 0:
                raise ValueObjectValidationError("Pending status must have 0% progress")
            if self.error_details is not None:
                raise ValueObjectValidationError("Pending status cannot have error details")
        
        elif self.status == ProcessingStatusType.COMPLETED:
            if self.progress_percentage != 100:
                raise ValueObjectValidationError("Completed status must have 100% progress")
            if self.error_details is not None:
                raise ValueObjectValidationError("Completed status cannot have error details")
        
        elif self.status == ProcessingStatusType.FAILED:
            if self.progress_percentage == 100:
                raise ValueObjectValidationError("Failed status cannot have 100% progress")
            # Failed status should have error details, but it's not strictly required
        
        elif self.status == ProcessingStatusType.CANCELLED:
            if self.progress_percentage == 100:
                raise ValueObjectValidationError("Cancelled status cannot have 100% progress")
    
    def _validate_timestamps(self) -> None:
        """Validate timestamp constraints."""
        if self.started_at and self.completed_at:
            if self.completed_at < self.started_at:
                raise ValueObjectValidationError("Completion time cannot be before start time")
        
        # Only completed, failed, or cancelled statuses should have completion time
        if self.completed_at and self.status not in [
            ProcessingStatusType.COMPLETED,
            ProcessingStatusType.FAILED,
            ProcessingStatusType.CANCELLED
        ]:
            raise ValueObjectValidationError(
                f"Status {self.status.value} cannot have completion time"
            )
    
    @property
    def is_terminal(self) -> bool:
        """Check if status is terminal (no further processing)."""
        return self.status in [
            ProcessingStatusType.COMPLETED,
            ProcessingStatusType.FAILED,
            ProcessingStatusType.CANCELLED
        ]
    
    @property
    def is_active(self) -> bool:
        """Check if processing is currently active."""
        return self.status in [
            ProcessingStatusType.QUEUED,
            ProcessingStatusType.PROCESSING
        ]
    
    @property
    def is_successful(self) -> bool:
        """Check if processing completed successfully."""
        return self.status == ProcessingStatusType.COMPLETED
    
    @property
    def has_error(self) -> bool:
        """Check if status indicates an error."""
        return self.status == ProcessingStatusType.FAILED
    
    @property
    def duration_seconds(self) -> Optional[int]:
        """Get processing duration in seconds."""
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds())
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "progress_percentage": self.progress_percentage,
            "message": self.message,
            "error_details": self.error_details,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "is_terminal": self.is_terminal,
            "is_active": self.is_active,
            "is_successful": self.is_successful,
            "has_error": self.has_error,
            "duration_seconds": self.duration_seconds
        }
    
    @classmethod
    def create_pending(cls, message: Optional[str] = None) -> 'ProcessingStatus':
        """Create pending status."""
        return cls(
            status=ProcessingStatusType.PENDING,
            progress_percentage=0,
            message=message or "Waiting to start processing"
        )
    
    @classmethod
    def create_queued(cls, message: Optional[str] = None) -> 'ProcessingStatus':
        """Create queued status."""
        return cls(
            status=ProcessingStatusType.QUEUED,
            progress_percentage=0,
            message=message or "Queued for processing"
        )
    
    @classmethod
    def create_processing(
        cls,
        progress: int = 0,
        message: Optional[str] = None,
        started_at: Optional[datetime] = None
    ) -> 'ProcessingStatus':
        """Create processing status."""
        return cls(
            status=ProcessingStatusType.PROCESSING,
            progress_percentage=progress,
            message=message or "Processing in progress",
            started_at=started_at or datetime.now(timezone.utc)
        )
    
    @classmethod
    def create_completed(
        cls,
        message: Optional[str] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None
    ) -> 'ProcessingStatus':
        """Create completed status."""
        return cls(
            status=ProcessingStatusType.COMPLETED,
            progress_percentage=100,
            message=message or "Processing completed successfully",
            started_at=started_at,
            completed_at=completed_at or datetime.now(timezone.utc)
        )
    
    @classmethod
    def create_failed(
        cls,
        error_details: str,
        progress: int = 0,
        message: Optional[str] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None
    ) -> 'ProcessingStatus':
        """Create failed status."""
        return cls(
            status=ProcessingStatusType.FAILED,
            progress_percentage=progress,
            message=message or "Processing failed",
            error_details=error_details,
            started_at=started_at,
            completed_at=completed_at or datetime.now(timezone.utc)
        )
    
    @classmethod
    def create_cancelled(
        cls,
        progress: int = 0,
        message: Optional[str] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None
    ) -> 'ProcessingStatus':
        """Create cancelled status."""
        return cls(
            status=ProcessingStatusType.CANCELLED,
            progress_percentage=progress,
            message=message or "Processing cancelled",
            started_at=started_at,
            completed_at=completed_at or datetime.now(timezone.utc)
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingStatus':
        """Create processing status from dictionary."""
        return cls(
            status=ProcessingStatusType(data["status"]),
            progress_percentage=data.get("progress_percentage", 0),
            message=data.get("message"),
            error_details=data.get("error_details"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
        ) 