"""
Error Context for Piel Mejorador AI SAM3
=========================================

Enhanced error handling with context.
"""

import logging
import traceback
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Error categories."""
    VALIDATION = "validation"
    PROCESSING = "processing"
    NETWORK = "network"
    STORAGE = "storage"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Error context information."""
    error_type: str
    category: ErrorCategory
    message: str
    task_id: Optional[str] = None
    file_path: Optional[str] = None
    user_id: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_type": self.error_type,
            "category": self.category.value,
            "message": self.message,
            "task_id": self.task_id,
            "file_path": self.file_path,
            "user_id": self.user_id,
            "stack_trace": self.stack_trace,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class EnhancedError(Exception):
    """Enhanced exception with context."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.category = category
        self.context = context or {}
        self.timestamp = datetime.now()
    
    def to_error_context(self, task_id: Optional[str] = None) -> ErrorContext:
        """Convert to error context."""
        return ErrorContext(
            error_type=self.__class__.__name__,
            category=self.category,
            message=str(self),
            task_id=task_id,
            stack_trace=traceback.format_exc(),
            metadata=self.context,
        )


class ValidationError(EnhancedError):
    """Validation error."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.VALIDATION, context)


class ProcessingError(EnhancedError):
    """Processing error."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.PROCESSING, context)


class NetworkError(EnhancedError):
    """Network error."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.NETWORK, context)


class StorageError(EnhancedError):
    """Storage error."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.STORAGE, context)


def capture_error_context(
    error: Exception,
    task_id: Optional[str] = None,
    file_path: Optional[str] = None,
    user_id: Optional[str] = None,
    **metadata
) -> ErrorContext:
    """
    Capture error context from exception.
    
    Args:
        error: Exception to capture
        task_id: Optional task ID
        file_path: Optional file path
        user_id: Optional user ID
        **metadata: Additional metadata
        
    Returns:
        ErrorContext
    """
    category = ErrorCategory.UNKNOWN
    
    if isinstance(error, EnhancedError):
        category = error.category
        metadata.update(error.context)
    elif isinstance(error, ValueError):
        category = ErrorCategory.VALIDATION
    elif isinstance(error, (ConnectionError, TimeoutError)):
        category = ErrorCategory.NETWORK
    elif isinstance(error, (IOError, OSError)):
        category = ErrorCategory.STORAGE
    
    return ErrorContext(
        error_type=type(error).__name__,
        category=category,
        message=str(error),
        task_id=task_id,
        file_path=file_path,
        user_id=user_id,
        stack_trace=traceback.format_exc(),
        metadata=metadata,
    )




