from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Analysis Status Value Object
Enum for SEO analysis status with additional metadata
"""



class AnalysisStatus(Enum):
    """SEO Analysis Status Enum"""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    INVALID_URL = "invalid_url"
    NETWORK_ERROR = "network_error"
    PARSER_ERROR = "parser_error"
    ANALYSIS_ERROR = "analysis_error"


@dataclass(frozen=True)
class AnalysisStatusInfo:
    """Analysis Status Information with metadata"""
    
    status: AnalysisStatus
    message: str
    error_code: Optional[str] = None
    error_details: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self) -> Any:
        """Set timestamps if not provided"""
        if self.created_at is None:
            object.__setattr__(self, 'created_at', datetime.utcnow())
        if self.updated_at is None:
            object.__setattr__(self, 'updated_at', datetime.utcnow())
    
    @property
    def is_final(self) -> bool:
        """Check if status is final (no more transitions expected)"""
        final_statuses = {
            AnalysisStatus.COMPLETED,
            AnalysisStatus.FAILED,
            AnalysisStatus.CANCELLED,
            AnalysisStatus.TIMEOUT,
            AnalysisStatus.INVALID_URL
        }
        return self.status in final_statuses
    
    @property
    def is_error(self) -> bool:
        """Check if status represents an error"""
        error_statuses = {
            AnalysisStatus.FAILED,
            AnalysisStatus.TIMEOUT,
            AnalysisStatus.RATE_LIMITED,
            AnalysisStatus.INVALID_URL,
            AnalysisStatus.NETWORK_ERROR,
            AnalysisStatus.PARSER_ERROR,
            AnalysisStatus.ANALYSIS_ERROR
        }
        return self.status in error_statuses
    
    @property
    def is_retryable(self) -> bool:
        """Check if status allows retry"""
        retryable_statuses = {
            AnalysisStatus.NETWORK_ERROR,
            AnalysisStatus.PARSER_ERROR,
            AnalysisStatus.ANALYSIS_ERROR,
            AnalysisStatus.TIMEOUT
        }
        return self.status in retryable_statuses and self.retry_count < self.max_retries
    
    @property
    def can_retry(self) -> bool:
        """Check if retry is possible"""
        return self.is_retryable and self.retry_count < self.max_retries
    
    def increment_retry(self) -> 'AnalysisStatusInfo':
        """Increment retry count"""
        return self.__class__(
            status=self.status,
            message=self.message,
            error_code=self.error_code,
            error_details=self.error_details,
            retry_count=self.retry_count + 1,
            max_retries=self.max_retries,
            created_at=self.created_at,
            updated_at=datetime.utcnow()
        )
    
    def update_status(self, new_status: AnalysisStatus, message: str = None) -> 'AnalysisStatusInfo':
        """Update status with new information"""
        return self.__class__(
            status=new_status,
            message=message or self.message,
            error_code=self.error_code,
            error_details=self.error_details,
            retry_count=self.retry_count,
            max_retries=self.max_retries,
            created_at=self.created_at,
            updated_at=datetime.utcnow()
        )
    
    def add_error(self, error_code: str, error_details: str) -> 'AnalysisStatusInfo':
        """Add error information"""
        return self.__class__(
            status=self.status,
            message=self.message,
            error_code=error_code,
            error_details=error_details,
            retry_count=self.retry_count,
            max_retries=self.max_retries,
            created_at=self.created_at,
            updated_at=datetime.utcnow()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'status': self.status.value,
            'message': self.message,
            'error_code': self.error_code,
            'error_details': self.error_details,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'is_final': self.is_final,
            'is_error': self.is_error,
            'is_retryable': self.is_retryable,
            'can_retry': self.can_retry,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisStatusInfo':
        """Create from dictionary"""
        return cls(
            status=AnalysisStatus(data['status']),
            message=data['message'],
            error_code=data.get('error_code'),
            error_details=data.get('error_details'),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at'])
        )
    
    @classmethod
    def pending(cls) -> 'AnalysisStatusInfo':
        """Create pending status"""
        return cls(
            status=AnalysisStatus.PENDING,
            message="Analysis is pending"
        )
    
    @classmethod
    def in_progress(cls) -> 'AnalysisStatusInfo':
        """Create in progress status"""
        return cls(
            status=AnalysisStatus.IN_PROGRESS,
            message="Analysis is in progress"
        )
    
    @classmethod
    def completed(cls) -> 'AnalysisStatusInfo':
        """Create completed status"""
        return cls(
            status=AnalysisStatus.COMPLETED,
            message="Analysis completed successfully"
        )
    
    @classmethod
    def failed(cls, error_code: str = None, error_details: str = None) -> 'AnalysisStatusInfo':
        """Create failed status"""
        return cls(
            status=AnalysisStatus.FAILED,
            message="Analysis failed",
            error_code=error_code,
            error_details=error_details
        )
    
    @classmethod
    def timeout(cls) -> 'AnalysisStatusInfo':
        """Create timeout status"""
        return cls(
            status=AnalysisStatus.TIMEOUT,
            message="Analysis timed out"
        )
    
    @classmethod
    def rate_limited(cls) -> 'AnalysisStatusInfo':
        """Create rate limited status"""
        return cls(
            status=AnalysisStatus.RATE_LIMITED,
            message="Analysis rate limited"
        )
    
    @classmethod
    def invalid_url(cls) -> 'AnalysisStatusInfo':
        """Create invalid URL status"""
        return cls(
            status=AnalysisStatus.INVALID_URL,
            message="Invalid URL provided"
        )
    
    @classmethod
    def network_error(cls, error_details: str = None) -> 'AnalysisStatusInfo':
        """Create network error status"""
        return cls(
            status=AnalysisStatus.NETWORK_ERROR,
            message="Network error occurred",
            error_code="NETWORK_ERROR",
            error_details=error_details
        )
    
    @classmethod
    def parser_error(cls, error_details: str = None) -> 'AnalysisStatusInfo':
        """Create parser error status"""
        return cls(
            status=AnalysisStatus.PARSER_ERROR,
            message="HTML parsing error occurred",
            error_code="PARSER_ERROR",
            error_details=error_details
        )
    
    @classmethod
    def analysis_error(cls, error_details: str = None) -> 'AnalysisStatusInfo':
        """Create analysis error status"""
        return cls(
            status=AnalysisStatus.ANALYSIS_ERROR,
            message="SEO analysis error occurred",
            error_code="ANALYSIS_ERROR",
            error_details=error_details
        )
    
    def __str__(self) -> str:
        """String representation"""
        return f"AnalysisStatusInfo(status={self.status.value}, message='{self.message}')"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (
            f"AnalysisStatusInfo("
            f"status={self.status.value}, "
            f"message='{self.message}', "
            f"error_code='{self.error_code}', "
            f"retry_count={self.retry_count}, "
            f"is_final={self.is_final}"
            f")"
        )
    
    def __eq__(self, other: object) -> bool:
        """Compare status info"""
        if not isinstance(other, AnalysisStatusInfo):
            return False
        
        return (
            self.status == other.status and
            self.message == other.message and
            self.error_code == other.error_code and
            self.retry_count == other.retry_count
        )
    
    def __hash__(self) -> int:
        """Hash based on status and message"""
        return hash((self.status, self.message, self.error_code, self.retry_count)) 