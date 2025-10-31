"""
Custom Exceptions
=================

Custom exceptions for PDF variantes feature.
"""

from typing import Optional, Dict, Any


class PDFVariantesError(Exception):
    """Base exception for PDF variantes feature."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class PDFNotFoundError(PDFVariantesError):
    """Raised when a PDF file is not found."""
    
    def __init__(self, file_id: str):
        super().__init__(
            message=f"PDF file not found: {file_id}",
            error_code="PDF_NOT_FOUND",
            details={"file_id": file_id}
        )


class InvalidFileError(PDFVariantesError):
    """Raised when file validation fails."""
    
    def __init__(self, message: str, file_type: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="INVALID_FILE",
            details={"file_type": file_type}
        )


class FileSizeError(PDFVariantesError):
    """Raised when file size exceeds limits."""
    
    def __init__(self, file_size: int, max_size: int):
        super().__init__(
            message=f"File size {file_size} exceeds maximum limit {max_size}",
            error_code="FILE_TOO_LARGE",
            details={"file_size": file_size, "max_size": max_size}
        )


class ProcessingError(PDFVariantesError):
    """Raised when document processing fails."""
    
    def __init__(
        self,
        message: str,
        processing_type: Optional[str] = None,
        file_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="PROCESSING_ERROR",
            details={
                "processing_type": processing_type,
                "file_id": file_id
            }
        )


class VariantGenerationError(ProcessingError):
    """Raised when variant generation fails."""
    
    def __init__(
        self,
        message: str,
        variant_type: Optional[str] = None,
        file_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            processing_type="variant_generation",
            file_id=file_id
        )
        self.error_code = "VARIANT_GENERATION_ERROR"
        self.details["variant_type"] = variant_type


class TopicExtractionError(ProcessingError):
    """Raised when topic extraction fails."""
    
    def __init__(self, message: str, file_id: Optional[str] = None):
        super().__init__(
            message=message,
            processing_type="topic_extraction",
            file_id=file_id
        )
        self.error_code = "TOPIC_EXTRACTION_ERROR"


class BrainstormingError(ProcessingError):
    """Raised when brainstorming fails."""
    
    def __init__(self, message: str, file_id: Optional[str] = None):
        super().__init__(
            message=message,
            processing_type="brainstorming",
            file_id=file_id
        )
        self.error_code = "BRAINSTORMING_ERROR"


class AnnotationError(PDFVariantesError):
    """Raised when annotation operations fail."""
    
    def __init__(
        self,
        message: str,
        annotation_id: Optional[str] = None,
        file_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="ANNOTATION_ERROR",
            details={
                "annotation_id": annotation_id,
                "file_id": file_id
            }
        )


class ConfigurationError(PDFVariantesError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key}
        )


class AuthenticationError(PDFVariantesError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR"
        )


class AuthorizationError(PDFVariantesError):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR"
        )


class RateLimitError(PDFVariantesError):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str,
        limit: Optional[int] = None,
        window: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            details={"limit": limit, "window": window}
        )


class StorageError(PDFVariantesError):
    """Raised when storage operations fail."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        file_path: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="STORAGE_ERROR",
            details={
                "operation": operation,
                "file_path": file_path
            }
        )


class AIProcessingError(PDFVariantesError):
    """Raised when AI processing fails."""
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="AI_PROCESSING_ERROR",
            details={
                "provider": provider,
                "model": model
            }
        )


class ValidationError(PDFVariantesError):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None
    ):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={
                "field": field,
                "value": str(value) if value is not None else None
            }
        )


class CollaborationError(PDFVariantesError):
    """Raised when collaboration operations fail."""
    
    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="COLLABORATION_ERROR",
            details={
                "session_id": session_id,
                "user_id": user_id
            }
        )


class WorkflowError(PDFVariantesError):
    """Raised when workflow operations fail."""
    
    def __init__(
        self,
        message: str,
        workflow_id: Optional[str] = None,
        step: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="WORKFLOW_ERROR",
            details={
                "workflow_id": workflow_id,
                "step": step
            }
        )


class CacheError(PDFVariantesError):
    """Raised when cache operations fail."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        key: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            details={
                "operation": operation,
                "key": key
            }
        )


class MonitoringError(PDFVariantesError):
    """Raised when monitoring operations fail."""
    
    def __init__(self, message: str, metric_name: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="MONITORING_ERROR",
            details={"metric_name": metric_name}
        )


# Error code mappings for HTTP status codes
ERROR_STATUS_MAPPING = {
    "PDF_NOT_FOUND": 404,
    "INVALID_FILE": 400,
    "FILE_TOO_LARGE": 413,
    "PROCESSING_ERROR": 422,
    "VARIANT_GENERATION_ERROR": 422,
    "TOPIC_EXTRACTION_ERROR": 422,
    "BRAINSTORMING_ERROR": 422,
    "ANNOTATION_ERROR": 422,
    "CONFIGURATION_ERROR": 500,
    "AUTHENTICATION_ERROR": 401,
    "AUTHORIZATION_ERROR": 403,
    "RATE_LIMIT_EXCEEDED": 429,
    "STORAGE_ERROR": 500,
    "AI_PROCESSING_ERROR": 503,
    "VALIDATION_ERROR": 422,
    "COLLABORATION_ERROR": 422,
    "WORKFLOW_ERROR": 422,
    "CACHE_ERROR": 500,
    "MONITORING_ERROR": 500,
}


def get_http_status_code(error_code: str) -> int:
    """Get HTTP status code for error code."""
    return ERROR_STATUS_MAPPING.get(error_code, 500)
