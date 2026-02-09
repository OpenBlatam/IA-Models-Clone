"""
Validation utilities for professional documents.

Centralized validation logic and error messages.
"""

from typing import Optional
from fastapi import HTTPException
from .models import ProfessionalDocument, DocumentExportRequest
from .exceptions import ValidationError, DocumentNotFoundError
from .constants import MIN_QUERY_LENGTH, EXPORT_FORMAT_EXTENSIONS


class InvalidFormatError(ValidationError):
    """Raised when an invalid format is specified."""
    pass


def validate_document_exists(document: Optional[ProfessionalDocument], document_id: str) -> None:
    """
    Validate that a document exists.
    
    Args:
        document: Document object (may be None)
        document_id: Document ID for error message
        
    Raises:
        DocumentNotFoundError: If document doesn't exist
    """
    if not document:
        raise DocumentNotFoundError(f"Document with ID '{document_id}' not found")


def validate_export_request(request: DocumentExportRequest) -> None:
    """Validate export request parameters."""
    if request.format.value not in EXPORT_FORMAT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported export format: {request.format.value}"
        )
    
    if request.password_protect and not request.password:
        raise HTTPException(
            status_code=400,
            detail="Password is required when password protection is enabled"
        )
    
    if request.custom_filename:
        if ".." in request.custom_filename or "/" in request.custom_filename or "\\" in request.custom_filename:
            raise HTTPException(
                status_code=400,
                detail="Invalid filename: cannot contain path separators or parent directory references"
            )


def validate_filename(filename: str) -> None:
    """
    Validate filename for security.
    
    Args:
        filename: Filename to validate
        
    Raises:
        ValidationError: If filename is invalid
    """
    if not filename or not filename.strip():
        raise ValidationError("Filename cannot be empty")
    
    if ".." in filename or "/" in filename or "\\" in filename:
        raise ValidationError(
            "Invalid filename: cannot contain path separators or parent directory references"
        )
    
    if len(filename) > 255:
        raise ValidationError("Filename is too long (max 255 characters)")


def validate_query_length(query: str, min_length: int = MIN_QUERY_LENGTH) -> str:
    """
    Validate and sanitize query string.
    
    Args:
        query: Query string to validate
        min_length: Minimum required length
        
    Returns:
        Sanitized query string
        
    Raises:
        ValueError: If query is too short
    """
    if not query:
        raise ValueError("Query cannot be empty")
    
    query = query.strip()
    
    if len(query) < min_length:
        raise ValueError(
            f"Query must be at least {min_length} characters long "
            f"(current: {len(query)} characters)"
        )
    
    return query

