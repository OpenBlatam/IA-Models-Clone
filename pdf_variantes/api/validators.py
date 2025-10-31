"""
PDF Variantes API - Request Validators
Centralized validation for requests
"""

from typing import Optional, List
from fastapi import UploadFile
from pydantic import BaseModel, field_validator

from .exceptions import ValidationError


class FileValidator:
    """File upload validation"""
    
    ALLOWED_TYPES = {"application/pdf"}
    MAX_SIZE_MB = 100
    MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024
    
    @classmethod
    def validate_upload_file(cls, file: UploadFile) -> None:
        """Validate uploaded file"""
        errors = []
        
        # Check content type
        if file.content_type not in cls.ALLOWED_TYPES:
            errors.append(
                f"Invalid file type: {file.content_type}. Only PDF files are allowed."
            )
        
        # Check file size (if available in headers)
        if hasattr(file, 'headers'):
            content_length = file.headers.get("content-length")
            if content_length:
                try:
                    size = int(content_length)
                    if size > cls.MAX_SIZE_BYTES:
                        errors.append(
                            f"File too large: {size} bytes. Maximum size is {cls.MAX_SIZE_MB}MB."
                        )
                    if size == 0:
                        errors.append("File is empty")
                except ValueError:
                    pass
        
        if errors:
            raise ValidationError(detail="; ".join(errors), metadata={"errors": errors})
    
    @classmethod
    def validate_filename(cls, filename: str) -> str:
        """Validate and sanitize filename"""
        if not filename or not filename.strip():
            raise ValidationError(detail="Filename cannot be empty", field="filename")
        
        # Sanitize filename
        cleaned = filename.strip()
        # Remove path components for security
        cleaned = cleaned.replace('..', '').replace('/', '').replace('\\', '')
        
        if not cleaned:
            raise ValidationError(detail="Invalid filename after sanitization", field="filename")
        
        # Check length
        if len(cleaned) > 500:
            raise ValidationError(
                detail="Filename too long. Maximum 500 characters.",
                field="filename"
            )
        
        return cleaned


class PaginationValidator:
    """Pagination parameters validation"""
    
    MAX_LIMIT = 100
    DEFAULT_LIMIT = 20
    
    @classmethod
    def validate_pagination(cls, limit: Optional[int] = None, offset: Optional[int] = None) -> tuple[int, int]:
        """Validate and normalize pagination parameters"""
        # Validate limit
        if limit is None:
            limit = cls.DEFAULT_LIMIT
        elif limit < 1:
            raise ValidationError(
                detail="Limit must be at least 1",
                field="limit"
            )
        elif limit > cls.MAX_LIMIT:
            raise ValidationError(
                detail=f"Limit cannot exceed {cls.MAX_LIMIT}",
                field="limit"
            )
        
        # Validate offset
        if offset is None:
            offset = 0
        elif offset < 0:
            raise ValidationError(
                detail="Offset cannot be negative",
                field="offset"
            )
        
        return limit, offset


class DocumentIDValidator:
    """Document ID validation"""
    
    @classmethod
    def validate(cls, document_id: str) -> str:
        """Validate document ID"""
        if not document_id or not document_id.strip():
            raise ValidationError(
                detail="Document ID cannot be empty",
                field="document_id"
            )
        
        return document_id.strip()


class VariantCountValidator:
    """Variant count validation"""
    
    MIN_COUNT = 1
    MAX_COUNT = 1000
    
    @classmethod
    def validate(cls, count: int) -> int:
        """Validate variant count"""
        if count < cls.MIN_COUNT:
            raise ValidationError(
                detail=f"Variant count must be at least {cls.MIN_COUNT}",
                field="number_of_variants"
            )
        
        if count > cls.MAX_COUNT:
            raise ValidationError(
                detail=f"Variant count cannot exceed {cls.MAX_COUNT}",
                field="number_of_variants"
            )
        
        return count


class RelevanceScoreValidator:
    """Relevance score validation"""
    
    MIN_SCORE = 0.0
    MAX_SCORE = 1.0
    
    @classmethod
    def validate(cls, score: float) -> float:
        """Validate relevance score"""
        if score < cls.MIN_SCORE or score > cls.MAX_SCORE:
            raise ValidationError(
                detail=f"Relevance score must be between {cls.MIN_SCORE} and {cls.MAX_SCORE}",
                field="min_relevance"
            )
        
        return score
