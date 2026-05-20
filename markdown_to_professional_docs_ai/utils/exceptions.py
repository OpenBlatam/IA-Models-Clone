"""Custom exceptions for Markdown to Professional Documents AI"""
from typing import Optional


class MarkdownConverterException(Exception):
    """Base exception for all converter errors"""
    def __init__(self, message: str, status_code: int = 500, details: Optional[dict] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class InvalidFormatException(MarkdownConverterException):
    """Raised when an invalid output format is requested"""
    def __init__(self, format_name: str, supported_formats: list):
        super().__init__(
            f"Unsupported format: {format_name}",
            status_code=400,
            details={"format": format_name, "supported_formats": supported_formats}
        )


class ParsingException(MarkdownConverterException):
    """Raised when Markdown parsing fails"""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            f"Failed to parse Markdown: {message}",
            status_code=422,
            details=details
        )


class ConversionException(MarkdownConverterException):
    """Raised when document conversion fails"""
    def __init__(self, format_name: str, message: str, details: Optional[dict] = None):
        super().__init__(
            f"Failed to convert to {format_name}: {message}",
            status_code=500,
            details={"format": format_name, **details or {}}
        )


class FileSizeException(MarkdownConverterException):
    """Raised when file size exceeds limit"""
    def __init__(self, file_size: int, max_size: int):
        super().__init__(
            f"File size {file_size} exceeds maximum {max_size} bytes",
            status_code=413,
            details={"file_size": file_size, "max_size": max_size}
        )


class ValidationException(MarkdownConverterException):
    """Raised when validation fails"""
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(
            f"Validation failed: {message}",
            status_code=400,
            details={"field": field} if field else {}
        )

