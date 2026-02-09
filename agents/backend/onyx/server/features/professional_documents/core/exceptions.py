"""
Custom exceptions for professional documents module.

Centralized exception definitions for better error handling.
"""


class ProfessionalDocumentsError(Exception):
    """Base exception for professional documents module."""
    pass


class DocumentNotFoundError(ProfessionalDocumentsError):
    """Exception raised when a document is not found."""
    pass


class DocumentGenerationError(ProfessionalDocumentsError):
    """Exception raised when document generation fails."""
    pass


class DocumentExportError(ProfessionalDocumentsError):
    """Exception raised when document export fails."""
    pass


class TemplateNotFoundError(ProfessionalDocumentsError):
    """Exception raised when a template is not found."""
    pass


class ValidationError(ProfessionalDocumentsError):
    """Exception raised when validation fails."""
    pass


class AIServiceError(ProfessionalDocumentsError):
    """Exception raised when AI service fails."""
    pass


class StorageError(ProfessionalDocumentsError):
    """Exception raised when storage operations fail."""
    pass






