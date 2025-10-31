"""
Domain Layer - Core Business Logic
=================================

Domain entities, value objects, and business rules.
"""

from .entities import Document, ProcessingResult, User, Organization
from .value_objects import DocumentId, ProcessingStatus, DocumentType, UserId
from .repositories import DocumentRepository, ProcessingResultRepository, UserRepository
from .services import DocumentDomainService, ProcessingDomainService
from .exceptions import DomainException, DocumentNotFoundError, ProcessingError

__all__ = [
    # Entities
    "Document",
    "ProcessingResult", 
    "User",
    "Organization",
    
    # Value Objects
    "DocumentId",
    "ProcessingStatus",
    "DocumentType",
    "UserId",
    
    # Repositories (interfaces)
    "DocumentRepository",
    "ProcessingResultRepository",
    "UserRepository",
    
    # Domain Services
    "DocumentDomainService",
    "ProcessingDomainService",
    
    # Exceptions
    "DomainException",
    "DocumentNotFoundError",
    "ProcessingError",
]

















