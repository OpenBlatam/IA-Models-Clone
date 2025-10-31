"""
Business Layer - Application Services
===================================

Application services implementing business logic and orchestrating domain operations.
"""

from .services import (
    DocumentService,
    ProcessingService,
    UserService,
    OrganizationService,
    NotificationService,
    MetricsService
)
from .dto import (
    CreateDocumentRequest,
    UpdateDocumentRequest,
    ProcessDocumentRequest,
    CreateUserRequest,
    UpdateUserRequest,
    CreateOrganizationRequest,
    UpdateOrganizationRequest
)
from .exceptions import (
    BusinessException,
    DocumentNotFoundError,
    UserNotFoundError,
    OrganizationNotFoundError,
    ProcessingFailedError,
    ValidationError,
    AuthorizationError
)

__all__ = [
    # Services
    "DocumentService",
    "ProcessingService",
    "UserService",
    "OrganizationService",
    "NotificationService",
    "MetricsService",
    
    # DTOs
    "CreateDocumentRequest",
    "UpdateDocumentRequest",
    "ProcessDocumentRequest",
    "CreateUserRequest",
    "UpdateUserRequest",
    "CreateOrganizationRequest",
    "UpdateOrganizationRequest",
    
    # Exceptions
    "BusinessException",
    "DocumentNotFoundError",
    "UserNotFoundError",
    "OrganizationNotFoundError",
    "ProcessingFailedError",
    "ValidationError",
    "AuthorizationError",
]

















