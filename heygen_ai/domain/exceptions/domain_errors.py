from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Any, Dict, Optional
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Domain-Specific Exceptions

Custom exceptions for domain layer business rule violations and validation errors.
"""



class DomainError(Exception):
    """
    Base exception for all domain errors.
    
    Contains error code, message, and optional context data.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.error_code}: {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "type": self.__class__.__name__
        }


class ValueObjectValidationError(DomainError):
    """
    Exception raised when value object validation fails.
    
    Used for invalid email formats, invalid enums, etc.
    """
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None
    ):
        
    """__init__ function."""
context = {}
        if field_name:
            context["field_name"] = field_name
        if field_value is not None:
            context["field_value"] = str(field_value)
        
        super().__init__(
            message=message,
            error_code="VALUE_OBJECT_VALIDATION_ERROR",
            context=context
        )


class UserValidationError(DomainError):
    """
    Exception raised when user entity validation fails.
    
    Used for invalid usernames, invalid user data, etc.
    """
    
    def __init__(
        self,
        message: str,
        user_id: Optional[str] = None,
        field_name: Optional[str] = None
    ):
        
    """__init__ function."""
context = {}
        if user_id:
            context["user_id"] = user_id
        if field_name:
            context["field_name"] = field_name
        
        super().__init__(
            message=message,
            error_code="USER_VALIDATION_ERROR",
            context=context
        )


class VideoValidationError(DomainError):
    """
    Exception raised when video entity validation fails.
    
    Used for invalid video parameters, invalid video data, etc.
    """
    
    def __init__(
        self,
        message: str,
        video_id: Optional[str] = None,
        field_name: Optional[str] = None
    ):
        
    """__init__ function."""
context = {}
        if video_id:
            context["video_id"] = video_id
        if field_name:
            context["field_name"] = field_name
        
        super().__init__(
            message=message,
            error_code="VIDEO_VALIDATION_ERROR",
            context=context
        )


class BusinessRuleViolationError(DomainError):
    """
    Exception raised when business rules are violated.
    
    Used for operations that violate business constraints.
    """
    
    def __init__(
        self,
        message: str,
        rule_name: Optional[str] = None,
        entity_id: Optional[str] = None
    ):
        
    """__init__ function."""
context = {}
        if rule_name:
            context["rule_name"] = rule_name
        if entity_id:
            context["entity_id"] = entity_id
        
        super().__init__(
            message=message,
            error_code="BUSINESS_RULE_VIOLATION",
            context=context
        )


class DomainNotFoundException(DomainError):
    """
    Exception raised when a domain entity is not found.
    """
    
    def __init__(
        self,
        message: str,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None
    ):
        
    """__init__ function."""
context = {}
        if entity_type:
            context["entity_type"] = entity_type
        if entity_id:
            context["entity_id"] = entity_id
        
        super().__init__(
            message=message,
            error_code="DOMAIN_NOT_FOUND",
            context=context
        )


class DomainConflictError(DomainError):
    """
    Exception raised when there's a conflict in domain operations.
    
    Used for unique constraint violations, concurrent modification, etc.
    """
    
    def __init__(
        self,
        message: str,
        conflict_type: Optional[str] = None,
        entity_id: Optional[str] = None
    ):
        
    """__init__ function."""
context = {}
        if conflict_type:
            context["conflict_type"] = conflict_type
        if entity_id:
            context["entity_id"] = entity_id
        
        super().__init__(
            message=message,
            error_code="DOMAIN_CONFLICT",
            context=context
        )


class DomainForbiddenError(DomainError):
    """
    Exception raised when an operation is forbidden by business rules.
    
    Used for authorization failures, insufficient permissions, etc.
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        
    """__init__ function."""
context = {}
        if operation:
            context["operation"] = operation
        if user_id:
            context["user_id"] = user_id
        
        super().__init__(
            message=message,
            error_code="DOMAIN_FORBIDDEN",
            context=context
        ) 