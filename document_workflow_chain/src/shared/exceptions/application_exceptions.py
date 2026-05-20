"""
Application Exceptions
======================

Custom exceptions for the application layer.
"""

from __future__ import annotations
from typing import Optional, Dict, Any


class ApplicationException(Exception):
    """Base application exception"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "APPLICATION_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationException(ApplicationException):
    """Validation exception"""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.field = field
        self.value = value
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={
                "field": field,
                "value": str(value) if value is not None else None,
                **(details or {})
            }
        )


class BusinessRuleException(ApplicationException):
    """Business rule exception"""
    
    def __init__(
        self,
        message: str,
        rule_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.rule_name = rule_name
        super().__init__(
            message=message,
            error_code="BUSINESS_RULE_ERROR",
            details={
                "rule_name": rule_name,
                **(details or {})
            }
        )


class ResourceNotFoundException(ApplicationException):
    """Resource not found exception"""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(
            message=f"{resource_type} with ID '{resource_id}' not found",
            error_code="RESOURCE_NOT_FOUND",
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
                **(details or {})
            }
        )


class DuplicateResourceException(ApplicationException):
    """Duplicate resource exception"""
    
    def __init__(
        self,
        resource_type: str,
        field: str,
        value: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.resource_type = resource_type
        self.field = field
        self.value = value
        super().__init__(
            message=f"{resource_type} with {field} '{value}' already exists",
            error_code="DUPLICATE_RESOURCE",
            details={
                "resource_type": resource_type,
                "field": field,
                "value": value,
                **(details or {})
            }
        )


class ConcurrencyException(ApplicationException):
    """Concurrency exception"""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        expected_version: int,
        actual_version: int,
        details: Optional[Dict[str, Any]] = None
    ):
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            message=f"Concurrency conflict for {resource_type} '{resource_id}': expected version {expected_version}, got {actual_version}",
            error_code="CONCURRENCY_ERROR",
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
                "expected_version": expected_version,
                "actual_version": actual_version,
                **(details or {})
            }
        )


class ExternalServiceException(ApplicationException):
    """External service exception"""
    
    def __init__(
        self,
        service_name: str,
        operation: str,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.service_name = service_name
        self.operation = operation
        self.status_code = status_code
        super().__init__(
            message=f"External service '{service_name}' error during '{operation}': {message}",
            error_code="EXTERNAL_SERVICE_ERROR",
            details={
                "service_name": service_name,
                "operation": operation,
                "status_code": status_code,
                **(details or {})
            }
        )


class AuthenticationException(ApplicationException):
    """Authentication exception"""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            details=details or {}
        )


class AuthorizationException(ApplicationException):
    """Authorization exception"""
    
    def __init__(
        self,
        message: str = "Access denied",
        required_permission: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.required_permission = required_permission
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            details={
                "required_permission": required_permission,
                **(details or {})
            }
        )


class RateLimitException(ApplicationException):
    """Rate limit exception"""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.retry_after = retry_after
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            details={
                "retry_after": retry_after,
                **(details or {})
            }
        )


class ConfigurationException(ApplicationException):
    """Configuration exception"""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.config_key = config_key
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details={
                "config_key": config_key,
                **(details or {})
            }
        )


class InfrastructureException(ApplicationException):
    """Infrastructure exception"""
    
    def __init__(
        self,
        component: str,
        operation: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.component = component
        self.operation = operation
        super().__init__(
            message=f"Infrastructure error in '{component}' during '{operation}': {message}",
            error_code="INFRASTRUCTURE_ERROR",
            details={
                "component": component,
                "operation": operation,
                **(details or {})
            }
        )


class EventProcessingException(ApplicationException):
    """Event processing exception"""
    
    def __init__(
        self,
        event_type: str,
        event_id: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.event_type = event_type
        self.event_id = event_id
        super().__init__(
            message=f"Event processing error for '{event_type}' (ID: {event_id}): {message}",
            error_code="EVENT_PROCESSING_ERROR",
            details={
                "event_type": event_type,
                "event_id": event_id,
                **(details or {})
            }
        )


class CacheException(ApplicationException):
    """Cache exception"""
    
    def __init__(
        self,
        operation: str,
        key: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.operation = operation
        self.key = key
        super().__init__(
            message=f"Cache error during '{operation}' for key '{key}': {message}",
            error_code="CACHE_ERROR",
            details={
                "operation": operation,
                "key": key,
                **(details or {})
            }
        )


class NotificationException(ApplicationException):
    """Notification exception"""
    
    def __init__(
        self,
        channel: str,
        recipient: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.channel = channel
        self.recipient = recipient
        super().__init__(
            message=f"Notification error via '{channel}' to '{recipient}': {message}",
            error_code="NOTIFICATION_ERROR",
            details={
                "channel": channel,
                "recipient": recipient,
                **(details or {})
            }
        )


class AnalyticsException(ApplicationException):
    """Analytics exception"""
    
    def __init__(
        self,
        operation: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.operation = operation
        super().__init__(
            message=f"Analytics error during '{operation}': {message}",
            error_code="ANALYTICS_ERROR",
            details={
                "operation": operation,
                **(details or {})
            }
        )


class AuditException(ApplicationException):
    """Audit exception"""
    
    def __init__(
        self,
        operation: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.operation = operation
        super().__init__(
            message=f"Audit error during '{operation}': {message}",
            error_code="AUDIT_ERROR",
            details={
                "operation": operation,
                **(details or {})
            }
        )




