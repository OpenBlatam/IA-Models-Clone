"""
Custom Exceptions for Email Sequence System

This module defines custom exception classes for better error handling
and more specific error responses in the API.
"""

from typing import Optional, Dict, Any
from uuid import UUID


class EmailSequenceError(Exception):
    """Base exception for email sequence errors"""
    
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


class SequenceNotFoundError(EmailSequenceError):
    """Raised when a sequence is not found"""
    
    def __init__(self, sequence_id: UUID):
        self.sequence_id = sequence_id
        super().__init__(
            message=f"Sequence {sequence_id} not found",
            error_code="SEQUENCE_NOT_FOUND",
            details={"sequence_id": str(sequence_id)}
        )


class InvalidSequenceError(EmailSequenceError):
    """Raised when sequence data is invalid"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="INVALID_SEQUENCE",
            details=details
        )


class SequenceAlreadyActiveError(EmailSequenceError):
    """Raised when trying to activate an already active sequence"""
    
    def __init__(self, sequence_id: UUID):
        self.sequence_id = sequence_id
        super().__init__(
            message=f"Sequence {sequence_id} is already active",
            error_code="SEQUENCE_ALREADY_ACTIVE",
            details={"sequence_id": str(sequence_id)}
        )


class SequenceNotActiveError(EmailSequenceError):
    """Raised when trying to perform operations on inactive sequence"""
    
    def __init__(self, sequence_id: UUID):
        self.sequence_id = sequence_id
        super().__init__(
            message=f"Sequence {sequence_id} is not active",
            error_code="SEQUENCE_NOT_ACTIVE",
            details={"sequence_id": str(sequence_id)}
        )


class SubscriberNotFoundError(EmailSequenceError):
    """Raised when a subscriber is not found"""
    
    def __init__(self, subscriber_id: UUID):
        self.subscriber_id = subscriber_id
        super().__init__(
            message=f"Subscriber {subscriber_id} not found",
            error_code="SUBSCRIBER_NOT_FOUND",
            details={"subscriber_id": str(subscriber_id)}
        )


class InvalidSubscriberError(EmailSequenceError):
    """Raised when subscriber data is invalid"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="INVALID_SUBSCRIBER",
            details=details
        )


class DuplicateSubscriberError(EmailSequenceError):
    """Raised when trying to add a duplicate subscriber"""
    
    def __init__(self, email: str):
        self.email = email
        super().__init__(
            message=f"Subscriber with email {email} already exists",
            error_code="DUPLICATE_SUBSCRIBER",
            details={"email": email}
        )


class TemplateNotFoundError(EmailSequenceError):
    """Raised when a template is not found"""
    
    def __init__(self, template_id: UUID):
        self.template_id = template_id
        super().__init__(
            message=f"Template {template_id} not found",
            error_code="TEMPLATE_NOT_FOUND",
            details={"template_id": str(template_id)}
        )


class InvalidTemplateError(EmailSequenceError):
    """Raised when template data is invalid"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="INVALID_TEMPLATE",
            details=details
        )


class CampaignNotFoundError(EmailSequenceError):
    """Raised when a campaign is not found"""
    
    def __init__(self, campaign_id: UUID):
        self.campaign_id = campaign_id
        super().__init__(
            message=f"Campaign {campaign_id} not found",
            error_code="CAMPAIGN_NOT_FOUND",
            details={"campaign_id": str(campaign_id)}
        )


class InvalidCampaignError(EmailSequenceError):
    """Raised when campaign data is invalid"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="INVALID_CAMPAIGN",
            details=details
        )


class EmailDeliveryError(EmailSequenceError):
    """Raised when email delivery fails"""
    
    def __init__(
        self,
        message: str,
        email_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="EMAIL_DELIVERY_ERROR",
            details={**(details or {}), "email_address": email_address}
        )


class LangChainServiceError(EmailSequenceError):
    """Raised when LangChain service fails"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="LANGCHAIN_SERVICE_ERROR",
            details=details
        )


class AnalyticsServiceError(EmailSequenceError):
    """Raised when analytics service fails"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="ANALYTICS_SERVICE_ERROR",
            details=details
        )


class DatabaseError(EmailSequenceError):
    """Raised when database operations fail"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details=details
        )


class CacheError(EmailSequenceError):
    """Raised when cache operations fail"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            details=details
        )


class RateLimitError(EmailSequenceError):
    """Raised when rate limit is exceeded"""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            details={**(details or {}), "retry_after": retry_after}
        )


class ValidationError(EmailSequenceError):
    """Raised when data validation fails"""
    
    def __init__(
        self,
        message: str,
        field_errors: Optional[Dict[str, str]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={**(details or {}), "field_errors": field_errors or {}}
        )


class AuthenticationError(EmailSequenceError):
    """Raised when authentication fails"""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            details=details
        )


class AuthorizationError(EmailSequenceError):
    """Raised when authorization fails"""
    
    def __init__(
        self,
        message: str = "Authorization failed",
        required_permission: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            details={**(details or {}), "required_permission": required_permission}
        )


class ConfigurationError(EmailSequenceError):
    """Raised when configuration is invalid"""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details={**(details or {}), "config_key": config_key}
        )


class ExternalServiceError(EmailSequenceError):
    """Raised when external service calls fail"""
    
    def __init__(
        self,
        message: str,
        service_name: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            details={**(details or {}), "service_name": service_name}
        )


class WebhookError(EmailSequenceError):
    """Raised when webhook operations fail"""
    
    def __init__(
        self,
        message: str,
        webhook_url: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="WEBHOOK_ERROR",
            details={**(details or {}), "webhook_url": webhook_url}
        )


class A/BTestError(EmailSequenceError):
    """Raised when A/B test operations fail"""
    
    def __init__(
        self,
        message: str,
        test_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="AB_TEST_ERROR",
            details={**(details or {}), "test_id": test_id}
        )






























