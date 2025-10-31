from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Optional, Dict, Any
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Post Domain Exceptions
=====================

Domain-specific exceptions for LinkedIn posts with proper error handling.
"""



class PostDomainError(Exception):
    """Base exception for post domain errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(message)
        self.message = message
        self.error_code = error_code or "POST_DOMAIN_ERROR"
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class InvalidPostStateError(PostDomainError):
    """Exception raised when post state is invalid for the requested operation."""
    
    def __init__(self, message: str, current_state: Optional[str] = None, 
                 required_state: Optional[str] = None):
        
    """__init__ function."""
details = {}
        if current_state:
            details["current_state"] = current_state
        if required_state:
            details["required_state"] = required_state
        
        super().__init__(message, "INVALID_POST_STATE", details)


class ContentValidationError(PostDomainError):
    """Exception raised when post content validation fails."""
    
    def __init__(self, message: str, content: Optional[str] = None, 
                 validation_errors: Optional[list] = None):
        
    """__init__ function."""
details = {}
        if content:
            details["content"] = content
        if validation_errors:
            details["validation_errors"] = validation_errors
        
        super().__init__(message, "CONTENT_VALIDATION_ERROR", details)


class PostAlreadyPublishedError(PostDomainError):
    """Exception raised when trying to publish an already published post."""
    
    def __init__(self, post_id: str, published_at: Optional[str] = None):
        
    """__init__ function."""
message = f"Post {post_id} is already published"
        details = {"post_id": post_id}
        if published_at:
            details["published_at"] = published_at
        
        super().__init__(message, "POST_ALREADY_PUBLISHED", details)


class PostNotFoundError(PostDomainError):
    """Exception raised when a post is not found."""
    
    def __init__(self, post_id: str):
        
    """__init__ function."""
message = f"Post {post_id} not found"
        super().__init__(message, "POST_NOT_FOUND", {"post_id": post_id})


class PostDeletionError(PostDomainError):
    """Exception raised when post deletion fails."""
    
    def __init__(self, post_id: str, reason: Optional[str] = None):
        
    """__init__ function."""
message = f"Failed to delete post {post_id}"
        details = {"post_id": post_id}
        if reason:
            details["reason"] = reason
        
        super().__init__(message, "POST_DELETION_ERROR", details)


class PostOptimizationError(PostDomainError):
    """Exception raised when post optimization fails."""
    
    def __init__(self, post_id: str, optimization_type: str, reason: Optional[str] = None):
        
    """__init__ function."""
message = f"Failed to optimize post {post_id} with {optimization_type}"
        details = {
            "post_id": post_id,
            "optimization_type": optimization_type
        }
        if reason:
            details["reason"] = reason
        
        super().__init__(message, "POST_OPTIMIZATION_ERROR", details)


class PostSchedulingError(PostDomainError):
    """Exception raised when post scheduling fails."""
    
    def __init__(self, post_id: str, scheduled_time: Optional[str] = None, 
                 reason: Optional[str] = None):
        
    """__init__ function."""
message = f"Failed to schedule post {post_id}"
        details = {"post_id": post_id}
        if scheduled_time:
            details["scheduled_time"] = scheduled_time
        if reason:
            details["reason"] = reason
        
        super().__init__(message, "POST_SCHEDULING_ERROR", details)


class PostEngagementError(PostDomainError):
    """Exception raised when post engagement update fails."""
    
    def __init__(self, post_id: str, engagement_type: str, reason: Optional[str] = None):
        
    """__init__ function."""
message = f"Failed to update {engagement_type} for post {post_id}"
        details = {
            "post_id": post_id,
            "engagement_type": engagement_type
        }
        if reason:
            details["reason"] = reason
        
        super().__init__(message, "POST_ENGAGEMENT_ERROR", details)


class PostPermissionError(PostDomainError):
    """Exception raised when user doesn't have permission to perform action on post."""
    
    def __init__(self, post_id: str, user_id: str, action: str):
        
    """__init__ function."""
message = f"User {user_id} doesn't have permission to {action} post {post_id}"
        details = {
            "post_id": post_id,
            "user_id": user_id,
            "action": action
        }
        
        super().__init__(message, "POST_PERMISSION_ERROR", details)


class PostRateLimitError(PostDomainError):
    """Exception raised when post creation rate limit is exceeded."""
    
    def __init__(self, user_id: str, limit: int, window: str):
        
    """__init__ function."""
message = f"Rate limit exceeded for user {user_id}. Limit: {limit} posts per {window}"
        details = {
            "user_id": user_id,
            "limit": limit,
            "window": window
        }
        
        super().__init__(message, "POST_RATE_LIMIT_ERROR", details)


class PostContentTooLongError(ContentValidationError):
    """Exception raised when post content exceeds maximum length."""
    
    def __init__(self, content_length: int, max_length: int):
        
    """__init__ function."""
message = f"Content length {content_length} exceeds maximum length {max_length}"
        details = {
            "content_length": content_length,
            "max_length": max_length,
            "excess_length": content_length - max_length
        }
        
        super().__init__(message, details=details)


class PostContentTooShortError(ContentValidationError):
    """Exception raised when post content is too short."""
    
    def __init__(self, content_length: int, min_length: int):
        
    """__init__ function."""
message = f"Content length {content_length} is below minimum length {min_length}"
        details = {
            "content_length": content_length,
            "min_length": min_length,
            "missing_length": min_length - content_length
        }
        
        super().__init__(message, details=details)


class PostContentProfanityError(ContentValidationError):
    """Exception raised when post content contains profanity."""
    
    def __init__(self, profane_words: list):
        
    """__init__ function."""
message = f"Content contains inappropriate language: {', '.join(profane_words)}"
        details = {
            "profane_words": profane_words,
            "profane_word_count": len(profane_words)
        }
        
        super().__init__(message, details=details)


class PostDuplicateError(PostDomainError):
    """Exception raised when trying to create a duplicate post."""
    
    def __init__(self, content_hash: str, existing_post_id: str):
        
    """__init__ function."""
message = f"Duplicate post detected. Similar content already exists in post {existing_post_id}"
        details = {
            "content_hash": content_hash,
            "existing_post_id": existing_post_id
        }
        
        super().__init__(message, "POST_DUPLICATE_ERROR", details)


class PostExternalServiceError(PostDomainError):
    """Exception raised when external service integration fails."""
    
    def __init__(self, service_name: str, operation: str, reason: Optional[str] = None):
        
    """__init__ function."""
message = f"External service {service_name} failed during {operation}"
        details = {
            "service_name": service_name,
            "operation": operation
        }
        if reason:
            details["reason"] = reason
        
        super().__init__(message, "POST_EXTERNAL_SERVICE_ERROR", details)


class PostCacheError(PostDomainError):
    """Exception raised when cache operations fail."""
    
    def __init__(self, operation: str, cache_key: str, reason: Optional[str] = None):
        
    """__init__ function."""
message = f"Cache operation {operation} failed for key {cache_key}"
        details = {
            "operation": operation,
            "cache_key": cache_key
        }
        if reason:
            details["reason"] = reason
        
        super().__init__(message, "POST_CACHE_ERROR", details)


# Exception handler utilities
class PostExceptionHandler:
    """Utility class for handling post domain exceptions."""
    
    @staticmethod
    def handle_exception(exception: Exception) -> Dict[str, Any]:
        """Handle any exception and return standardized error response."""
        if isinstance(exception, PostDomainError):
            return exception.to_dict()
        else:
            return {
                "error_type": "UNKNOWN_ERROR",
                "message": str(exception),
                "error_code": "UNKNOWN_ERROR",
                "details": {}
            }
    
    @staticmethod
    def is_recoverable(exception: Exception) -> bool:
        """Check if the exception is recoverable."""
        non_recoverable_exceptions = [
            PostNotFoundError,
            PostPermissionError,
            PostContentProfanityError,
            PostDuplicateError
        ]
        
        return not any(isinstance(exception, exc_type) for exc_type in non_recoverable_exceptions)
    
    @staticmethod
    def should_retry(exception: Exception) -> bool:
        """Check if the operation should be retried."""
        retryable_exceptions = [
            PostExternalServiceError,
            PostCacheError,
            PostOptimizationError
        ]
        
        return any(isinstance(exception, exc_type) for exc_type in retryable_exceptions)
    
    @staticmethod
    async def get_http_status_code(exception: Exception) -> int:
        """Get appropriate HTTP status code for the exception."""
        if isinstance(exception, PostNotFoundError):
            return 404
        elif isinstance(exception, PostPermissionError):
            return 403
        elif isinstance(exception, PostRateLimitError):
            return 429
        elif isinstance(exception, ContentValidationError):
            return 400
        elif isinstance(exception, PostAlreadyPublishedError):
            return 409
        elif isinstance(exception, PostDuplicateError):
            return 409
        else:
            return 500 