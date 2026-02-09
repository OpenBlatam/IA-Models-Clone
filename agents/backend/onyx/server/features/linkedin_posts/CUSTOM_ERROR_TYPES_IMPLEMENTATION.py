from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import logging
import traceback
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Dict, Optional
"""
Custom Error Types Implementation: Consistent Error Handling

This module demonstrates how to use custom error types and error factories
for consistent, structured error handling across the application.
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CUSTOM ERROR TYPES
# ============================================================================

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    BUSINESS_RULE = "business_rule"
    DATABASE = "database"
    NETWORK = "network"
    SYSTEM = "system"
    EXTERNAL_SERVICE = "external_service"

@dataclass
class ErrorContext:
    """Context information for errors"""
    user_id: Optional[str] = None
    operation: str = "unknown"
    resource_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None

class BaseLinkedInError(Exception):
    """Base error class for LinkedIn posts system"""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        user_message: Optional[str] = None
    ):
        
    """__init__ function."""
super().__init__(message)
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.user_message = user_message or message
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses"""
        return {
            "error_code": self.error_code,
            "message": self.user_message,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "context": {
                "user_id": self.context.user_id,
                "operation": self.context.operation,
                "resource_id": self.context.resource_id,
                "additional_data": self.context.additional_data
            }
        }

class ValidationError(BaseLinkedInError):
    """Validation-related errors"""
    def __init__(self, message: str, error_code: str, context: Optional[ErrorContext] = None, user_message: Optional[str] = None):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            context=context,
            user_message=user_message
        )

class AuthenticationError(BaseLinkedInError):
    """Authentication-related errors"""
    def __init__(self, message: str, error_code: str, context: Optional[ErrorContext] = None, user_message: Optional[str] = None):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            user_message=user_message
        )

class AuthorizationError(BaseLinkedInError):
    """Authorization-related errors"""
    def __init__(self, message: str, error_code: str, context: Optional[ErrorContext] = None, user_message: Optional[str] = None):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            user_message=user_message
        )

class BusinessRuleError(BaseLinkedInError):
    """Business rule violation errors"""
    def __init__(self, message: str, error_code: str, context: Optional[ErrorContext] = None, user_message: Optional[str] = None):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.BUSINESS_RULE,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            user_message=user_message
        )

class DatabaseError(BaseLinkedInError):
    """Database-related errors"""
    def __init__(self, message: str, error_code: str, context: Optional[ErrorContext] = None, user_message: Optional[str] = None):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            context=context,
            user_message=user_message
        )

class NetworkError(BaseLinkedInError):
    """Network-related errors"""
    def __init__(self, message: str, error_code: str, context: Optional[ErrorContext] = None, user_message: Optional[str] = None):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            user_message=user_message
        )

class SystemError(BaseLinkedInError):
    """System-level errors"""
    def __init__(self, message: str, error_code: str, context: Optional[ErrorContext] = None, user_message: Optional[str] = None):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            user_message=user_message
        )

class RetryableError(BaseLinkedInError):
    """Errors that can be retried"""
    def __init__(self, message: str, error_code: str, max_retries: int = 3, context: Optional[ErrorContext] = None, user_message: Optional[str] = None):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            user_message=user_message
        )
        self.max_retries = max_retries

# ============================================================================
# ERROR FACTORY
# ============================================================================

class ErrorFactory:
    """Factory for creating consistent errors"""
    
    # Validation errors
    @staticmethod
    def missing_parameter(param_name: str, context: Optional[ErrorContext] = None) -> ValidationError:
        return ValidationError(
            message=f"{param_name} is required",
            error_code=f"MISSING_{param_name.upper()}",
            context=context,
            user_message=f"Please provide {param_name}"
        )
    
    @staticmethod
    def invalid_format(field_name: str, expected_format: str, context: Optional[ErrorContext] = None) -> ValidationError:
        return ValidationError(
            message=f"{field_name} has invalid format. Expected: {expected_format}",
            error_code=f"INVALID_{field_name.upper()}_FORMAT",
            context=context,
            user_message=f"Please provide {field_name} in the correct format"
        )
    
    @staticmethod
    def content_too_short(min_length: int, context: Optional[ErrorContext] = None) -> ValidationError:
        return ValidationError(
            message=f"Content too short (minimum {min_length} characters)",
            error_code="CONTENT_TOO_SHORT",
            context=context,
            user_message=f"Your post is too short. Please add more content (minimum {min_length} characters)."
        )
    
    @staticmethod
    def content_too_long(max_length: int, context: Optional[ErrorContext] = None) -> ValidationError:
        return ValidationError(
            message=f"Content too long (maximum {max_length} characters)",
            error_code="CONTENT_TOO_LONG",
            context=context,
            user_message=f"Your post is too long. Please shorten it (maximum {max_length} characters)."
        )
    
    # Authentication errors
    @staticmethod
    def user_not_found(user_id: str, context: Optional[ErrorContext] = None) -> AuthenticationError:
        return AuthenticationError(
            message=f"User not found: {user_id}",
            error_code="USER_NOT_FOUND",
            context=context,
            user_message="We couldn't find your account. Please check your login and try again."
        )
    
    @staticmethod
    def account_deactivated(user_id: str, context: Optional[ErrorContext] = None) -> AuthenticationError:
        return AuthenticationError(
            message=f"Account deactivated: {user_id}",
            error_code="ACCOUNT_DEACTIVATED",
            context=context,
            user_message="Your account has been deactivated. Please contact support for assistance."
        )
    
    # Authorization errors
    @staticmethod
    def unauthorized_access(user_id: str, resource_id: str, context: Optional[ErrorContext] = None) -> AuthorizationError:
        return AuthorizationError(
            message=f"User {user_id} not authorized to access {resource_id}",
            error_code="UNAUTHORIZED_ACCESS",
            context=context,
            user_message="You don't have permission to perform this action."
        )
    
    @staticmethod
    def posts_private(user_id: str, context: Optional[ErrorContext] = None) -> AuthorizationError:
        return AuthorizationError(
            message=f"Posts are private for user: {user_id}",
            error_code="POSTS_PRIVATE",
            context=context,
            user_message="This content is private and not available for viewing."
        )
    
    # Business rule errors
    @staticmethod
    def daily_limit_exceeded(user_id: str, limit: int, context: Optional[ErrorContext] = None) -> BusinessRuleError:
        return BusinessRuleError(
            message=f"Daily limit exceeded for user {user_id} (limit: {limit})",
            error_code="DAILY_LIMIT_EXCEEDED",
            context=context,
            user_message=f"You've reached your daily post limit ({limit} posts). Please try again tomorrow."
        )
    
    @staticmethod
    def rate_limit_exceeded(user_id: str, action: str, context: Optional[ErrorContext] = None) -> BusinessRuleError:
        return BusinessRuleError(
            message=f"Rate limit exceeded for user {user_id} on action {action}",
            error_code="RATE_LIMIT_EXCEEDED",
            context=context,
            user_message="You're posting too quickly. Please wait a moment before trying again."
        )
    
    @staticmethod
    def duplicate_content(user_id: str, context: Optional[ErrorContext] = None) -> BusinessRuleError:
        return BusinessRuleError(
            message=f"Duplicate content detected for user {user_id}",
            error_code="DUPLICATE_CONTENT",
            context=context,
            user_message="This content appears to be a duplicate. Please create unique content."
        )
    
    # Database errors
    @staticmethod
    def database_connection_error(operation: str, context: Optional[ErrorContext] = None) -> DatabaseError:
        return DatabaseError(
            message=f"Database connection failed during {operation}",
            error_code="DATABASE_CONNECTION_ERROR",
            context=context,
            user_message="We're experiencing technical difficulties. Please try again in a few minutes."
        )
    
    @staticmethod
    def record_not_found(resource_type: str, resource_id: str, context: Optional[ErrorContext] = None) -> DatabaseError:
        return DatabaseError(
            message=f"{resource_type} not found: {resource_id}",
            error_code=f"{resource_type.upper()}_NOT_FOUND",
            context=context,
            user_message=f"The {resource_type.lower()} you're looking for doesn't exist."
        )
    
    # Network errors
    @staticmethod
    def external_service_unavailable(service_name: str, context: Optional[ErrorContext] = None) -> NetworkError:
        return NetworkError(
            message=f"External service {service_name} is unavailable",
            error_code="EXTERNAL_SERVICE_UNAVAILABLE",
            context=context,
            user_message="This service is temporarily unavailable. Please try again later."
        )
    
    @staticmethod
    def timeout_error(operation: str, timeout_seconds: int, context: Optional[ErrorContext] = None) -> NetworkError:
        return NetworkError(
            message=f"Operation {operation} timed out after {timeout_seconds} seconds",
            error_code="TIMEOUT_ERROR",
            context=context,
            user_message="The operation took too long. Please try again."
        )
    
    # System errors
    @staticmethod
    def unexpected_error(operation: str, original_error: str, context: Optional[ErrorContext] = None) -> SystemError:
        return SystemError(
            message=f"Unexpected error during {operation}: {original_error}",
            error_code="UNEXPECTED_ERROR",
            context=context,
            user_message="Something unexpected happened. Please try again or contact support if the problem persists."
        )

# ============================================================================
# ERROR HANDLER
# ============================================================================

class ErrorHandler:
    """Centralized error handler for consistent error processing"""
    
    @staticmethod
    def handle_error(error: Exception, context: Optional[ErrorContext] = None) -> Dict[str, Any]:
        """Handle any error and return consistent response"""
        
        # Handle custom LinkedIn errors
        if isinstance(error, BaseLinkedInError):
            return ErrorHandler._handle_linkedin_error(error)
        
        # Handle specific exception types
        if isinstance(error, ValidationError):
            return ErrorHandler._handle_validation_error(error)
        
        if isinstance(error, DatabaseError):
            return ErrorHandler._handle_database_error(error)
        
        if isinstance(error, NetworkError):
            return ErrorHandler._handle_network_error(error)
        
        # Handle generic exceptions
        return ErrorHandler._handle_generic_error(error, context)
    
    @staticmethod
    def _handle_linkedin_error(error: BaseLinkedInError) -> Dict[str, Any]:
        """Handle LinkedIn-specific errors"""
        # Log the error
        logger.error(f"LinkedIn Error: {error.error_code} - {error.message}")
        
        # Return structured response
        return {
            "status": "failed",
            "error": error.to_dict()
        }
    
    @staticmethod
    def _handle_validation_error(error: ValidationError) -> Dict[str, Any]:
        """Handle validation errors"""
        logger.warning(f"Validation Error: {error.error_code} - {error.message}")
        
        return {
            "status": "failed",
            "error": {
                "code": error.error_code,
                "message": error.user_message,
                "category": "validation",
                "severity": error.severity.value,
                "timestamp": error.timestamp.isoformat()
            }
        }
    
    @staticmethod
    def _handle_database_error(error: DatabaseError) -> Dict[str, Any]:
        """Handle database errors"""
        logger.error(f"Database Error: {error.error_code} - {error.message}")
        
        return {
            "status": "failed",
            "error": {
                "code": error.error_code,
                "message": error.user_message,
                "category": "database",
                "severity": error.severity.value,
                "timestamp": error.timestamp.isoformat()
            }
        }
    
    @staticmethod
    def _handle_network_error(error: NetworkError) -> Dict[str, Any]:
        """Handle network errors"""
        logger.error(f"Network Error: {error.error_code} - {error.message}")
        
        return {
            "status": "failed",
            "error": {
                "code": error.error_code,
                "message": error.user_message,
                "category": "network",
                "severity": error.severity.value,
                "timestamp": error.timestamp.isoformat()
            }
        }
    
    @staticmethod
    def _handle_generic_error(error: Exception, context: Optional[ErrorContext] = None) -> Dict[str, Any]:
        """Handle generic exceptions"""
        logger.error(f"Unexpected Error: {type(error).__name__} - {str(error)}")
        
        system_error = ErrorFactory.unexpected_error(
            context.operation if context else "unknown",
            str(error),
            context
        )
        
        return {
            "status": "failed",
            "error": system_error.to_dict()
        }

# ============================================================================
# ERROR RECOVERY
# ============================================================================

class ErrorRecovery:
    """Error recovery and retry logic"""
    
    @staticmethod
    async def retry_operation(
        operation: callable,
        max_retries: int = 3,
        delay: float = 1.0,
        context: Optional[ErrorContext] = None
    ) -> Any:
        """Retry operation with exponential backoff"""
        
        for attempt in range(max_retries + 1):
            try:
                return await operation()
                
            except RetryableError as e:
                if attempt == max_retries:
                    raise e
                
                wait_time = delay * (2 ** attempt)
                logger.warning(f"Retryable error on attempt {attempt + 1}: {e.message}. Retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
                
            except BaseLinkedInError as e:
                # Non-retryable error, re-raise immediately
                raise e
                
            except Exception as e:
                # Convert to system error and re-raise
                raise SystemError(
                    message=f"Unexpected error during retry: {str(e)}",
                    error_code="RETRY_FAILED",
                    context=context
                )

# ============================================================================
# MOCK DATABASE FUNCTIONS
# ============================================================================

class DatabaseConnectionError(Exception):
    """Custom exception for database connection errors"""
    pass

async def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Mock function to get user by ID"""
    await asyncio.sleep(0.01)
    
    # Simulate database connection error
    if user_id == "db_error_user":
        raise DatabaseConnectionError("Database connection failed")
    
    if user_id == "valid_user":
        return {
            "id": user_id,
            "name": "John Doe",
            "is_active": True,
            "email": "john@example.com",
            "role": "user",
            "posts_public": True
        }
    elif user_id == "inactive_user":
        return {
            "id": user_id,
            "name": "Jane Doe",
            "is_active": False,
            "email": "jane@example.com",
            "role": "user",
            "posts_public": True
        }
    elif user_id == "private_user":
        return {
            "id": user_id,
            "name": "Private User",
            "is_active": True,
            "email": "private@example.com",
            "role": "user",
            "posts_public": False
        }
    return None

async def get_post_by_id(post_id: str) -> Optional[Dict[str, Any]]:
    """Mock function to get post by ID"""
    await asyncio.sleep(0.01)
    
    # Simulate database connection error
    if post_id == "db_error_post":
        raise DatabaseConnectionError("Database connection failed")
    
    if post_id == "valid_post":
        return {
            "id": post_id,
            "user_id": "valid_user",
            "content": "Original content",
            "created_at": datetime.now() - timedelta(hours=2),
            "is_deleted": False,
            "is_public": True,
            "is_being_edited": False,
            "status": "published"
        }
    return None

async def create_post_in_database(user_id: str, content: str, hashtags: List[str] = None) -> Dict[str, Any]:
    """Mock function to create post in database"""
    await asyncio.sleep(0.05)
    
    # Simulate database error for specific content
    if "error" in content.lower():
        raise DatabaseConnectionError("Failed to create post in database")
    
    # Simulate retryable network error
    if "network_error" in content.lower():
        raise RetryableError(
            message="Network connection failed",
            error_code="NETWORK_CONNECTION_ERROR",
            max_retries=3
        )
    
    return {
        "id": f"post_{hash(content) % 10000}",
        "user_id": user_id,
        "content": content,
        "hashtags": hashtags or [],
        "created_at": datetime.now(),
        "status": "published"
    }

async def is_duplicate_content(content: str, user_id: str) -> bool:
    """Mock function to check for duplicate content"""
    await asyncio.sleep(0.01)
    return "duplicate" in content.lower()

async def check_rate_limit(user_id: str, action: str) -> bool:
    """Mock function to check rate limit"""
    await asyncio.sleep(0.01)
    return user_id != "rate_limited_user"

# ============================================================================
# SERVICE IMPLEMENTATIONS
# ============================================================================

class PostService:
    """Service class demonstrating custom error types"""
    
    @staticmethod
    async def create_post_with_custom_errors(user_id: str, content: str, hashtags: List[str] = None) -> Dict[str, Any]:
        """
        Create a new post using custom error types.
        
        Demonstrates structured error handling with custom error types.
        """
        context = ErrorContext(
            user_id=user_id,
            operation="create_post",
            additional_data={"content_length": len(content) if content else 0}
        )
        
        try:
            # ============================================================================
            # VALIDATION PHASE (Using error factory)
            # ============================================================================
            
            # Input validation
            if not user_id:
                raise ErrorFactory.missing_parameter("user_id", context)
            
            if not content:
                raise ErrorFactory.missing_parameter("content", context)
            
            # Content validation
            if len(content) < 10:
                raise ErrorFactory.content_too_short(10, context)
            
            if len(content) > 3000:
                raise ErrorFactory.content_too_long(3000, context)
            
            # User validation
            user = await get_user_by_id(user_id)
            if not user:
                raise ErrorFactory.user_not_found(user_id, context)
            
            if not user["is_active"]:
                raise ErrorFactory.account_deactivated(user_id, context)
            
            # Business rule validation
            if await is_duplicate_content(content, user_id):
                raise ErrorFactory.duplicate_content(user_id, context)
            
            if not await check_rate_limit(user_id, "post_creation"):
                raise ErrorFactory.rate_limit_exceeded(user_id, "post_creation", context)
            
            # ============================================================================
            # MAIN BUSINESS LOGIC
            # ============================================================================
            
            post = await create_post_in_database(user_id, content, hashtags)
            
            logger.info(f"Post created successfully | User: {user_id} | Post ID: {post['id']}")
            return {"status": "success", "post_id": post["id"]}
            
        except BaseLinkedInError as e:
            return {"status": "failed", "error": e.to_dict()}
        except DatabaseConnectionError as e:
            db_error = ErrorFactory.database_connection_error("create_post", context)
            return {"status": "failed", "error": db_error.to_dict()}
        except Exception as e:
            system_error = ErrorFactory.unexpected_error("create_post", str(e), context)
            return {"status": "failed", "error": system_error.to_dict()}
    
    @staticmethod
    async def update_post_with_custom_errors(post_id: str, user_id: str, new_content: str) -> Dict[str, Any]:
        """
        Update post using custom error types.
        
        Demonstrates error handling for update operations.
        """
        context = ErrorContext(
            user_id=user_id,
            operation="update_post",
            resource_id=post_id,
            additional_data={"content_length": len(new_content) if new_content else 0}
        )
        
        try:
            # Input validation
            if not post_id:
                raise ErrorFactory.missing_parameter("post_id", context)
            
            if not user_id:
                raise ErrorFactory.missing_parameter("user_id", context)
            
            if not new_content:
                raise ErrorFactory.missing_parameter("new_content", context)
            
            # Content validation
            if len(new_content) < 10:
                raise ErrorFactory.content_too_short(10, context)
            
            if len(new_content) > 3000:
                raise ErrorFactory.content_too_long(3000, context)
            
            # Post validation
            post = await get_post_by_id(post_id)
            if not post:
                raise ErrorFactory.record_not_found("post", post_id, context)
            
            # Authorization
            if post["user_id"] != user_id:
                raise ErrorFactory.unauthorized_access(user_id, post_id, context)
            
            # Update post (simplified for demo)
            logger.info(f"Post updated successfully | User: {user_id} | Post: {post_id}")
            return {"status": "success", "message": "Post updated successfully"}
            
        except BaseLinkedInError as e:
            return {"status": "failed", "error": e.to_dict()}
        except Exception as e:
            system_error = ErrorFactory.unexpected_error("update_post", str(e), context)
            return {"status": "failed", "error": system_error.to_dict()}
    
    @staticmethod
    async def get_user_posts_with_custom_errors(user_id: str, requester_id: str) -> Dict[str, Any]:
        """
        Get user posts using custom error types.
        
        Demonstrates error handling for read operations.
        """
        context = ErrorContext(
            user_id=user_id,
            operation="get_user_posts",
            additional_data={"requester_id": requester_id}
        )
        
        try:
            # Input validation
            if not requester_id:
                raise ErrorFactory.missing_parameter("requester_id", context)
            
            if not user_id:
                raise ErrorFactory.missing_parameter("user_id", context)
            
            # Authentication
            requester = await get_user_by_id(requester_id)
            if not requester:
                raise ErrorFactory.user_not_found(requester_id, context)
            
            if not requester["is_active"]:
                raise ErrorFactory.account_deactivated(requester_id, context)
            
            # Authorization
            if requester_id != user_id and not requester["posts_public"]:
                raise ErrorFactory.posts_private(user_id, context)
            
            # Get posts (simplified for demo)
            posts = [{"id": "post1", "content": "Sample post"}]
            
            logger.info(f"Posts retrieved successfully | User: {user_id} | Count: {len(posts)}")
            return {"status": "success", "posts": posts, "count": len(posts)}
            
        except BaseLinkedInError as e:
            return {"status": "failed", "error": e.to_dict()}
        except Exception as e:
            system_error = ErrorFactory.unexpected_error("get_user_posts", str(e), context)
            return {"status": "failed", "error": system_error.to_dict()}

class PostServiceWithRetry:
    """Service class demonstrating retry logic with custom errors"""
    
    @staticmethod
    async def create_post_with_retry(user_id: str, content: str) -> Dict[str, Any]:
        """
        Create post with retry logic for network operations.
        
        Demonstrates error recovery and retry mechanisms.
        """
        context = ErrorContext(user_id=user_id, operation="create_post_with_retry")
        
        async def create_post_operation():
            
    """create_post_operation function."""
# Simulate network operation that might fail
            if "network_error" in content.lower():
                raise RetryableError(
                    message="Network connection failed",
                    error_code="NETWORK_CONNECTION_ERROR",
                    max_retries=3,
                    context=context
                )
            
            return await create_post_in_database(user_id, content)
        
        try:
            post = await ErrorRecovery.retry_operation(create_post_operation, context=context)
            return {"status": "success", "post_id": post["id"]}
            
        except BaseLinkedInError as e:
            return {"status": "failed", "error": e.to_dict()}

class PostServiceWithHandler:
    """Service class demonstrating centralized error handling"""
    
    @staticmethod
    async def create_post_with_handler(user_id: str, content: str) -> Dict[str, Any]:
        """
        Create post with centralized error handling.
        
        Demonstrates using the ErrorHandler for consistent error processing.
        """
        context = ErrorContext(user_id=user_id, operation="create_post_with_handler")
        
        try:
            # Input validation
            if not user_id:
                raise ErrorFactory.missing_parameter("user_id", context)
            
            if not content:
                raise ErrorFactory.missing_parameter("content", context)
            
            # User validation
            user = await get_user_by_id(user_id)
            if not user:
                raise ErrorFactory.user_not_found(user_id, context)
            
            # Create post
            post = await create_post_in_database(user_id, content)
            return {"status": "success", "post_id": post["id"]}
            
        except Exception as e:
            return ErrorHandler.handle_error(e, context)

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def demonstrate_custom_error_types():
    """Demonstrate the custom error types pattern with various scenarios"""
    
    post_service = PostService()
    retry_service = PostServiceWithRetry()
    handler_service = PostServiceWithHandler()
    
    print("=" * 80)
    print("CUSTOM ERROR TYPES PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Test 1: Successful post creation
    print("\n1. SUCCESSFUL POST CREATION:")
    result = await post_service.create_post_with_custom_errors(
        user_id="valid_user",
        content="This is a test post with sufficient content length to pass validation.",
        hashtags=["test", "demo"]
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 2: Failed post creation (validation error)
    print("\n2. FAILED POST CREATION (Content too short):")
    result = await post_service.create_post_with_custom_errors(
        user_id="valid_user",
        content="Short",
        hashtags=["test"]
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 3: Failed post creation (user not found)
    print("\n3. FAILED POST CREATION (User not found):")
    result = await post_service.create_post_with_custom_errors(
        user_id="nonexistent_user",
        content="This is a test post with sufficient content length.",
        hashtags=["test"]
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 4: Failed post creation (inactive user)
    print("\n4. FAILED POST CREATION (Inactive user):")
    result = await post_service.create_post_with_custom_errors(
        user_id="inactive_user",
        content="This is a test post with sufficient content length.",
        hashtags=["test"]
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 5: Failed post creation (duplicate content)
    print("\n5. FAILED POST CREATION (Duplicate content):")
    result = await post_service.create_post_with_custom_errors(
        user_id="valid_user",
        content="This post contains duplicate content to trigger business rule error.",
        hashtags=["test"]
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 6: Failed post creation (rate limit)
    print("\n6. FAILED POST CREATION (Rate limit):")
    result = await post_service.create_post_with_custom_errors(
        user_id="rate_limited_user",
        content="This is a test post with sufficient content length.",
        hashtags=["test"]
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 7: Failed post creation (database error)
    print("\n7. FAILED POST CREATION (Database error):")
    result = await post_service.create_post_with_custom_errors(
        user_id="valid_user",
        content="This post contains error to simulate database failure.",
        hashtags=["test"]
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 8: Successful post update
    print("\n8. SUCCESSFUL POST UPDATE:")
    result = await post_service.update_post_with_custom_errors(
        post_id="valid_post",
        user_id="valid_user",
        new_content="This is the updated content with sufficient length to pass all validations."
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 9: Failed post update (unauthorized)
    print("\n9. FAILED POST UPDATE (Unauthorized):")
    result = await post_service.update_post_with_custom_errors(
        post_id="valid_post",
        user_id="private_user",
        new_content="This should fail due to unauthorized access."
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 10: Successful post retrieval
    print("\n10. SUCCESSFUL POST RETRIEVAL:")
    result = await post_service.get_user_posts_with_custom_errors(
        user_id="valid_user",
        requester_id="valid_user"
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 11: Failed post retrieval (private posts)
    print("\n11. FAILED POST RETRIEVAL (Private posts):")
    result = await post_service.get_user_posts_with_custom_errors(
        user_id="private_user",
        requester_id="valid_user"
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 12: Retry logic (successful retry)
    print("\n12. RETRY LOGIC (Successful retry):")
    result = await retry_service.create_post_with_retry(
        user_id="valid_user",
        content="This post contains network_error to trigger retry logic."
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 13: Centralized error handling
    print("\n13. CENTRALIZED ERROR HANDLING:")
    result = await handler_service.create_post_with_handler(
        user_id="valid_user",
        content="This is a test post using centralized error handling."
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 14: Error type hierarchy demonstration
    print("\n14. ERROR TYPE HIERARCHY DEMONSTRATION:")
    
    # Create different types of errors
    validation_error = ErrorFactory.content_too_short(10)
    auth_error = ErrorFactory.user_not_found("test_user")
    business_error = ErrorFactory.duplicate_content("test_user")
    
    print(f"Validation Error Category: {validation_error.category.value}")
    print(f"Authentication Error Category: {auth_error.category.value}")
    print(f"Business Rule Error Category: {business_error.category.value}")
    
    print(f"Validation Error Severity: {validation_error.severity.value}")
    print(f"Authentication Error Severity: {auth_error.severity.value}")
    print(f"Business Rule Error Severity: {business_error.severity.value}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_custom_error_types()) 