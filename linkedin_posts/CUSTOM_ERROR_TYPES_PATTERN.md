# Custom Error Types Pattern: Consistent Error Handling

## Core Principle: Structured Error Types for Consistent Handling

Use custom error types and error factories to create consistent, structured error handling across your application. This creates:
- **Type-safe error handling** with specific error categories
- **Consistent error structure** across all operations
- **Easy error categorization** for monitoring and debugging
- **Centralized error creation** with error factories

## 1. Custom Error Types Hierarchy

### ❌ **Generic Exception Handling (Bad)**
```python
async def create_post_bad(user_id: str, content: str) -> Dict[str, Any]:
    try:
        if not user_id:
            return {"error": "User ID required"}
        
        user = await get_user_by_id(user_id)
        if not user:
            return {"error": "User not found"}
        
        if not user.is_active:
            return {"error": "Account deactivated"}
        
        post = await create_post_in_database(user_id, content)
        return {"status": "success", "post_id": post.id}
        
    except Exception as e:
        return {"error": "Something went wrong"}
```

### ✅ **Custom Error Types (Good)**
```python
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass

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
    def __init__(self, message: str, error_code: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            context=context
        )

class AuthenticationError(BaseLinkedInError):
    """Authentication-related errors"""
    def __init__(self, message: str, error_code: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            context=context
        )

class AuthorizationError(BaseLinkedInError):
    """Authorization-related errors"""
    def __init__(self, message: str, error_code: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            context=context
        )

class BusinessRuleError(BaseLinkedInError):
    """Business rule violation errors"""
    def __init__(self, message: str, error_code: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.BUSINESS_RULE,
            severity=ErrorSeverity.MEDIUM,
            context=context
        )

class DatabaseError(BaseLinkedInError):
    """Database-related errors"""
    def __init__(self, message: str, error_code: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            context=context
        )

class NetworkError(BaseLinkedInError):
    """Network-related errors"""
    def __init__(self, message: str, error_code: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            context=context
        )

class SystemError(BaseLinkedInError):
    """System-level errors"""
    def __init__(self, message: str, error_code: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            context=context
        )

async def create_post_good(user_id: str, content: str) -> Dict[str, Any]:
    """Create post with custom error types"""
    context = ErrorContext(user_id=user_id, operation="create_post")
    
    try:
        # Input validation
        if not user_id:
            raise ValidationError(
                message="User ID is required",
                error_code="MISSING_USER_ID",
                context=context
            )
        
        if not content:
            raise ValidationError(
                message="Content is required",
                error_code="MISSING_CONTENT",
                context=context
            )
        
        # User validation
        user = await get_user_by_id(user_id)
        if not user:
            raise AuthenticationError(
                message=f"User not found: {user_id}",
                error_code="USER_NOT_FOUND",
                context=context
            )
        
        if not user.is_active:
            raise AuthenticationError(
                message=f"Account deactivated: {user_id}",
                error_code="ACCOUNT_DEACTIVATED",
                context=context
            )
        
        # Create post
        post = await create_post_in_database(user_id, content)
        return {"status": "success", "post_id": post.id}
        
    except BaseLinkedInError as e:
        return {"status": "failed", "error": e.to_dict()}
    except Exception as e:
        system_error = SystemError(
            message=f"Unexpected error: {str(e)}",
            error_code="UNKNOWN_ERROR",
            context=context
        )
        return {"status": "failed", "error": system_error.to_dict()}
```

## 2. Error Factory Pattern

### **Error Factory Implementation**
```python
class ErrorFactory:
    """Factory for creating consistent errors"""
    
    # Validation errors
    @staticmethod
    def missing_parameter(param_name: str, context: Optional[ErrorContext] = None) -> ValidationError:
        return ValidationError(
            message=f"{param_name} is required",
            error_code=f"MISSING_{param_name.upper()}",
            context=context
        )
    
    @staticmethod
    def invalid_format(field_name: str, expected_format: str, context: Optional[ErrorContext] = None) -> ValidationError:
        return ValidationError(
            message=f"{field_name} has invalid format. Expected: {expected_format}",
            error_code=f"INVALID_{field_name.upper()}_FORMAT",
            context=context
        )
    
    @staticmethod
    def content_too_short(min_length: int, context: Optional[ErrorContext] = None) -> ValidationError:
        return ValidationError(
            message=f"Content too short (minimum {min_length} characters)",
            error_code="CONTENT_TOO_SHORT",
            context=context
        )
    
    @staticmethod
    def content_too_long(max_length: int, context: Optional[ErrorContext] = None) -> ValidationError:
        return ValidationError(
            message=f"Content too long (maximum {max_length} characters)",
            error_code="CONTENT_TOO_LONG",
            context=context
        )
    
    # Authentication errors
    @staticmethod
    def user_not_found(user_id: str, context: Optional[ErrorContext] = None) -> AuthenticationError:
        return AuthenticationError(
            message=f"User not found: {user_id}",
            error_code="USER_NOT_FOUND",
            context=context
        )
    
    @staticmethod
    def account_deactivated(user_id: str, context: Optional[ErrorContext] = None) -> AuthenticationError:
        return AuthenticationError(
            message=f"Account deactivated: {user_id}",
            error_code="ACCOUNT_DEACTIVATED",
            context=context
        )
    
    # Authorization errors
    @staticmethod
    def unauthorized_access(user_id: str, resource_id: str, context: Optional[ErrorContext] = None) -> AuthorizationError:
        return AuthorizationError(
            message=f"User {user_id} not authorized to access {resource_id}",
            error_code="UNAUTHORIZED_ACCESS",
            context=context
        )
    
    @staticmethod
    def posts_private(user_id: str, context: Optional[ErrorContext] = None) -> AuthorizationError:
        return AuthorizationError(
            message=f"Posts are private for user: {user_id}",
            error_code="POSTS_PRIVATE",
            context=context
        )
    
    # Business rule errors
    @staticmethod
    def daily_limit_exceeded(user_id: str, limit: int, context: Optional[ErrorContext] = None) -> BusinessRuleError:
        return BusinessRuleError(
            message=f"Daily limit exceeded for user {user_id} (limit: {limit})",
            error_code="DAILY_LIMIT_EXCEEDED",
            context=context
        )
    
    @staticmethod
    def rate_limit_exceeded(user_id: str, action: str, context: Optional[ErrorContext] = None) -> BusinessRuleError:
        return BusinessRuleError(
            message=f"Rate limit exceeded for user {user_id} on action {action}",
            error_code="RATE_LIMIT_EXCEEDED",
            context=context
        )
    
    @staticmethod
    def duplicate_content(user_id: str, context: Optional[ErrorContext] = None) -> BusinessRuleError:
        return BusinessRuleError(
            message=f"Duplicate content detected for user {user_id}",
            error_code="DUPLICATE_CONTENT",
            context=context
        )
    
    # Database errors
    @staticmethod
    def database_connection_error(operation: str, context: Optional[ErrorContext] = None) -> DatabaseError:
        return DatabaseError(
            message=f"Database connection failed during {operation}",
            error_code="DATABASE_CONNECTION_ERROR",
            context=context
        )
    
    @staticmethod
    def record_not_found(resource_type: str, resource_id: str, context: Optional[ErrorContext] = None) -> DatabaseError:
        return DatabaseError(
            message=f"{resource_type} not found: {resource_id}",
            error_code=f"{resource_type.upper()}_NOT_FOUND",
            context=context
        )
    
    # Network errors
    @staticmethod
    def external_service_unavailable(service_name: str, context: Optional[ErrorContext] = None) -> NetworkError:
        return NetworkError(
            message=f"External service {service_name} is unavailable",
            error_code="EXTERNAL_SERVICE_UNAVAILABLE",
            context=context
        )
    
    @staticmethod
    def timeout_error(operation: str, timeout_seconds: int, context: Optional[ErrorContext] = None) -> NetworkError:
        return NetworkError(
            message=f"Operation {operation} timed out after {timeout_seconds} seconds",
            error_code="TIMEOUT_ERROR",
            context=context
        )
    
    # System errors
    @staticmethod
    def unexpected_error(operation: str, original_error: str, context: Optional[ErrorContext] = None) -> SystemError:
        return SystemError(
            message=f"Unexpected error during {operation}: {original_error}",
            error_code="UNEXPECTED_ERROR",
            context=context
        )

# Usage with error factory
async def create_post_with_factory(user_id: str, content: str) -> Dict[str, Any]:
    """Create post using error factory"""
    context = ErrorContext(user_id=user_id, operation="create_post")
    
    try:
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
        
        if not user.is_active:
            raise ErrorFactory.account_deactivated(user_id, context)
        
        # Business rule validation
        if await is_duplicate_content(content, user_id):
            raise ErrorFactory.duplicate_content(user_id, context)
        
        # Create post
        post = await create_post_in_database(user_id, content)
        return {"status": "success", "post_id": post.id}
        
    except BaseLinkedInError as e:
        return {"status": "failed", "error": e.to_dict()}
    except Exception as e:
        system_error = ErrorFactory.unexpected_error("create_post", str(e), context)
        return {"status": "failed", "error": system_error.to_dict()}
```

## 3. Error Handler Pattern

### **Centralized Error Handler**
```python
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
                "message": "Please check your input and try again",
                "category": "validation",
                "details": error.message
            }
        }
    
    @staticmethod
    def _handle_database_error(error: DatabaseError) -> Dict[str, Any]:
        """Handle database errors"""
        logger.error(f"Database Error: {error.error_code} - {error.message}")
        
        return {
            "status": "failed",
            "error": {
                "code": "DATABASE_ERROR",
                "message": "We're experiencing technical difficulties. Please try again later.",
                "category": "database"
            }
        }
    
    @staticmethod
    def _handle_network_error(error: NetworkError) -> Dict[str, Any]:
        """Handle network errors"""
        logger.error(f"Network Error: {error.error_code} - {error.message}")
        
        return {
            "status": "failed",
            "error": {
                "code": "NETWORK_ERROR",
                "message": "Connection issue detected. Please check your internet and try again.",
                "category": "network"
            }
        }
    
    @staticmethod
    def _handle_generic_error(error: Exception, context: Optional[ErrorContext] = None) -> Dict[str, Any]:
        """Handle generic exceptions"""
        logger.error(f"Unexpected Error: {type(error).__name__} - {str(error)}")
        
        return {
            "status": "failed",
            "error": {
                "code": "UNKNOWN_ERROR",
                "message": "Something unexpected happened. Please try again or contact support.",
                "category": "system"
            }
        }

# Usage with error handler
async def create_post_with_handler(user_id: str, content: str) -> Dict[str, Any]:
    """Create post with centralized error handling"""
    context = ErrorContext(user_id=user_id, operation="create_post")
    
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
        return {"status": "success", "post_id": post.id}
        
    except Exception as e:
        return ErrorHandler.handle_error(e, context)
```

## 4. Error Mapping and Translation

### **Error Mapping System**
```python
class ErrorMapper:
    """Map technical errors to user-friendly messages"""
    
    ERROR_MESSAGES = {
        # Validation errors
        "MISSING_USER_ID": "Please provide your user ID",
        "MISSING_CONTENT": "Please provide post content",
        "CONTENT_TOO_SHORT": "Your post is too short. Please add more content (minimum 10 characters).",
        "CONTENT_TOO_LONG": "Your post is too long. Please shorten it (maximum 3000 characters).",
        
        # Authentication errors
        "USER_NOT_FOUND": "We couldn't find your account. Please check your login and try again.",
        "ACCOUNT_DEACTIVATED": "Your account has been deactivated. Please contact support for assistance.",
        
        # Authorization errors
        "UNAUTHORIZED_ACCESS": "You don't have permission to perform this action.",
        "POSTS_PRIVATE": "This content is private and not available for viewing.",
        
        # Business rule errors
        "DAILY_LIMIT_EXCEEDED": "You've reached your daily post limit. Please try again tomorrow.",
        "RATE_LIMIT_EXCEEDED": "You're posting too quickly. Please wait a moment before trying again.",
        "DUPLICATE_CONTENT": "This content appears to be a duplicate. Please create unique content.",
        
        # System errors
        "DATABASE_ERROR": "We're experiencing technical difficulties. Please try again in a few minutes.",
        "NETWORK_ERROR": "Connection issue detected. Please check your internet and try again.",
        "UNKNOWN_ERROR": "Something unexpected happened. Please try again or contact support if the problem persists."
    }
    
    @staticmethod
    def get_user_message(error_code: str, default: str = None) -> str:
        """Get user-friendly message for error code"""
        return ErrorMapper.ERROR_MESSAGES.get(error_code, default or "An error occurred. Please try again.")
    
    @staticmethod
    def map_error(error: BaseLinkedInError) -> Dict[str, Any]:
        """Map error to user-friendly response"""
        return {
            "status": "failed",
            "error": {
                "code": error.error_code,
                "message": ErrorMapper.get_user_message(error.error_code),
                "category": error.category.value,
                "severity": error.severity.value,
                "timestamp": error.timestamp.isoformat(),
                "context": {
                    "operation": error.context.operation,
                    "user_id": error.context.user_id
                }
            }
        }

# Enhanced error factory with user-friendly messages
class UserFriendlyErrorFactory:
    """Error factory that creates errors with user-friendly messages"""
    
    @staticmethod
    def missing_parameter(param_name: str, context: Optional[ErrorContext] = None) -> ValidationError:
        user_message = ErrorMapper.get_user_message(f"MISSING_{param_name.upper()}")
        return ValidationError(
            message=f"{param_name} is required",
            error_code=f"MISSING_{param_name.upper()}",
            context=context,
            user_message=user_message
        )
    
    @staticmethod
    def content_too_short(min_length: int, context: Optional[ErrorContext] = None) -> ValidationError:
        user_message = ErrorMapper.get_user_message("CONTENT_TOO_SHORT")
        return ValidationError(
            message=f"Content too short (minimum {min_length} characters)",
            error_code="CONTENT_TOO_SHORT",
            context=context,
            user_message=user_message
        )
    
    @staticmethod
    def user_not_found(user_id: str, context: Optional[ErrorContext] = None) -> AuthenticationError:
        user_message = ErrorMapper.get_user_message("USER_NOT_FOUND")
        return AuthenticationError(
            message=f"User not found: {user_id}",
            error_code="USER_NOT_FOUND",
            context=context,
            user_message=user_message
        )
```

## 5. Error Recovery and Retry Logic

### **Error Recovery with Custom Types**
```python
class RetryableError(BaseLinkedInError):
    """Errors that can be retried"""
    def __init__(self, message: str, error_code: str, max_retries: int = 3, context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            context=context
        )
        self.max_retries = max_retries

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

# Usage with retry logic
async def create_post_with_retry(user_id: str, content: str) -> Dict[str, Any]:
    """Create post with retry logic for network operations"""
    context = ErrorContext(user_id=user_id, operation="create_post")
    
    async def create_post_operation():
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
        return {"status": "success", "post_id": post.id}
        
    except BaseLinkedInError as e:
        return {"status": "failed", "error": e.to_dict()}
```

## 6. Function Structure Template

### **Standard Custom Error Structure**
```python
async def function_with_custom_errors(param1: str, param2: str) -> Dict[str, Any]:
    """Function using custom error types for consistent handling"""
    context = ErrorContext(
        user_id=param1 if param1 else None,
        operation="function_name",
        additional_data={"param2": param2}
    )
    
    try:
        # ============================================================================
        # VALIDATION PHASE (Using error factory)
        # ============================================================================
        
        if not param1:
            raise UserFriendlyErrorFactory.missing_parameter("param1", context)
        
        if not param2:
            raise UserFriendlyErrorFactory.missing_parameter("param2", context)
        
        # ============================================================================
        # BUSINESS LOGIC PHASE
        # ============================================================================
        
        result = await perform_operation(param1, param2)
        
        return {"status": "success", "data": result}
        
    except BaseLinkedInError as e:
        # Handle LinkedIn-specific errors
        return ErrorMapper.map_error(e)
        
    except Exception as e:
        # Handle unexpected errors
        system_error = ErrorFactory.unexpected_error("function_name", str(e), context)
        return ErrorMapper.map_error(system_error)
```

## 7. Benefits of Custom Error Types

### **Type Safety**
- **Specific error types** for different categories
- **Compile-time error checking** for error handling
- **Clear error hierarchy** for inheritance

### **Consistency**
- **Standardized error structure** across the application
- **Consistent error codes** and messages
- **Uniform error handling** patterns

### **Maintainability**
- **Centralized error creation** with factories
- **Easy error categorization** for monitoring
- **Simple error mapping** to user messages

### **Debugging**
- **Rich error context** for troubleshooting
- **Error categorization** for filtering
- **Structured error logging** for analysis

## 8. When to Use Custom Error Types

### **✅ Use For:**
- Applications with multiple error categories
- Systems requiring consistent error handling
- APIs that need structured error responses
- Applications with complex business logic
- Systems requiring error monitoring and alerting

### **❌ Avoid For:**
- Simple scripts or utilities
- Applications with minimal error handling needs
- Performance-critical systems where overhead matters
- Temporary or prototype code

Custom error types transform your error handling into a structured, maintainable system that provides consistent user experience and enables effective debugging and monitoring. 