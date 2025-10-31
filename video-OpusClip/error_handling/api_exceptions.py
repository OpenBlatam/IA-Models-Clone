#!/usr/bin/env python3
"""
API Custom Exceptions for Video-OpusClip
Custom exceptions with user-friendly API message mapping
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass
from http import HTTPStatus

from .custom_exceptions import VideoOpusClipException


# ============================================================================
# API EXCEPTION CATEGORIES
# ============================================================================

class APISeverity(Enum):
    """API error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class APICategory(Enum):
    """API error categories"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RESOURCE = "resource"
    NETWORK = "network"
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"
    EXECUTION = "execution"
    SYSTEM = "system"


@dataclass
class APIMessage:
    """API message structure"""
    title: str
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None
    code: Optional[str] = None
    severity: APISeverity = APISeverity.ERROR
    category: APICategory = APICategory.EXECUTION
    http_status: int = 500
    retry_after: Optional[int] = None
    rate_limit_reset: Optional[int] = None


# ============================================================================
# API CUSTOM EXCEPTIONS
# ============================================================================

class APITimeoutError(VideoOpusClipException):
    """API timeout error with user-friendly message"""
    
    def __init__(
        self,
        operation: str,
        timeout: float,
        target: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        self.operation = operation
        self.timeout = timeout
        self.target = target
        self.suggestion = suggestion or "Try increasing the timeout value or check network connectivity"
        
        super().__init__(
            f"Operation '{operation}' timed out after {timeout} seconds",
            error_code="API_TIMEOUT",
            severity="ERROR",
            details={
                "operation": operation,
                "timeout": timeout,
                "target": target,
                "suggestion": self.suggestion
            }
        )
    
    def to_api_message(self) -> APIMessage:
        """Convert to API message"""
        return APIMessage(
            title="Operation Timeout",
            message=f"The {self.operation} operation timed out after {self.timeout} seconds.",
            details={
                "operation": self.operation,
                "timeout": self.timeout,
                "target": self.target
            },
            suggestion=self.suggestion,
            code=self.error_code,
            severity=APISeverity.ERROR,
            category=APICategory.TIMEOUT,
            http_status=HTTPStatus.REQUEST_TIMEOUT
        )


class APIInvalidTargetError(VideoOpusClipException):
    """API invalid target error with user-friendly message"""
    
    def __init__(
        self,
        target: str,
        reason: str,
        valid_examples: Optional[List[str]] = None,
        suggestion: Optional[str] = None
    ):
        self.target = target
        self.reason = reason
        self.valid_examples = valid_examples or []
        self.suggestion = suggestion or "Please provide a valid target address"
        
        super().__init__(
            f"Invalid target '{target}': {reason}",
            error_code="API_INVALID_TARGET",
            severity="ERROR",
            details={
                "target": target,
                "reason": reason,
                "valid_examples": self.valid_examples,
                "suggestion": self.suggestion
            }
        )
    
    def to_api_message(self) -> APIMessage:
        """Convert to API message"""
        return APIMessage(
            title="Invalid Target",
            message=f"The target '{self.target}' is invalid: {self.reason}",
            details={
                "target": self.target,
                "reason": self.reason,
                "valid_examples": self.valid_examples
            },
            suggestion=self.suggestion,
            code=self.error_code,
            severity=APISeverity.ERROR,
            category=APICategory.VALIDATION,
            http_status=HTTPStatus.BAD_REQUEST
        )


class APIInvalidPortError(VideoOpusClipException):
    """API invalid port error with user-friendly message"""
    
    def __init__(
        self,
        port: Union[int, str],
        reason: str,
        valid_range: Tuple[int, int] = (1, 65535),
        suggestion: Optional[str] = None
    ):
        self.port = port
        self.reason = reason
        self.valid_range = valid_range
        self.suggestion = suggestion or f"Please provide a port number between {valid_range[0]} and {valid_range[1]}"
        
        super().__init__(
            f"Invalid port '{port}': {reason}",
            error_code="API_INVALID_PORT",
            severity="ERROR",
            details={
                "port": port,
                "reason": reason,
                "valid_range": valid_range,
                "suggestion": self.suggestion
            }
        )
    
    def to_api_message(self) -> APIMessage:
        """Convert to API message"""
        return APIMessage(
            title="Invalid Port",
            message=f"The port '{self.port}' is invalid: {self.reason}",
            details={
                "port": self.port,
                "reason": self.reason,
                "valid_range": self.valid_range
            },
            suggestion=self.suggestion,
            code=self.error_code,
            severity=APISeverity.ERROR,
            category=APICategory.VALIDATION,
            http_status=HTTPStatus.BAD_REQUEST
        )


class APIInvalidScanTypeError(VideoOpusClipException):
    """API invalid scan type error with user-friendly message"""
    
    def __init__(
        self,
        scan_type: str,
        valid_types: List[str],
        suggestion: Optional[str] = None
    ):
        self.scan_type = scan_type
        self.valid_types = valid_types
        self.suggestion = suggestion or f"Please choose from: {', '.join(valid_types)}"
        
        super().__init__(
            f"Invalid scan type '{scan_type}'",
            error_code="API_INVALID_SCAN_TYPE",
            severity="ERROR",
            details={
                "scan_type": scan_type,
                "valid_types": valid_types,
                "suggestion": self.suggestion
            }
        )
    
    def to_api_message(self) -> APIMessage:
        """Convert to API message"""
        return APIMessage(
            title="Invalid Scan Type",
            message=f"The scan type '{self.scan_type}' is not supported.",
            details={
                "scan_type": self.scan_type,
                "valid_types": self.valid_types
            },
            suggestion=self.suggestion,
            code=self.error_code,
            severity=APISeverity.ERROR,
            category=APICategory.VALIDATION,
            http_status=HTTPStatus.BAD_REQUEST
        )


class APINetworkConnectionError(VideoOpusClipException):
    """API network connection error with user-friendly message"""
    
    def __init__(
        self,
        target: str,
        reason: str,
        suggestion: Optional[str] = None
    ):
        self.target = target
        self.reason = reason
        self.suggestion = suggestion or "Check your network connection and try again"
        
        super().__init__(
            f"Failed to connect to '{target}': {reason}",
            error_code="API_NETWORK_CONNECTION",
            severity="ERROR",
            details={
                "target": target,
                "reason": reason,
                "suggestion": self.suggestion
            }
        )
    
    def to_api_message(self) -> APIMessage:
        """Convert to API message"""
        return APIMessage(
            title="Network Connection Failed",
            message=f"Unable to connect to '{self.target}': {self.reason}",
            details={
                "target": self.target,
                "reason": self.reason
            },
            suggestion=self.suggestion,
            code=self.error_code,
            severity=APISeverity.ERROR,
            category=APICategory.NETWORK,
            http_status=HTTPStatus.SERVICE_UNAVAILABLE
        )


class APIAuthenticationError(VideoOpusClipException):
    """API authentication error with user-friendly message"""
    
    def __init__(
        self,
        username: Optional[str] = None,
        reason: str = "Invalid credentials",
        suggestion: Optional[str] = None
    ):
        self.username = username
        self.reason = reason
        self.suggestion = suggestion or "Please check your username and password"
        
        super().__init__(
            f"Authentication failed: {reason}",
            error_code="API_AUTHENTICATION",
            severity="ERROR",
            details={
                "username": username,
                "reason": reason,
                "suggestion": self.suggestion
            }
        )
    
    def to_api_message(self) -> APIMessage:
        """Convert to API message"""
        return APIMessage(
            title="Authentication Failed",
            message=f"Authentication failed: {self.reason}",
            details={
                "username": self.username,
                "reason": self.reason
            },
            suggestion=self.suggestion,
            code=self.error_code,
            severity=APISeverity.ERROR,
            category=APICategory.AUTHENTICATION,
            http_status=HTTPStatus.UNAUTHORIZED
        )


class APIAuthorizationError(VideoOpusClipException):
    """API authorization error with user-friendly message"""
    
    def __init__(
        self,
        resource: str,
        action: str,
        reason: str = "Insufficient permissions",
        suggestion: Optional[str] = None
    ):
        self.resource = resource
        self.action = action
        self.reason = reason
        self.suggestion = suggestion or "Contact your administrator for proper permissions"
        
        super().__init__(
            f"Authorization failed for {action} on {resource}: {reason}",
            error_code="API_AUTHORIZATION",
            severity="ERROR",
            details={
                "resource": resource,
                "action": action,
                "reason": reason,
                "suggestion": self.suggestion
            }
        )
    
    def to_api_message(self) -> APIMessage:
        """Convert to API message"""
        return APIMessage(
            title="Access Denied",
            message=f"You don't have permission to {self.action} on {self.resource}: {self.reason}",
            details={
                "resource": self.resource,
                "action": self.action,
                "reason": self.reason
            },
            suggestion=self.suggestion,
            code=self.error_code,
            severity=APISeverity.ERROR,
            category=APICategory.AUTHORIZATION,
            http_status=HTTPStatus.FORBIDDEN
        )


class APIConfigurationError(VideoOpusClipException):
    """API configuration error with user-friendly message"""
    
    def __init__(
        self,
        config_file: Optional[str] = None,
        missing_key: Optional[str] = None,
        reason: str = "Configuration error",
        suggestion: Optional[str] = None
    ):
        self.config_file = config_file
        self.missing_key = missing_key
        self.reason = reason
        self.suggestion = suggestion or "Please check your configuration file"
        
        super().__init__(
            f"Configuration error: {reason}",
            error_code="API_CONFIGURATION",
            severity="ERROR",
            details={
                "config_file": config_file,
                "missing_key": missing_key,
                "reason": reason,
                "suggestion": self.suggestion
            }
        )
    
    def to_api_message(self) -> APIMessage:
        """Convert to API message"""
        return APIMessage(
            title="Configuration Error",
            message=f"Configuration error: {self.reason}",
            details={
                "config_file": self.config_file,
                "missing_key": self.missing_key,
                "reason": self.reason
            },
            suggestion=self.suggestion,
            code=self.error_code,
            severity=APISeverity.ERROR,
            category=APICategory.CONFIGURATION,
            http_status=HTTPStatus.INTERNAL_SERVER_ERROR
        )


class APIResourceError(VideoOpusClipException):
    """API resource error with user-friendly message"""
    
    def __init__(
        self,
        resource_type: str,
        resource_name: str,
        reason: str,
        suggestion: Optional[str] = None
    ):
        self.resource_type = resource_type
        self.resource_name = resource_name
        self.reason = reason
        self.suggestion = suggestion or f"Please check if the {resource_type} exists and is accessible"
        
        super().__init__(
            f"{resource_type} error: {reason}",
            error_code="API_RESOURCE",
            severity="ERROR",
            details={
                "resource_type": resource_type,
                "resource_name": resource_name,
                "reason": reason,
                "suggestion": self.suggestion
            }
        )
    
    def to_api_message(self) -> APIMessage:
        """Convert to API message"""
        return APIMessage(
            title="Resource Error",
            message=f"{self.resource_type} error for '{self.resource_name}': {self.reason}",
            details={
                "resource_type": self.resource_type,
                "resource_name": self.resource_name,
                "reason": self.reason
            },
            suggestion=self.suggestion,
            code=self.error_code,
            severity=APISeverity.ERROR,
            category=APICategory.RESOURCE,
            http_status=HTTPStatus.NOT_FOUND
        )


class APISystemError(VideoOpusClipException):
    """API system error with user-friendly message"""
    
    def __init__(
        self,
        operation: str,
        reason: str,
        suggestion: Optional[str] = None
    ):
        self.operation = operation
        self.reason = reason
        self.suggestion = suggestion or "Please try again or contact support if the problem persists"
        
        super().__init__(
            f"System error during {operation}: {reason}",
            error_code="API_SYSTEM",
            severity="CRITICAL",
            details={
                "operation": operation,
                "reason": reason,
                "suggestion": self.suggestion
            }
        )
    
    def to_api_message(self) -> APIMessage:
        """Convert to API message"""
        return APIMessage(
            title="System Error",
            message=f"A system error occurred during {self.operation}: {self.reason}",
            details={
                "operation": self.operation,
                "reason": self.reason
            },
            suggestion=self.suggestion,
            code=self.error_code,
            severity=APISeverity.CRITICAL,
            category=APICategory.SYSTEM,
            http_status=HTTPStatus.INTERNAL_SERVER_ERROR
        )


class APIValidationError(VideoOpusClipException):
    """API validation error with user-friendly message"""
    
    def __init__(
        self,
        field: str,
        value: Any,
        reason: str,
        valid_examples: Optional[List[Any]] = None,
        suggestion: Optional[str] = None
    ):
        self.field = field
        self.value = value
        self.reason = reason
        self.valid_examples = valid_examples or []
        self.suggestion = suggestion or f"Please provide a valid value for {field}"
        
        super().__init__(
            f"Validation error for {field}: {reason}",
            error_code="API_VALIDATION",
            severity="ERROR",
            details={
                "field": field,
                "value": value,
                "reason": reason,
                "valid_examples": self.valid_examples,
                "suggestion": self.suggestion
            }
        )
    
    def to_api_message(self) -> APIMessage:
        """Convert to API message"""
        return APIMessage(
            title="Validation Error",
            message=f"Invalid value for '{self.field}': {self.reason}",
            details={
                "field": self.field,
                "value": self.value,
                "reason": self.reason,
                "valid_examples": self.valid_examples
            },
            suggestion=self.suggestion,
            code=self.error_code,
            severity=APISeverity.ERROR,
            category=APICategory.VALIDATION,
            http_status=HTTPStatus.BAD_REQUEST
        )


class APIRateLimitError(VideoOpusClipException):
    """API rate limit error with user-friendly message"""
    
    def __init__(
        self,
        limit: int,
        window: int,
        retry_after: Optional[int] = None,
        suggestion: Optional[str] = None
    ):
        self.limit = limit
        self.window = window
        self.retry_after = retry_after
        self.suggestion = suggestion or f"Please wait {retry_after} seconds before trying again"
        
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window} seconds",
            error_code="API_RATE_LIMIT",
            severity="WARNING",
            details={
                "limit": limit,
                "window": window,
                "retry_after": retry_after,
                "suggestion": self.suggestion
            }
        )
    
    def to_api_message(self) -> APIMessage:
        """Convert to API message"""
        return APIMessage(
            title="Rate Limit Exceeded",
            message=f"Rate limit exceeded: {self.limit} requests per {self.window} seconds",
            details={
                "limit": self.limit,
                "window": self.window,
                "retry_after": self.retry_after
            },
            suggestion=self.suggestion,
            code=self.error_code,
            severity=APISeverity.WARNING,
            category=APICategory.AUTHORIZATION,
            http_status=HTTPStatus.TOO_MANY_REQUESTS,
            retry_after=self.retry_after
        )


class APIResourceNotFoundError(VideoOpusClipException):
    """API resource not found error with user-friendly message"""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        suggestion: Optional[str] = None
    ):
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.suggestion = suggestion or f"Please check if the {resource_type} exists"
        
        super().__init__(
            f"{resource_type} with ID '{resource_id}' not found",
            error_code="API_RESOURCE_NOT_FOUND",
            severity="ERROR",
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
                "suggestion": self.suggestion
            }
        )
    
    def to_api_message(self) -> APIMessage:
        """Convert to API message"""
        return APIMessage(
            title="Resource Not Found",
            message=f"{self.resource_type} with ID '{self.resource_id}' not found",
            details={
                "resource_type": self.resource_type,
                "resource_id": self.resource_id
            },
            suggestion=self.suggestion,
            code=self.error_code,
            severity=APISeverity.ERROR,
            category=APICategory.RESOURCE,
            http_status=HTTPStatus.NOT_FOUND
        )


class APIInvalidRequestError(VideoOpusClipException):
    """API invalid request error with user-friendly message"""
    
    def __init__(
        self,
        field: str,
        reason: str,
        suggestion: Optional[str] = None
    ):
        self.field = field
        self.reason = reason
        self.suggestion = suggestion or "Please check your request format"
        
        super().__init__(
            f"Invalid request field '{field}': {reason}",
            error_code="API_INVALID_REQUEST",
            severity="ERROR",
            details={
                "field": field,
                "reason": reason,
                "suggestion": self.suggestion
            }
        )
    
    def to_api_message(self) -> APIMessage:
        """Convert to API message"""
        return APIMessage(
            title="Invalid Request",
            message=f"Invalid request field '{self.field}': {self.reason}",
            details={
                "field": self.field,
                "reason": self.reason
            },
            suggestion=self.suggestion,
            code=self.error_code,
            severity=APISeverity.ERROR,
            category=APICategory.VALIDATION,
            http_status=HTTPStatus.BAD_REQUEST
        )


# ============================================================================
# API MESSAGE SERIALIZERS
# ============================================================================

class APIMessageSerializer:
    """Serializer for API messages with different output formats"""
    
    @staticmethod
    def to_json(api_message: APIMessage) -> Dict[str, Any]:
        """Convert API message to JSON format"""
        response = {
            "error": {
                "title": api_message.title,
                "message": api_message.message,
                "code": api_message.code,
                "severity": api_message.severity.value,
                "category": api_message.category.value,
                "http_status": api_message.http_status
            }
        }
        
        if api_message.details:
            response["error"]["details"] = api_message.details
        
        if api_message.suggestion:
            response["error"]["suggestion"] = api_message.suggestion
        
        if api_message.retry_after:
            response["error"]["retry_after"] = api_message.retry_after
        
        if api_message.rate_limit_reset:
            response["error"]["rate_limit_reset"] = api_message.rate_limit_reset
        
        return response
    
    @staticmethod
    def to_fastapi_response(api_message: APIMessage) -> Dict[str, Any]:
        """Convert API message to FastAPI response format"""
        response = {
            "detail": api_message.message,
            "error_code": api_message.code,
            "severity": api_message.severity.value,
            "category": api_message.category.value
        }
        
        if api_message.details:
            response["details"] = api_message.details
        
        if api_message.suggestion:
            response["suggestion"] = api_message.suggestion
        
        return response
    
    @staticmethod
    def to_rest_response(api_message: APIMessage) -> Dict[str, Any]:
        """Convert API message to REST API response format"""
        response = {
            "success": False,
            "error": {
                "title": api_message.title,
                "message": api_message.message,
                "code": api_message.code
            },
            "timestamp": None,  # Will be set by caller
            "request_id": None  # Will be set by caller
        }
        
        if api_message.details:
            response["error"]["details"] = api_message.details
        
        if api_message.suggestion:
            response["error"]["suggestion"] = api_message.suggestion
        
        return response


# ============================================================================
# API EXCEPTION HANDLER
# ============================================================================

class APIExceptionHandler:
    """Handler for API exceptions with user-friendly message mapping"""
    
    def __init__(self, logger=None):
        self.logger = logger
    
    def handle_exception(self, exception: Exception) -> Tuple[Dict[str, Any], int]:
        """
        Handle exception and return API response
        
        Args:
            exception: Exception to handle
            
        Returns:
            Tuple of (response_dict, http_status_code)
        """
        # Log the exception if logger is available
        if self.logger:
            self.logger.error(
                f"API exception occurred",
                error=exception,
                tags=['api', 'exception', 'handled']
            )
        
        # Convert to API message
        api_message = self._exception_to_api_message(exception)
        
        # Serialize to JSON response
        response = APIMessageSerializer.to_json(api_message)
        
        return response, api_message.http_status
    
    def _exception_to_api_message(self, exception: Exception) -> APIMessage:
        """Convert exception to API message"""
        # Handle custom API exceptions
        if isinstance(exception, (
            APITimeoutError, APIInvalidTargetError, APIInvalidPortError,
            APIInvalidScanTypeError, APINetworkConnectionError,
            APIAuthenticationError, APIAuthorizationError,
            APIConfigurationError, APIResourceError, APISystemError,
            APIValidationError, APIRateLimitError, APIResourceNotFoundError,
            APIInvalidRequestError
        )):
            return exception.to_api_message()
        
        # Handle VideoOpusClipException
        if isinstance(exception, VideoOpusClipException):
            return APIMessage(
                title="Application Error",
                message=str(exception),
                details=getattr(exception, 'details', {}),
                suggestion="Please check the error details and try again",
                code=getattr(exception, 'error_code', 'UNKNOWN'),
                severity=APISeverity.ERROR,
                category=APICategory.EXECUTION,
                http_status=HTTPStatus.INTERNAL_SERVER_ERROR
            )
        
        # Handle standard exceptions
        if isinstance(exception, ValueError):
            return APIMessage(
                title="Invalid Value",
                message=f"Invalid value: {str(exception)}",
                suggestion="Please check your input and try again",
                code="API_INVALID_VALUE",
                severity=APISeverity.ERROR,
                category=APICategory.VALIDATION,
                http_status=HTTPStatus.BAD_REQUEST
            )
        
        if isinstance(exception, TypeError):
            return APIMessage(
                title="Type Error",
                message=f"Type error: {str(exception)}",
                suggestion="Please check your input format and try again",
                code="API_TYPE_ERROR",
                severity=APISeverity.ERROR,
                category=APICategory.VALIDATION,
                http_status=HTTPStatus.BAD_REQUEST
            )
        
        if isinstance(exception, FileNotFoundError):
            return APIMessage(
                title="File Not Found",
                message=f"The file '{exception.filename}' was not found",
                suggestion="Please check the file path and try again",
                code="API_FILE_NOT_FOUND",
                severity=APISeverity.ERROR,
                category=APICategory.RESOURCE,
                http_status=HTTPStatus.NOT_FOUND
            )
        
        if isinstance(exception, PermissionError):
            return APIMessage(
                title="Permission Denied",
                message=f"Permission denied: {exception.filename}",
                suggestion="Please check file permissions or run with appropriate privileges",
                code="API_PERMISSION_DENIED",
                severity=APISeverity.ERROR,
                category=APICategory.AUTHORIZATION,
                http_status=HTTPStatus.FORBIDDEN
            )
        
        # Generic exception
        return APIMessage(
            title="Unexpected Error",
            message=f"An unexpected error occurred: {str(exception)}",
            suggestion="Please try again or contact support if the problem persists",
            code="API_UNEXPECTED_ERROR",
            severity=APISeverity.CRITICAL,
            category=APICategory.SYSTEM,
            http_status=HTTPStatus.INTERNAL_SERVER_ERROR
        )


# ============================================================================
# FASTAPI INTEGRATION
# ============================================================================

def create_fastapi_exception_handler(logger=None):
    """Create FastAPI exception handler"""
    handler = APIExceptionHandler(logger)
    
    def fastapi_exception_handler(request, exc):
        """FastAPI exception handler"""
        response, status_code = handler.handle_exception(exc)
        return response, status_code
    
    return fastapi_exception_handler


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage of API exceptions
    print("🌐 API Exceptions Example")
    
    # Create exception handler
    handler = APIExceptionHandler()
    
    # Test different exception types
    exceptions_to_test = [
        APITimeoutError("port scan", 30.0, "192.168.1.100"),
        APIInvalidTargetError("invalid-target", "Not a valid IP address or domain", ["192.168.1.100", "example.com"]),
        APIInvalidPortError(99999, "Port number out of range"),
        APIInvalidScanTypeError("invalid_scan", ["port_scan", "vulnerability_scan", "web_scan"]),
        APINetworkConnectionError("192.168.1.100", "Connection refused"),
        APIAuthenticationError("admin", "Invalid password"),
        APIAuthorizationError("/admin", "read", "Admin access required"),
        APIConfigurationError("config.json", "api_key", "Missing API key"),
        APIResourceError("file", "output.txt", "Cannot create file"),
        APISystemError("scan", "Out of memory"),
        APIValidationError("email", "invalid-email", "Invalid email format", ["user@example.com", "admin@domain.org"]),
        APIRateLimitError(100, 3600, 60),
        APIResourceNotFoundError("scan", "scan_123"),
        APIInvalidRequestError("target", "Missing required field")
    ]
    
    print("\n" + "="*60)
    print("API EXCEPTION EXAMPLES")
    print("="*60)
    
    for exception in exceptions_to_test:
        print(f"\n{'-'*40}")
        response, status_code = handler.handle_exception(exception)
        print(f"Status Code: {status_code}")
        print(f"Response: {response}")
    
    # Test serialization formats
    print(f"\n{'-'*40}")
    print("SERIALIZATION FORMATS")
    print(f"{'-'*40}")
    
    example_exception = APIInvalidTargetError("test-target", "Invalid format", ["192.168.1.100"])
    api_message = example_exception.to_api_message()
    
    print("JSON Format:")
    print(APIMessageSerializer.to_json(api_message))
    
    print("\nFastAPI Format:")
    print(APIMessageSerializer.to_fastapi_response(api_message))
    
    print("\nREST Format:")
    print(APIMessageSerializer.to_rest_response(api_message))
    
    print("\n✅ API exception examples completed!") 