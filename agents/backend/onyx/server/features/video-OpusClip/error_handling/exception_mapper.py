#!/usr/bin/env python3
"""
Exception Mapper for Video-OpusClip
Maps custom exceptions to user-friendly CLI/API messages
"""

import sys
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Type
from enum import Enum
from dataclasses import dataclass
from http import HTTPStatus

from .custom_exceptions import VideoOpusClipException
from .cli_exceptions import (
    CLIMessage, CLISeverity, CLICategory,
    CLITimeoutError, CLIInvalidTargetError, CLIInvalidPortError,
    CLIInvalidScanTypeError, CLINetworkConnectionError,
    CLIAuthenticationError, CLIAuthorizationError,
    CLIConfigurationError, CLIResourceError, CLISystemError,
    CLIValidationError
)
from .api_exceptions import (
    APIMessage, APISeverity, APICategory,
    APITimeoutError, APIInvalidTargetError, APIInvalidPortError,
    APIInvalidScanTypeError, APINetworkConnectionError,
    APIAuthenticationError, APIAuthorizationError,
    APIConfigurationError, APIResourceError, APISystemError,
    APIValidationError, APIRateLimitError, APIResourceNotFoundError,
    APIInvalidRequestError
)


# ============================================================================
# EXCEPTION MAPPING CONFIGURATION
# ============================================================================

class OutputFormat(Enum):
    """Output format types"""
    CLI = "cli"
    API = "api"
    JSON = "json"
    TEXT = "text"


@dataclass
class ExceptionMapping:
    """Exception mapping configuration"""
    exception_type: Type[Exception]
    cli_message: Optional[CLIMessage] = None
    api_message: Optional[APIMessage] = None
    exit_code: int = 1
    http_status: int = 500
    log_level: str = "ERROR"
    retryable: bool = False
    user_friendly: bool = True


# ============================================================================
# EXCEPTION MAPPER CLASS
# ============================================================================

class ExceptionMapper:
    """Maps exceptions to user-friendly messages for different output formats"""
    
    def __init__(self):
        self.mappings: Dict[Type[Exception], ExceptionMapping] = {}
        self.logger = None
        self._setup_default_mappings()
    
    def set_logger(self, logger):
        """Set logger for error logging"""
        self.logger = logger
    
    def _setup_default_mappings(self):
        """Setup default exception mappings"""
        # Timeout exceptions
        self.add_mapping(
            TimeoutError,
            cli_message=CLIMessage(
                title="⏰ Operation Timed Out",
                message="The operation timed out",
                suggestion="Try increasing the timeout value or check network connectivity",
                code="TIMEOUT_ERROR",
                severity=CLISeverity.ERROR,
                category=CLICategory.TIMEOUT,
                exit_code=124
            ),
            api_message=APIMessage(
                title="Operation Timeout",
                message="The operation timed out",
                suggestion="Try increasing the timeout value or check network connectivity",
                code="TIMEOUT_ERROR",
                severity=APISeverity.ERROR,
                category=APICategory.TIMEOUT,
                http_status=HTTPStatus.REQUEST_TIMEOUT
            ),
            exit_code=124,
            http_status=HTTPStatus.REQUEST_TIMEOUT,
            retryable=True
        )
        
        # Connection exceptions
        self.add_mapping(
            ConnectionError,
            cli_message=CLIMessage(
                title="🌐 Connection Failed",
                message="Failed to establish connection",
                suggestion="Check your network connection and try again",
                code="CONNECTION_ERROR",
                severity=CLISeverity.ERROR,
                category=CLICategory.NETWORK,
                exit_code=101
            ),
            api_message=APIMessage(
                title="Connection Failed",
                message="Failed to establish connection",
                suggestion="Check your network connection and try again",
                code="CONNECTION_ERROR",
                severity=APISeverity.ERROR,
                category=APICategory.NETWORK,
                http_status=HTTPStatus.SERVICE_UNAVAILABLE
            ),
            exit_code=101,
            http_status=HTTPStatus.SERVICE_UNAVAILABLE,
            retryable=True
        )
        
        # File not found exceptions
        self.add_mapping(
            FileNotFoundError,
            cli_message=CLIMessage(
                title="📁 File Not Found",
                message="The specified file was not found",
                suggestion="Please check the file path and try again",
                code="FILE_NOT_FOUND",
                severity=CLISeverity.ERROR,
                category=CLICategory.RESOURCE,
                exit_code=2
            ),
            api_message=APIMessage(
                title="File Not Found",
                message="The specified file was not found",
                suggestion="Please check the file path and try again",
                code="FILE_NOT_FOUND",
                severity=APISeverity.ERROR,
                category=APICategory.RESOURCE,
                http_status=HTTPStatus.NOT_FOUND
            ),
            exit_code=2,
            http_status=HTTPStatus.NOT_FOUND
        )
        
        # Permission exceptions
        self.add_mapping(
            PermissionError,
            cli_message=CLIMessage(
                title="🚫 Permission Denied",
                message="Permission denied for the requested operation",
                suggestion="Please check file permissions or run with appropriate privileges",
                code="PERMISSION_DENIED",
                severity=CLISeverity.ERROR,
                category=CLICategory.AUTHORIZATION,
                exit_code=126
            ),
            api_message=APIMessage(
                title="Permission Denied",
                message="Permission denied for the requested operation",
                suggestion="Please check file permissions or run with appropriate privileges",
                code="PERMISSION_DENIED",
                severity=APISeverity.ERROR,
                category=APICategory.AUTHORIZATION,
                http_status=HTTPStatus.FORBIDDEN
            ),
            exit_code=126,
            http_status=HTTPStatus.FORBIDDEN
        )
        
        # Value exceptions
        self.add_mapping(
            ValueError,
            cli_message=CLIMessage(
                title="✅ Invalid Value",
                message="An invalid value was provided",
                suggestion="Please check your input and try again",
                code="INVALID_VALUE",
                severity=CLISeverity.ERROR,
                category=CLICategory.INPUT,
                exit_code=22
            ),
            api_message=APIMessage(
                title="Invalid Value",
                message="An invalid value was provided",
                suggestion="Please check your input and try again",
                code="INVALID_VALUE",
                severity=APISeverity.ERROR,
                category=APICategory.VALIDATION,
                http_status=HTTPStatus.BAD_REQUEST
            ),
            exit_code=22,
            http_status=HTTPStatus.BAD_REQUEST
        )
        
        # Type exceptions
        self.add_mapping(
            TypeError,
            cli_message=CLIMessage(
                title="🔧 Type Error",
                message="A type error occurred",
                suggestion="Please check your input format and try again",
                code="TYPE_ERROR",
                severity=CLISeverity.ERROR,
                category=CLICategory.INPUT,
                exit_code=22
            ),
            api_message=APIMessage(
                title="Type Error",
                message="A type error occurred",
                suggestion="Please check your input format and try again",
                code="TYPE_ERROR",
                severity=APISeverity.ERROR,
                category=APICategory.VALIDATION,
                http_status=HTTPStatus.BAD_REQUEST
            ),
            exit_code=22,
            http_status=HTTPStatus.BAD_REQUEST
        )
        
        # Keyboard interrupt
        self.add_mapping(
            KeyboardInterrupt,
            cli_message=CLIMessage(
                title="⏹️ Operation Cancelled",
                message="The operation was cancelled by the user",
                suggestion="Run the command again to retry",
                code="OPERATION_CANCELLED",
                severity=CLISeverity.INFO,
                category=CLICategory.EXECUTION,
                exit_code=130,
                show_help=False
            ),
            api_message=APIMessage(
                title="Operation Cancelled",
                message="The operation was cancelled by the user",
                suggestion="Run the command again to retry",
                code="OPERATION_CANCELLED",
                severity=APISeverity.INFO,
                category=APICategory.EXECUTION,
                http_status=HTTPStatus.REQUEST_TIMEOUT
            ),
            exit_code=130,
            http_status=HTTPStatus.REQUEST_TIMEOUT
        )
    
    def add_mapping(self, exception_type: Type[Exception], **kwargs):
        """Add exception mapping"""
        mapping = ExceptionMapping(exception_type=exception_type, **kwargs)
        self.mappings[exception_type] = mapping
    
    def get_mapping(self, exception: Exception) -> Optional[ExceptionMapping]:
        """Get mapping for exception"""
        # Check exact type match
        if type(exception) in self.mappings:
            return self.mappings[type(exception)]
        
        # Check parent class matches
        for exception_type, mapping in self.mappings.items():
            if isinstance(exception, exception_type):
                return mapping
        
        return None
    
    def map_exception(
        self,
        exception: Exception,
        output_format: OutputFormat = OutputFormat.CLI,
        context: Optional[Dict[str, Any]] = None
    ) -> Union[CLIMessage, APIMessage, Dict[str, Any]]:
        """
        Map exception to user-friendly message
        
        Args:
            exception: Exception to map
            output_format: Desired output format
            context: Additional context information
            
        Returns:
            Mapped message in requested format
        """
        # Log the exception if logger is available
        if self.logger:
            self.logger.error(
                f"Exception mapping requested",
                error=exception,
                output_format=output_format.value,
                context=context,
                tags=['exception_mapper', 'mapping']
            )
        
        # Handle custom CLI/API exceptions
        if hasattr(exception, 'to_cli_message') and output_format == OutputFormat.CLI:
            return exception.to_cli_message()
        
        if hasattr(exception, 'to_api_message') and output_format == OutputFormat.API:
            return exception.to_api_message()
        
        # Get mapping
        mapping = self.get_mapping(exception)
        
        if mapping is None:
            # Default mapping for unknown exceptions
            if output_format == OutputFormat.CLI:
                return CLIMessage(
                    title="💥 Unexpected Error",
                    message=f"An unexpected error occurred: {str(exception)}",
                    suggestion="Please try again or contact support if the problem persists",
                    code="UNKNOWN_ERROR",
                    severity=CLISeverity.CRITICAL,
                    category=CLICategory.SYSTEM,
                    exit_code=125
                )
            else:
                return APIMessage(
                    title="Unexpected Error",
                    message=f"An unexpected error occurred: {str(exception)}",
                    suggestion="Please try again or contact support if the problem persists",
                    code="UNKNOWN_ERROR",
                    severity=APISeverity.CRITICAL,
                    category=APICategory.SYSTEM,
                    http_status=HTTPStatus.INTERNAL_SERVER_ERROR
                )
        
        # Return appropriate message format
        if output_format == OutputFormat.CLI:
            return mapping.cli_message or self._create_default_cli_message(exception, mapping)
        elif output_format == OutputFormat.API:
            return mapping.api_message or self._create_default_api_message(exception, mapping)
        elif output_format == OutputFormat.JSON:
            return self._to_json_format(exception, mapping, context)
        else:
            return self._to_text_format(exception, mapping, context)
    
    def _create_default_cli_message(self, exception: Exception, mapping: ExceptionMapping) -> CLIMessage:
        """Create default CLI message"""
        return CLIMessage(
            title="🚨 Error",
            message=str(exception),
            suggestion="Please try again or contact support if the problem persists",
            code=getattr(exception, 'error_code', 'UNKNOWN'),
            severity=CLISeverity.ERROR,
            category=CLICategory.EXECUTION,
            exit_code=mapping.exit_code
        )
    
    def _create_default_api_message(self, exception: Exception, mapping: ExceptionMapping) -> APIMessage:
        """Create default API message"""
        return APIMessage(
            title="Error",
            message=str(exception),
            suggestion="Please try again or contact support if the problem persists",
            code=getattr(exception, 'error_code', 'UNKNOWN'),
            severity=APISeverity.ERROR,
            category=APICategory.EXECUTION,
            http_status=mapping.http_status
        )
    
    def _to_json_format(self, exception: Exception, mapping: ExceptionMapping, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert to JSON format"""
        return {
            "error": {
                "type": type(exception).__name__,
                "message": str(exception),
                "code": getattr(exception, 'error_code', 'UNKNOWN'),
                "exit_code": mapping.exit_code,
                "http_status": mapping.http_status,
                "retryable": mapping.retryable,
                "user_friendly": mapping.user_friendly
            },
            "context": context or {}
        }
    
    def _to_text_format(self, exception: Exception, mapping: ExceptionMapping, context: Optional[Dict[str, Any]]) -> str:
        """Convert to text format"""
        lines = [
            f"Error Type: {type(exception).__name__}",
            f"Message: {str(exception)}",
            f"Code: {getattr(exception, 'error_code', 'UNKNOWN')}",
            f"Exit Code: {mapping.exit_code}",
            f"HTTP Status: {mapping.http_status}",
            f"Retryable: {mapping.retryable}",
            f"User Friendly: {mapping.user_friendly}"
        ]
        
        if context:
            lines.append(f"Context: {context}")
        
        return "\n".join(lines)
    
    def handle_exception(
        self,
        exception: Exception,
        output_format: OutputFormat = OutputFormat.CLI,
        context: Optional[Dict[str, Any]] = None,
        exit_on_error: bool = True
    ) -> Union[CLIMessage, APIMessage, Dict[str, Any], int]:
        """
        Handle exception with mapping and optional exit
        
        Args:
            exception: Exception to handle
            output_format: Desired output format
            context: Additional context information
            exit_on_error: Whether to exit after handling
            
        Returns:
            Mapped message or exit code
        """
        # Map exception
        mapped_message = self.map_exception(exception, output_format, context)
        
        # Handle CLI output
        if output_format == OutputFormat.CLI:
            if isinstance(mapped_message, CLIMessage):
                # Render and display message
                from .cli_exceptions import CLIMessageRenderer
                renderer = CLIMessageRenderer()
                rendered_message = renderer.render(mapped_message)
                print(rendered_message, file=sys.stderr)
                
                if exit_on_error:
                    sys.exit(mapped_message.exit_code)
                
                return mapped_message.exit_code
        
        # Handle API output
        elif output_format == OutputFormat.API:
            if isinstance(mapped_message, APIMessage):
                # Return API response
                from .api_exceptions import APIMessageSerializer
                response = APIMessageSerializer.to_json(mapped_message)
                return response, mapped_message.http_status
        
        # Handle other formats
        else:
            return mapped_message
        
        return mapped_message


# ============================================================================
# CONTEXT MANAGERS
# ============================================================================

class ExceptionMapperContext:
    """Context manager for exception mapping"""
    
    def __init__(
        self,
        mapper: ExceptionMapper,
        output_format: OutputFormat = OutputFormat.CLI,
        context: Optional[Dict[str, Any]] = None,
        exit_on_error: bool = True
    ):
        self.mapper = mapper
        self.output_format = output_format
        self.context = context or {}
        self.exit_on_error = exit_on_error
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.mapper.handle_exception(
                exc_val,
                self.output_format,
                self.context,
                self.exit_on_error
            )
            return True  # Suppress the exception
        return False


# ============================================================================
# DECORATORS
# ============================================================================

def map_exceptions(
    output_format: OutputFormat = OutputFormat.CLI,
    context: Optional[Dict[str, Any]] = None,
    exit_on_error: bool = True
):
    """Decorator to map exceptions"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            mapper = ExceptionMapper()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Add function context
                func_context = {
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                }
                
                if context:
                    func_context.update(context)
                
                return mapper.handle_exception(
                    e,
                    output_format,
                    func_context,
                    exit_on_error
                )
        
        return wrapper
    return decorator


def map_async_exceptions(
    output_format: OutputFormat = OutputFormat.CLI,
    context: Optional[Dict[str, Any]] = None,
    exit_on_error: bool = True
):
    """Decorator to map async exceptions"""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            mapper = ExceptionMapper()
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Add function context
                func_context = {
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs),
                    "async": True
                }
                
                if context:
                    func_context.update(context)
                
                return mapper.handle_exception(
                    e,
                    output_format,
                    func_context,
                    exit_on_error
                )
        
        return wrapper
    return decorator


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_exception_mapper(
    custom_mappings: Optional[Dict[Type[Exception], ExceptionMapping]] = None,
    logger=None
) -> ExceptionMapper:
    """Create exception mapper with custom mappings"""
    mapper = ExceptionMapper()
    
    if logger:
        mapper.set_logger(logger)
    
    if custom_mappings:
        for exception_type, mapping in custom_mappings.items():
            mapper.add_mapping(exception_type, **mapping.__dict__)
    
    return mapper


def map_video_opusclip_exception(
    exception: VideoOpusClipException,
    output_format: OutputFormat = OutputFormat.CLI
) -> Union[CLIMessage, APIMessage, Dict[str, Any]]:
    """Map VideoOpusClipException to user-friendly message"""
    mapper = ExceptionMapper()
    
    # Add custom mapping for VideoOpusClipException
    if output_format == OutputFormat.CLI:
        return CLIMessage(
            title="🚨 Application Error",
            message=str(exception),
            suggestion="Please check the error details and try again",
            code=getattr(exception, 'error_code', 'UNKNOWN'),
            severity=CLISeverity.ERROR,
            category=CLICategory.EXECUTION,
            exit_code=1
        )
    elif output_format == OutputFormat.API:
        return APIMessage(
            title="Application Error",
            message=str(exception),
            suggestion="Please check the error details and try again",
            code=getattr(exception, 'error_code', 'UNKNOWN'),
            severity=APISeverity.ERROR,
            category=APICategory.EXECUTION,
            http_status=HTTPStatus.INTERNAL_SERVER_ERROR
        )
    else:
        return {
            "error": {
                "type": "VideoOpusClipException",
                "message": str(exception),
                "code": getattr(exception, 'error_code', 'UNKNOWN'),
                "details": getattr(exception, 'details', {})
            }
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage of exception mapper
    print("🗺️ Exception Mapper Example")
    
    # Create exception mapper
    mapper = create_exception_mapper()
    
    # Test different exception types
    exceptions_to_test = [
        TimeoutError("Operation timed out"),
        ConnectionError("Connection failed"),
        FileNotFoundError("config.json"),
        PermissionError("output.txt"),
        ValueError("Invalid input"),
        TypeError("Invalid type"),
        KeyboardInterrupt(),
        VideoOpusClipException("Custom application error", error_code="CUSTOM_ERROR")
    ]
    
    print("\n" + "="*60)
    print("EXCEPTION MAPPING EXAMPLES")
    print("="*60)
    
    for exception in exceptions_to_test:
        print(f"\n{'-'*40}")
        print(f"Exception: {type(exception).__name__}")
        print(f"Message: {str(exception)}")
        
        # Test CLI format
        print("\nCLI Format:")
        cli_message = mapper.map_exception(exception, OutputFormat.CLI)
        print(f"Title: {cli_message.title}")
        print(f"Message: {cli_message.message}")
        print(f"Exit Code: {cli_message.exit_code}")
        
        # Test API format
        print("\nAPI Format:")
        api_message = mapper.map_exception(exception, OutputFormat.API)
        print(f"Title: {api_message.title}")
        print(f"Message: {api_message.message}")
        print(f"HTTP Status: {api_message.http_status}")
        
        # Test JSON format
        print("\nJSON Format:")
        json_response = mapper.map_exception(exception, OutputFormat.JSON)
        print(json_response)
    
    # Test context manager
    print(f"\n{'-'*40}")
    print("CONTEXT MANAGER EXAMPLE")
    print(f"{'-'*40}")
    
    with ExceptionMapperContext(mapper, OutputFormat.CLI, {"operation": "test"}, exit_on_error=False):
        raise ValueError("Test error in context")
    
    # Test decorator
    print(f"\n{'-'*40}")
    print("DECORATOR EXAMPLE")
    print(f"{'-'*40}")
    
    @map_exceptions(OutputFormat.CLI, {"decorated": True}, exit_on_error=False)
    def test_function():
        raise FileNotFoundError("test.txt")
    
    test_function()
    
    print("\n✅ Exception mapper examples completed!") 