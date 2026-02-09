#!/usr/bin/env python3
"""
CLI Custom Exceptions for Video-OpusClip
Custom exceptions with user-friendly CLI message mapping
"""

import sys
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass

from .custom_exceptions import VideoOpusClipException


# ============================================================================
# CLI EXCEPTION CATEGORIES
# ============================================================================

class CLISeverity(Enum):
    """CLI error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CLICategory(Enum):
    """CLI error categories"""
    INPUT = "input"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    CONFIGURATION = "configuration"
    EXECUTION = "execution"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    SYSTEM = "system"


@dataclass
class CLIMessage:
    """CLI message structure"""
    title: str
    message: str
    suggestion: Optional[str] = None
    code: Optional[str] = None
    severity: CLISeverity = CLISeverity.ERROR
    category: CLICategory = CLICategory.EXECUTION
    show_help: bool = False
    exit_code: int = 1


# ============================================================================
# CLI CUSTOM EXCEPTIONS
# ============================================================================

class CLITimeoutError(VideoOpusClipException):
    """CLI timeout error with user-friendly message"""
    
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
        self.suggestion = suggestion or f"Try increasing the timeout value or check network connectivity"
        
        super().__init__(
            f"Operation '{operation}' timed out after {timeout} seconds",
            error_code="CLI_TIMEOUT",
            severity="ERROR",
            details={
                "operation": operation,
                "timeout": timeout,
                "target": target,
                "suggestion": self.suggestion
            }
        )
    
    def to_cli_message(self) -> CLIMessage:
        """Convert to CLI message"""
        return CLIMessage(
            title="⏰ Operation Timed Out",
            message=f"The {self.operation} operation timed out after {self.timeout} seconds.",
            suggestion=self.suggestion,
            code=self.error_code,
            severity=CLISeverity.ERROR,
            category=CLICategory.TIMEOUT,
            show_help=True,
            exit_code=124
        )


class CLIInvalidTargetError(VideoOpusClipException):
    """CLI invalid target error with user-friendly message"""
    
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
            error_code="CLI_INVALID_TARGET",
            severity="ERROR",
            details={
                "target": target,
                "reason": reason,
                "valid_examples": self.valid_examples,
                "suggestion": self.suggestion
            }
        )
    
    def to_cli_message(self) -> CLIMessage:
        """Convert to CLI message"""
        message = f"The target '{self.target}' is invalid: {self.reason}"
        
        if self.valid_examples:
            message += f"\nValid examples: {', '.join(self.valid_examples)}"
        
        return CLIMessage(
            title="🎯 Invalid Target",
            message=message,
            suggestion=self.suggestion,
            code=self.error_code,
            severity=CLISeverity.ERROR,
            category=CLICategory.INPUT,
            show_help=True,
            exit_code=22
        )


class CLIInvalidPortError(VideoOpusClipException):
    """CLI invalid port error with user-friendly message"""
    
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
            error_code="CLI_INVALID_PORT",
            severity="ERROR",
            details={
                "port": port,
                "reason": reason,
                "valid_range": valid_range,
                "suggestion": self.suggestion
            }
        )
    
    def to_cli_message(self) -> CLIMessage:
        """Convert to CLI message"""
        return CLIMessage(
            title="🔌 Invalid Port",
            message=f"The port '{self.port}' is invalid: {self.reason}\nValid range: {self.valid_range[0]}-{self.valid_range[1]}",
            suggestion=self.suggestion,
            code=self.error_code,
            severity=CLISeverity.ERROR,
            category=CLICategory.INPUT,
            show_help=True,
            exit_code=22
        )


class CLIInvalidScanTypeError(VideoOpusClipException):
    """CLI invalid scan type error with user-friendly message"""
    
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
            error_code="CLI_INVALID_SCAN_TYPE",
            severity="ERROR",
            details={
                "scan_type": scan_type,
                "valid_types": valid_types,
                "suggestion": self.suggestion
            }
        )
    
    def to_cli_message(self) -> CLIMessage:
        """Convert to CLI message"""
        return CLIMessage(
            title="🔍 Invalid Scan Type",
            message=f"The scan type '{self.scan_type}' is not supported.\nAvailable types: {', '.join(self.valid_types)}",
            suggestion=self.suggestion,
            code=self.error_code,
            severity=CLISeverity.ERROR,
            category=CLICategory.INPUT,
            show_help=True,
            exit_code=22
        )


class CLINetworkConnectionError(VideoOpusClipException):
    """CLI network connection error with user-friendly message"""
    
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
            error_code="CLI_NETWORK_CONNECTION",
            severity="ERROR",
            details={
                "target": target,
                "reason": reason,
                "suggestion": self.suggestion
            }
        )
    
    def to_cli_message(self) -> CLIMessage:
        """Convert to CLI message"""
        return CLIMessage(
            title="🌐 Network Connection Failed",
            message=f"Unable to connect to '{self.target}': {self.reason}",
            suggestion=self.suggestion,
            code=self.error_code,
            severity=CLISeverity.ERROR,
            category=CLICategory.NETWORK,
            show_help=True,
            exit_code=101
        )


class CLIAuthenticationError(VideoOpusClipException):
    """CLI authentication error with user-friendly message"""
    
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
            error_code="CLI_AUTHENTICATION",
            severity="ERROR",
            details={
                "username": username,
                "reason": reason,
                "suggestion": self.suggestion
            }
        )
    
    def to_cli_message(self) -> CLIMessage:
        """Convert to CLI message"""
        message = f"Authentication failed: {self.reason}"
        if self.username:
            message += f" (User: {self.username})"
        
        return CLIMessage(
            title="🔐 Authentication Failed",
            message=message,
            suggestion=self.suggestion,
            code=self.error_code,
            severity=CLISeverity.ERROR,
            category=CLICategory.AUTHENTICATION,
            show_help=True,
            exit_code=126
        )


class CLIAuthorizationError(VideoOpusClipException):
    """CLI authorization error with user-friendly message"""
    
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
            error_code="CLI_AUTHORIZATION",
            severity="ERROR",
            details={
                "resource": resource,
                "action": action,
                "reason": reason,
                "suggestion": self.suggestion
            }
        )
    
    def to_cli_message(self) -> CLIMessage:
        """Convert to CLI message"""
        return CLIMessage(
            title="🚫 Access Denied",
            message=f"You don't have permission to {self.action} on {self.resource}: {self.reason}",
            suggestion=self.suggestion,
            code=self.error_code,
            severity=CLISeverity.ERROR,
            category=CLICategory.AUTHORIZATION,
            show_help=True,
            exit_code=126
        )


class CLIConfigurationError(VideoOpusClipException):
    """CLI configuration error with user-friendly message"""
    
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
            error_code="CLI_CONFIGURATION",
            severity="ERROR",
            details={
                "config_file": config_file,
                "missing_key": missing_key,
                "reason": reason,
                "suggestion": self.suggestion
            }
        )
    
    def to_cli_message(self) -> CLIMessage:
        """Convert to CLI message"""
        message = f"Configuration error: {self.reason}"
        if self.config_file:
            message += f"\nConfig file: {self.config_file}"
        if self.missing_key:
            message += f"\nMissing key: {self.missing_key}"
        
        return CLIMessage(
            title="⚙️ Configuration Error",
            message=message,
            suggestion=self.suggestion,
            code=self.error_code,
            severity=CLISeverity.ERROR,
            category=CLICategory.CONFIGURATION,
            show_help=True,
            exit_code=78
        )


class CLIResourceError(VideoOpusClipException):
    """CLI resource error with user-friendly message"""
    
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
            error_code="CLI_RESOURCE",
            severity="ERROR",
            details={
                "resource_type": resource_type,
                "resource_name": resource_name,
                "reason": reason,
                "suggestion": self.suggestion
            }
        )
    
    def to_cli_message(self) -> CLIMessage:
        """Convert to CLI message"""
        return CLIMessage(
            title="📁 Resource Error",
            message=f"{self.resource_type} error for '{self.resource_name}': {self.reason}",
            suggestion=self.suggestion,
            code=self.error_code,
            severity=CLISeverity.ERROR,
            category=CLICategory.RESOURCE,
            show_help=True,
            exit_code=2
        )


class CLISystemError(VideoOpusClipException):
    """CLI system error with user-friendly message"""
    
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
            error_code="CLI_SYSTEM",
            severity="CRITICAL",
            details={
                "operation": operation,
                "reason": reason,
                "suggestion": self.suggestion
            }
        )
    
    def to_cli_message(self) -> CLIMessage:
        """Convert to CLI message"""
        return CLIMessage(
            title="💥 System Error",
            message=f"A system error occurred during {self.operation}: {self.reason}",
            suggestion=self.suggestion,
            code=self.error_code,
            severity=CLISeverity.CRITICAL,
            category=CLICategory.SYSTEM,
            show_help=True,
            exit_code=125
        )


class CLIValidationError(VideoOpusClipException):
    """CLI validation error with user-friendly message"""
    
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
            error_code="CLI_VALIDATION",
            severity="ERROR",
            details={
                "field": field,
                "value": value,
                "reason": reason,
                "valid_examples": self.valid_examples,
                "suggestion": self.suggestion
            }
        )
    
    def to_cli_message(self) -> CLIMessage:
        """Convert to CLI message"""
        message = f"Invalid value for '{self.field}': {self.reason}\nValue provided: {self.value}"
        
        if self.valid_examples:
            message += f"\nValid examples: {', '.join(str(ex) for ex in self.valid_examples)}"
        
        return CLIMessage(
            title="✅ Validation Error",
            message=message,
            suggestion=self.suggestion,
            code=self.error_code,
            severity=CLISeverity.ERROR,
            category=CLICategory.INPUT,
            show_help=True,
            exit_code=22
        )


# ============================================================================
# CLI MESSAGE RENDERERS
# ============================================================================

class CLIMessageRenderer:
    """Renderer for CLI messages with different output formats"""
    
    def __init__(self, use_colors: bool = True, use_unicode: bool = True):
        self.use_colors = use_colors and self._supports_colors()
        self.use_unicode = use_unicode and self._supports_unicode()
    
    def _supports_colors(self) -> bool:
        """Check if terminal supports colors"""
        try:
            import colorama
            return True
        except ImportError:
            return False
    
    def _supports_unicode(self) -> bool:
        """Check if terminal supports unicode"""
        try:
            return sys.stdout.encoding.lower().startswith('utf')
        except:
            return False
    
    def render(self, cli_message: CLIMessage) -> str:
        """Render CLI message with appropriate formatting"""
        if self.use_colors:
            return self._render_colored(cli_message)
        else:
            return self._render_plain(cli_message)
    
    def _render_colored(self, cli_message: CLIMessage) -> str:
        """Render colored CLI message"""
        try:
            from colorama import Fore, Back, Style, init
            init()
            
            # Color mapping
            colors = {
                CLISeverity.INFO: Fore.BLUE,
                CLISeverity.WARNING: Fore.YELLOW,
                CLISeverity.ERROR: Fore.RED,
                CLISeverity.CRITICAL: Fore.RED + Style.BRIGHT
            }
            
            color = colors.get(cli_message.severity, Fore.WHITE)
            
            # Build message
            lines = []
            
            # Title
            title_icon = "ℹ️" if cli_message.severity == CLISeverity.INFO else \
                        "⚠️" if cli_message.severity == CLISeverity.WARNING else \
                        "❌" if cli_message.severity == CLISeverity.ERROR else "💥"
            
            if not self.use_unicode:
                title_icon = "[INFO]" if cli_message.severity == CLISeverity.INFO else \
                           "[WARN]" if cli_message.severity == CLISeverity.WARNING else \
                           "[ERROR]" if cli_message.severity == CLISeverity.ERROR else "[CRIT]"
            
            lines.append(f"{color}{title_icon} {cli_message.title}{Style.RESET_ALL}")
            lines.append("")
            
            # Message
            lines.append(f"{color}{cli_message.message}{Style.RESET_ALL}")
            lines.append("")
            
            # Suggestion
            if cli_message.suggestion:
                lines.append(f"{Fore.CYAN}💡 Suggestion: {cli_message.suggestion}{Style.RESET_ALL}")
                lines.append("")
            
            # Code
            if cli_message.code:
                lines.append(f"{Fore.MAGENTA}🔧 Error Code: {cli_message.code}{Style.RESET_ALL}")
                lines.append("")
            
            # Help hint
            if cli_message.show_help:
                lines.append(f"{Fore.GREEN}📖 Run 'video-opusclip --help' for more information{Style.RESET_ALL}")
            
            return "\n".join(lines)
            
        except ImportError:
            return self._render_plain(cli_message)
    
    def _render_plain(self, cli_message: CLIMessage) -> str:
        """Render plain text CLI message"""
        lines = []
        
        # Title
        title_icon = "[INFO]" if cli_message.severity == CLISeverity.INFO else \
                   "[WARN]" if cli_message.severity == CLISeverity.WARNING else \
                   "[ERROR]" if cli_message.severity == CLISeverity.ERROR else "[CRIT]"
        
        lines.append(f"{title_icon} {cli_message.title}")
        lines.append("")
        
        # Message
        lines.append(cli_message.message)
        lines.append("")
        
        # Suggestion
        if cli_message.suggestion:
            lines.append(f"Suggestion: {cli_message.suggestion}")
            lines.append("")
        
        # Code
        if cli_message.code:
            lines.append(f"Error Code: {cli_message.code}")
            lines.append("")
        
        # Help hint
        if cli_message.show_help:
            lines.append("Run 'video-opusclip --help' for more information")
        
        return "\n".join(lines)


# ============================================================================
# CLI EXCEPTION HANDLER
# ============================================================================

class CLIExceptionHandler:
    """Handler for CLI exceptions with user-friendly message mapping"""
    
    def __init__(self, renderer: Optional[CLIMessageRenderer] = None):
        self.renderer = renderer or CLIMessageRenderer()
        self.logger = None  # Will be set by set_logger method
    
    def set_logger(self, logger):
        """Set logger for error logging"""
        self.logger = logger
    
    def handle_exception(self, exception: Exception, exit_on_error: bool = True) -> int:
        """
        Handle exception and display user-friendly message
        
        Args:
            exception: Exception to handle
            exit_on_error: Whether to exit after displaying error
            
        Returns:
            Exit code
        """
        # Log the exception if logger is available
        if self.logger:
            self.logger.error(
                f"CLI exception occurred",
                error=exception,
                tags=['cli', 'exception', 'handled']
            )
        
        # Convert to CLI message
        cli_message = self._exception_to_cli_message(exception)
        
        # Render and display message
        rendered_message = self.renderer.render(cli_message)
        print(rendered_message, file=sys.stderr)
        
        # Exit if requested
        if exit_on_error:
            sys.exit(cli_message.exit_code)
        
        return cli_message.exit_code
    
    def _exception_to_cli_message(self, exception: Exception) -> CLIMessage:
        """Convert exception to CLI message"""
        # Handle custom CLI exceptions
        if isinstance(exception, (
            CLITimeoutError, CLIInvalidTargetError, CLIInvalidPortError,
            CLIInvalidScanTypeError, CLINetworkConnectionError,
            CLIAuthenticationError, CLIAuthorizationError,
            CLIConfigurationError, CLIResourceError, CLISystemError,
            CLIValidationError
        )):
            return exception.to_cli_message()
        
        # Handle VideoOpusClipException
        if isinstance(exception, VideoOpusClipException):
            return CLIMessage(
                title="🚨 Application Error",
                message=str(exception),
                suggestion="Please check the error details and try again",
                code=getattr(exception, 'error_code', 'UNKNOWN'),
                severity=CLISeverity.ERROR,
                category=CLICategory.EXECUTION,
                show_help=True,
                exit_code=1
            )
        
        # Handle standard exceptions
        if isinstance(exception, KeyboardInterrupt):
            return CLIMessage(
                title="⏹️ Operation Cancelled",
                message="The operation was cancelled by the user",
                suggestion="Run the command again to retry",
                code="CLI_CANCELLED",
                severity=CLISeverity.INFO,
                category=CLICategory.EXECUTION,
                show_help=False,
                exit_code=130
            )
        
        if isinstance(exception, FileNotFoundError):
            return CLIMessage(
                title="📁 File Not Found",
                message=f"The file '{exception.filename}' was not found",
                suggestion="Please check the file path and try again",
                code="CLI_FILE_NOT_FOUND",
                severity=CLISeverity.ERROR,
                category=CLICategory.RESOURCE,
                show_help=True,
                exit_code=2
            )
        
        if isinstance(exception, PermissionError):
            return CLIMessage(
                title="🚫 Permission Denied",
                message=f"Permission denied: {exception.filename}",
                suggestion="Please check file permissions or run with appropriate privileges",
                code="CLI_PERMISSION_DENIED",
                severity=CLISeverity.ERROR,
                category=CLICategory.AUTHORIZATION,
                show_help=True,
                exit_code=126
            )
        
        if isinstance(exception, ValueError):
            return CLIMessage(
                title="✅ Invalid Value",
                message=f"Invalid value: {str(exception)}",
                suggestion="Please check your input and try again",
                code="CLI_INVALID_VALUE",
                severity=CLISeverity.ERROR,
                category=CLICategory.INPUT,
                show_help=True,
                exit_code=22
            )
        
        if isinstance(exception, TypeError):
            return CLIMessage(
                title="🔧 Type Error",
                message=f"Type error: {str(exception)}",
                suggestion="Please check your input format and try again",
                code="CLI_TYPE_ERROR",
                severity=CLISeverity.ERROR,
                category=CLICategory.INPUT,
                show_help=True,
                exit_code=22
            )
        
        # Generic exception
        return CLIMessage(
            title="💥 Unexpected Error",
            message=f"An unexpected error occurred: {str(exception)}",
            suggestion="Please try again or contact support if the problem persists",
            code="CLI_UNEXPECTED_ERROR",
            severity=CLISeverity.CRITICAL,
            category=CLICategory.SYSTEM,
            show_help=True,
            exit_code=125
        )


# ============================================================================
# CLI EXCEPTION CONTEXT MANAGER
# ============================================================================

class CLIExceptionContext:
    """Context manager for handling CLI exceptions"""
    
    def __init__(self, handler: CLIExceptionHandler, exit_on_error: bool = True):
        self.handler = handler
        self.exit_on_error = exit_on_error
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.handler.handle_exception(exc_val, self.exit_on_error)
            return True  # Suppress the exception
        return False


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage of CLI exceptions
    print("🎯 CLI Exceptions Example")
    
    # Create exception handler
    handler = CLIExceptionHandler()
    
    # Test different exception types
    exceptions_to_test = [
        CLITimeoutError("port scan", 30.0, "192.168.1.100"),
        CLIInvalidTargetError("invalid-target", "Not a valid IP address or domain", ["192.168.1.100", "example.com"]),
        CLIInvalidPortError(99999, "Port number out of range"),
        CLIInvalidScanTypeError("invalid_scan", ["port_scan", "vulnerability_scan", "web_scan"]),
        CLINetworkConnectionError("192.168.1.100", "Connection refused"),
        CLIAuthenticationError("admin", "Invalid password"),
        CLIAuthorizationError("/admin", "read", "Admin access required"),
        CLIConfigurationError("config.json", "api_key", "Missing API key"),
        CLIResourceError("file", "output.txt", "Cannot create file"),
        CLISystemError("scan", "Out of memory"),
        CLIValidationError("email", "invalid-email", "Invalid email format", ["user@example.com", "admin@domain.org"])
    ]
    
    print("\n" + "="*60)
    print("CLI EXCEPTION EXAMPLES")
    print("="*60)
    
    for exception in exceptions_to_test:
        print(f"\n{'-'*40}")
        handler.handle_exception(exception, exit_on_error=False)
    
    # Test context manager
    print(f"\n{'-'*40}")
    print("CONTEXT MANAGER EXAMPLE")
    print(f"{'-'*40}")
    
    with CLIExceptionContext(handler, exit_on_error=False):
        raise CLIInvalidTargetError("test-target", "Invalid format")
    
    print("\n✅ CLI exception examples completed!") 