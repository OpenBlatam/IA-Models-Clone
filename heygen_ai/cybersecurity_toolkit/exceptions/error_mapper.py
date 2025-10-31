from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime
import json
import sys
from pathlib import Path
from .custom_exceptions import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Error Mapper Module
==================

Maps custom exceptions to user-friendly CLI/API messages with:
- Exception to message mapping
- Error code generation
- Context-aware messages
- Localization support
- Message formatting
"""


    CybersecurityToolkitError,
    ValidationError,
    NetworkError,
    ScanningError,
    CryptographicError,
    ConfigurationError,
    ResourceError,
    SecurityError,
    create_user_friendly_message,
    map_exception_to_http_status
)


class ErrorMapper:
    """
    Maps exceptions to user-friendly messages for CLI and API interfaces.
    
    Provides consistent error message formatting, context-aware messages,
    and support for different output formats.
    """
    
    def __init__(self, 
                 include_stack_trace: bool = False,
                 include_error_codes: bool = True,
                 include_timestamps: bool = True,
                 output_format: str = "text"):
        """
        Initialize error mapper.
        
        Args:
            include_stack_trace: Include stack trace in error output
            include_error_codes: Include error codes in output
            include_timestamps: Include timestamps in output
            output_format: Output format (text, json, structured)
        """
        self.include_stack_trace = include_stack_trace
        self.include_error_codes = include_error_codes
        self.include_timestamps = include_timestamps
        self.output_format = output_format
        
        # Message templates for different exception types
        self.message_templates = self._initialize_message_templates()
        
        # Error code mappings
        self.error_code_mappings = self._initialize_error_code_mappings()
    
    def _initialize_message_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize message templates for different exception types."""
        return {
            "ValidationError": {
                "missing_required_field": "❌ Required field '{field_name}' is missing. Please provide this field and try again.",
                "invalid_field_type": "❌ Field '{field_name}' has invalid type. Expected {expected_type}, got {actual_type}.",
                "field_value_out_of_range": "❌ Field '{field_name}' value {field_value} is out of range. Must be between {min_value} and {max_value}.",
                "invalid_format": "❌ Field '{field_name}' has invalid format. Expected: {expected_format}",
                "default": "❌ Validation error: {message}"
            },
            "NetworkError": {
                "connection_timeout": "🌐 Connection to {target} timed out after {timeout} seconds. Please check your network connection and try again.",
                "connection_refused": "🌐 Connection to {target} was refused. The service may not be running or the port may be closed.",
                "invalid_target": "🌐 Invalid target '{target}': {reason}. Please check the target specification and try again.",
                "dns_resolution": "🌐 Failed to resolve hostname '{hostname}'. Please check the hostname and try again.",
                "default": "🌐 Network error: {message}"
            },
            "ScanningError": {
                "port_scan": "🔍 Port scan of {target} failed: {reason}. Please check the target and scan parameters.",
                "vulnerability_scan": "🔍 Vulnerability scan of {target} failed: {reason}. Please check the target and scan configuration.",
                "scan_configuration": "🔍 Invalid scan configuration for {target}: {configuration_error}. Please check your scan settings.",
                "default": "🔍 Scanning error: {message}"
            },
            "CryptographicError": {
                "encryption": "🔐 Encryption using {algorithm} failed: {reason}. Please check your encryption parameters.",
                "decryption": "🔐 Decryption using {algorithm} failed: {reason}. Please check your decryption parameters.",
                "invalid_key": "🔐 Invalid {key_type} key: {reason}. Please check your key and try again.",
                "default": "🔐 Cryptographic error: {message}"
            },
            "ConfigurationError": {
                "missing_configuration": "⚙️ Missing required configuration: {config_key}. Please check your configuration file.",
                "invalid_configuration": "⚙️ Invalid configuration for {config_key}: {reason}. Please check your configuration settings.",
                "default": "⚙️ Configuration error: {message}"
            },
            "ResourceError": {
                "resource_limit_exceeded": "📊 {resource_type} limit exceeded: {current} > {limit}. Please reduce the request size or contact support.",
                "resource_not_found": "📊 {resource_type} not found: {resource_name}. Please check the resource specification.",
                "default": "📊 Resource error: {message}"
            },
            "SecurityError": {
                "authentication_failure": "🔒 Authentication using {auth_method} failed: {reason}. Please check your credentials.",
                "authorization_failure": "🔒 Authorization failed for {required_permission}: {reason}. Please check your permissions.",
                "default": "🔒 Security error: {message}"
            },
            "default": {
                "default": "❌ An error occurred: {message}"
            }
        }
    
    def _initialize_error_code_mappings(self) -> Dict[str, str]:
        """Initialize error code mappings for different exception types."""
        return {
            "ValidationError": "VAL",
            "NetworkError": "NET",
            "ScanningError": "SCN",
            "CryptographicError": "CRY",
            "ConfigurationError": "CFG",
            "ResourceError": "RES",
            "SecurityError": "SEC",
            "default": "GEN"
        }
    
    def map_exception(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Map exception to user-friendly message and metadata.
        
        Args:
            exception: Exception to map
            context: Additional context information
            
        Returns:
            Dictionary with mapped error information
        """
        context = context or {}
        
        # Get exception type
        exception_type = type(exception).__name__
        
        # Get error code
        error_code = self._get_error_code(exception, exception_type)
        
        # Get user-friendly message
        user_message = self._get_user_friendly_message(exception, exception_type, context)
        
        # Get technical message
        technical_message = str(exception)
        
        # Get severity
        severity = self._get_severity(exception_type)
        
        # Build result
        result = {
            "error_type": exception_type,
            "error_code": error_code,
            "user_message": user_message,
            "technical_message": technical_message,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add optional fields
        if self.include_error_codes:
            result["error_code"] = error_code
        
        if self.include_timestamps:
            result["timestamp"] = datetime.utcnow().isoformat()
        
        if self.include_stack_trace and hasattr(exception, 'stack_trace'):
            result["stack_trace"] = exception.stack_trace
        
        # Add context if available
        if hasattr(exception, 'context') and exception.context:
            result["context"] = exception.context
        
        if context:
            result["additional_context"] = context
        
        return result
    
    def _get_error_code(self, exception: Exception, exception_type: str) -> str:
        """Get error code for exception."""
        if isinstance(exception, CybersecurityToolkitError):
            return exception.error_code
        
        # Generate error code based on exception type
        base_code = self.error_code_mappings.get(exception_type, self.error_code_mappings["default"])
        return f"{base_code}_{exception_type.upper()}"
    
    def _get_user_friendly_message(self, 
                                 exception: Exception, 
                                 exception_type: str, 
                                 context: Dict[str, Any]) -> str:
        """Get user-friendly message for exception."""
        # Get templates for exception type
        templates = self.message_templates.get(exception_type, self.message_templates["default"])
        
        # Determine specific template key
        template_key = self._get_template_key(exception, exception_type)
        template = templates.get(template_key, templates["default"])
        
        # Format template with context
        try:
            # Combine exception context with additional context
            format_context = {}
            
            if hasattr(exception, 'context'):
                format_context.update(exception.context)
            
            format_context.update(context)
            format_context["message"f"] = str(exception)
            
            return template"
        except (KeyError, ValueError):
            # Fallback to default message
            return create_user_friendly_message(exception)
    
    def _get_template_key(self, exception: Exception, exception_type: str) -> str:
        """Get template key for exception."""
        if isinstance(exception, ValidationError):
            return exception.context.get("validation_type", "default")
        elif isinstance(exception, NetworkError):
            if "timeout" in str(exception).lower():
                return "connection_timeout"
            elif "refused" in str(exception).lower():
                return "connection_refused"
            elif "invalid" in str(exception).lower():
                return "invalid_target"
            elif "resolve" in str(exception).lower():
                return "dns_resolution"
            else:
                return "default"
        elif isinstance(exception, ScanningError):
            if "port" in str(exception).lower():
                return "port_scan"
            elif "vulnerability" in str(exception).lower():
                return "vulnerability_scan"
            elif "configuration" in str(exception).lower():
                return "scan_configuration"
            else:
                return "default"
        elif isinstance(exception, CryptographicError):
            if "encrypt" in str(exception).lower():
                return "encryption"
            elif "decrypt" in str(exception).lower():
                return "decryption"
            elif "key" in str(exception).lower():
                return "invalid_key"
            else:
                return "default"
        elif isinstance(exception, ConfigurationError):
            if "missing" in str(exception).lower():
                return "missing_configuration"
            elif "invalid" in str(exception).lower():
                return "invalid_configuration"
            else:
                return "default"
        elif isinstance(exception, ResourceError):
            if "limit" in str(exception).lower():
                return "resource_limit_exceeded"
            elif "not found" in str(exception).lower():
                return "resource_not_found"
            else:
                return "default"
        elif isinstance(exception, SecurityError):
            if "auth" in str(exception).lower():
                return "authentication_failure"
            elif "permission" in str(exception).lower():
                return "authorization_failure"
            else:
                return "default"
        else:
            return "default"
    
    def _get_severity(self, exception_type: str) -> str:
        """Get severity level for exception type."""
        severity_mapping = {
            "ValidationError": "LOW",
            "ConfigurationError": "MEDIUM",
            "NetworkError": "MEDIUM",
            "ResourceError": "MEDIUM",
            "ScanningError": "HIGH",
            "CryptographicError": "HIGH",
            "SecurityError": "CRITICAL"
        }
        return severity_mapping.get(exception_type, "MEDIUM")
    
    def format_error_for_cli(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Format error for CLI output.
        
        Args:
            exception: Exception to format
            context: Additional context
            
        Returns:
            Formatted CLI error message
        """
        mapped_error = self.map_exception(exception, context)
        
        if self.output_format == "json":
            return json.dumps(mapped_error, indent=2)
        
        # Text format
        lines = []
        
        # Main error message
        lines.append(mapped_error["user_message"])
        
        # Error code
        if self.include_error_codes:
            lines.append(f"Error Code: {mapped_error['error_code']}")
        
        # Technical details (if different from user message)
        if mapped_error["technical_message"] != mapped_error["user_message"]:
            lines.append(f"Technical Details: {mapped_error['technical_message']}")
        
        # Severity
        lines.append(f"Severity: {mapped_error['severity']}")
        
        # Timestamp
        if self.include_timestamps:
            lines.append(f"Timestamp: {mapped_error['timestamp']}")
        
        # Context information
        if "context" in mapped_error and mapped_error["context"]:
            lines.append("Context:")
            for key, value in mapped_error["context"].items():
                lines.append(f"  {key}: {value}")
        
        # Stack trace
        if self.include_stack_trace and "stack_trace" in mapped_error:
            lines.append("Stack Trace:")
            lines.append(mapped_error["stack_trace"])
        
        return "\n".join(lines)
    
    async def format_error_for_api(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Format error for API response.
        
        Args:
            exception: Exception to format
            context: Additional context
            
        Returns:
            Formatted API error response
        """
        mapped_error = self.map_exception(exception, context)
        
        # Add HTTP status code
        mapped_error["status_code"] = map_exception_to_http_status(exception)
        
        # Remove stack trace for API responses (security)
        if "stack_trace" in mapped_error:
            del mapped_error["stack_trace"]
        
        return mapped_error
    
    def log_error(self, 
                 exception: Exception, 
                 logger: Any,
                 context: Optional[Dict[str, Any]] = None,
                 level: str = "ERROR"):
        """
        Log error with structured information.
        
        Args:
            exception: Exception to log
            logger: Logger instance
            context: Additional context
            level: Log level
        """
        mapped_error = self.map_exception(exception, context)
        
        # Create log message
        log_message = f"{mapped_error['error_type']}: {mapped_error['user_message']}"
        
        # Add extra fields for structured logging
        extra = {
            "error_code": mapped_error["error_code"],
            "severity": mapped_error["severity"],
            "technical_message": mapped_error["technical_message"],
            "context": mapped_error.get("context", {}),
            "additional_context": mapped_error.get("additional_context", {})
        }
        
        # Log with appropriate level
        log_method = getattr(logger, level.lower(), logger.error)
        log_method(log_message, extra=extra)


# Global error mapper instance
_error_mapper: Optional[ErrorMapper] = None

def get_error_mapper(**kwargs) -> ErrorMapper:
    """
    Get or create global error mapper instance.
    
    Args:
        **kwargs: Error mapper configuration
        
    Returns:
        Error mapper instance
    """
    global _error_mapper
    
    if _error_mapper is None:
        _error_mapper = ErrorMapper(**kwargs)
    
    return _error_mapper


def format_cli_error(exception: Exception, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Format error for CLI output using global error mapper.
    
    Args:
        exception: Exception to format
        context: Additional context
        
    Returns:
        Formatted CLI error message
    """
    mapper = get_error_mapper()
    return mapper.format_error_for_cli(exception, context)


async def format_api_error(exception: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Format error for API response using global error mapper.
    
    Args:
        exception: Exception to format
        context: Additional context
        
    Returns:
        Formatted API error response
    """
    mapper = get_error_mapper()
    return mapper.format_error_for_api(exception, context)


def log_error(exception: Exception, logger: Any, context: Optional[Dict[str, Any]] = None):
    """
    Log error with structured information using global error mapper.
    
    Args:
        exception: Exception to log
        logger: Logger instance
        context: Additional context
    """
    mapper = get_error_mapper()
    mapper.log_error(exception, logger, context)


# --- Named Exports ---

__all__ = [
    'ErrorMapper',
    'get_error_mapper',
    'format_cli_error',
    'format_api_error',
    'log_error'
] 