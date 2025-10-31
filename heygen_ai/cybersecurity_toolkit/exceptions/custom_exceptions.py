from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import traceback
import uuid
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Custom Exceptions Module
=======================

Comprehensive exception hierarchy for the cybersecurity toolkit with:
- Base exception classes
- Domain-specific exceptions
- User-friendly error messages
- Error code mapping
- Context preservation
"""



class CybersecurityToolkitError(Exception):
    """
    Base exception class for all cybersecurity toolkit errors.
    
    Provides common functionality for error tracking, context preservation,
    and user-friendly message generation.
    """
    
    def __init__(self, 
                 message: str,
                 error_code: str = None,
                 error_type: str = None,
                 context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        """
        Initialize cybersecurity toolkit error.
        
        Args:
            message: Human-readable error message
            error_code: Unique error code for programmatic handling
            error_type: Error type for categorization
            context: Additional context information
            original_exception: Original exception that caused this error
        """
        super().__init__(message)
        
        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.error_type = error_type or self.__class__.__name__
        self.context = context or {}
        self.original_exception = original_exception
        self.timestamp = datetime.utcnow().isoformat()
        self.error_id = str(uuid.uuid4())
        
        # Capture stack trace
        self.stack_trace = traceback.format_exc()
    
    def _generate_error_code(self) -> str:
        """Generate unique error code based on exception class."""
        class_name = self.__class__.__name__
        return f"{class_name.upper()}_{self.error_id[:8].upper()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.error_type,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp,
            "error_id": self.error_id,
            "stack_trace": self.stack_trace if self.context.get("include_stack_trace", False) else None
        }
    
    def get_user_friendly_message(self) -> str:
        """Get user-friendly error message."""
        return self.message


# =============================================================================
# VALIDATION EXCEPTIONS
# =============================================================================

class ValidationError(CybersecurityToolkitError):
    """Base class for validation errors."""
    
    def __init__(self, 
                 field_name: str,
                 field_value: Any,
                 validation_type: str,
                 message: str,
                 context: Optional[Dict[str, Any]] = None):
        """
        Initialize validation error.
        
        Args:
            field_name: Name of the field that failed validation
            field_value: Value that failed validation
            validation_type: Type of validation that failed
            message: Validation error message
            context: Additional context
        """
        super().__init__(
            message=message,
            error_type="ValidationError",
            context={
                "field_name": field_name,
                "field_value": str(field_value)[:100],  # Truncate long values
                "validation_type": validation_type,
                **(context or {})
            }
        )


class MissingRequiredFieldError(ValidationError):
    """Raised when a required field is missing."""
    
    def __init__(self, field_name: str, context: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            field_name=field_name,
            field_value=None,
            validation_type="missing_required_field",
            message=f"Required field '{field_name}' is missing",
            context=context
        )


class InvalidFieldTypeError(ValidationError):
    """Raised when a field has an invalid type."""
    
    def __init__(self, 
                 field_name: str, 
                 field_value: Any, 
                 expected_type: str,
                 context: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            field_name=field_name,
            field_value=field_value,
            validation_type="invalid_field_type",
            message=f"Field '{field_name}' must be of type {expected_type}, got {type(field_value).__name__}",
            context={"expected_type": expected_type, **(context or {})}
        )


class FieldValueOutOfRangeError(ValidationError):
    """Raised when a field value is out of acceptable range."""
    
    def __init__(self, 
                 field_name: str, 
                 field_value: Any, 
                 min_value: Optional[Union[int, float]] = None,
                 max_value: Optional[Union[int, float]] = None,
                 context: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
range_desc = []
        if min_value is not None:
            range_desc.append(f"minimum {min_value}")
        if max_value is not None:
            range_desc.append(f"maximum {max_value}")
        
        range_text = f" between {' and '.join(range_desc)}" if range_desc else ""
        
        super().__init__(
            field_name=field_name,
            field_value=field_value,
            validation_type="field_value_out_of_range",
            message=f"Field '{field_name}' value {field_value} is out of range{range_text}",
            context={
                "min_value": min_value,
                "max_value": max_value,
                **(context or {})
            }
        )


class InvalidFormatError(ValidationError):
    """Raised when a field has an invalid format."""
    
    def __init__(self, 
                 field_name: str, 
                 field_value: Any, 
                 expected_format: str,
                 context: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            field_name=field_name,
            field_value=field_value,
            validation_type="invalid_format",
            message=f"Field '{field_name}' has invalid format. Expected: {expected_format}",
            context={"expected_format": expected_format, **(context or {})}
        )


# =============================================================================
# NETWORK EXCEPTIONS
# =============================================================================

class NetworkError(CybersecurityToolkitError):
    """Base class for network-related errors."""
    
    def __init__(self, 
                 target: str,
                 operation: str,
                 message: str,
                 context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_type="NetworkError",
            context={
                "target": target,
                "operation": operation,
                **(context or {})
            },
            original_exception=original_exception
        )


class ConnectionTimeoutError(NetworkError):
    """Raised when a network connection times out."""
    
    def __init__(self, 
                 target: str, 
                 port: Optional[int] = None,
                 timeout: Optional[float] = None,
                 context: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
port_info = f":{port}" if port else ""
        timeout_info = f" (timeout: {timeout}s)" if timeout else ""
        
        super().__init__(
            target=target,
            operation="connection",
            message=f"Connection to {target}{port_info} timed out{timeout_info}",
            context={
                "port": port,
                "timeout": timeout,
                **(context or {})
            }
        )


class ConnectionRefusedError(NetworkError):
    """Raised when a network connection is refused."""
    
    def __init__(self, 
                 target: str, 
                 port: Optional[int] = None,
                 context: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
port_info = f":{port}" if port else ""
        
        super().__init__(
            target=target,
            operation="connection",
            message=f"Connection to {target}{port_info} was refused",
            context={
                "port": port,
                **(context or {})
            }
        )


class InvalidTargetError(NetworkError):
    """Raised when a network target is invalid."""
    
    def __init__(self, 
                 target: str, 
                 reason: str,
                 context: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            target=target,
            operation="validation",
            message=f"Invalid target '{target}': {reason}",
            context={
                "reason": reason,
                **(context or {})
            }
        )


class DNSResolutionError(NetworkError):
    """Raised when DNS resolution fails."""
    
    def __init__(self, 
                 hostname: str,
                 context: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            target=hostname,
            operation="dns_resolution",
            message=f"Failed to resolve hostname '{hostname}'",
            context=context
        )


# =============================================================================
# SCANNING EXCEPTIONS
# =============================================================================

class ScanningError(CybersecurityToolkitError):
    """Base class for scanning-related errors."""
    
    def __init__(self, 
                 scan_type: str,
                 target: str,
                 message: str,
                 context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_type="ScanningError",
            context={
                "scan_type": scan_type,
                "target": target,
                **(context or {})
            },
            original_exception=original_exception
        )


class PortScanError(ScanningError):
    """Raised when port scanning fails."""
    
    def __init__(self, 
                 target: str,
                 ports: Optional[List[int]] = None,
                 reason: str = "Port scanning failed",
                 context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        
    """__init__ function."""
super().__init__(
            scan_type="port_scan",
            target=target,
            message=f"Port scan of {target} failed: {reason}",
            context={
                "ports": ports,
                "reason": reason,
                **(context or {})
            },
            original_exception=original_exception
        )


class VulnerabilityScanError(ScanningError):
    """Raised when vulnerability scanning fails."""
    
    def __init__(self, 
                 target: str,
                 scan_type: str,
                 reason: str = "Vulnerability scanning failed",
                 context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        
    """__init__ function."""
super().__init__(
            scan_type=f"vulnerability_scan_{scan_type}",
            target=target,
            message=f"Vulnerability scan of {target} failed: {reason}",
            context={
                "vulnerability_scan_type": scan_type,
                "reason": reason,
                **(context or {})
            },
            original_exception=original_exception
        )


class ScanConfigurationError(ScanningError):
    """Raised when scan configuration is invalid."""
    
    def __init__(self, 
                 target: str,
                 configuration_error: str,
                 context: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            scan_type="configuration",
            target=target,
            message=f"Invalid scan configuration for {target}: {configuration_error}",
            context={
                "configuration_error": configuration_error,
                **(context or {})
            }
        )


# =============================================================================
# CRYPTOGRAPHIC EXCEPTIONS
# =============================================================================

class CryptographicError(CybersecurityToolkitError):
    """Base class for cryptographic errors."""
    
    def __init__(self, 
                 operation: str,
                 message: str,
                 context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_type="CryptographicError",
            context={
                "operation": operation,
                **(context or {})
            },
            original_exception=original_exception
        )


class EncryptionError(CryptographicError):
    """Raised when encryption fails."""
    
    def __init__(self, 
                 algorithm: str,
                 reason: str = "Encryption failed",
                 context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        
    """__init__ function."""
super().__init__(
            operation="encryption",
            message=f"Encryption using {algorithm} failed: {reason}",
            context={
                "algorithm": algorithm,
                "reason": reason,
                **(context or {})
            },
            original_exception=original_exception
        )


class DecryptionError(CryptographicError):
    """Raised when decryption fails."""
    
    def __init__(self, 
                 algorithm: str,
                 reason: str = "Decryption failed",
                 context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        
    """__init__ function."""
super().__init__(
            operation="decryption",
            message=f"Decryption using {algorithm} failed: {reason}",
            context={
                "algorithm": algorithm,
                "reason": reason,
                **(context or {})
            },
            original_exception=original_exception
        )


class InvalidKeyError(CryptographicError):
    """Raised when an encryption key is invalid."""
    
    def __init__(self, 
                 key_type: str,
                 reason: str,
                 context: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            operation="key_validation",
            message=f"Invalid {key_type} key: {reason}",
            context={
                "key_type": key_type,
                "reason": reason,
                **(context or {})
            }
        )


# =============================================================================
# CONFIGURATION EXCEPTIONS
# =============================================================================

class ConfigurationError(CybersecurityToolkitError):
    """Base class for configuration errors."""
    
    def __init__(self, 
                 config_type: str,
                 message: str,
                 context: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_type="ConfigurationError",
            context={
                "config_type": config_type,
                **(context or {})
            }
        )


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""
    
    def __init__(self, 
                 config_key: str,
                 config_type: str = "general",
                 context: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            config_type=config_type,
            message=f"Missing required configuration: {config_key}",
            context={
                "config_key": config_key,
                **(context or {})
            }
        )


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration is invalid."""
    
    def __init__(self, 
                 config_key: str,
                 config_value: Any,
                 reason: str,
                 config_type: str = "general",
                 context: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            config_type=config_type,
            message=f"Invalid configuration for {config_key}: {reason}",
            context={
                "config_key": config_key,
                "config_value": str(config_value),
                "reason": reason,
                **(context or {})
            }
        )


# =============================================================================
# RESOURCE EXCEPTIONS
# =============================================================================

class ResourceError(CybersecurityToolkitError):
    """Base class for resource-related errors."""
    
    def __init__(self, 
                 resource_type: str,
                 resource_name: str,
                 message: str,
                 context: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_type="ResourceError",
            context={
                "resource_type": resource_type,
                "resource_name": resource_name,
                **(context or {})
            }
        )


class ResourceLimitExceededError(ResourceError):
    """Raised when a resource limit is exceeded."""
    
    def __init__(self, 
                 resource_type: str,
                 resource_name: str,
                 limit: Union[int, float],
                 current: Union[int, float],
                 context: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            resource_type=resource_type,
            resource_name=resource_name,
            message=f"{resource_type} limit exceeded: {current} > {limit}",
            context={
                "limit": limit,
                "current": current,
                **(context or {})
            }
        )


class ResourceNotFoundError(ResourceError):
    """Raised when a required resource is not found."""
    
    def __init__(self, 
                 resource_type: str,
                 resource_name: str,
                 context: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            resource_type=resource_type,
            resource_name=resource_name,
            message=f"{resource_type} not found: {resource_name}",
            context=context
        )


# =============================================================================
# SECURITY EXCEPTIONS
# =============================================================================

class SecurityError(CybersecurityToolkitError):
    """Base class for security-related errors."""
    
    def __init__(self, 
                 security_event: str,
                 message: str,
                 severity: str = "MEDIUM",
                 context: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_type="SecurityError",
            context={
                "security_event": security_event,
                "severity": severity,
                **(context or {})
            }
        )


class AuthenticationError(SecurityError):
    """Raised when authentication fails."""
    
    def __init__(self, 
                 auth_method: str,
                 reason: str = "Authentication failed",
                 context: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            security_event="authentication_failure",
            message=f"Authentication using {auth_method} failed: {reason}",
            severity="HIGH",
            context={
                "auth_method": auth_method,
                "reason": reason,
                **(context or {})
            }
        )


class AuthorizationError(SecurityError):
    """Raised when authorization fails."""
    
    def __init__(self, 
                 required_permission: str,
                 reason: str = "Insufficient permissions",
                 context: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            security_event="authorization_failure",
            message=f"Authorization failed for {required_permission}: {reason}",
            severity="HIGH",
            context={
                "required_permission": required_permission,
                "reason": reason,
                **(context or {})
            }
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_user_friendly_message(exception: Exception) -> str:
    """
    Create user-friendly error message from exception.
    
    Args:
        exception: Exception to convert to user-friendly message
        
    Returns:
        User-friendly error message
    """
    if isinstance(exception, CybersecurityToolkitError):
        return exception.get_user_friendly_message()
    
    # Handle standard exceptions
    if isinstance(exception, TimeoutError):
        return "Operation timed out. Please try again or increase timeout value."
    elif isinstance(exception, ConnectionError):
        return "Connection failed. Please check your network connection and try again."
    elif isinstance(exception, ValueError):
        return f"Invalid value provided: {str(exception)}"
    elif isinstance(exception, TypeError):
        return f"Invalid type provided: {str(exception)}"
    elif isinstance(exception, FileNotFoundError):
        return "File not found. Please check the file path and try again."
    elif isinstance(exception, PermissionError):
        return "Permission denied. Please check your permissions and try again."
    else:
        return f"An unexpected error occurred: {str(exception)}"


async def map_exception_to_http_status(exception: Exception) -> int:
    """
    Map exception to appropriate HTTP status code.
    
    Args:
        exception: Exception to map
        
    Returns:
        HTTP status code
    """
    if isinstance(exception, ValidationError):
        return 400  # Bad Request
    elif isinstance(exception, (ConnectionTimeoutError, ConnectionRefusedError)):
        return 408  # Request Timeout
    elif isinstance(exception, (AuthenticationError, AuthorizationError)):
        return 401  # Unauthorized
    elif isinstance(exception, ResourceNotFoundError):
        return 404  # Not Found
    elif isinstance(exception, ResourceLimitExceededError):
        return 429  # Too Many Requests
    elif isinstance(exception, ConfigurationError):
        return 500  # Internal Server Error
    elif isinstance(exception, (ScanningError, CryptographicError, NetworkError)):
        return 500  # Internal Server Error
    else:
        return 500  # Internal Server Error


# =============================================================================
# NAMED EXPORTS
# =============================================================================

__all__ = [
    # Base exceptions
    'CybersecurityToolkitError',
    
    # Validation exceptions
    'ValidationError',
    'MissingRequiredFieldError',
    'InvalidFieldTypeError',
    'FieldValueOutOfRangeError',
    'InvalidFormatError',
    
    # Network exceptions
    'NetworkError',
    'ConnectionTimeoutError',
    'ConnectionRefusedError',
    'InvalidTargetError',
    'DNSResolutionError',
    
    # Scanning exceptions
    'ScanningError',
    'PortScanError',
    'VulnerabilityScanError',
    'ScanConfigurationError',
    
    # Cryptographic exceptions
    'CryptographicError',
    'EncryptionError',
    'DecryptionError',
    'InvalidKeyError',
    
    # Configuration exceptions
    'ConfigurationError',
    'MissingConfigurationError',
    'InvalidConfigurationError',
    
    # Resource exceptions
    'ResourceError',
    'ResourceLimitExceededError',
    'ResourceNotFoundError',
    
    # Security exceptions
    'SecurityError',
    'AuthenticationError',
    'AuthorizationError',
    
    # Utility functions
    'create_user_friendly_message',
    'map_exception_to_http_status'
] 