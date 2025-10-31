from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import re
import shlex
import os
import urllib.parse
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import logging
import hashlib
import base64
from typing import Any, List, Dict, Optional
import asyncio
"""
Input Sanitizer Module
=====================

Comprehensive input sanitization for the cybersecurity toolkit:
- Validate and sanitize all external inputs
- Prevent injection attacks (SQL, command, path)
- Secure string handling
- Input validation with whitelisting
- Safe command execution
"""


# Get logger
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom validation error."""
    def __init__(self, field: str, value: Any, error_type: str, message: str, context: Optional[Dict] = None):
        
    """__init__ function."""
self.field = field
        self.value = value
        self.error_type = error_type
        self.message = message
        self.context = context or {}
        super().__init__(message)

class SecurityError(Exception):
    """Custom security error."""
    def __init__(self, error_type: str, message: str, severity: str = "MEDIUM", context: Optional[Dict] = None):
        
    """__init__ function."""
self.error_type = error_type
        self.message = message
        self.severity = severity
        self.context = context or {}
        super().__init__(message)

class InputSanitizer:
    """
    Comprehensive input sanitizer for preventing injection attacks.
    """
    
    def __init__(self) -> Any:
        """Initialize input sanitizer with secure defaults."""
        # Define allowed characters for different input types
        self.allowed_chars = {
            'hostname': re.compile(r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$'),
            'ip_address': re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'),
            'ipv6_address': re.compile(r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'),
            'port_number': re.compile(r'^(?:[1-9]\d{0,3}|[1-5]\d{4}|6[0-4]\d{3}|65[0-4]\d{2}|655[0-2]\d|6553[0-5])$'),
            'filename': re.compile(r'^[a-zA-Z0-9._-]+$'),
            'path': re.compile(r'^[a-zA-Z0-9/._-]+$'),
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'url': re.compile(r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[a-zA-Z0-9._/-]*)?$'),
            'command': re.compile(r'^[a-zA-Z0-9._-]+$'),
            'parameter': re.compile(r'^[a-zA-Z0-9._-]+$')
        }
        
        # Define dangerous patterns to block
        self.dangerous_patterns = [
            # Command injection patterns
            re.compile(r'[;&|`$(){}[\]]'),
            re.compile(r'\b(?:rm|del|format|mkfs|dd|cat|chmod|chown|sudo|su|exec|eval|system|subprocess)\b', re.IGNORECASE),
            
            # SQL injection patterns
            re.compile(r'\b(?:SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|UNION|OR|AND)\b', re.IGNORECASE),
            re.compile(r'[\'";]'),
            
            # Path traversal patterns
            re.compile(r'\.\./|\.\.\\'),
            re.compile(r'%2e%2e|%2e%2e%5c'),
            
            # XSS patterns
            re.compile(r'<script|javascript:|vbscript:|onload=|onerror=', re.IGNORECASE),
            
            # File inclusion patterns
            re.compile(r'file://|ftp://|gopher://'),
            
            # Shell metacharacters
            re.compile(r'[<>|&$`!*?[]{}()~]')
        ]
        
        # Define maximum lengths for different input types
        self.max_lengths = {
            'hostname': 253,
            'ip_address': 15,
            'ipv6_address': 39,
            'port_number': 5,
            'filename': 255,
            'path': 4096,
            'email': 254,
            'url': 2048,
            'command': 100,
            'parameter': 100,
            'general': 1000
        }
    
    def sanitize_string(self, input_string: str, input_type: str = 'general') -> str:
        """
        Sanitize a string input with type-specific validation.
        
        Args:
            input_string: String to sanitize
            input_type: Type of input for specific validation
            
        Returns:
            Sanitized string
            
        Raises:
            ValidationError: When input is invalid or dangerous
        """
        # Guard clause 1: Check if input is provided
        if input_string is None:
            raise ValidationError(
                "input_string",
                None,
                "missing_input",
                "Input string is required",
                context={"operation": "string_sanitization"}
            )
        
        # Guard clause 2: Check if input is a string
        if not isinstance(input_string, str):
            raise ValidationError(
                "input_string",
                input_string,
                "invalid_type",
                "Input must be a string",
                context={"operation": "string_sanitization"}
            )
        
        # Guard clause 3: Check input length
        max_length = self.max_lengths.get(input_type, self.max_lengths['general'])
        if len(input_string) > max_length:
            raise ValidationError(
                "input_string",
                len(input_string),
                "input_too_long",
                f"Input too long (max {max_length} characters)",
                context={"operation": "string_sanitization"}
            )
        
        # Guard clause 4: Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if pattern.search(input_string):
                raise SecurityError(
                    "dangerous_input_detected",
                    f"Dangerous pattern detected in input: {pattern.pattern}",
                    severity="HIGH",
                    context={"operation": "string_sanitization", "pattern": pattern.pattern}
                )
        
        # Guard clause 5: Type-specific validation
        if input_type in self.allowed_chars:
            if not self.allowed_chars[input_type].match(input_string):
                raise ValidationError(
                    "input_string",
                    input_string,
                    "invalid_format",
                    f"Input does not match expected format for {input_type}",
                    context={"operation": "string_sanitization"}
                )
        
        # Happy path: Return sanitized string
        sanitized = input_string.strip()
        
        logger.info(f"String sanitized successfully: {input_type}", extra={
            "input_type": input_type,
            "original_length": len(input_string),
            "sanitized_length": len(sanitized)
        })
        
        return sanitized
    
    def sanitize_hostname(self, hostname: str) -> str:
        """
        Sanitize hostname input.
        
        Args:
            hostname: Hostname to sanitize
            
        Returns:
            Sanitized hostname
        """
        return self.sanitize_string(hostname, 'hostname').lower()
    
    def sanitize_ip_address(self, ip_address: str) -> str:
        """
        Sanitize IP address input.
        
        Args:
            ip_address: IP address to sanitize
            
        Returns:
            Sanitized IP address
        """
        sanitized = self.sanitize_string(ip_address, 'ip_address')
        
        # Additional IP validation
        if not self.allowed_chars['ip_address'].match(sanitized):
            raise ValidationError(
                "ip_address",
                sanitized,
                "invalid_ip_format",
                "Invalid IP address format",
                context={"operation": "ip_sanitization"}
            )
        
        return sanitized
    
    def sanitize_port(self, port: Union[str, int]) -> int:
        """
        Sanitize port number input.
        
        Args:
            port: Port number to sanitize
            
        Returns:
            Sanitized port number as integer
        """
        # Convert to string for validation
        if isinstance(port, int):
            port_str = str(port)
        else:
            port_str = self.sanitize_string(port, 'port_number')
        
        # Validate port format
        if not self.allowed_chars['port_number'].match(port_str):
            raise ValidationError(
                "port",
                port_str,
                "invalid_port_format",
                "Invalid port number format",
                context={"operation": "port_sanitization"}
            )
        
        port_int = int(port_str)
        
        # Validate port range
        if port_int < 1 or port_int > 65535:
            raise ValidationError(
                "port",
                port_int,
                "port_out_of_range",
                "Port must be between 1 and 65535",
                context={"operation": "port_sanitization"}
            )
        
        return port_int
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename input.
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            Sanitized filename
        """
        sanitized = self.sanitize_string(filename, 'filename')
        
        # Additional filename validation
        if not self.allowed_chars['filename'].match(sanitized):
            raise ValidationError(
                "filename",
                sanitized,
                "invalid_filename_format",
                "Invalid filename format",
                context={"operation": "filename_sanitization"}
            )
        
        # Check for reserved names (Windows)
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL'] + [f'COM{i}' for i in range(1, 10)] + [f'LPT{i}' for i in range(1, 10)]
        if sanitized.upper() in reserved_names:
            raise ValidationError(
                "filename",
                sanitized,
                "reserved_filename",
                "Filename is a reserved system name",
                context={"operation": "filename_sanitization"}
            )
        
        return sanitized
    
    def sanitize_path(self, path: str, base_path: Optional[str] = None) -> str:
        """
        Sanitize file path input with path traversal protection.
        
        Args:
            path: Path to sanitize
            base_path: Base path to restrict access
            
        Returns:
            Sanitized path
        """
        sanitized = self.sanitize_string(path, 'path')
        
        # Additional path validation
        if not self.allowed_chars['path'].match(sanitized):
            raise ValidationError(
                "path",
                sanitized,
                "invalid_path_format",
                "Invalid path format",
                context={"operation": "path_sanitization"}
            )
        
        # Check for path traversal attempts
        if '..' in sanitized or '\\' in sanitized:
            raise SecurityError(
                "path_traversal_attempt",
                "Path traversal attempt detected",
                severity="HIGH",
                context={"operation": "path_sanitization"}
            )
        
        # Normalize path
        normalized_path = os.path.normpath(sanitized)
        
        # Restrict to base path if provided
        if base_path:
            base_path = os.path.abspath(base_path)
            normalized_path = os.path.abspath(normalized_path)
            
            if not normalized_path.startswith(base_path):
                raise SecurityError(
                    "path_outside_base",
                    "Path is outside allowed base directory",
                    severity="HIGH",
                    context={"operation": "path_sanitization"}
                )
        
        return normalized_path
    
    def sanitize_command(self, command: str, allowed_commands: Optional[List[str]] = None) -> str:
        """
        Sanitize command input with whitelist validation.
        
        Args:
            command: Command to sanitize
            allowed_commands: List of allowed commands
            
        Returns:
            Sanitized command
        """
        sanitized = self.sanitize_string(command, 'command')
        
        # Additional command validation
        if not self.allowed_chars['command'].match(sanitized):
            raise ValidationError(
                "command",
                sanitized,
                "invalid_command_format",
                "Invalid command format",
                context={"operation": "command_sanitization"}
            )
        
        # Whitelist validation
        if allowed_commands and sanitized not in allowed_commands:
            raise SecurityError(
                "command_not_allowed",
                f"Command '{sanitized}' is not in allowed list",
                severity="HIGH",
                context={"operation": "command_sanitization", "allowed_commands": allowed_commands}
            )
        
        return sanitized
    
    def sanitize_url(self, url: str) -> str:
        """
        Sanitize URL input.
        
        Args:
            url: URL to sanitize
            
        Returns:
            Sanitized URL
        """
        sanitized = self.sanitize_string(url, 'url')
        
        # Additional URL validation
        if not self.allowed_chars['url'].match(sanitized):
            raise ValidationError(
                "url",
                sanitized,
                "invalid_url_format",
                "Invalid URL format",
                context={"operation": "url_sanitization"}
            )
        
        # Parse and validate URL
        try:
            parsed = urllib.parse.urlparse(sanitized)
            if not parsed.scheme or not parsed.netloc:
                raise ValidationError(
                    "url",
                    sanitized,
                    "invalid_url_structure",
                    "Invalid URL structure",
                    context={"operation": "url_sanitization"}
                )
        except Exception as e:
            raise ValidationError(
                "url",
                sanitized,
                "url_parsing_error",
                f"URL parsing failed: {str(e)}",
                context={"operation": "url_sanitization"}
            )
        
        return sanitized
    
    def sanitize_email(self, email: str) -> str:
        """
        Sanitize email input.
        
        Args:
            email: Email to sanitize
            
        Returns:
            Sanitized email
        """
        sanitized = self.sanitize_string(email, 'email').lower()
        
        # Additional email validation
        if not self.allowed_chars['email'].match(sanitized):
            raise ValidationError(
                "email",
                sanitized,
                "invalid_email_format",
                "Invalid email format",
                context={"operation": "email_sanitization"}
            )
        
        return sanitized
    
    def sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize a dictionary of parameters.
        
        Args:
            parameters: Parameters to sanitize
            
        Returns:
            Sanitized parameters
        """
        if not isinstance(parameters, dict):
            raise ValidationError(
                "parameters",
                parameters,
                "invalid_type",
                "Parameters must be a dictionary",
                context={"operation": "parameter_sanitization"}
            )
        
        sanitized_params = {}
        
        for key, value in parameters.items():
            # Sanitize key
            sanitized_key = self.sanitize_string(key, 'parameter')
            
            # Sanitize value based on type
            if isinstance(value, str):
                sanitized_value = self.sanitize_string(value)
            elif isinstance(value, (int, float, bool)):
                sanitized_value = value
            elif isinstance(value, list):
                sanitized_value = [self.sanitize_string(str(item)) if isinstance(item, str) else item for item in value]
            elif isinstance(value, dict):
                sanitized_value = self.sanitize_parameters(value)
            else:
                sanitized_value = str(value)
            
            sanitized_params[sanitized_key] = sanitized_value
        
        return sanitized_params
    
    def create_safe_command(self, command: str, *args) -> List[str]:
        """
        Create a safe command list for subprocess execution.
        
        Args:
            command: Base command
            *args: Command arguments
            
        Returns:
            Safe command list
        """
        # Sanitize command
        safe_command = self.sanitize_command(command)
        
        # Sanitize arguments
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                safe_args.append(self.sanitize_string(arg))
            else:
                safe_args.append(str(arg))
        
        # Use shlex to properly split command
        command_list = [safe_command] + safe_args
        
        logger.info(f"Safe command created: {command_list}", extra={
            "original_command": command,
            "safe_command": command_list
        })
        
        return command_list
    
    async def validate_file_upload(self, filename: str, content_type: str, max_size: int = 10 * 1024 * 1024) -> Tuple[str, str]:
        """
        Validate and sanitize file upload.
        
        Args:
            filename: Original filename
            content_type: File content type
            max_size: Maximum file size in bytes
            
        Returns:
            Tuple of (sanitized_filename, sanitized_content_type)
        """
        # Sanitize filename
        safe_filename = self.sanitize_filename(filename)
        
        # Validate content type
        allowed_types = [
            'text/plain', 'text/csv', 'application/json', 'application/xml',
            'image/jpeg', 'image/png', 'image/gif', 'application/pdf'
        ]
        
        if content_type not in allowed_types:
            raise ValidationError(
                "content_type",
                content_type,
                "unsupported_content_type",
                f"Content type not allowed. Allowed: {allowed_types}",
                context={"operation": "file_upload_validation"}
            )
        
        return safe_filename, content_type
    
    def generate_safe_filename(self, original_filename: str, prefix: str = "file_") -> str:
        """
        Generate a safe filename with hash.
        
        Args:
            original_filename: Original filename
            prefix: Prefix for generated filename
            
        Returns:
            Safe filename
        """
        # Get file extension
        path = Path(original_filename)
        extension = path.suffix if path.suffix else ""
        
        # Generate hash of original filename
        filename_hash = hashlib.sha256(original_filename.encode()).hexdigest()[:16]
        
        # Create safe filename
        safe_filename = f"{prefix}{filename_hash}{extension}"
        
        return self.sanitize_filename(safe_filename)

# Global input sanitizer instance
_input_sanitizer = InputSanitizer()

def sanitize_string(input_string: str, input_type: str = 'general') -> str:
    """
    Sanitize a string input using global sanitizer.
    
    Args:
        input_string: String to sanitize
        input_type: Type of input for specific validation
        
    Returns:
        Sanitized string
    """
    return _input_sanitizer.sanitize_string(input_string, input_type)

def sanitize_hostname(hostname: str) -> str:
    """Sanitize hostname input."""
    return _input_sanitizer.sanitize_hostname(hostname)

def sanitize_ip_address(ip_address: str) -> str:
    """Sanitize IP address input."""
    return _input_sanitizer.sanitize_ip_address(ip_address)

def sanitize_port(port: Union[str, int]) -> int:
    """Sanitize port number input."""
    return _input_sanitizer.sanitize_port(port)

def sanitize_filename(filename: str) -> str:
    """Sanitize filename input."""
    return _input_sanitizer.sanitize_filename(filename)

def sanitize_path(path: str, base_path: Optional[str] = None) -> str:
    """Sanitize file path input."""
    return _input_sanitizer.sanitize_path(path, base_path)

def sanitize_command(command: str, allowed_commands: Optional[List[str]] = None) -> str:
    """Sanitize command input."""
    return _input_sanitizer.sanitize_command(command, allowed_commands)

def sanitize_url(url: str) -> str:
    """Sanitize URL input."""
    return _input_sanitizer.sanitize_url(url)

def sanitize_email(email: str) -> str:
    """Sanitize email input."""
    return _input_sanitizer.sanitize_email(email)

def sanitize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize parameters dictionary."""
    return _input_sanitizer.sanitize_parameters(parameters)

def create_safe_command(command: str, *args) -> List[str]:
    """Create a safe command list for subprocess execution."""
    return _input_sanitizer.create_safe_command(command, *args)

async def validate_file_upload(filename: str, content_type: str, max_size: int = 10 * 1024 * 1024) -> Tuple[str, str]:
    """Validate and sanitize file upload."""
    return _input_sanitizer.validate_file_upload(filename, content_type, max_size)

def generate_safe_filename(original_filename: str, prefix: str = "file_") -> str:
    """Generate a safe filename with hash."""
    return _input_sanitizer.generate_safe_filename(original_filename, prefix)

# --- Named Exports ---

__all__ = [
    'InputSanitizer',
    'ValidationError',
    'SecurityError',
    'sanitize_string',
    'sanitize_hostname',
    'sanitize_ip_address',
    'sanitize_port',
    'sanitize_filename',
    'sanitize_path',
    'sanitize_command',
    'sanitize_url',
    'sanitize_email',
    'sanitize_parameters',
    'create_safe_command',
    'validate_file_upload',
    'generate_safe_filename'
] 