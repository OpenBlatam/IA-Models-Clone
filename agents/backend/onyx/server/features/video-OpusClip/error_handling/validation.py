#!/usr/bin/env python3
"""
Validation Utilities for Video-OpusClip
Comprehensive input validation and sanitization functions
"""

import re
import ipaddress
import socket
import dns.resolver
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin
from email_validator import validate_email, EmailNotValidError

from .custom_exceptions import (
    ValidationError, InputValidationError, TypeValidationError
)


# ============================================================================
# VALIDATION PATTERNS
# ============================================================================

VALIDATION_PATTERNS = {
    "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "username": r"^[a-zA-Z0-9_]{3,50}$",
    "password": r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$",
    "ip_address": r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
    "domain": r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$",
    "url": r"^https?://[^\s/$.?#].[^\s]*$",
    "port": r"^(?:[1-9]\d{0,3}|[1-5]\d{4}|6[0-4]\d{3}|65[0-4]\d{2}|655[0-2]\d|6553[0-5])$",
    "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    "hex_color": r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$",
    "phone": r"^\+?1?\d{9,15}$",
    "date_iso": r"^\d{4}-\d{2}-\d{2}$",
    "datetime_iso": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?(?:Z|[+-]\d{2}:\d{2})?$",
    "filename": r"^[a-zA-Z0-9._-]+$",
    "file_path": r"^[a-zA-Z0-9/._-]+$",
    "api_key": r"^[a-zA-Z0-9\-]{32,64}$",
    "jwt_token": r"^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]*$"
}


# ============================================================================
# BASIC VALIDATION FUNCTIONS
# ============================================================================

def validate_string(
    value: Any,
    field_name: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
    required: bool = True
) -> str:
    """
    Validate string value
    
    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        min_length: Minimum length
        max_length: Maximum length
        pattern: Regex pattern to match
        required: Whether the field is required
        
    Returns:
        Validated string value
        
    Raises:
        InputValidationError: If validation fails
    """
    if value is None:
        if required:
            raise InputValidationError(f"{field_name} is required", field_name, value)
        return ""
    
    if not isinstance(value, str):
        raise TypeValidationError(
            f"{field_name} must be a string",
            field_name,
            value,
            "string",
            type(value).__name__
        )
    
    value = value.strip()
    
    if min_length is not None and len(value) < min_length:
        raise InputValidationError(
            f"{field_name} must be at least {min_length} characters long",
            field_name,
            value,
            [f"min_length: {min_length}"]
        )
    
    if max_length is not None and len(value) > max_length:
        raise InputValidationError(
            f"{field_name} must be at most {max_length} characters long",
            field_name,
            value,
            [f"max_length: {max_length}"]
        )
    
    if pattern and not re.match(pattern, value):
        raise InputValidationError(
            f"{field_name} format is invalid",
            field_name,
            value,
            [f"pattern: {pattern}"]
        )
    
    return value


def validate_integer(
    value: Any,
    field_name: str,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    required: bool = True
) -> int:
    """
    Validate integer value
    
    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        min_value: Minimum value
        max_value: Maximum value
        required: Whether the field is required
        
    Returns:
        Validated integer value
        
    Raises:
        InputValidationError: If validation fails
    """
    if value is None:
        if required:
            raise InputValidationError(f"{field_name} is required", field_name, value)
        return 0
    
    try:
        int_value = int(value)
    except (ValueError, TypeError):
        raise TypeValidationError(
            f"{field_name} must be an integer",
            field_name,
            value,
            "integer",
            type(value).__name__
        )
    
    if min_value is not None and int_value < min_value:
        raise InputValidationError(
            f"{field_name} must be at least {min_value}",
            field_name,
            int_value,
            [f"min_value: {min_value}"]
        )
    
    if max_value is not None and int_value > max_value:
        raise InputValidationError(
            f"{field_name} must be at most {max_value}",
            field_name,
            int_value,
            [f"max_value: {max_value}"]
        )
    
    return int_value


def validate_float(
    value: Any,
    field_name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    required: bool = True
) -> float:
    """
    Validate float value
    
    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        min_value: Minimum value
        max_value: Maximum value
        required: Whether the field is required
        
    Returns:
        Validated float value
        
    Raises:
        InputValidationError: If validation fails
    """
    if value is None:
        if required:
            raise InputValidationError(f"{field_name} is required", field_name, value)
        return 0.0
    
    try:
        float_value = float(value)
    except (ValueError, TypeError):
        raise TypeValidationError(
            f"{field_name} must be a number",
            field_name,
            value,
            "float",
            type(value).__name__
        )
    
    if min_value is not None and float_value < min_value:
        raise InputValidationError(
            f"{field_name} must be at least {min_value}",
            field_name,
            float_value,
            [f"min_value: {min_value}"]
        )
    
    if max_value is not None and float_value > max_value:
        raise InputValidationError(
            f"{field_name} must be at most {max_value}",
            field_name,
            float_value,
            [f"max_value: {max_value}"]
        )
    
    return float_value


def validate_boolean(
    value: Any,
    field_name: str,
    required: bool = True
) -> bool:
    """
    Validate boolean value
    
    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        required: Whether the field is required
        
    Returns:
        Validated boolean value
        
    Raises:
        InputValidationError: If validation fails
    """
    if value is None:
        if required:
            raise InputValidationError(f"{field_name} is required", field_name, value)
        return False
    
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        value_lower = value.lower()
        if value_lower in ('true', '1', 'yes', 'on'):
            return True
        elif value_lower in ('false', '0', 'no', 'off'):
            return False
    
    if isinstance(value, (int, float)):
        return bool(value)
    
    raise TypeValidationError(
        f"{field_name} must be a boolean",
        field_name,
        value,
        "boolean",
        type(value).__name__
    )


# ============================================================================
# SPECIALIZED VALIDATION FUNCTIONS
# ============================================================================

def validate_email_address(
    email: Any,
    field_name: str = "email",
    required: bool = True
) -> str:
    """
    Validate email address
    
    Args:
        email: Email address to validate
        field_name: Name of the field for error messages
        required: Whether the field is required
        
    Returns:
        Validated email address
        
    Raises:
        InputValidationError: If validation fails
    """
    email_str = validate_string(email, field_name, required=required)
    
    if not email_str:
        return email_str
    
    try:
        validated_email = validate_email(email_str)
        return validated_email.email
    except EmailNotValidError as e:
        raise InputValidationError(
            f"Invalid email address: {str(e)}",
            field_name,
            email_str,
            ["valid_email_format"]
        )


def validate_ip_address(
    ip: Any,
    field_name: str = "ip_address",
    required: bool = True,
    allow_private: bool = True,
    allow_reserved: bool = False
) -> str:
    """
    Validate IP address
    
    Args:
        ip: IP address to validate
        field_name: Name of the field for error messages
        required: Whether the field is required
        allow_private: Whether to allow private IP addresses
        allow_reserved: Whether to allow reserved IP addresses
        
    Returns:
        Validated IP address
        
    Raises:
        InputValidationError: If validation fails
    """
    ip_str = validate_string(ip, field_name, required=required)
    
    if not ip_str:
        return ip_str
    
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        
        if not allow_private and ip_obj.is_private:
            raise InputValidationError(
                f"Private IP addresses are not allowed",
                field_name,
                ip_str,
                ["no_private_ips"]
            )
        
        if not allow_reserved and ip_obj.is_reserved:
            raise InputValidationError(
                f"Reserved IP addresses are not allowed",
                field_name,
                ip_str,
                ["no_reserved_ips"]
            )
        
        return str(ip_obj)
    except ValueError:
        raise InputValidationError(
            f"Invalid IP address format",
            field_name,
            ip_str,
            ["valid_ip_format"]
        )


def validate_domain_name(
    domain: Any,
    field_name: str = "domain",
    required: bool = True,
    check_dns: bool = False
) -> str:
    """
    Validate domain name
    
    Args:
        domain: Domain name to validate
        field_name: Name of the field for error messages
        required: Whether the field is required
        check_dns: Whether to check DNS resolution
        
    Returns:
        Validated domain name
        
    Raises:
        InputValidationError: If validation fails
    """
    domain_str = validate_string(
        domain,
        field_name,
        pattern=VALIDATION_PATTERNS["domain"],
        required=required
    )
    
    if not domain_str:
        return domain_str
    
    if check_dns:
        try:
            dns.resolver.resolve(domain_str, 'A')
        except dns.resolver.NXDOMAIN:
            raise InputValidationError(
                f"Domain does not exist",
                field_name,
                domain_str,
                ["domain_exists"]
            )
        except dns.resolver.NoAnswer:
            raise InputValidationError(
                f"Domain has no A records",
                field_name,
                domain_str,
                ["domain_has_a_records"]
            )
        except Exception as e:
            raise InputValidationError(
                f"DNS resolution failed: {str(e)}",
                field_name,
                domain_str,
                ["dns_resolution"]
            )
    
    return domain_str.lower()


def validate_url(
    url: Any,
    field_name: str = "url",
    required: bool = True,
    allowed_schemes: Optional[List[str]] = None,
    check_connectivity: bool = False
) -> str:
    """
    Validate URL
    
    Args:
        url: URL to validate
        field_name: Name of the field for error messages
        required: Whether the field is required
        allowed_schemes: List of allowed URL schemes
        check_connectivity: Whether to check URL connectivity
        
    Returns:
        Validated URL
        
    Raises:
        InputValidationError: If validation fails
    """
    url_str = validate_string(url, field_name, required=required)
    
    if not url_str:
        return url_str
    
    try:
        parsed_url = urlparse(url_str)
        
        if not parsed_url.scheme:
            raise InputValidationError(
                f"URL must have a scheme (http:// or https://)",
                field_name,
                url_str,
                ["url_scheme_required"]
            )
        
        if not parsed_url.netloc:
            raise InputValidationError(
                f"URL must have a valid hostname",
                field_name,
                url_str,
                ["url_hostname_required"]
            )
        
        if allowed_schemes and parsed_url.scheme not in allowed_schemes:
            raise InputValidationError(
                f"URL scheme must be one of: {', '.join(allowed_schemes)}",
                field_name,
                url_str,
                [f"allowed_schemes: {allowed_schemes}"]
            )
        
        if check_connectivity:
            try:
                response = socket.gethostbyname(parsed_url.netloc)
            except socket.gaierror:
                raise InputValidationError(
                    f"URL hostname is not reachable",
                    field_name,
                    url_str,
                    ["url_connectivity"]
                )
        
        return url_str
    except Exception as e:
        raise InputValidationError(
            f"Invalid URL format: {str(e)}",
            field_name,
            url_str,
            ["valid_url_format"]
        )


def validate_port_number(
    port: Any,
    field_name: str = "port",
    required: bool = True,
    allow_privileged: bool = False
) -> int:
    """
    Validate port number
    
    Args:
        port: Port number to validate
        field_name: Name of the field for error messages
        required: Whether the field is required
        allow_privileged: Whether to allow privileged ports (1-1023)
        
    Returns:
        Validated port number
        
    Raises:
        InputValidationError: If validation fails
    """
    port_int = validate_integer(port, field_name, required=required)
    
    if port_int < 1 or port_int > 65535:
        raise InputValidationError(
            f"Port number must be between 1 and 65535",
            field_name,
            port_int,
            ["port_range: 1-65535"]
        )
    
    if not allow_privileged and port_int <= 1023:
        raise InputValidationError(
            f"Privileged ports (1-1023) are not allowed",
            field_name,
            port_int,
            ["no_privileged_ports"]
        )
    
    return port_int


def validate_password_strength(
    password: Any,
    field_name: str = "password",
    min_length: int = 8,
    require_uppercase: bool = True,
    require_lowercase: bool = True,
    require_digits: bool = True,
    require_special: bool = True,
    required: bool = True
) -> Dict[str, Any]:
    """
    Validate password strength
    
    Args:
        password: Password to validate
        field_name: Name of the field for error messages
        min_length: Minimum password length
        require_uppercase: Whether to require uppercase letters
        require_lowercase: Whether to require lowercase letters
        require_digits: Whether to require digits
        require_special: Whether to require special characters
        required: Whether the field is required
        
    Returns:
        Dictionary with validation results and score
        
    Raises:
        InputValidationError: If validation fails
    """
    password_str = validate_string(password, field_name, required=required)
    
    if not password_str:
        return {
            "is_valid": False,
            "score": 0,
            "strength": "empty",
            "issues": ["password_required"]
        }
    
    issues = []
    score = 0
    
    # Length check
    if len(password_str) < min_length:
        issues.append(f"min_length: {min_length}")
    else:
        score += min(len(password_str) - min_length + 1, 10)
    
    # Character type checks
    if require_uppercase and not re.search(r'[A-Z]', password_str):
        issues.append("uppercase_required")
    else:
        score += 2
    
    if require_lowercase and not re.search(r'[a-z]', password_str):
        issues.append("lowercase_required")
    else:
        score += 2
    
    if require_digits and not re.search(r'\d', password_str):
        issues.append("digits_required")
    else:
        score += 2
    
    if require_special and not re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]', password_str):
        issues.append("special_chars_required")
    else:
        score += 2
    
    # Determine strength
    if score >= 8:
        strength = "strong"
    elif score >= 6:
        strength = "medium"
    elif score >= 4:
        strength = "weak"
    else:
        strength = "very_weak"
    
    is_valid = len(issues) == 0
    
    if not is_valid:
        raise InputValidationError(
            f"Password does not meet strength requirements",
            field_name,
            password_str,
            issues
        )
    
    return {
        "is_valid": is_valid,
        "score": score,
        "strength": strength,
        "issues": issues
    }


# ============================================================================
# COMPOSITE VALIDATION FUNCTIONS
# ============================================================================

def validate_host_port(
    host: Any,
    port: Any,
    host_field: str = "host",
    port_field: str = "port",
    required: bool = True
) -> Tuple[str, int]:
    """
    Validate host and port combination
    
    Args:
        host: Host to validate
        port: Port to validate
        host_field: Name of the host field
        port_field: Name of the port field
        required: Whether the fields are required
        
    Returns:
        Tuple of (validated_host, validated_port)
        
    Raises:
        InputValidationError: If validation fails
    """
    validated_host = validate_string(host, host_field, required=required)
    validated_port = validate_port_number(port, port_field, required=required)
    
    return validated_host, validated_port


def validate_network_range(
    network: Any,
    field_name: str = "network",
    required: bool = True,
    allow_private: bool = True
) -> str:
    """
    Validate network range (CIDR notation)
    
    Args:
        network: Network range to validate
        field_name: Name of the field for error messages
        required: Whether the field is required
        allow_private: Whether to allow private networks
        
    Returns:
        Validated network range
        
    Raises:
        InputValidationError: If validation fails
    """
    network_str = validate_string(network, field_name, required=required)
    
    if not network_str:
        return network_str
    
    try:
        network_obj = ipaddress.ip_network(network_str, strict=False)
        
        if not allow_private and network_obj.is_private:
            raise InputValidationError(
                f"Private networks are not allowed",
                field_name,
                network_str,
                ["no_private_networks"]
            )
        
        return str(network_obj)
    except ValueError:
        raise InputValidationError(
            f"Invalid network range format (use CIDR notation)",
            field_name,
            network_str,
            ["valid_cidr_format"]
        )


def validate_file_info(
    filename: Any,
    file_size: Any,
    file_type: Any,
    filename_field: str = "filename",
    size_field: str = "file_size",
    type_field: str = "file_type",
    max_size: Optional[int] = None,
    allowed_types: Optional[List[str]] = None,
    required: bool = True
) -> Dict[str, Any]:
    """
    Validate file information
    
    Args:
        filename: Filename to validate
        file_size: File size to validate
        file_type: File type to validate
        filename_field: Name of the filename field
        size_field: Name of the size field
        type_field: Name of the type field
        max_size: Maximum file size in bytes
        allowed_types: List of allowed file types
        required: Whether the fields are required
        
    Returns:
        Dictionary with validated file information
        
    Raises:
        InputValidationError: If validation fails
    """
    validated_filename = validate_string(
        filename,
        filename_field,
        pattern=VALIDATION_PATTERNS["filename"],
        required=required
    )
    
    validated_size = validate_integer(
        file_size,
        size_field,
        min_value=0,
        required=required
    )
    
    validated_type = validate_string(
        file_type,
        type_field,
        required=required
    )
    
    if max_size and validated_size > max_size:
        raise InputValidationError(
            f"File size exceeds maximum allowed size of {max_size} bytes",
            size_field,
            validated_size,
            [f"max_size: {max_size}"]
        )
    
    if allowed_types and validated_type.lower() not in [t.lower() for t in allowed_types]:
        raise InputValidationError(
            f"File type must be one of: {', '.join(allowed_types)}",
            type_field,
            validated_type,
            [f"allowed_types: {allowed_types}"]
        )
    
    return {
        "filename": validated_filename,
        "file_size": validated_size,
        "file_type": validated_type.lower()
    }


# ============================================================================
# BULK VALIDATION FUNCTIONS
# ============================================================================

def validate_multiple_fields(
    data: Dict[str, Any],
    field_validations: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Validate multiple fields at once
    
    Args:
        data: Dictionary containing field values
        field_validations: Dictionary mapping field names to validation rules
        
    Returns:
        Dictionary with validation results for each field
        
    Raises:
        InputValidationError: If any validation fails
    """
    results = {}
    all_errors = []
    
    for field_name, validation_rules in field_validations.items():
        try:
            field_value = data.get(field_name)
            validation_type = validation_rules.get("type", "string")
            
            if validation_type == "string":
                result = validate_string(
                    field_value,
                    field_name,
                    min_length=validation_rules.get("min_length"),
                    max_length=validation_rules.get("max_length"),
                    pattern=validation_rules.get("pattern"),
                    required=validation_rules.get("required", True)
                )
            elif validation_type == "integer":
                result = validate_integer(
                    field_value,
                    field_name,
                    min_value=validation_rules.get("min_value"),
                    max_value=validation_rules.get("max_value"),
                    required=validation_rules.get("required", True)
                )
            elif validation_type == "email":
                result = validate_email_address(
                    field_value,
                    field_name,
                    required=validation_rules.get("required", True)
                )
            elif validation_type == "ip":
                result = validate_ip_address(
                    field_value,
                    field_name,
                    required=validation_rules.get("required", True),
                    allow_private=validation_rules.get("allow_private", True)
                )
            elif validation_type == "url":
                result = validate_url(
                    field_value,
                    field_name,
                    required=validation_rules.get("required", True),
                    allowed_schemes=validation_rules.get("allowed_schemes")
                )
            elif validation_type == "port":
                result = validate_port_number(
                    field_value,
                    field_name,
                    required=validation_rules.get("required", True),
                    allow_privileged=validation_rules.get("allow_privileged", False)
                )
            elif validation_type == "password":
                result = validate_password_strength(
                    field_value,
                    field_name,
                    min_length=validation_rules.get("min_length", 8),
                    require_uppercase=validation_rules.get("require_uppercase", True),
                    require_lowercase=validation_rules.get("require_lowercase", True),
                    require_digits=validation_rules.get("require_digits", True),
                    require_special=validation_rules.get("require_special", True),
                    required=validation_rules.get("required", True)
                )
            else:
                raise ValueError(f"Unknown validation type: {validation_type}")
            
            results[field_name] = {
                "is_valid": True,
                "value": result,
                "errors": []
            }
            
        except (InputValidationError, TypeValidationError) as e:
            results[field_name] = {
                "is_valid": False,
                "value": field_value,
                "errors": [str(e)]
            }
            all_errors.append(f"{field_name}: {str(e)}")
    
    if all_errors:
        raise InputValidationError(
            f"Multiple validation errors: {'; '.join(all_errors)}",
            "multiple_fields",
            data,
            all_errors
        )
    
    return results


# ============================================================================
# VALIDATION DECORATORS
# ============================================================================

def validate_input(
    validation_rules: Dict[str, Dict[str, Any]],
    error_handler: Optional[Callable] = None
) -> Callable:
    """
    Decorator to validate function input parameters
    
    Args:
        validation_rules: Dictionary mapping parameter names to validation rules
        error_handler: Custom error handler function
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                # Combine args and kwargs for validation
                param_dict = {}
                
                # Get function parameter names
                import inspect
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                
                # Map positional arguments
                for i, arg in enumerate(args):
                    if i < len(param_names):
                        param_dict[param_names[i]] = arg
                
                # Add keyword arguments
                param_dict.update(kwargs)
                
                # Validate parameters
                validation_results = validate_multiple_fields(param_dict, validation_rules)
                
                # Call original function
                return func(*args, **kwargs)
                
            except (InputValidationError, TypeValidationError) as e:
                if error_handler:
                    return error_handler(e)
                else:
                    raise
        
        return wrapper
    return decorator


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage of validation functions
    print("🔍 Validation Utilities Example")
    
    # Test basic validation
    try:
        email = validate_email_address("user@example.com")
        print(f"Valid email: {email}")
    except InputValidationError as e:
        print(f"Email validation error: {e}")
    
    # Test IP validation
    try:
        ip = validate_ip_address("192.168.1.1")
        print(f"Valid IP: {ip}")
    except InputValidationError as e:
        print(f"IP validation error: {e}")
    
    # Test password strength
    try:
        password_result = validate_password_strength("SecureP@ss123")
        print(f"Password validation: {password_result}")
    except InputValidationError as e:
        print(f"Password validation error: {e}")
    
    # Test multiple field validation
    try:
        data = {
            "username": "john_doe",
            "email": "john@example.com",
            "age": "25"
        }
        
        validation_rules = {
            "username": {"type": "string", "min_length": 3, "max_length": 50},
            "email": {"type": "email", "required": True},
            "age": {"type": "integer", "min_value": 18, "max_value": 100}
        }
        
        results = validate_multiple_fields(data, validation_rules)
        print(f"Multiple field validation: {results}")
    except InputValidationError as e:
        print(f"Multiple field validation error: {e}")
    
    # Test validation decorator
    @validate_input({
        "name": {"type": "string", "min_length": 2, "max_length": 50},
        "age": {"type": "integer", "min_value": 0, "max_value": 150}
    })
    def create_user(name: str, age: int):
        return {"name": name, "age": age}
    
    try:
        user = create_user("John", 25)
        print(f"Created user: {user}")
    except InputValidationError as e:
        print(f"User creation error: {e}") 