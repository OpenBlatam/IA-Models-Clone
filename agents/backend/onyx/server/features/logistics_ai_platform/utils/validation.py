"""
Input validation and sanitization utilities

This module provides utilities for validating and sanitizing user input.
"""

import re
from typing import Any, Optional, List, Dict
from urllib.parse import quote, unquote

from utils.exceptions import ValidationError
from utils.logger import logger


def sanitize_string(value: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize string input
    
    Args:
        value: String to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
        
    Raises:
        ValidationError: If value is invalid
    """
    if not isinstance(value, str):
        raise ValidationError("Value must be a string", field="value")
    
    # Remove leading/trailing whitespace
    sanitized = value.strip()
    
    # Remove null bytes
    sanitized = sanitized.replace('\x00', '')
    
    # Check length
    if max_length and len(sanitized) > max_length:
        raise ValidationError(
            f"String exceeds maximum length of {max_length}",
            field="value"
        )
    
    return sanitized


def validate_email(email: str) -> str:
    """
    Validate email address
    
    Args:
        email: Email to validate
        
    Returns:
        Validated email
        
    Raises:
        ValidationError: If email is invalid
    """
    if not email:
        raise ValidationError("Email is required", field="email")
    
    email = sanitize_string(email, max_length=254)
    
    # Basic email regex
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        raise ValidationError("Invalid email format", field="email")
    
    return email.lower()


def validate_phone(phone: str) -> str:
    """
    Validate phone number
    
    Args:
        phone: Phone number to validate
        
    Returns:
        Validated phone number
        
    Raises:
        ValidationError: If phone is invalid
    """
    if not phone:
        raise ValidationError("Phone is required", field="phone")
    
    phone = sanitize_string(phone, max_length=20)
    
    # Remove common formatting characters
    phone_clean = re.sub(r'[\s\-\(\)\+]', '', phone)
    
    # Check if it's numeric (with optional + prefix)
    if not re.match(r'^\+?[0-9]{7,15}$', phone_clean):
        raise ValidationError("Invalid phone number format", field="phone")
    
    return phone


def validate_port_code(port_code: str) -> str:
    """
    Validate port code (UN/LOCODE format)
    
    Args:
        port_code: Port code to validate
        
    Returns:
        Validated port code
        
    Raises:
        ValidationError: If port code is invalid
    """
    if not port_code:
        raise ValidationError("Port code is required", field="port_code")
    
    port_code = sanitize_string(port_code, max_length=10).upper()
    
    # Port codes are typically 5 characters (2 country + 3 location)
    if not re.match(r'^[A-Z]{2}[A-Z0-9]{3}$', port_code):
        raise ValidationError(
            "Invalid port code format. Expected format: XXYYY (e.g., MXVER)",
            field="port_code"
        )
    
    return port_code


def validate_positive_number(
    value: float,
    field_name: str = "value",
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> float:
    """
    Validate positive number
    
    Args:
        value: Number to validate
        field_name: Field name for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Validated number
        
    Raises:
        ValidationError: If number is invalid
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"{field_name} must be a number",
            field=field_name
        )
    
    if value < 0:
        raise ValidationError(
            f"{field_name} must be positive",
            field=field_name
        )
    
    if min_value is not None and value < min_value:
        raise ValidationError(
            f"{field_name} must be at least {min_value}",
            field=field_name
        )
    
    if max_value is not None and value > max_value:
        raise ValidationError(
            f"{field_name} must be at most {max_value}",
            field=field_name
        )
    
    return float(value)


def sanitize_dict(data: Dict[str, Any], max_depth: int = 10) -> Dict[str, Any]:
    """
    Recursively sanitize dictionary values
    
    Args:
        data: Dictionary to sanitize
        max_depth: Maximum recursion depth
        
    Returns:
        Sanitized dictionary
    """
    if max_depth <= 0:
        logger.warning("Maximum sanitization depth reached")
        return {}
    
    sanitized = {}
    for key, value in data.items():
        # Sanitize key
        sanitized_key = sanitize_string(str(key), max_length=100)
        
        # Sanitize value
        if isinstance(value, str):
            sanitized_value = sanitize_string(value, max_length=10000)
        elif isinstance(value, dict):
            sanitized_value = sanitize_dict(value, max_depth - 1)
        elif isinstance(value, list):
            sanitized_value = [
                sanitize_dict(item, max_depth - 1) if isinstance(item, dict)
                else sanitize_string(item, max_length=10000) if isinstance(item, str)
                else item
                for item in value[:100]  # Limit list size
            ]
        else:
            sanitized_value = value
        
        sanitized[sanitized_key] = sanitized_value
    
    return sanitized


def validate_country_code(country_code: str) -> str:
    """
    Validate ISO country code
    
    Args:
        country_code: Country code to validate
        
    Returns:
        Validated country code
        
    Raises:
        ValidationError: If country code is invalid
    """
    if not country_code:
        raise ValidationError("Country code is required", field="country_code")
    
    country_code = sanitize_string(country_code, max_length=3).upper()
    
    # ISO 3166-1 alpha-2 or alpha-3
    if not re.match(r'^[A-Z]{2,3}$', country_code):
        raise ValidationError(
            "Invalid country code format. Expected ISO 3166-1 format (e.g., MX, USA)",
            field="country_code"
        )
    
    return country_code


def validate_transportation_mode(mode: str) -> str:
    """
    Validate transportation mode
    
    Args:
        mode: Transportation mode to validate
        
    Returns:
        Validated transportation mode
        
    Raises:
        ValidationError: If mode is invalid
    """
    valid_modes = ["air", "maritime", "ground", "multimodal"]
    
    if not mode:
        raise ValidationError("Transportation mode is required", field="transportation_mode")
    
    mode = sanitize_string(mode, max_length=20).lower()
    
    if mode not in valid_modes:
        raise ValidationError(
            f"Invalid transportation mode. Must be one of: {', '.join(valid_modes)}",
            field="transportation_mode"
        )
    
    return mode

