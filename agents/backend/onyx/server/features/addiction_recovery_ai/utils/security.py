"""
Security utilities
"""

import hashlib
import secrets
from typing import Optional
import re


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token
    
    Args:
        length: Length of token in bytes
    
    Returns:
        Hexadecimal token string
    """
    return secrets.token_hex(length)


def hash_string(value: str, algorithm: str = "sha256") -> str:
    """
    Hash a string using specified algorithm
    
    Args:
        value: String to hash
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)
    
    Returns:
        Hexadecimal hash string
    """
    hash_algorithms = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512
    }
    
    hash_func = hash_algorithms.get(algorithm.lower(), hashlib.sha256)
    return hash_func(value.encode()).hexdigest()


def validate_password_strength(password: str) -> tuple[bool, list[str]]:
    """
    Validate password strength
    
    Args:
        password: Password to validate
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if len(password) < 8:
        issues.append("Password must be at least 8 characters long")
    
    if not re.search(r'[A-Z]', password):
        issues.append("Password must contain at least one uppercase letter")
    
    if not re.search(r'[a-z]', password):
        issues.append("Password must contain at least one lowercase letter")
    
    if not re.search(r'\d', password):
        issues.append("Password must contain at least one digit")
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        issues.append("Password must contain at least one special character")
    
    return len(issues) == 0, issues


def sanitize_input(value: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize user input
    
    Args:
        value: Input string to sanitize
        max_length: Maximum allowed length
    
    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        raise ValueError("Input must be a string")
    
    # Remove leading/trailing whitespace
    sanitized = value.strip()
    
    # Remove null bytes
    sanitized = sanitized.replace('\x00', '')
    
    # Limit length
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized


def mask_sensitive_data(value: str, visible_chars: int = 4) -> str:
    """
    Mask sensitive data (e.g., email, phone)
    
    Args:
        value: Value to mask
        visible_chars: Number of characters to keep visible
    
    Returns:
        Masked string
    """
    if not value or len(value) <= visible_chars:
        return "*" * len(value) if value else ""
    
    return value[:visible_chars] + "*" * (len(value) - visible_chars)
