"""Security utilities."""

import hashlib
import secrets
import hmac
from typing import Optional
from datetime import datetime, timedelta
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)


def generate_token(length: int = 32) -> str:
    """
    Generate a secure random token.
    
    Args:
        length: Token length in bytes
        
    Returns:
        Hexadecimal token string
    """
    return secrets.token_hex(length)


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a secure random token (alias for generate_token).
    
    Args:
        length: Token length in bytes
        
    Returns:
        Secure random token as hex string
    """
    return generate_token(length)


def generate_api_key(prefix: str = "psk") -> str:
    """
    Generate a secure API key.
    
    Args:
        prefix: Prefix for the API key
        
    Returns:
        API key string
    """
    token = secrets.token_urlsafe(32)
    return f"{prefix}_{token}"


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """
    Hash a password using SHA-256.
    
    Args:
        password: Password to hash
        salt: Optional salt (generated if not provided)
        
    Returns:
        Tuple of (hashed_password, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    hashed = hashlib.sha256((password + salt).encode()).hexdigest()
    return hashed, salt


def verify_password(password: str, hashed: str, salt: str) -> bool:
    """
    Verify a password against a hash.
    
    Args:
        password: Password to verify
        hashed: Hashed password
        salt: Salt used for hashing
        
    Returns:
        True if password matches
    """
    new_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(new_hash, hashed)


def generate_secure_filename(original_filename: str) -> str:
    """
    Generate a secure filename.
    
    Args:
        original_filename: Original filename
        
    Returns:
        Secure filename
    """
    # Get extension
    if '.' in original_filename:
        name, ext = original_filename.rsplit('.', 1)
        ext = f".{ext}"
    else:
        name = original_filename
        ext = ""
    
    # Generate secure name
    secure_name = secrets.token_urlsafe(16)
    return f"{secure_name}{ext}"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal.
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        Sanitized filename
    """
    import os
    
    # Remove path components
    filename = os.path.basename(filename)
    
    # Remove dangerous characters
    dangerous_chars = ['/', '\\', '..', '\x00']
    for char in dangerous_chars:
        filename = filename.replace(char, '')
    
    # Limit length
    max_length = 255
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length - len(ext)] + ext
    
    return filename


def validate_api_key_format(api_key: str, prefix: str = "psk") -> bool:
    """
    Validate API key format.
    
    Args:
        api_key: API key to validate
        prefix: Expected prefix
        
    Returns:
        True if format is valid
    """
    if not api_key.startswith(f"{prefix}_"):
        return False
    
    parts = api_key.split('_', 1)
    if len(parts) != 2:
        return False
    
    token = parts[1]
    if len(token) < 32:  # Minimum token length
        return False
    
    return True


def mask_sensitive_data(data: str, visible_chars: int = 4) -> str:
    """
    Mask sensitive data (e.g., API keys, tokens).
    
    Args:
        data: Data to mask
        visible_chars: Number of characters to show at start and end
        
    Returns:
        Masked string
    """
    if len(data) <= visible_chars * 2:
        return "*" * len(data)
    
    start = data[:visible_chars]
    end = data[-visible_chars:]
    middle = "*" * (len(data) - visible_chars * 2)
    
    return f"{start}{middle}{end}"


def generate_csrf_token() -> str:
    """
    Generate a CSRF token.
    
    Returns:
        CSRF token
    """
    return secrets.token_urlsafe(32)


def constant_time_compare(a: str, b: str) -> bool:
    """
    Compare two strings in constant time to prevent timing attacks.
    
    Args:
        a: First string
        b: Second string
        
    Returns:
        True if strings are equal
    """
    return hmac.compare_digest(a.encode(), b.encode())


def validate_file_extension(filename: str, allowed_extensions: list[str]) -> bool:
    """
    Validate file extension.
    
    Args:
        filename: Filename to validate
        allowed_extensions: List of allowed extensions (with or without dot)
        
    Returns:
        True if extension is allowed
    """
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    allowed = [e.lower().lstrip('.') for e in allowed_extensions]
    
    return ext in allowed


def check_file_size(file_size: int, max_size_mb: int) -> bool:
    """
    Check if file size is within limits.
    
    Args:
        file_size: File size in bytes
        max_size_mb: Maximum size in MB
        
    Returns:
        True if size is within limits
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    return file_size <= max_size_bytes


def hash_sensitive_data(data: str) -> str:
    """
    Hash sensitive data (one-way).
    
    Args:
        data: Data to hash
        
    Returns:
        SHA256 hash as hex string
    """
    return hashlib.sha256(data.encode()).hexdigest()


def validate_file_path(file_path: Path, allowed_dir: Path) -> bool:
    """
    Validate that file path is within allowed directory.
    
    Args:
        file_path: Path to validate
        allowed_dir: Allowed base directory
        
    Returns:
        True if path is safe
    """
    try:
        resolved_path = file_path.resolve()
        resolved_allowed = allowed_dir.resolve()
        return str(resolved_path).startswith(str(resolved_allowed))
    except Exception:
        return False


def check_file_type(file_path: Path, allowed_extensions: list) -> bool:
    """
    Check if file has allowed extension.
    
    Args:
        file_path: Path to file
        allowed_extensions: List of allowed extensions (without dot)
        
    Returns:
        True if extension is allowed
    """
    ext = file_path.suffix.lstrip('.').lower()
    return ext in [e.lower() for e in allowed_extensions]

