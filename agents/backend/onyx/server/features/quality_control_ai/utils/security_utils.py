"""
Security Utilities

Utility functions for security operations.
"""

import hashlib
import secrets
from typing import Optional
import base64


def generate_token(length: int = 32) -> str:
    """
    Generate a secure random token.
    
    Args:
        length: Token length in bytes
    
    Returns:
        Random token as hex string
    """
    return secrets.token_hex(length)


def hash_string(text: str, algorithm: str = "sha256") -> str:
    """
    Hash a string.
    
    Args:
        text: Text to hash
        algorithm: Hash algorithm (sha256, sha512, md5)
    
    Returns:
        Hashed string
    """
    hash_func = getattr(hashlib, algorithm, hashlib.sha256)
    return hash_func(text.encode()).hexdigest()


def verify_hash(text: str, hash_value: str, algorithm: str = "sha256") -> bool:
    """
    Verify a hash against text.
    
    Args:
        text: Text to verify
        hash_value: Hash to compare
        algorithm: Hash algorithm
    
    Returns:
        True if hash matches
    """
    computed_hash = hash_string(text, algorithm)
    return secrets.compare_digest(computed_hash, hash_value)


def sanitize_input(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize user input.
    
    Args:
        text: Input text
        max_length: Optional maximum length
    
    Returns:
        Sanitized text
    """
    # Remove null bytes
    sanitized = text.replace('\x00', '')
    
    # Remove control characters except newlines and tabs
    sanitized = ''.join(
        char for char in sanitized
        if ord(char) >= 32 or char in '\n\t'
    )
    
    # Truncate if needed
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized.strip()


def encode_base64(data: bytes) -> str:
    """
    Encode bytes to base64 string.
    
    Args:
        data: Bytes to encode
    
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(data).decode('utf-8')


def decode_base64(encoded: str) -> bytes:
    """
    Decode base64 string to bytes.
    
    Args:
        encoded: Base64 encoded string
    
    Returns:
        Decoded bytes
    """
    return base64.b64decode(encoded)


def mask_sensitive_data(text: str, visible_chars: int = 4) -> str:
    """
    Mask sensitive data (e.g., API keys, tokens).
    
    Args:
        text: Text to mask
        visible_chars: Number of characters to show at start and end
    
    Returns:
        Masked text
    """
    if len(text) <= visible_chars * 2:
        return '*' * len(text)
    
    start = text[:visible_chars]
    end = text[-visible_chars:]
    middle = '*' * (len(text) - visible_chars * 2)
    
    return f"{start}{middle}{end}"



