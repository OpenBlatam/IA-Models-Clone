"""
Security Utilities
==================

Security-related utilities and helpers.
"""

import hashlib
import hmac
import secrets
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def generate_api_key(length: int = 32) -> str:
    """
    Generate a secure API key.
    
    Args:
        length: Length of the API key (default: 32)
        
    Returns:
        Secure random API key string
    """
    return secrets.token_urlsafe(length)


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key for storage.
    
    Args:
        api_key: API key to hash
        
    Returns:
        Hashed API key (SHA256)
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(api_key: str, hashed_key: str) -> bool:
    """
    Verify an API key against a hash.
    
    Args:
        api_key: API key to verify
        hashed_key: Hashed API key to compare against
        
    Returns:
        True if API key matches, False otherwise
    """
    return hmac.compare_digest(
        hash_api_key(api_key),
        hashed_key
    )


def generate_webhook_secret(length: int = 32) -> str:
    """
    Generate a secure webhook secret.
    
    Args:
        length: Length of the secret (default: 32)
        
    Returns:
        Secure random secret string
    """
    return secrets.token_urlsafe(length)


def verify_webhook_signature(
    payload: str,
    signature: str,
    secret: str
) -> bool:
    """
    Verify webhook signature.
    
    Args:
        payload: Webhook payload string
        signature: Signature from header (format: sha256=...)
        secret: Webhook secret
        
    Returns:
        True if signature is valid, False otherwise
    """
    try:
        # Extract signature value (remove "sha256=" prefix)
        if signature.startswith("sha256="):
            signature_value = signature[7:]
        else:
            signature_value = signature
        
        # Generate expected signature
        expected_signature = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        return hmac.compare_digest(signature_value, expected_signature)
        
    except Exception as e:
        logger.error(f"Error verifying webhook signature: {e}")
        return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import re
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:250] + ('.' + ext if ext else '')
    return filename


def validate_url_safe(url: str) -> bool:
    """
    Validate URL is safe (basic check).
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL appears safe, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
    
    # Basic checks
    if len(url) > 2048:  # Max URL length
        return False
    
    # Check for dangerous patterns
    dangerous_patterns = [
        'javascript:',
        'data:',
        'vbscript:',
        'file:',
        'about:'
    ]
    
    url_lower = url.lower()
    for pattern in dangerous_patterns:
        if pattern in url_lower:
            return False
    
    return True

