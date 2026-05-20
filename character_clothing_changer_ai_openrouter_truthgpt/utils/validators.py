"""
Validation utilities for the Character Clothing Changer service.
"""

import re
from typing import Optional, Tuple
from urllib.parse import urlparse


def validate_image_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate image URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url or not url.strip():
        return False, "URL cannot be empty"
    
    url = url.strip()
    
    # Check if it's a valid URL
    try:
        parsed = urlparse(url)
        if not parsed.scheme:
            # Might be a file path
            if not url.startswith('/') and not url.startswith('./'):
                return False, "Invalid URL format"
        elif parsed.scheme not in ['http', 'https', 'file']:
            return False, f"Unsupported URL scheme: {parsed.scheme}"
    except Exception as e:
        return False, f"Invalid URL: {str(e)}"
    
    # Check file extension if it's a URL
    if url.startswith('http'):
        valid_extensions = ['.png', '.jpg', '.jpeg', '.webp']
        url_lower = url.lower()
        if not any(url_lower.endswith(ext) for ext in valid_extensions):
            # Might be a URL without extension, which is OK
            pass
    
    return True, None


def validate_prompt(prompt: str, max_length: int = 1000) -> Tuple[bool, Optional[str]]:
    """
    Validate prompt text.
    
    Args:
        prompt: Prompt to validate
        max_length: Maximum prompt length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not prompt or not prompt.strip():
        return False, "Prompt cannot be empty"
    
    if len(prompt) > max_length:
        return False, f"Prompt exceeds maximum length of {max_length} characters"
    
    return True, None


def validate_guidance_scale(guidance_scale: float) -> Tuple[bool, Optional[str]]:
    """
    Validate guidance scale value.
    
    Args:
        guidance_scale: Guidance scale to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(guidance_scale, (int, float)):
        return False, "Guidance scale must be a number"
    
    if not (1.0 <= guidance_scale <= 100.0):
        return False, f"Guidance scale must be between 1.0 and 100.0, got {guidance_scale}"
    
    return True, None


def validate_num_steps(num_steps: int) -> Tuple[bool, Optional[str]]:
    """
    Validate number of inference steps.
    
    Args:
        num_steps: Number of steps to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(num_steps, int):
        return False, "Number of steps must be an integer"
    
    if not (1 <= num_steps <= 100):
        return False, f"Number of steps must be between 1 and 100, got {num_steps}"
    
    return True, None


def validate_seed(seed: Optional[int]) -> Tuple[bool, Optional[str]]:
    """
    Validate seed value.
    
    Args:
        seed: Seed to validate (can be None)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if seed is None:
        return True, None
    
    if not isinstance(seed, int):
        return False, "Seed must be an integer"
    
    if seed < 0:
        return False, "Seed must be non-negative"
    
    if seed > 2**31 - 1:
        return False, f"Seed exceeds maximum value of {2**31 - 1}"
    
    return True, None

