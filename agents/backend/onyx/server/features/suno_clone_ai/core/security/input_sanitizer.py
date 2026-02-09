"""
Input Sanitization

Utilities for sanitizing and validating inputs.
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


class InputSanitizer:
    """Sanitize and validate inputs."""
    
    @staticmethod
    def sanitize_text(
        text: str,
        max_length: int = 1000,
        remove_special: bool = False
    ) -> str:
        """
        Sanitize text input.
        
        Args:
            text: Input text
            max_length: Maximum length
            remove_special: Remove special characters
            
        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Truncate
        text = text[:max_length]
        
        # Remove special characters if requested
        if remove_special:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def sanitize_path(
        path: str,
        allowed_extensions: Optional[list] = None
    ) -> str:
        """
        Sanitize file path.
        
        Args:
            path: File path
            allowed_extensions: Allowed file extensions
            
        Returns:
            Sanitized path
        """
        # Remove path traversal attempts
        path = path.replace('..', '').replace('//', '/')
        
        # Check extension
        if allowed_extensions:
            ext = Path(path).suffix.lower()
            if ext not in allowed_extensions:
                raise ValueError(f"File extension {ext} not allowed")
        
        return path
    
    @staticmethod
    def validate_numeric(
        value: Any,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> float:
        """
        Validate numeric value.
        
        Args:
            value: Value to validate
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Validated numeric value
        """
        try:
            num = float(value)
            
            if min_val is not None and num < min_val:
                raise ValueError(f"Value {num} below minimum {min_val}")
            
            if max_val is not None and num > max_val:
                raise ValueError(f"Value {num} above maximum {max_val}")
            
            return num
        except (ValueError, TypeError):
            raise ValueError(f"Invalid numeric value: {value}")


def sanitize_input(
    input_data: Any,
    input_type: str = "text",
    **kwargs
) -> Any:
    """
    Sanitize input based on type.
    
    Args:
        input_data: Input data
        input_type: Type of input ('text', 'path', 'numeric')
        **kwargs: Additional sanitization arguments
        
    Returns:
        Sanitized input
    """
    sanitizer = InputSanitizer()
    
    if input_type == "text":
        return sanitizer.sanitize_text(input_data, **kwargs)
    elif input_type == "path":
        return sanitizer.sanitize_path(input_data, **kwargs)
    elif input_type == "numeric":
        return sanitizer.validate_numeric(input_data, **kwargs)
    else:
        raise ValueError(f"Unknown input type: {input_type}")


def validate_input(
    input_data: Any,
    validation_rules: dict
) -> Tuple[bool, Optional[str]]:
    """
    Validate input against rules.
    
    Args:
        input_data: Input data
        validation_rules: Validation rules
        
    Returns:
        (is_valid, error_message)
    """
    # Implement validation logic
    return True, None

