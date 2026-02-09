"""
Advanced Validators
==================

Advanced validation utilities beyond basic validators.
"""

import re
from typing import Dict, Any, Optional, List
from pathlib import Path

from .validators import ValidationError


class AdvancedValidator:
    """
    Advanced validation utilities.
    """
    
    @staticmethod
    def validate_file_integrity(file_path: str) -> Dict[str, Any]:
        """
        Validate file integrity (checksum, size, format).
        
        Args:
            file_path: Path to file
            
        Returns:
            Validation result dictionary
            
        Raises:
            ValidationError: If validation fails
        """
        path = Path(file_path)
        
        if not path.exists():
            raise ValidationError(f"File does not exist: {file_path}")
        
        if not path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")
        
        # Check file is readable
        if not path.stat().st_size > 0:
            raise ValidationError(f"File is empty: {file_path}")
        
        return {
            "exists": True,
            "size_bytes": path.stat().st_size,
            "readable": True
        }
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate URL format.
        
        Args:
            url: URL string
            
        Returns:
            True if valid
        """
        pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return pattern.match(url) is not None
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email format.
        
        Args:
            email: Email string
            
        Returns:
            True if valid
        """
        pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        return pattern.match(email) is not None
    
    @staticmethod
    def validate_config_structure(config: Dict[str, Any], required_keys: List[str]) -> List[str]:
        """
        Validate configuration structure.
        
        Args:
            config: Configuration dictionary
            required_keys: List of required keys
            
        Returns:
            List of missing keys
        """
        missing = []
        for key in required_keys:
            if key not in config:
                missing.append(key)
        return missing
    
    @staticmethod
    def validate_numeric_range(value: float, min_val: float, max_val: float, name: str = "value"):
        """
        Validate numeric value is in range.
        
        Args:
            value: Value to validate
            min_val: Minimum value
            max_val: Maximum value
            name: Value name for error message
            
        Raises:
            ValidationError: If value is out of range
        """
        if value < min_val or value > max_val:
            raise ValidationError(
                f"{name} must be between {min_val} and {max_val}, got {value}"
            )
    
    @staticmethod
    def validate_string_length(value: str, min_length: int, max_length: int, name: str = "string"):
        """
        Validate string length.
        
        Args:
            value: String to validate
            min_length: Minimum length
            max_length: Maximum length
            name: String name for error message
            
        Raises:
            ValidationError: If length is invalid
        """
        if len(value) < min_length:
            raise ValidationError(
                f"{name} must be at least {min_length} characters, got {len(value)}"
            )
        if len(value) > max_length:
            raise ValidationError(
                f"{name} must be at most {max_length} characters, got {len(value)}"
            )




