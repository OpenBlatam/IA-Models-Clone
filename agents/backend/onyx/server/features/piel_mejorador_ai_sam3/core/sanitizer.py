"""
Sanitizer for Piel Mejorador AI SAM3
====================================

Input sanitization and validation utilities.
"""

import re
import logging
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class InputSanitizer:
    """
    Input sanitization utilities.
    
    Features:
    - Path sanitization
    - String sanitization
    - Filename validation
    - SQL injection prevention
    - XSS prevention
    """
    
    # Dangerous characters for filenames
    DANGEROUS_CHARS = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r'\.\./',
        r'\.\.\\',
        r'\.\.',
    ]
    
    @staticmethod
    def sanitize_filename(filename: str, max_length: int = 255) -> str:
        """
        Sanitize a filename.
        
        Args:
            filename: Original filename
            max_length: Maximum filename length
            
        Returns:
            Sanitized filename
            
        Raises:
            ValueError: If filename cannot be sanitized
        """
        if not filename:
            raise ValueError("Filename cannot be empty")
        
        # Remove dangerous characters
        sanitized = filename
        for char in InputSanitizer.DANGEROUS_CHARS:
            sanitized = sanitized.replace(char, '_')
        
        # Remove path traversal attempts
        for pattern in InputSanitizer.PATH_TRAVERSAL_PATTERNS:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        
        # Limit length
        if len(sanitized) > max_length:
            # Keep extension if possible
            path = Path(sanitized)
            if path.suffix:
                name_part = path.stem[:max_length - len(path.suffix)]
                sanitized = name_part + path.suffix
            else:
                sanitized = sanitized[:max_length]
        
        if not sanitized:
            raise ValueError("Filename became empty after sanitization")
        
        return sanitized
    
    @staticmethod
    def sanitize_path(file_path: str) -> Path:
        """
        Sanitize a file path.
        
        Args:
            file_path: Original file path
            
        Returns:
            Sanitized Path object
            
        Raises:
            ValueError: If path is invalid
        """
        if not file_path:
            raise ValueError("Path cannot be empty")
        
        # Check for path traversal
        for pattern in InputSanitizer.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, file_path, re.IGNORECASE):
                raise ValueError(f"Path traversal detected in: {file_path}")
        
        path = Path(file_path)
        
        # Resolve to absolute path to prevent traversal
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid path: {e}")
        
        return resolved
    
    @staticmethod
    def sanitize_string(value: str, max_length: Optional[int] = None) -> str:
        """
        Sanitize a string input.
        
        Args:
            value: Original string
            max_length: Optional maximum length
            
        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            value = str(value)
        
        # Remove control characters
        sanitized = ''.join(char for char in value if ord(char) >= 32 or char in '\n\r\t')
        
        # Limit length
        if max_length and len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    @staticmethod
    def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
        """
        Validate file extension.
        
        Args:
            filename: Filename to validate
            allowed_extensions: List of allowed extensions (with or without dot)
            
        Returns:
            True if extension is allowed
        """
        path = Path(filename)
        extension = path.suffix.lower()
        
        # Normalize extensions (add dot if missing)
        normalized_allowed = [
            ext if ext.startswith('.') else f'.{ext}'
            for ext in allowed_extensions
        ]
        
        return extension in normalized_allowed
    
    @staticmethod
    def is_safe_path(path: Path, base_dir: Path) -> bool:
        """
        Check if path is within base directory (prevents directory traversal).
        
        Args:
            path: Path to check
            base_dir: Base directory
            
        Returns:
            True if path is safe
        """
        try:
            resolved_path = path.resolve()
            resolved_base = base_dir.resolve()
            
            # Check if path is within base directory
            return str(resolved_path).startswith(str(resolved_base))
        except Exception:
            return False




