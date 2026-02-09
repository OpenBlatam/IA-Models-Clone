"""
String Utilities for Piel Mejorador AI SAM3
===========================================

Unified string manipulation utilities.
"""

import re
import logging
from typing import List, Optional, Callable

logger = logging.getLogger(__name__)


class StringUtils:
    """Unified string utilities."""
    
    @staticmethod
    def sanitize(
        text: str,
        allowed_chars: Optional[str] = None,
        replacement: str = "_"
    ) -> str:
        """
        Sanitize string by removing/replacing dangerous characters.
        
        Args:
            text: Text to sanitize
            allowed_chars: Optional regex pattern of allowed characters
            replacement: Replacement character
            
        Returns:
            Sanitized string
        """
        if not text:
            return ""
        
        if allowed_chars:
            # Keep only allowed characters
            pattern = f"[^{re.escape(allowed_chars)}]"
            return re.sub(pattern, replacement, text)
        else:
            # Remove common dangerous characters
            dangerous = ['<', '>', ':', '"', '|', '?', '*', '\\', '/', '\x00']
            result = text
            for char in dangerous:
                result = result.replace(char, replacement)
            return result
    
    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize string (lowercase, strip, replace spaces).
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized string
        """
        return text.lower().strip().replace(' ', '_')
    
    @staticmethod
    def truncate(
        text: str,
        max_length: int,
        suffix: str = "..."
    ) -> str:
        """
        Truncate string to max length.
        
        Args:
            text: Text to truncate
            max_length: Maximum length
            suffix: Suffix to add if truncated
            
        Returns:
            Truncated string
        """
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def slugify(text: str) -> str:
        """
        Convert string to URL-friendly slug.
        
        Args:
            text: Text to slugify
            
        Returns:
            Slug string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Replace spaces and underscores with hyphens
        text = re.sub(r'[\s_]+', '-', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\-]+', '', text)
        
        # Remove multiple hyphens
        text = re.sub(r'-+', '-', text)
        
        # Remove leading/trailing hyphens
        text = text.strip('-')
        
        return text
    
    @staticmethod
    def camel_to_snake(text: str) -> str:
        """
        Convert camelCase to snake_case.
        
        Args:
            text: CamelCase string
            
        Returns:
            snake_case string
        """
        # Insert underscore before uppercase letters
        text = re.sub(r'(?<!^)(?=[A-Z])', '_', text)
        return text.lower()
    
    @staticmethod
    def snake_to_camel(text: str) -> str:
        """
        Convert snake_case to camelCase.
        
        Args:
            text: snake_case string
            
        Returns:
            camelCase string
        """
        components = text.split('_')
        return components[0] + ''.join(x.capitalize() for x in components[1:])
    
    @staticmethod
    def remove_path_traversal(text: str) -> str:
        """
        Remove path traversal patterns.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        patterns = [
            r'\.\./',
            r'\.\.\\',
            r'\.\.',
        ]
        
        result = text
        for pattern in patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        
        return result
    
    @staticmethod
    def join_with_separator(
        items: List[str],
        separator: str = ", ",
        last_separator: Optional[str] = None
    ) -> str:
        """
        Join items with separator (supports different last separator).
        
        Args:
            items: List of strings
            separator: Separator between items
            last_separator: Optional separator before last item
            
        Returns:
            Joined string
        """
        if not items:
            return ""
        
        if len(items) == 1:
            return items[0]
        
        if last_separator:
            return separator.join(items[:-1]) + last_separator + items[-1]
        
        return separator.join(items)
    
    @staticmethod
    def split_safe(
        text: str,
        separator: str = ",",
        maxsplit: int = -1
    ) -> List[str]:
        """
        Split string and strip whitespace from each part.
        
        Args:
            text: Text to split
            separator: Separator
            maxsplit: Maximum splits
            
        Returns:
            List of stripped strings
        """
        if not text:
            return []
        
        parts = text.split(separator, maxsplit) if maxsplit >= 0 else text.split(separator)
        return [part.strip() for part in parts if part.strip()]


# Convenience functions
def sanitize(text: str, **kwargs) -> str:
    """Sanitize string."""
    return StringUtils.sanitize(text, **kwargs)


def slugify(text: str) -> str:
    """Convert to slug."""
    return StringUtils.slugify(text)


def truncate(text: str, max_length: int, **kwargs) -> str:
    """Truncate string."""
    return StringUtils.truncate(text, max_length, **kwargs)




