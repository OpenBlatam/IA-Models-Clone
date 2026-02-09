"""
String manipulation utilities.

Consolidates common string operations and formatting patterns.
"""

import re
import logging
from typing import Optional, List
from urllib.parse import quote, unquote

logger = logging.getLogger(__name__)


class StringUtils:
    """Utilities for string operations."""
    
    @staticmethod
    def slugify(text: str, separator: str = "-") -> str:
        """
        Convert string to URL-friendly slug.
        
        Args:
            text: Text to slugify
            separator: Separator character
        
        Returns:
            Slug string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Replace spaces and underscores with separator
        text = re.sub(r'[\s_]+', separator, text)
        
        # Remove special characters, keep alphanumeric and separator
        text = re.sub(r'[^\w\-]', '', text)
        
        # Remove multiple separators
        text = re.sub(rf'{re.escape(separator)}+', separator, text)
        
        # Remove leading/trailing separators
        text = text.strip(separator)
        
        return text
    
    @staticmethod
    def sanitize_filename(filename: str, max_length: int = 255) -> str:
        """
        Sanitize filename for safe filesystem use.
        
        Args:
            filename: Filename to sanitize
            max_length: Maximum filename length
        
        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = filename.replace('/', '_').replace('\\', '_')
        
        # Remove special characters
        filename = re.sub(r'[<>:"|?*]', '', filename)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Truncate if too long
        if len(filename) > max_length:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            max_name_length = max_length - len(ext) - 1 if ext else max_length
            filename = name[:max_name_length] + (f'.{ext}' if ext else '')
        
        return filename or 'file'
    
    @staticmethod
    def truncate(
        text: str,
        max_length: int,
        suffix: str = "..."
    ) -> str:
        """
        Truncate string to maximum length.
        
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
    def normalize_whitespace(text: str) -> str:
        """
        Normalize whitespace in string.
        
        Args:
            text: Text to normalize
        
        Returns:
            Normalized string
        """
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def remove_special_chars(
        text: str,
        keep: Optional[str] = None
    ) -> str:
        """
        Remove special characters from string.
        
        Args:
            text: Text to clean
            keep: Characters to keep (regex pattern)
        
        Returns:
            Cleaned string
        """
        if keep:
            pattern = f'[^\\w\\s{re.escape(keep)}]'
        else:
            pattern = r'[^\w\s]'
        
        return re.sub(pattern, '', text)
    
    @staticmethod
    def camel_to_snake(text: str) -> str:
        """Convert camelCase to snake_case."""
        # Insert underscore before uppercase letters
        text = re.sub(r'(?<!^)(?=[A-Z])', '_', text)
        return text.lower()
    
    @staticmethod
    def snake_to_camel(text: str, capitalize_first: bool = False) -> str:
        """
        Convert snake_case to camelCase.
        
        Args:
            text: Text to convert
            capitalize_first: Capitalize first letter (PascalCase)
        
        Returns:
            Converted string
        """
        parts = text.split('_')
        if capitalize_first:
            return ''.join(word.capitalize() for word in parts)
        else:
            return parts[0] + ''.join(word.capitalize() for word in parts[1:])
    
    @staticmethod
    def extract_emails(text: str) -> List[str]:
        """Extract email addresses from text."""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(pattern, text)
    
    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extract URLs from text."""
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])|[a-zA-Z0-9]+'
        return re.findall(pattern, text)
    
    @staticmethod
    def mask_sensitive(text: str, visible_chars: int = 4) -> str:
        """
        Mask sensitive information (e.g., credit cards, emails).
        
        Args:
            text: Text to mask
            visible_chars: Number of visible characters at end
        
        Returns:
            Masked string
        """
        if len(text) <= visible_chars:
            return '*' * len(text)
        
        return '*' * (len(text) - visible_chars) + text[-visible_chars:]
    
    @staticmethod
    def url_encode(text: str) -> str:
        """URL encode string."""
        return quote(text)
    
    @staticmethod
    def url_decode(text: str) -> str:
        """URL decode string."""
        return unquote(text)


# Convenience functions
def slugify(text: str, separator: str = "-") -> str:
    """Convert string to URL-friendly slug."""
    return StringUtils.slugify(text, separator)


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize filename for safe filesystem use."""
    return StringUtils.sanitize_filename(filename, max_length)


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate string to maximum length."""
    return StringUtils.truncate(text, max_length, suffix)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in string."""
    return StringUtils.normalize_whitespace(text)


def camel_to_snake(text: str) -> str:
    """Convert camelCase to snake_case."""
    return StringUtils.camel_to_snake(text)


def snake_to_camel(text: str, capitalize_first: bool = False) -> str:
    """Convert snake_case to camelCase."""
    return StringUtils.snake_to_camel(text, capitalize_first)


def mask_sensitive(text: str, visible_chars: int = 4) -> str:
    """Mask sensitive information."""
    return StringUtils.mask_sensitive(text, visible_chars)

