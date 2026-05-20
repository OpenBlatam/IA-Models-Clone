"""
Formatters
==========

Utilities for formatting data and strings.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from decimal import Decimal


class DataFormatter:
    """Formatter for various data types."""
    
    @staticmethod
    def format_bytes(size_bytes: int, precision: int = 2) -> str:
        """
        Format bytes to human-readable size.
        
        Args:
            size_bytes: Size in bytes
            precision: Decimal precision
            
        Returns:
            Formatted size string
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.{precision}f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.{precision}f} PB"
    
    @staticmethod
    def format_duration(seconds: float, precision: int = 2) -> str:
        """
        Format seconds to human-readable duration.
        
        Args:
            seconds: Duration in seconds
            precision: Decimal precision
            
        Returns:
            Formatted duration string
        """
        if seconds < 60:
            return f"{seconds:.{precision}f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.{precision}f}m"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.{precision}f}h"
        else:
            days = seconds / 86400
            return f"{days:.{precision}f}d"
    
    @staticmethod
    def format_number(number: float, precision: int = 2, use_separator: bool = True) -> str:
        """
        Format number with precision and optional separator.
        
        Args:
            number: Number to format
            precision: Decimal precision
            use_separator: Whether to use thousand separator
            
        Returns:
            Formatted number string
        """
        if use_separator:
            return f"{number:,.{precision}f}"
        return f"{number:.{precision}f}"
    
    @staticmethod
    def format_percentage(value: float, precision: int = 1) -> str:
        """
        Format value as percentage.
        
        Args:
            value: Value to format (0-1 or 0-100)
            precision: Decimal precision
            
        Returns:
            Formatted percentage string
        """
        # Assume value is 0-1 if less than 1, otherwise 0-100
        if value < 1:
            value *= 100
        return f"{value:.{precision}f}%"
    
    @staticmethod
    def format_datetime(dt: datetime, format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Format datetime to string.
        
        Args:
            dt: Datetime object
            format_string: Format string
            
        Returns:
            Formatted datetime string
        """
        return dt.strftime(format_string)
    
    @staticmethod
    def format_dict(
        data: Dict[str, Any],
        indent: int = 2,
        max_depth: Optional[int] = None,
        current_depth: int = 0
    ) -> str:
        """
        Format dictionary as readable string.
        
        Args:
            data: Dictionary to format
            indent: Indentation level
            max_depth: Maximum depth to format
            current_depth: Current depth (internal)
            
        Returns:
            Formatted string
        """
        if max_depth is not None and current_depth >= max_depth:
            return str(data)
        
        lines = []
        indent_str = " " * (indent * current_depth)
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}{key}:")
                lines.append(DataFormatter.format_dict(value, indent, max_depth, current_depth + 1))
            elif isinstance(value, list):
                lines.append(f"{indent_str}{key}: [")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(DataFormatter.format_dict(item, indent, max_depth, current_depth + 1))
                    else:
                        lines.append(f"{indent_str}  - {item}")
                lines.append(f"{indent_str}]")
            else:
                lines.append(f"{indent_str}{key}: {value}")
        
        return "\n".join(lines)


class StringFormatter:
    """String formatting utilities."""
    
    @staticmethod
    def truncate(text: str, max_length: int, suffix: str = "...") -> str:
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
        Convert text to URL-friendly slug.
        
        Args:
            text: Text to slugify
            
        Returns:
            Slug string
        """
        import re
        # Convert to lowercase
        text = text.lower()
        # Replace spaces and special characters with hyphens
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        # Remove leading/trailing hyphens
        return text.strip('-')
    
    @staticmethod
    def camel_to_snake(text: str) -> str:
        """
        Convert camelCase to snake_case.
        
        Args:
            text: CamelCase text
            
        Returns:
            snake_case text
        """
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    @staticmethod
    def snake_to_camel(text: str) -> str:
        """
        Convert snake_case to camelCase.
        
        Args:
            text: snake_case text
            
        Returns:
            camelCase text
        """
        components = text.split('_')
        return components[0] + ''.join(x.capitalize() for x in components[1:])




