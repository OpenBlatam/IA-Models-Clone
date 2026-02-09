"""
Message Formatting Utilities for Piel Mejorador AI SAM3
=======================================================

Unified message formatting and template utilities.
"""

import re
import logging
from typing import Any, Dict, Optional, List, Callable
from string import Template
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MessageTemplate:
    """Message template with placeholders."""
    template: str
    placeholders: List[str] = None
    
    def __post_init__(self):
        """Extract placeholders from template."""
        if self.placeholders is None:
            # Find {placeholder} patterns
            self.placeholders = re.findall(r'\{(\w+)\}', self.template)
    
    def format(self, **kwargs) -> str:
        """
        Format template with values.
        
        Args:
            **kwargs: Values for placeholders
            
        Returns:
            Formatted message
        """
        return self.template.format(**kwargs)
    
    def format_safe(self, **kwargs) -> str:
        """
        Format template safely (missing placeholders become empty).
        
        Args:
            **kwargs: Values for placeholders
            
        Returns:
            Formatted message
        """
        safe_kwargs = {k: kwargs.get(k, "") for k in self.placeholders}
        return self.template.format(**safe_kwargs)


class MessageFormattingUtils:
    """Unified message formatting utilities."""
    
    @staticmethod
    def format_template(
        template: str,
        **kwargs
    ) -> str:
        """
        Format string template.
        
        Args:
            template: Template string with {placeholders}
            **kwargs: Values for placeholders
            
        Returns:
            Formatted string
        """
        return template.format(**kwargs)
    
    @staticmethod
    def format_template_safe(
        template: str,
        default: str = "",
        **kwargs
    ) -> str:
        """
        Format template safely (missing placeholders use default).
        
        Args:
            template: Template string
            default: Default value for missing placeholders
            **kwargs: Values for placeholders
            
        Returns:
            Formatted string
        """
        # Replace missing placeholders with default
        def replace_missing(match):
            key = match.group(1)
            return str(kwargs.get(key, default))
        
        return re.sub(r'\{(\w+)\}', replace_missing, template)
    
    @staticmethod
    def create_template(template: str) -> MessageTemplate:
        """
        Create message template.
        
        Args:
            template: Template string
            
        Returns:
            MessageTemplate
        """
        return MessageTemplate(template)
    
    @staticmethod
    def format_list(
        items: List[Any],
        separator: str = ", ",
        prefix: str = "",
        suffix: str = "",
        formatter: Optional[Callable[[Any], str]] = None
    ) -> str:
        """
        Format list of items.
        
        Args:
            items: List of items
            separator: Separator between items
            prefix: Prefix for the list
            suffix: Suffix for the list
            formatter: Optional formatter function
            
        Returns:
            Formatted list string
        """
        if formatter:
            formatted_items = [formatter(item) for item in items]
        else:
            formatted_items = [str(item) for item in items]
        
        result = separator.join(formatted_items)
        return f"{prefix}{result}{suffix}"
    
    @staticmethod
    def format_dict(
        data: Dict[str, Any],
        separator: str = ", ",
        key_value_separator: str = ": ",
        prefix: str = "",
        suffix: str = ""
    ) -> str:
        """
        Format dictionary.
        
        Args:
            data: Dictionary
            separator: Separator between entries
            key_value_separator: Separator between key and value
            prefix: Prefix
            suffix: Suffix
            
        Returns:
            Formatted dictionary string
        """
        entries = [
            f"{k}{key_value_separator}{v}"
            for k, v in data.items()
        ]
        result = separator.join(entries)
        return f"{prefix}{result}{suffix}"
    
    @staticmethod
    def format_table(
        headers: List[str],
        rows: List[List[Any]],
        column_separator: str = " | ",
        header_separator: str = "-"
    ) -> str:
        """
        Format data as table.
        
        Args:
            headers: Column headers
            rows: Table rows
            column_separator: Separator between columns
            header_separator: Separator for header row
            
        Returns:
            Formatted table string
        """
        if not headers or not rows:
            return ""
        
        # Calculate column widths
        widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))
        
        # Format header
        header_row = column_separator.join(
            str(h).ljust(widths[i]) for i, h in enumerate(headers)
        )
        
        # Format separator
        separator_row = header_separator * len(header_row)
        
        # Format rows
        formatted_rows = []
        for row in rows:
            formatted_row = column_separator.join(
                str(cell).ljust(widths[i]) if i < len(widths) else str(cell)
                for i, cell in enumerate(row)
            )
            formatted_rows.append(formatted_row)
        
        return "\n".join([header_row, separator_row] + formatted_rows)
    
    @staticmethod
    def format_progress(
        current: int,
        total: int,
        prefix: str = "Progress",
        show_percentage: bool = True
    ) -> str:
        """
        Format progress message.
        
        Args:
            current: Current value
            total: Total value
            prefix: Prefix text
            show_percentage: Whether to show percentage
            
        Returns:
            Formatted progress string
        """
        if total == 0:
            percentage = 0.0
        else:
            percentage = (current / total) * 100
        
        progress_str = f"{prefix}: {current}/{total}"
        if show_percentage:
            progress_str += f" ({percentage:.1f}%)"
        
        return progress_str
    
    @staticmethod
    def format_duration(
        seconds: float,
        precision: int = 2
    ) -> str:
        """
        Format duration in human-readable format.
        
        Args:
            seconds: Duration in seconds
            precision: Decimal precision
            
        Returns:
            Formatted duration string
        """
        if seconds < 1:
            return f"{seconds * 1000:.{precision}f}ms"
        elif seconds < 60:
            return f"{seconds:.{precision}f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.{precision}f}min"
        else:
            hours = seconds / 3600
            return f"{hours:.{precision}f}h"
    
    @staticmethod
    def format_size(
        bytes_size: int,
        precision: int = 2
    ) -> str:
        """
        Format size in human-readable format.
        
        Args:
            bytes_size: Size in bytes
            precision: Decimal precision
            
        Returns:
            Formatted size string
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.{precision}f}{unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.{precision}f}PB"


# Convenience functions
def format_template(template: str, **kwargs) -> str:
    """Format template."""
    return MessageFormattingUtils.format_template(template, **kwargs)


def format_list(items: List[Any], **kwargs) -> str:
    """Format list."""
    return MessageFormattingUtils.format_list(items, **kwargs)


def format_progress(current: int, total: int, **kwargs) -> str:
    """Format progress."""
    return MessageFormattingUtils.format_progress(current, total, **kwargs)


def format_duration(seconds: float, **kwargs) -> str:
    """Format duration."""
    return MessageFormattingUtils.format_duration(seconds, **kwargs)


def format_size(bytes_size: int, **kwargs) -> str:
    """Format size."""
    return MessageFormattingUtils.format_size(bytes_size, **kwargs)




