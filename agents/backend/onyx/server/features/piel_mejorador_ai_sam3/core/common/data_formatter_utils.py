"""
Data Formatter Utilities for Piel Mejorador AI SAM3
===================================================

Unified data formatting utilities.
"""

import json
import logging
from typing import TypeVar, Callable, Any, Optional, Dict, List
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Formatter(ABC):
    """Base formatter interface."""
    
    @abstractmethod
    def format(self, data: Any) -> str:
        """Format data."""
        pass


class JSONFormatter(Formatter):
    """JSON formatter."""
    
    def __init__(self, indent: Optional[int] = None, ensure_ascii: bool = False):
        """
        Initialize JSON formatter.
        
        Args:
            indent: Optional indentation
            ensure_ascii: Whether to ensure ASCII
        """
        self.indent = indent
        self.ensure_ascii = ensure_ascii
    
    def format(self, data: Any) -> str:
        """
        Format data as JSON.
        
        Args:
            data: Data to format
            
        Returns:
            JSON string
        """
        return json.dumps(data, indent=self.indent, ensure_ascii=self.ensure_ascii, default=str)


class FunctionFormatter(Formatter):
    """Formatter using a function."""
    
    def __init__(
        self,
        format_func: Callable[[Any], str],
        name: Optional[str] = None
    ):
        """
        Initialize function formatter.
        
        Args:
            format_func: Formatting function
            name: Optional formatter name
        """
        self._format_func = format_func
        self.name = name or format_func.__name__
    
    def format(self, data: Any) -> str:
        """Format data."""
        return self._format_func(data)


class TemplateFormatter(Formatter):
    """Template-based formatter."""
    
    def __init__(self, template: str):
        """
        Initialize template formatter.
        
        Args:
            template: Format template with {placeholders}
        """
        self.template = template
    
    def format(self, data: Any) -> str:
        """
        Format data using template.
        
        Args:
            data: Data to format (dict or object with attributes)
            
        Returns:
            Formatted string
        """
        if isinstance(data, dict):
            return self.template.format(**data)
        elif hasattr(data, '__dict__'):
            return self.template.format(**data.__dict__)
        else:
            return self.template.format(data=data)


class CSVFormatter(Formatter):
    """CSV formatter."""
    
    def __init__(self, delimiter: str = ",", include_header: bool = True):
        """
        Initialize CSV formatter.
        
        Args:
            delimiter: CSV delimiter
            include_header: Whether to include header row
        """
        self.delimiter = delimiter
        self.include_header = include_header
    
    def format(self, data: List[Dict[str, Any]]) -> str:
        """
        Format data as CSV.
        
        Args:
            data: List of dictionaries
            
        Returns:
            CSV string
        """
        if not data:
            return ""
        
        # Get all keys from first dict
        keys = list(data[0].keys())
        
        lines = []
        
        # Header
        if self.include_header:
            lines.append(self.delimiter.join(keys))
        
        # Data rows
        for row in data:
            values = [str(row.get(key, "")) for key in keys]
            lines.append(self.delimiter.join(values))
        
        return "\n".join(lines)


class DataFormatterUtils:
    """Unified data formatter utilities."""
    
    @staticmethod
    def create_json_formatter(indent: Optional[int] = None, ensure_ascii: bool = False) -> JSONFormatter:
        """
        Create JSON formatter.
        
        Args:
            indent: Optional indentation
            ensure_ascii: Whether to ensure ASCII
            
        Returns:
            JSONFormatter
        """
        return JSONFormatter(indent, ensure_ascii)
    
    @staticmethod
    def create_template_formatter(template: str) -> TemplateFormatter:
        """
        Create template formatter.
        
        Args:
            template: Format template
            
        Returns:
            TemplateFormatter
        """
        return TemplateFormatter(template)
    
    @staticmethod
    def create_csv_formatter(delimiter: str = ",", include_header: bool = True) -> CSVFormatter:
        """
        Create CSV formatter.
        
        Args:
            delimiter: CSV delimiter
            include_header: Whether to include header
            
        Returns:
            CSVFormatter
        """
        return CSVFormatter(delimiter, include_header)
    
    @staticmethod
    def create_function_formatter(
        format_func: Callable[[Any], str],
        name: Optional[str] = None
    ) -> FunctionFormatter:
        """
        Create function formatter.
        
        Args:
            format_func: Formatting function
            name: Optional formatter name
            
        Returns:
            FunctionFormatter
        """
        return FunctionFormatter(format_func, name)
    
    @staticmethod
    def format_json(data: Any, indent: Optional[int] = None) -> str:
        """
        Format data as JSON.
        
        Args:
            data: Data to format
            indent: Optional indentation
            
        Returns:
            JSON string
        """
        return JSONFormatter(indent).format(data)
    
    @staticmethod
    def format_csv(data: List[Dict[str, Any]], delimiter: str = ",", include_header: bool = True) -> str:
        """
        Format data as CSV.
        
        Args:
            data: List of dictionaries
            delimiter: CSV delimiter
            include_header: Whether to include header
            
        Returns:
            CSV string
        """
        return CSVFormatter(delimiter, include_header).format(data)


# Convenience functions
def create_json_formatter(**kwargs) -> JSONFormatter:
    """Create JSON formatter."""
    return DataFormatterUtils.create_json_formatter(**kwargs)


def create_template_formatter(template: str) -> TemplateFormatter:
    """Create template formatter."""
    return DataFormatterUtils.create_template_formatter(template)


def format_json(data: Any, **kwargs) -> str:
    """Format data as JSON."""
    return DataFormatterUtils.format_json(data, **kwargs)




