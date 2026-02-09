"""
Converter Utilities for Piel Mejorador AI SAM3
=============================================

Unified type converter pattern utilities.
"""

import logging
from typing import TypeVar, Callable, Any, Optional, Type, Union
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class Converter(ABC):
    """Base converter interface."""
    
    @abstractmethod
    def convert(self, value: T) -> R:
        """Convert value."""
        pass


class TypeConverter(Converter):
    """Type converter."""
    
    def __init__(self, target_type: Type[R]):
        """
        Initialize type converter.
        
        Args:
            target_type: Target type
        """
        self.target_type = target_type
    
    def convert(self, value: T) -> R:
        """Convert value to target type."""
        return self.target_type(value)


class FunctionConverter(Converter):
    """Converter using a function."""
    
    def __init__(
        self,
        convert_func: Callable[[T], R],
        name: Optional[str] = None
    ):
        """
        Initialize function converter.
        
        Args:
            convert_func: Conversion function
            name: Optional converter name
        """
        self._convert_func = convert_func
        self.name = name or convert_func.__name__
    
    def convert(self, value: T) -> R:
        """Convert value."""
        return self._convert_func(value)


class ConverterUtils:
    """Unified converter utilities."""
    
    @staticmethod
    def create_type_converter(target_type: Type[R]) -> TypeConverter:
        """
        Create type converter.
        
        Args:
            target_type: Target type
            
        Returns:
            TypeConverter
        """
        return TypeConverter(target_type)
    
    @staticmethod
    def create_function_converter(
        convert_func: Callable[[T], R],
        name: Optional[str] = None
    ) -> FunctionConverter:
        """
        Create function converter.
        
        Args:
            convert_func: Conversion function
            name: Optional converter name
            
        Returns:
            FunctionConverter
        """
        return FunctionConverter(convert_func, name)
    
    @staticmethod
    def to_int(value: Any) -> int:
        """
        Convert to int.
        
        Args:
            value: Value to convert
            
        Returns:
            Integer value
        """
        return int(value)
    
    @staticmethod
    def to_float(value: Any) -> float:
        """
        Convert to float.
        
        Args:
            value: Value to convert
            
        Returns:
            Float value
        """
        return float(value)
    
    @staticmethod
    def to_str(value: Any) -> str:
        """
        Convert to string.
        
        Args:
            value: Value to convert
            
        Returns:
            String value
        """
        return str(value)
    
    @staticmethod
    def to_bool(value: Any) -> bool:
        """
        Convert to bool.
        
        Args:
            value: Value to convert
            
        Returns:
            Boolean value
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)
    
    @staticmethod
    def to_datetime(value: Union[str, datetime, int, float]) -> datetime:
        """
        Convert to datetime.
        
        Args:
            value: Value to convert (ISO string, timestamp, or datetime)
            
        Returns:
            Datetime value
        """
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        if isinstance(value, str):
            # Try ISO format
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                # Try common formats
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y']:
                    try:
                        return datetime.strptime(value, fmt)
                    except ValueError:
                        continue
                raise ValueError(f"Cannot parse datetime: {value}")
        raise TypeError(f"Cannot convert {type(value)} to datetime")
    
    @staticmethod
    def safe_convert(value: Any, target_type: Type[R], default: Optional[R] = None) -> Optional[R]:
        """
        Safely convert value with default fallback.
        
        Args:
            value: Value to convert
            target_type: Target type
            default: Default value if conversion fails
            
        Returns:
            Converted value or default
        """
        try:
            return target_type(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Conversion failed: {e}, using default")
            return default


# Convenience functions
def create_type_converter(target_type: Type[R]) -> TypeConverter:
    """Create type converter."""
    return ConverterUtils.create_type_converter(target_type)


def create_function_converter(convert_func: Callable[[T], R], **kwargs) -> FunctionConverter:
    """Create function converter."""
    return ConverterUtils.create_function_converter(convert_func, **kwargs)


def to_int(value: Any) -> int:
    """Convert to int."""
    return ConverterUtils.to_int(value)


def to_float(value: Any) -> float:
    """Convert to float."""
    return ConverterUtils.to_float(value)


def to_str(value: Any) -> str:
    """Convert to string."""
    return ConverterUtils.to_str(value)


def to_bool(value: Any) -> bool:
    """Convert to bool."""
    return ConverterUtils.to_bool(value)




