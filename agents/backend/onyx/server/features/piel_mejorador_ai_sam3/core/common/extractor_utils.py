"""
Extractor Utilities for Piel Mejorador AI SAM3
=============================================

Unified data extraction pattern utilities.
"""

import logging
from typing import TypeVar, Callable, Any, Optional, Dict, List, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class Extractor(ABC):
    """Base extractor interface."""
    
    @abstractmethod
    def extract(self, source: T) -> R:
        """Extract data from source."""
        pass


class FunctionExtractor(Extractor):
    """Extractor using a function."""
    
    def __init__(
        self,
        extract_func: Callable[[T], R],
        name: Optional[str] = None
    ):
        """
        Initialize function extractor.
        
        Args:
            extract_func: Extraction function
            name: Optional extractor name
        """
        self._extract_func = extract_func
        self.name = name or extract_func.__name__
    
    def extract(self, source: T) -> R:
        """Extract data."""
        return self._extract_func(source)


class KeyExtractor(Extractor):
    """Extractor for dictionary keys."""
    
    def __init__(self, key: Union[str, List[str]]):
        """
        Initialize key extractor.
        
        Args:
            key: Key or nested key path (e.g., "user.name" or ["user", "name"])
        """
        if isinstance(key, str):
            self._keys = key.split('.')
        else:
            self._keys = key
        self.key = key
    
    def extract(self, source: Dict[str, Any]) -> Any:
        """
        Extract value by key.
        
        Args:
            source: Dictionary source
            
        Returns:
            Extracted value
        """
        result = source
        for k in self._keys:
            if isinstance(result, dict):
                result = result.get(k)
            else:
                return None
        return result


class AttributeExtractor(Extractor):
    """Extractor for object attributes."""
    
    def __init__(self, attr: Union[str, List[str]]):
        """
        Initialize attribute extractor.
        
        Args:
            attr: Attribute or nested attribute path (e.g., "user.name" or ["user", "name"])
        """
        if isinstance(attr, str):
            self._attrs = attr.split('.')
        else:
            self._attrs = attr
        self.attr = attr
    
    def extract(self, source: Any) -> Any:
        """
        Extract value by attribute.
        
        Args:
            source: Object source
            
        Returns:
            Extracted value
        """
        result = source
        for attr in self._attrs:
            if hasattr(result, attr):
                result = getattr(result, attr)
            else:
                return None
        return result


class ExtractorUtils:
    """Unified extractor utilities."""
    
    @staticmethod
    def create_function_extractor(
        extract_func: Callable[[T], R],
        name: Optional[str] = None
    ) -> FunctionExtractor:
        """
        Create function extractor.
        
        Args:
            extract_func: Extraction function
            name: Optional extractor name
            
        Returns:
            FunctionExtractor
        """
        return FunctionExtractor(extract_func, name)
    
    @staticmethod
    def create_key_extractor(key: Union[str, List[str]]) -> KeyExtractor:
        """
        Create key extractor.
        
        Args:
            key: Key or nested key path
            
        Returns:
            KeyExtractor
        """
        return KeyExtractor(key)
    
    @staticmethod
    def create_attribute_extractor(attr: Union[str, List[str]]) -> AttributeExtractor:
        """
        Create attribute extractor.
        
        Args:
            attr: Attribute or nested attribute path
            
        Returns:
            AttributeExtractor
        """
        return AttributeExtractor(attr)
    
    @staticmethod
    def extract_key(source: Dict[str, Any], key: Union[str, List[str]]) -> Any:
        """
        Extract value by key.
        
        Args:
            source: Dictionary source
            key: Key or nested key path
            
        Returns:
            Extracted value
        """
        return KeyExtractor(key).extract(source)
    
    @staticmethod
    def extract_attribute(source: Any, attr: Union[str, List[str]]) -> Any:
        """
        Extract value by attribute.
        
        Args:
            source: Object source
            attr: Attribute or nested attribute path
            
        Returns:
            Extracted value
        """
        return AttributeExtractor(attr).extract(source)


# Convenience functions
def create_function_extractor(extract_func: Callable[[T], R], **kwargs) -> FunctionExtractor:
    """Create function extractor."""
    return ExtractorUtils.create_function_extractor(extract_func, **kwargs)


def create_key_extractor(key: Union[str, List[str]]) -> KeyExtractor:
    """Create key extractor."""
    return ExtractorUtils.create_key_extractor(key)


def extract_key(source: Dict[str, Any], key: Union[str, List[str]]) -> Any:
    """Extract value by key."""
    return ExtractorUtils.extract_key(source, key)




