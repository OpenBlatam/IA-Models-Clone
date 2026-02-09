"""
Transformer and Mapper Utilities for Piel Mejorador AI SAM3
==========================================================

Unified transformer and mapper pattern utilities.
"""

import logging
from typing import TypeVar, Callable, Any, Optional, Dict, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class Transformer(ABC):
    """Base transformer interface."""
    
    @abstractmethod
    def transform(self, input: T) -> R:
        """Transform input to output."""
        pass


class FunctionTransformer(Transformer):
    """Transformer using a function."""
    
    def __init__(
        self,
        transform_func: Callable[[T], R],
        name: Optional[str] = None
    ):
        """
        Initialize function transformer.
        
        Args:
            transform_func: Transformation function
            name: Optional transformer name
        """
        self._transform_func = transform_func
        self.name = name or transform_func.__name__
    
    def transform(self, input: T) -> R:
        """Transform input."""
        return self._transform_func(input)


class Mapper(ABC):
    """Base mapper interface."""
    
    @abstractmethod
    def map(self, source: T, target: Optional[Any] = None) -> R:
        """Map source to target."""
        pass


class FunctionMapper(Mapper):
    """Mapper using a function."""
    
    def __init__(
        self,
        map_func: Callable[[T], R],
        name: Optional[str] = None
    ):
        """
        Initialize function mapper.
        
        Args:
            map_func: Mapping function
            name: Optional mapper name
        """
        self._map_func = map_func
        self.name = name or map_func.__name__
    
    def map(self, source: T, target: Optional[Any] = None) -> R:
        """Map source."""
        return self._map_func(source)


class DictMapper(Mapper):
    """Mapper for dictionary key mapping."""
    
    def __init__(
        self,
        field_mapping: Dict[str, str],
        name: str = "dict_mapper"
    ):
        """
        Initialize dictionary mapper.
        
        Args:
            field_mapping: Mapping from source keys to target keys
            name: Mapper name
        """
        self._field_mapping = field_mapping
        self.name = name
    
    def map(self, source: Dict[str, Any], target: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Map dictionary."""
        result = target.copy() if target else {}
        for source_key, target_key in self._field_mapping.items():
            if source_key in source:
                result[target_key] = source[source_key]
        return result


class TransformerUtils:
    """Unified transformer utilities."""
    
    @staticmethod
    def create_transformer(
        transform_func: Callable[[T], R],
        name: Optional[str] = None
    ) -> FunctionTransformer:
        """
        Create transformer from function.
        
        Args:
            transform_func: Transformation function
            name: Optional transformer name
            
        Returns:
            FunctionTransformer
        """
        return FunctionTransformer(transform_func, name)
    
    @staticmethod
    def create_mapper(
        map_func: Callable[[T], R],
        name: Optional[str] = None
    ) -> FunctionMapper:
        """
        Create mapper from function.
        
        Args:
            map_func: Mapping function
            name: Optional mapper name
            
        Returns:
            FunctionMapper
        """
        return FunctionMapper(map_func, name)
    
    @staticmethod
    def create_dict_mapper(
        field_mapping: Dict[str, str],
        name: str = "dict_mapper"
    ) -> DictMapper:
        """
        Create dictionary mapper.
        
        Args:
            field_mapping: Field mapping
            name: Mapper name
            
        Returns:
            DictMapper
        """
        return DictMapper(field_mapping, name)
    
    @staticmethod
    def transform_list(
        items: List[T],
        transformer: Transformer
    ) -> List[R]:
        """
        Transform list of items.
        
        Args:
            items: Items to transform
            transformer: Transformer to use
            
        Returns:
            Transformed items
        """
        return [transformer.transform(item) for item in items]
    
    @staticmethod
    def map_list(
        items: List[T],
        mapper: Mapper
    ) -> List[R]:
        """
        Map list of items.
        
        Args:
            items: Items to map
            mapper: Mapper to use
            
        Returns:
            Mapped items
        """
        return [mapper.map(item) for item in items]


# Convenience functions
def create_transformer(transform_func: Callable[[T], R], **kwargs) -> FunctionTransformer:
    """Create transformer."""
    return TransformerUtils.create_transformer(transform_func, **kwargs)


def create_mapper(map_func: Callable[[T], R], **kwargs) -> FunctionMapper:
    """Create mapper."""
    return TransformerUtils.create_mapper(map_func, **kwargs)




