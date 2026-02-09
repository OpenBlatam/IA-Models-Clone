"""
Combiner Utilities for Piel Mejorador AI SAM3
============================================

Unified data combination pattern utilities.
"""

import logging
from typing import TypeVar, Callable, Any, Optional, List, Dict, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class Combiner(ABC):
    """Base combiner interface."""
    
    @abstractmethod
    def combine(self, *items: T) -> R:
        """Combine items."""
        pass


class FunctionCombiner(Combiner):
    """Combiner using a function."""
    
    def __init__(
        self,
        combine_func: Callable[[List[T]], R],
        name: Optional[str] = None
    ):
        """
        Initialize function combiner.
        
        Args:
            combine_func: Combination function
            name: Optional combiner name
        """
        self._combine_func = combine_func
        self.name = name or combine_func.__name__
    
    def combine(self, *items: T) -> R:
        """Combine items."""
        return self._combine_func(list(items))


class ListCombiner(Combiner):
    """List combiner."""
    
    def __init__(self, unique: bool = False, preserve_order: bool = True):
        """
        Initialize list combiner.
        
        Args:
            unique: Whether to keep only unique items
            preserve_order: Whether to preserve order
        """
        self.unique = unique
        self.preserve_order = preserve_order
    
    def combine(self, *items: List[T]) -> List[T]:
        """
        Combine lists.
        
        Args:
            *items: Lists to combine
            
        Returns:
            Combined list
        """
        if not items:
            return []
        
        result = []
        seen = set()
        
        for item_list in items:
            for item in item_list:
                if self.unique:
                    item_repr = str(item) if not isinstance(item, (str, int, float, bool, type(None))) else item
                    if item_repr in seen:
                        continue
                    seen.add(item_repr)
                
                result.append(item)
        
        return result


class DictCombiner(Combiner):
    """Dictionary combiner."""
    
    def __init__(self, deep: bool = True, overwrite: bool = True):
        """
        Initialize dictionary combiner.
        
        Args:
            deep: Whether to deep combine
            overwrite: Whether to overwrite existing keys
        """
        self.deep = deep
        self.overwrite = overwrite
    
    def combine(self, *items: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine dictionaries.
        
        Args:
            *items: Dictionaries to combine
            
        Returns:
            Combined dictionary
        """
        if not items:
            return {}
        
        result = items[0].copy()
        if self.deep:
            from copy import deepcopy
            result = deepcopy(result)
        
        for item_dict in items[1:]:
            item_copy = item_dict.copy()
            if self.deep:
                from copy import deepcopy
                item_copy = deepcopy(item_copy)
            
            for key, value in item_copy.items():
                if key in result:
                    if self.deep and isinstance(result[key], dict) and isinstance(value, dict):
                        # Recursive combine for nested dicts
                        nested_combiner = DictCombiner(deep=True, overwrite=self.overwrite)
                        result[key] = nested_combiner.combine(result[key], value)
                    elif self.overwrite:
                        result[key] = value
                else:
                    result[key] = value
        
        return result


class TupleCombiner(Combiner):
    """Tuple combiner."""
    
    def combine(self, *items: Tuple[T, ...]) -> Tuple[T, ...]:
        """
        Combine tuples.
        
        Args:
            *items: Tuples to combine
            
        Returns:
            Combined tuple
        """
        result = []
        for item_tuple in items:
            result.extend(item_tuple)
        return tuple(result)


class StringCombiner(Combiner):
    """String combiner."""
    
    def __init__(self, separator: str = ""):
        """
        Initialize string combiner.
        
        Args:
            separator: Separator between strings
        """
        self.separator = separator
    
    def combine(self, *items: str) -> str:
        """
        Combine strings.
        
        Args:
            *items: Strings to combine
            
        Returns:
            Combined string
        """
        return self.separator.join(items)


class CombinerUtils:
    """Unified combiner utilities."""
    
    @staticmethod
    def create_function_combiner(
        combine_func: Callable[[List[T]], R],
        name: Optional[str] = None
    ) -> FunctionCombiner:
        """
        Create function combiner.
        
        Args:
            combine_func: Combination function
            name: Optional combiner name
            
        Returns:
            FunctionCombiner
        """
        return FunctionCombiner(combine_func, name)
    
    @staticmethod
    def create_list_combiner(unique: bool = False, preserve_order: bool = True) -> ListCombiner:
        """
        Create list combiner.
        
        Args:
            unique: Whether to keep only unique items
            preserve_order: Whether to preserve order
            
        Returns:
            ListCombiner
        """
        return ListCombiner(unique, preserve_order)
    
    @staticmethod
    def create_dict_combiner(deep: bool = True, overwrite: bool = True) -> DictCombiner:
        """
        Create dictionary combiner.
        
        Args:
            deep: Whether to deep combine
            overwrite: Whether to overwrite existing keys
            
        Returns:
            DictCombiner
        """
        return DictCombiner(deep, overwrite)
    
    @staticmethod
    def create_tuple_combiner() -> TupleCombiner:
        """
        Create tuple combiner.
        
        Returns:
            TupleCombiner
        """
        return TupleCombiner()
    
    @staticmethod
    def create_string_combiner(separator: str = "") -> StringCombiner:
        """
        Create string combiner.
        
        Args:
            separator: Separator between strings
            
        Returns:
            StringCombiner
        """
        return StringCombiner(separator)
    
    @staticmethod
    def combine_lists(*items: List[T], unique: bool = False, preserve_order: bool = True) -> List[T]:
        """
        Combine lists.
        
        Args:
            *items: Lists to combine
            unique: Whether to keep only unique items
            preserve_order: Whether to preserve order
            
        Returns:
            Combined list
        """
        return ListCombiner(unique, preserve_order).combine(*items)
    
    @staticmethod
    def combine_dicts(*items: Dict[str, Any], deep: bool = True, overwrite: bool = True) -> Dict[str, Any]:
        """
        Combine dictionaries.
        
        Args:
            *items: Dictionaries to combine
            deep: Whether to deep combine
            overwrite: Whether to overwrite existing keys
            
        Returns:
            Combined dictionary
        """
        return DictCombiner(deep, overwrite).combine(*items)
    
    @staticmethod
    def combine_strings(*items: str, separator: str = "") -> str:
        """
        Combine strings.
        
        Args:
            *items: Strings to combine
            separator: Separator between strings
            
        Returns:
            Combined string
        """
        return StringCombiner(separator).combine(*items)


# Convenience functions
def create_function_combiner(combine_func: Callable[[List[T]], R], **kwargs) -> FunctionCombiner:
    """Create function combiner."""
    return CombinerUtils.create_function_combiner(combine_func, **kwargs)


def combine_lists(*items: List[T], **kwargs) -> List[T]:
    """Combine lists."""
    return CombinerUtils.combine_lists(*items, **kwargs)


def combine_strings(*items: str, **kwargs) -> str:
    """Combine strings."""
    return CombinerUtils.combine_strings(*items, **kwargs)




