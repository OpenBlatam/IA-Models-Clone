"""
Merger Utilities for Piel Mejorador AI SAM3
==========================================

Unified data merging pattern utilities.
"""

import logging
from typing import TypeVar, Callable, Any, Optional, Dict, List
from abc import ABC, abstractmethod
from copy import deepcopy

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Merger(ABC):
    """Base merger interface."""
    
    @abstractmethod
    def merge(self, *sources: T) -> T:
        """Merge sources."""
        pass


class FunctionMerger(Merger):
    """Merger using a function."""
    
    def __init__(
        self,
        merge_func: Callable[[T, T], T],
        name: Optional[str] = None
    ):
        """
        Initialize function merger.
        
        Args:
            merge_func: Merging function
            name: Optional merger name
        """
        self._merge_func = merge_func
        self.name = name or merge_func.__name__
    
    def merge(self, *sources: T) -> T:
        """Merge sources."""
        if not sources:
            raise ValueError("At least one source required")
        
        result = sources[0]
        for source in sources[1:]:
            result = self._merge_func(result, source)
        return result


class DictMerger(Merger):
    """Dictionary merger."""
    
    def __init__(self, deep: bool = True, overwrite: bool = True):
        """
        Initialize dictionary merger.
        
        Args:
            deep: Whether to deep merge
            overwrite: Whether to overwrite existing keys
        """
        self.deep = deep
        self.overwrite = overwrite
    
    def merge(self, *sources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge dictionaries.
        
        Args:
            *sources: Dictionaries to merge
            
        Returns:
            Merged dictionary
        """
        if not sources:
            return {}
        
        result = deepcopy(sources[0]) if self.deep else sources[0].copy()
        
        for source in sources[1:]:
            source_copy = deepcopy(source) if self.deep else source.copy()
            
            for key, value in source_copy.items():
                if key in result:
                    if self.deep and isinstance(result[key], dict) and isinstance(value, dict):
                        # Recursive merge for nested dicts
                        result[key] = DictMerger(deep=True, overwrite=self.overwrite).merge(
                            result[key], value
                        )
                    elif self.overwrite:
                        result[key] = value
                else:
                    result[key] = value
        
        return result


class ListMerger(Merger):
    """List merger."""
    
    def __init__(self, unique: bool = False, preserve_order: bool = True):
        """
        Initialize list merger.
        
        Args:
            unique: Whether to keep only unique items
            preserve_order: Whether to preserve order
        """
        self.unique = unique
        self.preserve_order = preserve_order
    
    def merge(self, *sources: List[Any]) -> List[Any]:
        """
        Merge lists.
        
        Args:
            *sources: Lists to merge
            
        Returns:
            Merged list
        """
        if not sources:
            return []
        
        result = []
        seen = set()
        
        for source in sources:
            for item in source:
                if self.unique:
                    # Use hashable representation for uniqueness check
                    item_repr = str(item) if not isinstance(item, (str, int, float, bool, type(None))) else item
                    if item_repr in seen:
                        continue
                    seen.add(item_repr)
                
                result.append(item)
        
        return result


class MergerUtils:
    """Unified merger utilities."""
    
    @staticmethod
    def create_function_merger(
        merge_func: Callable[[T, T], T],
        name: Optional[str] = None
    ) -> FunctionMerger:
        """
        Create function merger.
        
        Args:
            merge_func: Merging function
            name: Optional merger name
            
        Returns:
            FunctionMerger
        """
        return FunctionMerger(merge_func, name)
    
    @staticmethod
    def create_dict_merger(deep: bool = True, overwrite: bool = True) -> DictMerger:
        """
        Create dictionary merger.
        
        Args:
            deep: Whether to deep merge
            overwrite: Whether to overwrite existing keys
            
        Returns:
            DictMerger
        """
        return DictMerger(deep, overwrite)
    
    @staticmethod
    def create_list_merger(unique: bool = False, preserve_order: bool = True) -> ListMerger:
        """
        Create list merger.
        
        Args:
            unique: Whether to keep only unique items
            preserve_order: Whether to preserve order
            
        Returns:
            ListMerger
        """
        return ListMerger(unique, preserve_order)
    
    @staticmethod
    def merge_dicts(*sources: Dict[str, Any], deep: bool = True, overwrite: bool = True) -> Dict[str, Any]:
        """
        Merge dictionaries.
        
        Args:
            *sources: Dictionaries to merge
            deep: Whether to deep merge
            overwrite: Whether to overwrite existing keys
            
        Returns:
            Merged dictionary
        """
        return DictMerger(deep, overwrite).merge(*sources)
    
    @staticmethod
    def merge_lists(*sources: List[Any], unique: bool = False, preserve_order: bool = True) -> List[Any]:
        """
        Merge lists.
        
        Args:
            *sources: Lists to merge
            unique: Whether to keep only unique items
            preserve_order: Whether to preserve order
            
        Returns:
            Merged list
        """
        return ListMerger(unique, preserve_order).merge(*sources)


# Convenience functions
def create_function_merger(merge_func: Callable[[T, T], T], **kwargs) -> FunctionMerger:
    """Create function merger."""
    return MergerUtils.create_function_merger(merge_func, **kwargs)


def create_dict_merger(**kwargs) -> DictMerger:
    """Create dictionary merger."""
    return MergerUtils.create_dict_merger(**kwargs)


def merge_dicts(*sources: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Merge dictionaries."""
    return MergerUtils.merge_dicts(*sources, **kwargs)




