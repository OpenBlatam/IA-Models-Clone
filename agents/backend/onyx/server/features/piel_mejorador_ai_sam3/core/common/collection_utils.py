"""
Collection Utilities for Piel Mejorador AI SAM3
===============================================

Unified collection operations and utilities.
"""

import logging
from typing import List, Dict, Any, Callable, Optional, TypeVar, Iterable
from collections import defaultdict

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class CollectionUtils:
    """Unified collection utilities."""
    
    @staticmethod
    def chunk(
        items: List[T],
        chunk_size: int
    ) -> List[List[T]]:
        """
        Split list into chunks.
        
        Args:
            items: List to chunk
            chunk_size: Size of each chunk
            
        Returns:
            List of chunks
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    
    @staticmethod
    def group_by(
        items: Iterable[T],
        key_func: Callable[[T], K]
    ) -> Dict[K, List[T]]:
        """
        Group items by key function.
        
        Args:
            items: Items to group
            key_func: Function to extract key
            
        Returns:
            Dictionary mapping keys to lists of items
        """
        grouped = defaultdict(list)
        for item in items:
            key = key_func(item)
            grouped[key].append(item)
        return dict(grouped)
    
    @staticmethod
    def partition(
        items: List[T],
        predicate: Callable[[T], bool]
    ) -> tuple[List[T], List[T]]:
        """
        Partition list into two lists based on predicate.
        
        Args:
            items: List to partition
            predicate: Function that returns True for items in first list
            
        Returns:
            Tuple of (matching items, non-matching items)
        """
        matching = []
        non_matching = []
        
        for item in items:
            if predicate(item):
                matching.append(item)
            else:
                non_matching.append(item)
        
        return matching, non_matching
    
    @staticmethod
    def flatten(nested: List[List[T]]) -> List[T]:
        """
        Flatten nested list.
        
        Args:
            nested: Nested list
            
        Returns:
            Flattened list
        """
        return [item for sublist in nested for item in sublist]
    
    @staticmethod
    def unique(items: List[T], key: Optional[Callable[[T], Any]] = None) -> List[T]:
        """
        Get unique items from list.
        
        Args:
            items: List of items
            key: Optional key function for uniqueness
            
        Returns:
            List of unique items (preserves order)
        """
        if key is None:
            seen = set()
            result = []
            for item in items:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return result
        else:
            seen = set()
            result = []
            for item in items:
                key_value = key(item)
                if key_value not in seen:
                    seen.add(key_value)
                    result.append(item)
            return result
    
    @staticmethod
    def sort_by(
        items: List[T],
        key_func: Callable[[T], Any],
        reverse: bool = False
    ) -> List[T]:
        """
        Sort items by key function.
        
        Args:
            items: List to sort
            key_func: Key function
            reverse: Whether to reverse sort
            
        Returns:
            Sorted list
        """
        return sorted(items, key=key_func, reverse=reverse)
    
    @staticmethod
    def filter_map(
        items: Iterable[T],
        filter_func: Callable[[T], bool],
        map_func: Callable[[T], V]
    ) -> List[V]:
        """
        Filter and map in one operation.
        
        Args:
            items: Items to process
            filter_func: Filter predicate
            map_func: Map function
            
        Returns:
            List of mapped values for items that pass filter
        """
        return [map_func(item) for item in items if filter_func(item)]
    
    @staticmethod
    def batch_process(
        items: List[T],
        batch_size: int,
        process_func: Callable[[List[T]], Any]
    ) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            items: Items to process
            batch_size: Batch size
            process_func: Function to process each batch
            
        Returns:
            List of results
        """
        chunks = CollectionUtils.chunk(items, batch_size)
        return [process_func(chunk) for chunk in chunks]
    
    @staticmethod
    def find_first(
        items: Iterable[T],
        predicate: Callable[[T], bool],
        default: Optional[T] = None
    ) -> Optional[T]:
        """
        Find first item matching predicate.
        
        Args:
            items: Items to search
            predicate: Predicate function
            default: Default value if not found
            
        Returns:
            First matching item or default
        """
        for item in items:
            if predicate(item):
                return item
        return default
    
    @staticmethod
    def count_by(
        items: Iterable[T],
        key_func: Callable[[T], K]
    ) -> Dict[K, int]:
        """
        Count items by key function.
        
        Args:
            items: Items to count
            key_func: Key function
            
        Returns:
            Dictionary mapping keys to counts
        """
        counts = defaultdict(int)
        for item in items:
            key = key_func(item)
            counts[key] += 1
        return dict(counts)


# Convenience functions
def chunk(items: List[T], chunk_size: int) -> List[List[T]]:
    """Split list into chunks."""
    return CollectionUtils.chunk(items, chunk_size)


def group_by(items: Iterable[T], key_func: Callable[[T], K]) -> Dict[K, List[T]]:
    """Group items by key."""
    return CollectionUtils.group_by(items, key_func)


def unique(items: List[T], key: Optional[Callable[[T], Any]] = None) -> List[T]:
    """Get unique items."""
    return CollectionUtils.unique(items, key)




