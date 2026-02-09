"""
Splitter Utilities for Piel Mejorador AI SAM3
============================================

Unified data splitting pattern utilities.
"""

import logging
from typing import TypeVar, Callable, Any, Optional, List, Tuple, Dict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Splitter(ABC):
    """Base splitter interface."""
    
    @abstractmethod
    def split(self, source: T) -> List[T]:
        """Split source into parts."""
        pass


class FunctionSplitter(Splitter):
    """Splitter using a function."""
    
    def __init__(
        self,
        split_func: Callable[[T], List[T]],
        name: Optional[str] = None
    ):
        """
        Initialize function splitter.
        
        Args:
            split_func: Splitting function
            name: Optional splitter name
        """
        self._split_func = split_func
        self.name = name or split_func.__name__
    
    def split(self, source: T) -> List[T]:
        """Split source."""
        return self._split_func(source)


class StringSplitter(Splitter):
    """String splitter."""
    
    def __init__(self, delimiter: str = " ", maxsplit: Optional[int] = None):
        """
        Initialize string splitter.
        
        Args:
            delimiter: Delimiter to split on
            maxsplit: Maximum number of splits
        """
        self.delimiter = delimiter
        self.maxsplit = maxsplit
    
    def split(self, source: str) -> List[str]:
        """
        Split string.
        
        Args:
            source: String to split
            
        Returns:
            List of parts
        """
        if self.maxsplit is not None:
            return source.split(self.delimiter, self.maxsplit)
        return source.split(self.delimiter)


class ChunkSplitter(Splitter):
    """Chunk splitter for collections."""
    
    def __init__(self, chunk_size: int):
        """
        Initialize chunk splitter.
        
        Args:
            chunk_size: Size of each chunk
        """
        self.chunk_size = chunk_size
    
    def split(self, source: List[T]) -> List[List[T]]:
        """
        Split into chunks.
        
        Args:
            source: List to split
            
        Returns:
            List of chunks
        """
        chunks = []
        for i in range(0, len(source), self.chunk_size):
            chunks.append(source[i:i + self.chunk_size])
        return chunks


class DictSplitter(Splitter):
    """Dictionary splitter."""
    
    def __init__(self, keys: Optional[List[str]] = None):
        """
        Initialize dictionary splitter.
        
        Args:
            keys: Optional list of keys to extract (None = split all keys)
        """
        self.keys = keys
    
    def split(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split dictionary into multiple dictionaries.
        
        Args:
            source: Dictionary to split
            
        Returns:
            List of dictionaries (one per key)
        """
        if self.keys is None:
            self.keys = list(source.keys())
        
        result = []
        for key in self.keys:
            if key in source:
                result.append({key: source[key]})
        return result


class SplitterUtils:
    """Unified splitter utilities."""
    
    @staticmethod
    def create_function_splitter(
        split_func: Callable[[T], List[T]],
        name: Optional[str] = None
    ) -> FunctionSplitter:
        """
        Create function splitter.
        
        Args:
            split_func: Splitting function
            name: Optional splitter name
            
        Returns:
            FunctionSplitter
        """
        return FunctionSplitter(split_func, name)
    
    @staticmethod
    def create_string_splitter(delimiter: str = " ", maxsplit: Optional[int] = None) -> StringSplitter:
        """
        Create string splitter.
        
        Args:
            delimiter: Delimiter to split on
            maxsplit: Maximum number of splits
            
        Returns:
            StringSplitter
        """
        return StringSplitter(delimiter, maxsplit)
    
    @staticmethod
    def create_chunk_splitter(chunk_size: int) -> ChunkSplitter:
        """
        Create chunk splitter.
        
        Args:
            chunk_size: Size of each chunk
            
        Returns:
            ChunkSplitter
        """
        return ChunkSplitter(chunk_size)
    
    @staticmethod
    def create_dict_splitter(keys: Optional[List[str]] = None) -> DictSplitter:
        """
        Create dictionary splitter.
        
        Args:
            keys: Optional list of keys to extract
            
        Returns:
            DictSplitter
        """
        return DictSplitter(keys)
    
    @staticmethod
    def split_string(source: str, delimiter: str = " ", maxsplit: Optional[int] = None) -> List[str]:
        """
        Split string.
        
        Args:
            source: String to split
            delimiter: Delimiter to split on
            maxsplit: Maximum number of splits
            
        Returns:
            List of parts
        """
        return StringSplitter(delimiter, maxsplit).split(source)
    
    @staticmethod
    def split_chunks(source: List[T], chunk_size: int) -> List[List[T]]:
        """
        Split into chunks.
        
        Args:
            source: List to split
            chunk_size: Size of each chunk
            
        Returns:
            List of chunks
        """
        return ChunkSplitter(chunk_size).split(source)


# Convenience functions
def create_function_splitter(split_func: Callable[[T], List[T]], **kwargs) -> FunctionSplitter:
    """Create function splitter."""
    return SplitterUtils.create_function_splitter(split_func, **kwargs)


def create_string_splitter(**kwargs) -> StringSplitter:
    """Create string splitter."""
    return SplitterUtils.create_string_splitter(**kwargs)


def split_string(source: str, **kwargs) -> List[str]:
    """Split string."""
    return SplitterUtils.split_string(source, **kwargs)




