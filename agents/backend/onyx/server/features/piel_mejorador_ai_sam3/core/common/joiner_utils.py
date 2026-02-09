"""
Joiner Utilities for Piel Mejorador AI SAM3
==========================================

Unified data joining pattern utilities.
"""

import logging
from typing import TypeVar, Callable, Any, Optional, List, Dict, Iterable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class Joiner(ABC):
    """Base joiner interface."""
    
    @abstractmethod
    def join(self, items: Iterable[T]) -> str:
        """Join items into a string."""
        pass


class FunctionJoiner(Joiner):
    """Joiner using a function."""
    
    def __init__(
        self,
        join_func: Callable[[Iterable[T]], str],
        name: Optional[str] = None
    ):
        """
        Initialize function joiner.
        
        Args:
            join_func: Joining function
            name: Optional joiner name
        """
        self._join_func = join_func
        self.name = name or join_func.__name__
    
    def join(self, items: Iterable[T]) -> str:
        """Join items."""
        return self._join_func(items)


class StringJoiner(Joiner):
    """String joiner with separator."""
    
    def __init__(self, separator: str = ", "):
        """
        Initialize string joiner.
        
        Args:
            separator: Separator between items
        """
        self.separator = separator
    
    def join(self, items: Iterable[T]) -> str:
        """
        Join items.
        
        Args:
            items: Items to join
            
        Returns:
            Joined string
        """
        return self.separator.join(str(item) for item in items)


class PrefixSuffixJoiner(Joiner):
    """Joiner with prefix and suffix."""
    
    def __init__(
        self,
        separator: str = ", ",
        prefix: str = "",
        suffix: str = ""
    ):
        """
        Initialize prefix/suffix joiner.
        
        Args:
            separator: Separator between items
            prefix: Prefix for the result
            suffix: Suffix for the result
        """
        self.separator = separator
        self.prefix = prefix
        self.suffix = suffix
    
    def join(self, items: Iterable[T]) -> str:
        """
        Join items with prefix and suffix.
        
        Args:
            items: Items to join
            
        Returns:
            Joined string with prefix and suffix
        """
        joined = self.separator.join(str(item) for item in items)
        return f"{self.prefix}{joined}{self.suffix}"


class KeyValueJoiner(Joiner):
    """Joiner for key-value pairs."""
    
    def __init__(
        self,
        key_value_separator: str = ": ",
        pair_separator: str = ", "
    ):
        """
        Initialize key-value joiner.
        
        Args:
            key_value_separator: Separator between key and value
            pair_separator: Separator between pairs
        """
        self.key_value_separator = key_value_separator
        self.pair_separator = pair_separator
    
    def join(self, items: Iterable[tuple[K, V]]) -> str:
        """
        Join key-value pairs.
        
        Args:
            items: Key-value pairs to join
            
        Returns:
            Joined string
        """
        pairs = [
            f"{str(key)}{self.key_value_separator}{str(value)}"
            for key, value in items
        ]
        return self.pair_separator.join(pairs)


class DictJoiner(Joiner):
    """Joiner for dictionaries."""
    
    def __init__(
        self,
        key_value_separator: str = ": ",
        pair_separator: str = ", ",
        prefix: str = "{",
        suffix: str = "}"
    ):
        """
        Initialize dictionary joiner.
        
        Args:
            key_value_separator: Separator between key and value
            pair_separator: Separator between pairs
            prefix: Prefix for the result
            suffix: Suffix for the result
        """
        self.key_value_separator = key_value_separator
        self.pair_separator = pair_separator
        self.prefix = prefix
        self.suffix = suffix
    
    def join(self, items: Dict[K, V]) -> str:
        """
        Join dictionary items.
        
        Args:
            items: Dictionary to join
            
        Returns:
            Joined string
        """
        pairs = [
            f"{str(key)}{self.key_value_separator}{str(value)}"
            for key, value in items.items()
        ]
        joined = self.pair_separator.join(pairs)
        return f"{self.prefix}{joined}{self.suffix}"


class JoinerUtils:
    """Unified joiner utilities."""
    
    @staticmethod
    def create_function_joiner(
        join_func: Callable[[Iterable[T]], str],
        name: Optional[str] = None
    ) -> FunctionJoiner:
        """
        Create function joiner.
        
        Args:
            join_func: Joining function
            name: Optional joiner name
            
        Returns:
            FunctionJoiner
        """
        return FunctionJoiner(join_func, name)
    
    @staticmethod
    def create_string_joiner(separator: str = ", ") -> StringJoiner:
        """
        Create string joiner.
        
        Args:
            separator: Separator between items
            
        Returns:
            StringJoiner
        """
        return StringJoiner(separator)
    
    @staticmethod
    def create_prefix_suffix_joiner(
        separator: str = ", ",
        prefix: str = "",
        suffix: str = ""
    ) -> PrefixSuffixJoiner:
        """
        Create prefix/suffix joiner.
        
        Args:
            separator: Separator between items
            prefix: Prefix for the result
            suffix: Suffix for the result
            
        Returns:
            PrefixSuffixJoiner
        """
        return PrefixSuffixJoiner(separator, prefix, suffix)
    
    @staticmethod
    def create_key_value_joiner(
        key_value_separator: str = ": ",
        pair_separator: str = ", "
    ) -> KeyValueJoiner:
        """
        Create key-value joiner.
        
        Args:
            key_value_separator: Separator between key and value
            pair_separator: Separator between pairs
            
        Returns:
            KeyValueJoiner
        """
        return KeyValueJoiner(key_value_separator, pair_separator)
    
    @staticmethod
    def create_dict_joiner(
        key_value_separator: str = ": ",
        pair_separator: str = ", ",
        prefix: str = "{",
        suffix: str = "}"
    ) -> DictJoiner:
        """
        Create dictionary joiner.
        
        Args:
            key_value_separator: Separator between key and value
            pair_separator: Separator between pairs
            prefix: Prefix for the result
            suffix: Suffix for the result
            
        Returns:
            DictJoiner
        """
        return DictJoiner(key_value_separator, pair_separator, prefix, suffix)
    
    @staticmethod
    def join(items: Iterable[T], separator: str = ", ") -> str:
        """
        Join items.
        
        Args:
            items: Items to join
            separator: Separator between items
            
        Returns:
            Joined string
        """
        return StringJoiner(separator).join(items)
    
    @staticmethod
    def join_key_value(
        items: Iterable[tuple[K, V]],
        key_value_separator: str = ": ",
        pair_separator: str = ", "
    ) -> str:
        """
        Join key-value pairs.
        
        Args:
            items: Key-value pairs to join
            key_value_separator: Separator between key and value
            pair_separator: Separator between pairs
            
        Returns:
            Joined string
        """
        return KeyValueJoiner(key_value_separator, pair_separator).join(items)
    
    @staticmethod
    def join_dict(
        items: Dict[K, V],
        key_value_separator: str = ": ",
        pair_separator: str = ", ",
        prefix: str = "{",
        suffix: str = "}"
    ) -> str:
        """
        Join dictionary items.
        
        Args:
            items: Dictionary to join
            key_value_separator: Separator between key and value
            pair_separator: Separator between pairs
            prefix: Prefix for the result
            suffix: Suffix for the result
            
        Returns:
            Joined string
        """
        return DictJoiner(key_value_separator, pair_separator, prefix, suffix).join(items)


# Convenience functions
def create_string_joiner(**kwargs) -> StringJoiner:
    """Create string joiner."""
    return JoinerUtils.create_string_joiner(**kwargs)


def join_items(items: Iterable[T], **kwargs) -> str:
    """Join items."""
    return JoinerUtils.join(items, **kwargs)




