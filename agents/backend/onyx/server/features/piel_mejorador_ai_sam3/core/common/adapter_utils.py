"""
Adapter Pattern Utilities for Piel Mejorador AI SAM3
====================================================

Unified adapter pattern implementation utilities.
"""

import logging
from typing import TypeVar, Callable, Any, Optional, Dict, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class Adapter(ABC):
    """Base adapter interface."""
    
    @abstractmethod
    def adapt(self, source: Any) -> Any:
        """Adapt source to target format."""
        pass


class FunctionAdapter(Adapter):
    """Adapter using a function."""
    
    def __init__(self, adapt_func: Callable[[T], R], name: Optional[str] = None):
        """
        Initialize function adapter.
        
        Args:
            adapt_func: Adaptation function
            name: Optional adapter name
        """
        self._adapt_func = adapt_func
        self.name = name or adapt_func.__name__
    
    def adapt(self, source: T) -> R:
        """Adapt source."""
        return self._adapt_func(source)


class AdapterChain:
    """Chain of adapters."""
    
    def __init__(self, *adapters: Adapter):
        """
        Initialize adapter chain.
        
        Args:
            *adapters: Adapter instances
        """
        self._adapters: List[Adapter] = list(adapters)
    
    def add(self, adapter: Adapter) -> "AdapterChain":
        """
        Add adapter to chain.
        
        Args:
            adapter: Adapter to add
            
        Returns:
            Self for chaining
        """
        self._adapters.append(adapter)
        return self
    
    def adapt(self, source: Any) -> Any:
        """
        Adapt through chain.
        
        Args:
            source: Source object
            
        Returns:
            Adapted object
        """
        result = source
        for adapter in self._adapters:
            result = adapter.adapt(result)
        return result


class AdapterUtils:
    """Unified adapter pattern utilities."""
    
    @staticmethod
    def create_adapter(
        adapt_func: Callable[[T], R],
        name: Optional[str] = None
    ) -> FunctionAdapter:
        """
        Create adapter from function.
        
        Args:
            adapt_func: Adaptation function
            name: Optional adapter name
            
        Returns:
            FunctionAdapter
        """
        return FunctionAdapter(adapt_func, name)
    
    @staticmethod
    def create_chain(*adapters: Adapter) -> AdapterChain:
        """
        Create adapter chain.
        
        Args:
            *adapters: Adapter instances
            
        Returns:
            AdapterChain
        """
        return AdapterChain(*adapters)
    
    @staticmethod
    def create_dict_adapter(
        mapping: Dict[str, str],
        name: str = "dict_adapter"
    ) -> Adapter:
        """
        Create adapter for dictionary key mapping.
        
        Args:
            mapping: Dictionary mapping old keys to new keys
            name: Adapter name
            
        Returns:
            Adapter
        """
        def adapt_dict(source: Dict[str, Any]) -> Dict[str, Any]:
            result = {}
            for old_key, new_key in mapping.items():
                if old_key in source:
                    result[new_key] = source[old_key]
            return result
        
        return FunctionAdapter(adapt_dict, name)
    
    @staticmethod
    def create_type_adapter(
        target_type: type,
        name: Optional[str] = None
    ) -> Adapter:
        """
        Create adapter for type conversion.
        
        Args:
            target_type: Target type
            name: Optional adapter name
            
        Returns:
            Adapter
        """
        def adapt_type(source: Any) -> Any:
            return target_type(source)
        
        return FunctionAdapter(adapt_type, name or f"{target_type.__name__}_adapter")


# Convenience functions
def create_adapter(adapt_func: Callable[[T], R], **kwargs) -> FunctionAdapter:
    """Create adapter."""
    return AdapterUtils.create_adapter(adapt_func, **kwargs)


def create_chain(*adapters: Adapter) -> AdapterChain:
    """Create adapter chain."""
    return AdapterUtils.create_chain(*adapters)




