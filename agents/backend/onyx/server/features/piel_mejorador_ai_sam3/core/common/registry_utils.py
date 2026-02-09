"""
Registry Utilities for Piel Mejorador AI SAM3
============================================

Unified registry pattern utilities.
"""

import logging
from typing import TypeVar, Dict, Optional, List, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')


@dataclass
class RegistryEntry:
    """Registry entry."""
    key: str
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": str(self.value),
            "metadata": self.metadata,
            "registered_at": self.registered_at.isoformat(),
        }


class Registry:
    """Generic registry for objects."""
    
    def __init__(self, name: str = "registry"):
        """
        Initialize registry.
        
        Args:
            name: Registry name
        """
        self.name = name
        self._entries: Dict[str, RegistryEntry] = {}
    
    def register(
        self,
        key: str,
        value: T,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ):
        """
        Register object.
        
        Args:
            key: Registration key
            value: Object to register
            metadata: Optional metadata
            overwrite: Whether to overwrite existing entry
            
        Raises:
            KeyError: If key exists and overwrite is False
        """
        if key in self._entries and not overwrite:
            raise KeyError(f"Key already registered: {key}")
        
        self._entries[key] = RegistryEntry(
            key=key,
            value=value,
            metadata=metadata or {}
        )
        logger.debug(f"Registered {key} in {self.name}")
    
    def get(self, key: str) -> Optional[T]:
        """
        Get registered object.
        
        Args:
            key: Registration key
            
        Returns:
            Registered object or None
        """
        entry = self._entries.get(key)
        return entry.value if entry else None
    
    def get_entry(self, key: str) -> Optional[RegistryEntry]:
        """
        Get registry entry.
        
        Args:
            key: Registration key
            
        Returns:
            RegistryEntry or None
        """
        return self._entries.get(key)
    
    def unregister(self, key: str) -> bool:
        """
        Unregister object.
        
        Args:
            key: Registration key
            
        Returns:
            True if unregistered, False if not found
        """
        if key in self._entries:
            del self._entries[key]
            logger.debug(f"Unregistered {key} from {self.name}")
            return True
        return False
    
    def has(self, key: str) -> bool:
        """
        Check if key is registered.
        
        Args:
            key: Registration key
            
        Returns:
            True if registered
        """
        return key in self._entries
    
    def list_keys(self) -> List[str]:
        """
        List all registered keys.
        
        Returns:
            List of keys
        """
        return list(self._entries.keys())
    
    def list_entries(self) -> List[RegistryEntry]:
        """
        List all entries.
        
        Returns:
            List of entries
        """
        return list(self._entries.values())
    
    def clear(self):
        """Clear all entries."""
        self._entries.clear()
        logger.debug(f"Cleared {self.name}")
    
    def size(self) -> int:
        """Get registry size."""
        return len(self._entries)
    
    def filter(self, predicate: Callable[[RegistryEntry], bool]) -> List[RegistryEntry]:
        """
        Filter entries by predicate.
        
        Args:
            predicate: Filter function
            
        Returns:
            Filtered entries
        """
        return [entry for entry in self._entries.values() if predicate(entry)]


class RegistryUtils:
    """Unified registry utilities."""
    
    @staticmethod
    def create_registry(name: str = "registry") -> Registry:
        """
        Create registry.
        
        Args:
            name: Registry name
            
        Returns:
            Registry
        """
        return Registry(name)
    
    @staticmethod
    def create_typed_registry(
        value_type: type,
        name: Optional[str] = None
    ) -> Registry:
        """
        Create typed registry with validation.
        
        Args:
            value_type: Expected value type
            name: Optional registry name
            
        Returns:
            Registry with type validation
        """
        registry = Registry(name or f"{value_type.__name__}_registry")
        original_register = registry.register
        
        def validated_register(key: str, value: Any, **kwargs):
            if not isinstance(value, value_type):
                raise TypeError(f"Expected {value_type}, got {type(value)}")
            original_register(key, value, **kwargs)
        
        registry.register = validated_register
        return registry


# Convenience functions
def create_registry(name: str = "registry") -> Registry:
    """Create registry."""
    return RegistryUtils.create_registry(name)


def create_typed_registry(value_type: type, **kwargs) -> Registry:
    """Create typed registry."""
    return RegistryUtils.create_typed_registry(value_type, **kwargs)




