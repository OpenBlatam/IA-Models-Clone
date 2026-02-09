"""
Cache key generation utilities.

Consolidates cache key generation patterns across the codebase.
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional, Union, List

logger = logging.getLogger(__name__)


class CacheKeyGenerator:
    """Utility for generating consistent cache keys."""
    
    @staticmethod
    def generate(
        prefix: str,
        *args,
        **kwargs
    ) -> str:
        """
        Generate cache key from prefix and arguments.
        
        Args:
            prefix: Key prefix
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Cache key string
        """
        key_parts = [prefix]
        
        # Add positional arguments
        if args:
            key_parts.append(CacheKeyGenerator._serialize_args(args))
        
        # Add keyword arguments
        if kwargs:
            key_parts.append(CacheKeyGenerator._serialize_kwargs(kwargs))
        
        key_string = ":".join(str(part) for part in key_parts if part)
        return CacheKeyGenerator._hash_key(key_string)
    
    @staticmethod
    def generate_from_dict(
        prefix: str,
        data: Dict[str, Any],
        exclude_keys: Optional[List[str]] = None
    ) -> str:
        """
        Generate cache key from dictionary.
        
        Args:
            prefix: Key prefix
            data: Dictionary data
            exclude_keys: Keys to exclude from key generation
        
        Returns:
            Cache key string
        """
        if exclude_keys:
            data = {k: v for k, v in data.items() if k not in exclude_keys}
        
        key_string = f"{prefix}:{CacheKeyGenerator._serialize_dict(data)}"
        return CacheKeyGenerator._hash_key(key_string)
    
    @staticmethod
    def generate_simple(
        prefix: str,
        *values: Any
    ) -> str:
        """
        Generate simple cache key from prefix and values.
        
        Args:
            prefix: Key prefix
            *values: Values to include in key
        
        Returns:
            Cache key string
        """
        key_parts = [prefix] + [str(v) for v in values]
        key_string = ":".join(key_parts)
        return CacheKeyGenerator._hash_key(key_string)
    
    @staticmethod
    def _serialize_args(args: tuple) -> str:
        """Serialize positional arguments."""
        return CacheKeyGenerator._serialize_value(args)
    
    @staticmethod
    def _serialize_kwargs(kwargs: Dict[str, Any]) -> str:
        """Serialize keyword arguments."""
        # Sort keys for consistent ordering
        sorted_items = sorted(kwargs.items())
        return CacheKeyGenerator._serialize_dict(dict(sorted_items))
    
    @staticmethod
    def _serialize_dict(data: Dict[str, Any]) -> str:
        """Serialize dictionary to string."""
        try:
            # Use JSON for consistent serialization
            return json.dumps(data, sort_keys=True, default=str)
        except (TypeError, ValueError):
            # Fallback to string representation
            return str(sorted(data.items()))
    
    @staticmethod
    def _serialize_value(value: Any) -> str:
        """Serialize a value to string."""
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, sort_keys=True, default=str)
            except (TypeError, ValueError):
                return str(value)
        return str(value)
    
    @staticmethod
    def _hash_key(key_string: str, algorithm: str = "md5") -> str:
        """
        Hash cache key string.
        
        Args:
            key_string: Key string to hash
            algorithm: Hash algorithm (md5, sha256)
        
        Returns:
            Hashed key string
        """
        if algorithm == "md5":
            return hashlib.md5(key_string.encode()).hexdigest()
        elif algorithm == "sha256":
            return hashlib.sha256(key_string.encode()).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")


# Convenience functions
def generate_cache_key(
    prefix: str,
    *args,
    **kwargs
) -> str:
    """Generate cache key from prefix and arguments."""
    return CacheKeyGenerator.generate(prefix, *args, **kwargs)


def generate_cache_key_from_dict(
    prefix: str,
    data: Dict[str, Any],
    exclude_keys: Optional[List[str]] = None
) -> str:
    """Generate cache key from dictionary."""
    return CacheKeyGenerator.generate_from_dict(prefix, data, exclude_keys)


def generate_simple_cache_key(
    prefix: str,
    *values: Any
) -> str:
    """Generate simple cache key from prefix and values."""
    return CacheKeyGenerator.generate_simple(prefix, *values)

