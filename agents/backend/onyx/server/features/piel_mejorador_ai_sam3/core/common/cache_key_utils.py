"""
Cache Key Utilities for Piel Mejorador AI SAM3
==============================================

Unified cache key generation utilities.
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional, Union, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheKeyUtils:
    """Unified cache key generation utilities."""
    
    @staticmethod
    def generate_hash(
        data: Union[str, Dict[str, Any], list],
        algorithm: str = "sha256"
    ) -> str:
        """
        Generate hash from data.
        
        Args:
            data: Data to hash (string, dict, or list)
            algorithm: Hash algorithm (sha256, md5, etc.)
            
        Returns:
            Hexadecimal hash string
        """
        if isinstance(data, (dict, list)):
            # Sort keys for consistent hashing
            key_string = json.dumps(data, sort_keys=True, ensure_ascii=False)
        else:
            key_string = str(data)
        
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(key_string.encode('utf-8'))
        return hash_obj.hexdigest()
    
    @staticmethod
    def generate_key(
        *args: Any,
        **kwargs: Any
    ) -> str:
        """
        Generate cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cache key (hash)
        """
        key_data = {}
        
        # Add positional args
        for i, arg in enumerate(args):
            key_data[f"arg_{i}"] = CacheKeyUtils._normalize_value(arg)
        
        # Add keyword args
        for key, value in sorted(kwargs.items()):
            key_data[key] = CacheKeyUtils._normalize_value(value)
        
        return CacheKeyUtils.generate_hash(key_data)
    
    @staticmethod
    def _normalize_value(value: Any) -> Any:
        """
        Normalize value for consistent hashing.
        
        Args:
            value: Value to normalize
            
        Returns:
            Normalized value
        """
        if isinstance(value, Path):
            return str(value.resolve())
        elif isinstance(value, (dict, list)):
            return value
        elif hasattr(value, '__dict__'):
            # Convert objects to dict
            return CacheKeyUtils._normalize_value(value.__dict__)
        else:
            return value
    
    @staticmethod
    def generate_file_key(
        file_path: Union[str, Path],
        *additional_params: Any,
        **kwargs: Any
    ) -> str:
        """
        Generate cache key for file with additional parameters.
        
        Args:
            file_path: Path to file
            *additional_params: Additional parameters
            **kwargs: Additional keyword parameters
            
        Returns:
            Cache key (hash)
        """
        path_obj = Path(file_path)
        key_data = {
            "file_path": str(path_obj.resolve()),
        }
        
        # Add additional params
        for i, param in enumerate(additional_params):
            key_data[f"param_{i}"] = CacheKeyUtils._normalize_value(param)
        
        # Add kwargs
        for key, value in sorted(kwargs.items()):
            key_data[key] = CacheKeyUtils._normalize_value(value)
        
        return CacheKeyUtils.generate_hash(key_data)
    
    @staticmethod
    def generate_prefix_key(
        prefix: str,
        *args: Any,
        **kwargs: Any
    ) -> str:
        """
        Generate cache key with prefix.
        
        Args:
            prefix: Key prefix
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Prefixed cache key
        """
        key = CacheKeyUtils.generate_key(*args, **kwargs)
        return f"{prefix}:{key}"
    
    @staticmethod
    def generate_namespaced_key(
        namespace: str,
        key: str
    ) -> str:
        """
        Generate namespaced cache key.
        
        Args:
            namespace: Namespace
            key: Cache key
            
        Returns:
            Namespaced key
        """
        return f"{namespace}:{key}"


# Convenience functions
def generate_key(*args: Any, **kwargs: Any) -> str:
    """Generate cache key."""
    return CacheKeyUtils.generate_key(*args, **kwargs)


def generate_file_key(file_path: Union[str, Path], **kwargs) -> str:
    """Generate file cache key."""
    return CacheKeyUtils.generate_file_key(file_path, **kwargs)


def generate_hash(data: Union[str, Dict[str, Any], list], **kwargs) -> str:
    """Generate hash."""
    return CacheKeyUtils.generate_hash(data, **kwargs)




