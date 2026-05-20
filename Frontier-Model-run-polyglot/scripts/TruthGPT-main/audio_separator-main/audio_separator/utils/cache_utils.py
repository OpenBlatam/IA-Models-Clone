"""
Caching utilities for audio separator.
Refactored to use constants and improve organization.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
import pickle

from .constants import DEFAULT_CACHE_DIR_NAME, DEFAULT_CACHE_EXTENSION
from ..logger import logger

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_CACHE_DIR = Path.home() / ".audio_separator" / "cache"
DEFAULT_CACHE_PATTERN = "*"

# ════════════════════════════════════════════════════════════════════════════
# CACHE MANAGER
# ════════════════════════════════════════════════════════════════════════════

class CacheManager:
    """Manage caching of model outputs and configurations."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache files (default: ~/.audio_separator/cache)
        """
        if cache_dir is None:
            cache_dir = DEFAULT_CACHE_DIR
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Cache directory: {self.cache_dir}")
    
    def _get_cache_key(self, key_data: Dict[str, Any]) -> str:
        """
        Generate cache key from data.
        
        Args:
            key_data: Data to generate key from
            
        Returns:
            Cache key string
        """
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_cache_path(self, key: str, extension: str = DEFAULT_CACHE_EXTENSION) -> Path:
        """
        Get cache file path for key.
        
        Args:
            key: Cache key
            extension: File extension
            
        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{key}{extension}"
    
    def exists(self, key_data: Dict[str, Any]) -> bool:
        """
        Check if cache entry exists.
        
        Args:
            key_data: Data to generate key from
            
        Returns:
            True if cache entry exists
        """
        key = self._get_cache_key(key_data)
        cache_path = self.get_cache_path(key)
        return cache_path.exists()
    
    def get(self, key_data: Dict[str, Any]) -> Optional[Any]:
        """
        Get cached value.
        
        Args:
            key_data: Data to generate key from
            
        Returns:
            Cached value or None if not found
        """
        if not self.exists(key_data):
            return None
        
        key = self._get_cache_key(key_data)
        cache_path = self.get_cache_path(key)
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {str(e)}")
            return None
    
    def set(self, key_data: Dict[str, Any], value: Any) -> bool:
        """
        Set cached value.
        
        Args:
            key_data: Data to generate key from
            value: Value to cache
            
        Returns:
            True if successful
        """
        key = self._get_cache_key(key_data)
        cache_path = self.get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            logger.debug(f"Cached value with key: {key}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache value: {str(e)}")
            return False
    
    def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            pattern: Optional pattern to match (e.g., "*.pkl")
            
        Returns:
            Number of files deleted
        """
        if pattern is None:
            pattern = DEFAULT_CACHE_PATTERN
        
        files = list(self.cache_dir.glob(pattern))
        
        deleted = 0
        for file in files:
            if file.is_file():
                try:
                    file.unlink()
                    deleted += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {file}: {str(e)}")
        
        logger.info(f"Cleared {deleted} cache files")
        return deleted
    
    def get_size(self) -> int:
        """
        Get total cache size in bytes.
        
        Returns:
            Total size in bytes
        """
        total_size = 0
        for file in self.cache_dir.glob(DEFAULT_CACHE_PATTERN):
            if file.is_file():
                total_size += file.stat().st_size
        return total_size
