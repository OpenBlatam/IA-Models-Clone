"""
Cache Manager for Music Generation
"""

import hashlib
import json
from typing import Optional, Dict, Any
from pathlib import Path
import logging
import pickle
import numpy as np

logger = logging.getLogger(__name__)


class MusicCache:
    """Cache for generated music"""
    
    def __init__(self, cache_dir: str = "cache/music"):
        """
        Initialize music cache
        
        Args:
            cache_dir: Cache directory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        logger.info(f"MusicCache initialized: {cache_dir}")
    
    def _get_cache_key(self, text: str, duration: int, **params) -> str:
        """Generate cache key"""
        key_data = {
            "text": text,
            "duration": duration,
            **params
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(
        self,
        text: str,
        duration: int,
        **params
    ) -> Optional[np.ndarray]:
        """
        Get cached audio
        
        Args:
            text: Song description
            duration: Duration
            **params: Additional parameters
        
        Returns:
            Cached audio or None
        """
        cache_key = self._get_cache_key(text, duration, **params)
        
        # Check memory cache
        if cache_key in self.memory_cache:
            logger.debug("Cache hit (memory)")
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    audio = pickle.load(f)
                self.memory_cache[cache_key] = audio
                logger.debug("Cache hit (disk)")
                return audio
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
        
        logger.debug("Cache miss")
        return None
    
    def set(
        self,
        text: str,
        duration: int,
        audio: np.ndarray,
        **params
    ):
        """
        Cache audio
        
        Args:
            text: Song description
            duration: Duration
            audio: Audio array
            **params: Additional parameters
        """
        cache_key = self._get_cache_key(text, duration, **params)
        
        # Store in memory
        self.memory_cache[cache_key] = audio
        
        # Store on disk
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(audio, f)
            logger.debug(f"Audio cached: {cache_key}")
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")
    
    def clear(self):
        """Clear cache"""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        disk_files = list(self.cache_dir.glob("*.pkl"))
        return {
            "memory_entries": len(self.memory_cache),
            "disk_entries": len(disk_files),
            "cache_dir": str(self.cache_dir)
        }

