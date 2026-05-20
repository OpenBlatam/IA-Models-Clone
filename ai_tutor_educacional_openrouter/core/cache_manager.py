"""
Cache manager for storing and retrieving tutor responses.
"""

import json
import hashlib
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages caching of tutor responses to improve performance and reduce API calls.
    """
    
    def __init__(self, cache_dir: str = ".cache", ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
    
    def _generate_key(self, question: str, subject: Optional[str] = None, difficulty: Optional[str] = None) -> str:
        """Generate cache key from question and context."""
        key_data = f"{question}:{subject}:{difficulty}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, question: str, subject: Optional[str] = None, difficulty: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired."""
        cache_key = self._generate_key(question, subject, difficulty)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            cached = self.memory_cache[cache_key]
            if datetime.now() < cached["expires_at"]:
                logger.debug(f"Cache hit (memory): {cache_key[:8]}")
                return cached["data"]
            else:
                del self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                
                expires_at = datetime.fromisoformat(cached["expires_at"])
                if datetime.now() < expires_at:
                    logger.debug(f"Cache hit (disk): {cache_key[:8]}")
                    self.memory_cache[cache_key] = cached
                    return cached["data"]
                else:
                    cache_file.unlink()
            except Exception as e:
                logger.warning(f"Error reading cache file: {e}")
        
        return None
    
    def set(self, question: str, data: Dict[str, Any], subject: Optional[str] = None, difficulty: Optional[str] = None):
        """Store response in cache."""
        cache_key = self._generate_key(question, subject, difficulty)
        expires_at = datetime.now() + timedelta(seconds=self.ttl)
        
        cached = {
            "data": data,
            "expires_at": expires_at.isoformat(),
            "created_at": datetime.now().isoformat()
        }
        
        # Store in memory
        self.memory_cache[cache_key] = cached
        
        # Store on disk
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cached, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Error writing cache file: {e}")
    
    def clear(self, pattern: Optional[str] = None):
        """Clear cache entries."""
        if pattern:
            # Clear specific pattern
            for cache_file in self.cache_dir.glob(f"{pattern}*.json"):
                cache_file.unlink()
        else:
            # Clear all
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            self.memory_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        disk_files = list(self.cache_dir.glob("*.json"))
        memory_entries = len(self.memory_cache)
        
        return {
            "disk_entries": len(disk_files),
            "memory_entries": memory_entries,
            "total_size_mb": sum(f.stat().st_size for f in disk_files) / (1024 * 1024)
        }






