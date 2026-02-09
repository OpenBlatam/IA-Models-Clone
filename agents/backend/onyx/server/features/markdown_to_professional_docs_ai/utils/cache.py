"""Cache utilities for Markdown to Professional Documents AI"""
import hashlib
import json
from typing import Optional, Any, Dict
from pathlib import Path
import os
from datetime import datetime, timedelta

from config import settings


class ConversionCache:
    """Cache for conversion results"""
    
    def __init__(self, cache_dir: Optional[str] = None, ttl_hours: int = 24):
        """
        Initialize cache
        
        Args:
            cache_dir: Cache directory (defaults to temp_dir/cache)
            ttl_hours: Time to live in hours
        """
        self.cache_dir = Path(cache_dir or os.path.join(settings.temp_dir, "cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
    
    def _get_cache_key(self, content: str, format_name: str, options: Dict[str, Any]) -> str:
        """Generate cache key from content and options"""
        cache_data = {
            "content": content,
            "format": format_name,
            "options": options
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.cache"
    
    def get(self, content: str, format_name: str, options: Dict[str, Any]) -> Optional[str]:
        """
        Get cached conversion result
        
        Args:
            content: Markdown content
            format_name: Output format
            options: Conversion options
            
        Returns:
            Path to cached file or None
        """
        cache_key = self._get_cache_key(content, format_name, options)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        # Check if cache is expired
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime > self.ttl:
            cache_path.unlink()
            return None
        
        # Read metadata
        try:
            with open(cache_path.with_suffix('.meta'), 'r') as f:
                metadata = json.load(f)
                output_path = metadata.get('output_path')
                
                if output_path and Path(output_path).exists():
                    return output_path
        except:
            pass
        
        return None
    
    def set(self, content: str, format_name: str, options: Dict[str, Any], output_path: str) -> None:
        """
        Cache conversion result
        
        Args:
            content: Markdown content
            format_name: Output format
            options: Conversion options
            output_path: Path to output file
        """
        cache_key = self._get_cache_key(content, format_name, options)
        cache_path = self._get_cache_path(cache_key)
        
        # Save metadata
        metadata = {
            "output_path": output_path,
            "format": format_name,
            "cached_at": datetime.now().isoformat(),
            "options": options
        }
        
        with open(cache_path.with_suffix('.meta'), 'w') as f:
            json.dump(metadata, f)
        
        # Create symlink or copy reference
        # For now, just store the path in metadata
    
    def clear(self) -> int:
        """
        Clear all cache entries
        
        Returns:
            Number of entries cleared
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()
            count += 1
        
        for meta_file in self.cache_dir.glob("*.meta"):
            meta_file.unlink()
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.cache"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "entries": len(cache_files),
            "total_size": total_size,
            "cache_dir": str(self.cache_dir)
        }


# Global cache instance
_cache_instance: Optional[ConversionCache] = None


def get_cache() -> ConversionCache:
    """Get global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ConversionCache()
    return _cache_instance

