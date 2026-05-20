"""
Cache Manager for video generation
Supports file-based and Redis caching
"""

import json
import hashlib
from typing import Optional, Any, Dict
from pathlib import Path
import logging
import pickle
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching for images, audio, and video metadata"""
    
    def __init__(self, cache_dir: Optional[str] = None, use_redis: bool = False):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("/tmp/faceless_video/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_redis = use_redis
        self.redis_client = None
        
        if use_redis:
            try:
                import redis
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=0,
                    decode_responses=False
                )
                self.redis_client.ping()
                logger.info("Redis cache enabled")
            except Exception as e:
                logger.warning(f"Redis not available, using file cache: {str(e)}")
                self.use_redis = False
    
    def _get_cache_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data"""
        data_str = json.dumps(data, sort_keys=True) if isinstance(data, dict) else str(data)
        hash_obj = hashlib.md5(data_str.encode())
        return f"{prefix}_{hash_obj.hexdigest()}"
    
    def get(self, key: str, default: Any = None) -> Optional[Any]:
        """Get value from cache"""
        if self.use_redis and self.redis_client:
            try:
                cached = self.redis_client.get(key)
                if cached:
                    return pickle.loads(cached)
            except Exception as e:
                logger.warning(f"Redis get failed: {str(e)}")
        
        # File-based cache
        cache_file = self.cache_dir / f"{key}.cache"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    # Check expiration
                    if cache_data.get('expires_at') and datetime.now() > cache_data['expires_at']:
                        cache_file.unlink()
                        return default
                    return cache_data.get('value')
            except Exception as e:
                logger.warning(f"Cache read failed: {str(e)}")
        
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL (seconds)"""
        expires_at = None
        if ttl:
            expires_at = datetime.now() + timedelta(seconds=ttl)
        
        cache_data = {
            'value': value,
            'created_at': datetime.now(),
            'expires_at': expires_at,
        }
        
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.setex(
                    key,
                    ttl or 86400,  # Default 24 hours
                    pickle.dumps(value)
                )
                return True
            except Exception as e:
                logger.warning(f"Redis set failed: {str(e)}")
        
        # File-based cache
        cache_file = self.cache_dir / f"{key}.cache"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            return True
        except Exception as e:
            logger.warning(f"Cache write failed: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception:
                pass
        
        cache_file = self.cache_dir / f"{key}.cache"
        if cache_file.exists():
            cache_file.unlink()
            return True
        
        return False
    
    def clear(self, prefix: Optional[str] = None) -> int:
        """Clear cache, optionally by prefix"""
        count = 0
        
        if self.use_redis and self.redis_client:
            try:
                if prefix:
                    keys = self.redis_client.keys(f"{prefix}*")
                    if keys:
                        self.redis_client.delete(*keys)
                        count = len(keys)
                else:
                    self.redis_client.flushdb()
                    count = -1  # Unknown count
            except Exception:
                pass
        
        # File-based cache
        for cache_file in self.cache_dir.glob("*.cache"):
            if prefix and not cache_file.name.startswith(prefix):
                continue
            cache_file.unlink()
            count += 1
        
        return count
    
    def cache_image(self, prompt: str, style: str, width: int, height: int, image_path: str, ttl: int = 86400) -> str:
        """Cache generated image metadata"""
        cache_key = self._get_cache_key("image", {
            "prompt": prompt,
            "style": style,
            "width": width,
            "height": height,
        })
        
        self.set(cache_key, {
            "image_path": image_path,
            "prompt": prompt,
            "style": style,
            "width": width,
            "height": height,
        }, ttl=ttl)
        
        return cache_key
    
    def get_cached_image(self, prompt: str, style: str, width: int, height: int) -> Optional[Dict[str, Any]]:
        """Get cached image metadata"""
        cache_key = self._get_cache_key("image", {
            "prompt": prompt,
            "style": style,
            "width": width,
            "height": height,
        })
        
        return self.get(cache_key)
    
    def cache_audio(self, text: str, voice: str, language: str, audio_path: str, ttl: int = 86400) -> str:
        """Cache generated audio metadata"""
        cache_key = self._get_cache_key("audio", {
            "text": text,
            "voice": voice,
            "language": language,
        })
        
        self.set(cache_key, {
            "audio_path": audio_path,
            "text": text,
            "voice": voice,
            "language": language,
        }, ttl=ttl)
        
        return cache_key
    
    def get_cached_audio(self, text: str, voice: str, language: str) -> Optional[Dict[str, Any]]:
        """Get cached audio metadata"""
        cache_key = self._get_cache_key("audio", {
            "text": text,
            "voice": voice,
            "language": language,
        })
        
        return self.get(cache_key)


_cache_manager: Optional[CacheManager] = None


def get_cache_manager(cache_dir: Optional[str] = None, use_redis: bool = False) -> CacheManager:
    """Get cache manager instance (singleton)"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(cache_dir=cache_dir, use_redis=use_redis)
    return _cache_manager

