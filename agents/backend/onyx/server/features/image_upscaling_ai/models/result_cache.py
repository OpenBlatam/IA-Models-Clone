"""
Result Cache for Upscaling
===========================

Cache system for upscaling results to avoid reprocessing.
"""

import logging
import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from PIL import Image
import pickle

logger = logging.getLogger(__name__)


class ResultCache:
    """
    Cache system for upscaling results.
    
    Features:
    - Hash-based cache keys
    - Metadata storage
    - Automatic cleanup
    - Statistics tracking
    """
    
    def __init__(
        self,
        cache_dir: str = "./upscaling_cache",
        max_size_mb: int = 1000,
        enabled: bool = True
    ):
        """
        Initialize result cache.
        
        Args:
            cache_dir: Cache directory
            max_size_mb: Maximum cache size in MB
            enabled: Whether cache is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.enabled = enabled
        
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_file = self.cache_dir / "metadata.json"
            self._load_metadata()
        
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "size_mb": 0.0
        }
    
    def _load_metadata(self) -> None:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving cache metadata: {e}")
    
    def _generate_key(
        self,
        image_path: str,
        scale_factor: float,
        quality_mode: str,
        use_ai: bool,
        use_optimization_core: bool
    ) -> str:
        """
        Generate cache key from parameters.
        
        Args:
            image_path: Path to image
            scale_factor: Scale factor
            quality_mode: Quality mode
            use_ai: Whether AI was used
            use_optimization_core: Whether optimization_core was used
            
        Returns:
            Cache key
        """
        # Create hash from parameters
        key_data = f"{image_path}:{scale_factor}:{quality_mode}:{use_ai}:{use_optimization_core}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return key_hash
    
    def get(
        self,
        image_path: str,
        scale_factor: float,
        quality_mode: str,
        use_ai: bool,
        use_optimization_core: bool
    ) -> Optional[Tuple[Image.Image, Dict[str, Any]]]:
        """
        Get cached result.
        
        Args:
            image_path: Path to image
            scale_factor: Scale factor
            quality_mode: Quality mode
            use_ai: Whether AI was used
            use_optimization_core: Whether optimization_core was used
            
        Returns:
            Tuple of (image, metrics) or None if not cached
        """
        if not self.enabled:
            return None
        
        key = self._generate_key(image_path, scale_factor, quality_mode, use_ai, use_optimization_core)
        
        if key not in self.metadata:
            self.stats["misses"] += 1
            return None
        
        cache_entry = self.metadata[key]
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if not cache_file.exists():
            # Remove stale metadata
            del self.metadata[key]
            self._save_metadata()
            self.stats["misses"] += 1
            return None
        
        try:
            # Load cached result
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            image = cached_data["image"]
            metrics = cached_data["metrics"]
            
            self.stats["hits"] += 1
            logger.debug(f"Cache hit for key: {key}")
            
            return image, metrics
            
        except Exception as e:
            logger.warning(f"Error loading cache entry: {e}")
            # Remove corrupted cache entry
            if cache_file.exists():
                cache_file.unlink()
            if key in self.metadata:
                del self.metadata[key]
                self._save_metadata()
            self.stats["misses"] += 1
            return None
    
    def save(
        self,
        image_path: str,
        scale_factor: float,
        quality_mode: str,
        use_ai: bool,
        use_optimization_core: bool,
        image: Image.Image,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Save result to cache.
        
        Args:
            image_path: Path to image
            scale_factor: Scale factor
            quality_mode: Quality mode
            use_ai: Whether AI was used
            use_optimization_core: Whether optimization_core was used
            image: Upscaled image
            metrics: Processing metrics
        """
        if not self.enabled:
            return
        
        key = self._generate_key(image_path, scale_factor, quality_mode, use_ai, use_optimization_core)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        try:
            # Save cached result
            cached_data = {
                "image": image,
                "metrics": metrics,
                "timestamp": time.time()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            
            # Update metadata
            self.metadata[key] = {
                "image_path": str(image_path),
                "scale_factor": scale_factor,
                "quality_mode": quality_mode,
                "use_ai": use_ai,
                "use_optimization_core": use_optimization_core,
                "timestamp": time.time(),
                "file_size": cache_file.stat().st_size
            }
            
            self._save_metadata()
            self.stats["saves"] += 1
            
            # Check cache size and cleanup if needed
            self._cleanup_if_needed()
            
            logger.debug(f"Cached result with key: {key}")
            
        except Exception as e:
            logger.warning(f"Error saving cache entry: {e}")
    
    def _cleanup_if_needed(self) -> None:
        """Cleanup cache if it exceeds max size."""
        try:
            total_size = sum(
                entry.get("file_size", 0)
                for entry in self.metadata.values()
            )
            total_size_mb = total_size / (1024 * 1024)
            self.stats["size_mb"] = total_size_mb
            
            if total_size_mb > self.max_size_mb:
                # Remove oldest entries
                sorted_entries = sorted(
                    self.metadata.items(),
                    key=lambda x: x[1].get("timestamp", 0)
                )
                
                # Remove 20% of oldest entries
                remove_count = max(1, len(sorted_entries) // 5)
                
                for key, entry in sorted_entries[:remove_count]:
                    cache_file = self.cache_dir / f"{key}.pkl"
                    if cache_file.exists():
                        cache_file.unlink()
                    del self.metadata[key]
                
                self._save_metadata()
                logger.info(f"Cleaned up {remove_count} cache entries")
                
        except Exception as e:
            logger.warning(f"Error during cache cleanup: {e}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        if not self.enabled:
            return
        
        try:
            # Remove all cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            
            # Clear metadata
            self.metadata = {}
            self._save_metadata()
            
            # Reset stats
            self.stats = {
                "hits": 0,
                "misses": 0,
                "saves": 0,
                "size_mb": 0.0
            }
            
            logger.info("Cache cleared")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = (
            self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
            if (self.stats["hits"] + self.stats["misses"]) > 0 else 0.0
        )
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "enabled": self.enabled,
            "cache_dir": str(self.cache_dir),
            "max_size_mb": self.max_size_mb,
            "entries": len(self.metadata)
        }

