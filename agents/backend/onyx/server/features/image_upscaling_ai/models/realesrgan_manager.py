"""
Real-ESRGAN Model Manager
=========================

Advanced model management with caching, auto-detection, and optimization.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from PIL import Image
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

try:
    from .realesrgan_integration import (
        RealESRGANWrapper,
        RealESRGANUpscaler,
        REALESRGAN_AVAILABLE
    )
except ImportError:
    REALESRGAN_AVAILABLE = False
    RealESRGANWrapper = None
    RealESRGANUpscaler = None


@dataclass
class ModelCacheEntry:
    """Cache entry for loaded models."""
    wrapper: Any
    last_used: float
    use_count: int
    memory_mb: float


class RealESRGANModelManager:
    """
    Advanced model manager with caching, auto-detection, and optimization.
    
    Features:
    - Model caching to avoid reloading
    - Automatic model selection based on image type
    - Memory management
    - Performance monitoring
    - Batch processing optimization
    """
    
    def __init__(
        self,
        max_cached_models: int = 3,
        cache_ttl: float = 3600.0,  # 1 hour
        auto_download: bool = False,
        device: Optional[str] = None,
    ):
        """
        Initialize model manager.
        
        Args:
            max_cached_models: Maximum number of models to cache
            cache_ttl: Time-to-live for cached models (seconds)
            auto_download: Auto-download models if not found
            device: Device to use (cuda/cpu)
        """
        self.max_cached_models = max_cached_models
        self.cache_ttl = cache_ttl
        self.auto_download = auto_download
        self.device = device
        
        # Model cache (LRU)
        self.model_cache: OrderedDict[str, ModelCacheEntry] = OrderedDict()
        
        # Statistics
        self.stats = {
            "models_loaded": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "models_evicted": 0,
            "total_upscales": 0,
        }
        
        logger.info(f"RealESRGANModelManager initialized (max_cache: {max_cached_models})")
    
    def _get_model_key(self, model_name: str, device: str) -> str:
        """Generate cache key for model."""
        return f"{model_name}_{device}"
    
    def _is_cache_valid(self, entry: ModelCacheEntry) -> bool:
        """Check if cache entry is still valid."""
        age = time.time() - entry.last_used
        return age < self.cache_ttl
    
    def _evict_oldest(self) -> None:
        """Evict oldest model from cache."""
        if not self.model_cache:
            return
        
        # Remove oldest entry
        oldest_key = next(iter(self.model_cache))
        entry = self.model_cache.pop(oldest_key)
        
        # Clean up model
        if hasattr(entry.wrapper, 'upsampler'):
            del entry.wrapper.upsampler
        del entry.wrapper
        
        self.stats["models_evicted"] += 1
        logger.debug(f"Evicted model from cache: {oldest_key}")
    
    def _load_model(
        self,
        model_name: str,
        device: Optional[str] = None
    ) -> RealESRGANWrapper:
        """Load a model (with caching)."""
        if not REALESRGAN_AVAILABLE:
            raise ImportError("Real-ESRGAN not available")
        
        device = device or self.device or "cpu"
        cache_key = self._get_model_key(model_name, device)
        
        # Check cache
        if cache_key in self.model_cache:
            entry = self.model_cache[cache_key]
            if self._is_cache_valid(entry):
                # Move to end (most recently used)
                self.model_cache.move_to_end(cache_key)
                entry.last_used = time.time()
                entry.use_count += 1
                self.stats["cache_hits"] += 1
                logger.debug(f"Cache hit: {cache_key}")
                return entry.wrapper
            else:
                # Remove expired entry
                del self.model_cache[cache_key]
        
        # Cache miss - load model
        self.stats["cache_misses"] += 1
        logger.info(f"Loading model: {model_name} on {device}")
        
        # Evict if cache is full
        if len(self.model_cache) >= self.max_cached_models:
            self._evict_oldest()
        
        # Load model
        wrapper = RealESRGANWrapper(
            model_name=model_name,
            device=device,
        )
        
        # Add to cache
        entry = ModelCacheEntry(
            wrapper=wrapper,
            last_used=time.time(),
            use_count=1,
            memory_mb=0.0  # Could estimate memory usage
        )
        self.model_cache[cache_key] = entry
        self.stats["models_loaded"] += 1
        
        return wrapper
    
    def get_model(
        self,
        model_name: str,
        device: Optional[str] = None
    ) -> RealESRGANWrapper:
        """
        Get a model (cached or load new).
        
        Args:
            model_name: Model name
            device: Device to use
            
        Returns:
            RealESRGANWrapper instance
        """
        return self._load_model(model_name, device)
    
    def detect_image_type(self, image: Image.Image) -> str:
        """
        Detect image type (anime/photo/artwork).
        
        Uses advanced detection if available, otherwise simple heuristics.
        
        Args:
            image: Input image
            
        Returns:
            Image type: 'anime', 'photo', 'artwork', or 'pixel_art'
        """
        try:
            # Try advanced detection
            try:
                from .advanced_image_detection import AdvancedImageDetector
                detector = AdvancedImageDetector()
                analysis = detector.analyze(image)
                return analysis.image_type
            except ImportError:
                pass
            
            # Fallback to simple heuristics
            img_array = np.array(image.convert("RGB"))
            
            # Calculate statistics
            mean_color = np.mean(img_array, axis=(0, 1))
            std_color = np.std(img_array, axis=(0, 1))
            
            # Saturation (distance from gray)
            gray = np.mean(mean_color)
            saturation = np.std([c - gray for c in mean_color])
            
            # Contrast
            contrast = np.mean(std_color)
            
            # Pixel art detection
            unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
            if unique_colors < 256 and image.size[0] < 512:
                return "pixel_art"
            
            # Heuristics
            if saturation > 50 and contrast > 40:
                return "anime"
            elif contrast < 30 and saturation < 30:
                return "photo"
            else:
                return "artwork"
                
        except Exception as e:
            logger.warning(f"Error detecting image type: {e}, defaulting to 'artwork'")
            return "artwork"
    
    def select_best_model(
        self,
        image: Image.Image,
        scale_factor: float,
        image_type: Optional[str] = None
    ) -> str:
        """
        Select best model for image and scale factor.
        
        Args:
            image: Input image
            scale_factor: Desired scale factor
            image_type: Image type (auto-detect if None)
            
        Returns:
            Recommended model name
        """
        if image_type is None:
            image_type = self.detect_image_type(image)
        
        # Select model based on type and scale
        if scale_factor <= 2.0:
            return "RealESRGAN_x2plus"
        elif scale_factor <= 4.0:
            if image_type == "anime":
                return "RealESRGAN_x4plus_anime_6B"
            else:
                return "RealESRGAN_x4plus"
        else:
            # For > 4x, use 4x model with multiple passes
            if image_type == "anime":
                return "RealESRGAN_x4plus_anime_6B"
            else:
                return "RealESRGAN_x4plus"
    
    async def upscale_async(
        self,
        image: Image.Image,
        scale_factor: float,
        model_name: Optional[str] = None,
        auto_select: bool = True,
        device: Optional[str] = None
    ) -> Image.Image:
        """
        Upscale image asynchronously.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            model_name: Specific model (auto-select if None)
            auto_select: Auto-select best model
            device: Device to use
            
        Returns:
            Upscaled image
        """
        # Select model
        if model_name is None and auto_select:
            model_name = self.select_best_model(image, scale_factor)
        
        if model_name is None:
            model_name = "RealESRGAN_x4plus"
        
        # Get model
        wrapper = self.get_model(model_name, device)
        
        # Upscale in thread pool (Real-ESRGAN is CPU/GPU bound)
        loop = asyncio.get_event_loop()
        upscaled = await loop.run_in_executor(
            None,
            wrapper.upscale,
            image,
            scale_factor
        )
        
        self.stats["total_upscales"] += 1
        return upscaled
    
    async def batch_upscale_async(
        self,
        images: List[Image.Image],
        scale_factor: float,
        model_name: Optional[str] = None,
        auto_select: bool = True,
        max_concurrent: int = 2
    ) -> List[Image.Image]:
        """
        Upscale multiple images in parallel.
        
        Args:
            images: List of images
            scale_factor: Scale factor
            model_name: Specific model
            auto_select: Auto-select best model
            max_concurrent: Maximum concurrent upscales
            
        Returns:
            List of upscaled images
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def upscale_with_semaphore(img):
            async with semaphore:
                return await self.upscale_async(
                    img,
                    scale_factor,
                    model_name,
                    auto_select
                )
        
        tasks = [upscale_with_semaphore(img) for img in images]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def clear_cache(self) -> None:
        """Clear model cache."""
        for entry in self.model_cache.values():
            if hasattr(entry.wrapper, 'upsampler'):
                del entry.wrapper.upsampler
            del entry.wrapper
        
        self.model_cache.clear()
        logger.info("Model cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            **self.stats,
            "cached_models": len(self.model_cache),
            "cache_hit_rate": (
                self.stats["cache_hits"] / max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
            ),
            "models": list(self.model_cache.keys())
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        return {
            "cached_models": len(self.model_cache),
            "max_models": self.max_cached_models,
            "cache_ttl": self.cache_ttl,
            "entries": [
                {
                    "key": key,
                    "model": entry.wrapper.model_name,
                    "device": entry.wrapper.device,
                    "last_used": entry.last_used,
                    "use_count": entry.use_count,
                    "age_seconds": time.time() - entry.last_used,
                }
                for key, entry in self.model_cache.items()
            ]
        }

