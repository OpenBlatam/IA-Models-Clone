from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

from typing import Dict, Any, Optional, List, Union
import httpx
import base64
import io
import numpy as np
from PIL import Image
import pyvips
import cv2
from rembg import remove, new_session
from uuid import UUID
import asyncio
from functools import lru_cache
try:
    import aioredis  # type: ignore
except Exception:  # pragma: no cover - optional in tests
    aioredis = None  # type: ignore[assignment]
from contextlib import asynccontextmanager
import time
import hashlib
import json
from onyx.llm.interface import (
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.storage import StorageService
from onyx.server.features.ads.config import settings
from .models import Ad
from .schemas import AdCreate
from typing import Any, List, Dict, Optional
import logging
"""
Optimized service for handling ads-related business logic.
"""

    generate_ads_lcel,
    generate_brand_kit_lcel,
    generate_custom_content_lcel
)

logger = setup_logger()

class OptimizedAdsService:
    """Optimized service for handling ads-related business logic."""
    
    def __init__(self) -> Any:
        """Initialize the service with optimized components."""
        self.rembg_session = new_session()
        self.storage_service = StorageService()
        self._redis_client = None
        self._http_client = None
        self._semaphore = asyncio.Semaphore(10)  # Limit concurrent operations
        
    @property
    async def redis_client(self) -> Any:
        """Lazy initialization of Redis client."""
        if self._redis_client is None:
            self._redis_client = await aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis_client
    
    @property
    async async def http_client(self) -> Any:
        """Lazy initialization of HTTP client with connection pooling."""
        if self._http_client is None:
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
            self._http_client = httpx.AsyncClient(
                limits=limits,
                timeout=httpx.Timeout(30.0)
            )
        return self._http_client
    
    async def _get_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters."""
        key_data = json.dumps(kwargs, sort_keys=True)
        return f"ads:{prefix}:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result from Redis."""
        try:
            redis = await self.redis_client
            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None
    
    async def _set_cached_result(self, cache_key: str, result: Dict[str, Any], ttl: int = 3600):
        """Set result in cache with TTL."""
        try:
            redis = await self.redis_client
            await redis.setex(cache_key, ttl, json.dumps(result))
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    @asynccontextmanager
    async def _rate_limit(self, user_id: int, operation: str):
        """Rate limiting context manager."""
        redis = await self.redis_client
        key = f"rate_limit:{user_id}:{operation}"
        
        # Check current rate
        current = await redis.get(key)
        if current and int(current) >= settings.RATE_LIMITS.get(operation, 100):
            raise ValueError(f"Rate limit exceeded for {operation}")
        
        # Increment counter
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 3600)  # 1 hour window
        await pipe.execute()
        
        try:
            yield
        finally:
            pass
    
    async def generate_ads(
        self,
        prompt: str,
        type: str,
        user_id: int,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Generate ads based on prompt with caching and rate limiting.
        
        Args:
            prompt: The prompt to generate ads from
            type: The type of content to generate (ads, brand-kit, custom)
            user_id: User ID for rate limiting
            use_cache: Whether to use caching
            
        Returns:
            Dict containing the generated content
        """
        async with self._rate_limit(user_id, "ads_generation"):
            # Check cache first
            if use_cache:
                cache_key = await self._get_cache_key("generation", prompt=prompt, type=type)
                cached_result = await self._get_cached_result(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for ads generation: {cache_key}")
                    return cached_result
            
            # Generate content
            async with self._semaphore:
                start_time = time.time()
                
                try:
                    if type == "ads":
                        content = await generate_ads_lcel(
                            prompt,
                            model=settings.LLM_MODEL,
                            temperature=settings.LLM_TEMPERATURE,
                            max_tokens=settings.LLM_MAX_TOKENS
                        )
                        result = {
                            "type": "ads",
                            "content": content,
                            "url": None,
                            "generation_time": time.time() - start_time
                        }
                    elif type == "brand-kit":
                        content = await generate_brand_kit_lcel(
                            prompt,
                            model=settings.LLM_MODEL,
                            temperature=settings.LLM_TEMPERATURE,
                            max_tokens=settings.LLM_MAX_TOKENS
                        )
                        result = {
                            "type": "brand_kit",
                            "content": content,
                            "url": None,
                            "generation_time": time.time() - start_time
                        }
                    elif type == "custom":
                        content = await generate_custom_content_lcel(
                            prompt,
                            model=settings.LLM_MODEL,
                            temperature=settings.LLM_TEMPERATURE,
                            max_tokens=settings.LLM_MAX_TOKENS
                        )
                        result = {
                            "type": "custom_content",
                            "content": content,
                            "url": None,
                            "generation_time": time.time() - start_time
                        }
                    else:
                        raise ValueError(f"Invalid content type: {type}")
                    
                    # Cache the result
                    if use_cache:
                        cache_key = await self._get_cache_key("generation", prompt=prompt, type=type)
                        await self._set_cached_result(cache_key, result, ttl=7200)  # 2 hours
                    
                    return result
                    
                except Exception as e:
                    logger.exception("Error generating ads")
                    raise
    
    async def remove_background(
        self,
        image_url: str,
        user_id: int,
        max_size: int = 1024
    ) -> Dict[str, Any]:
        """
        Remove background from an image with optimized processing.
        
        Args:
            image_url: URL of the image to process
            user_id: User ID for rate limiting
            max_size: Maximum image size for processing
            
        Returns:
            Dict containing the processed image URL and metadata
        """
        async with self._rate_limit(user_id, "background_removal"):
            # Check cache first
            cache_key = await self._get_cache_key("bg_removal", image_url=image_url)
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Cache hit for background removal: {cache_key}")
                return cached_result
            
            async with self._semaphore:
                start_time = time.time()
                
                try:
                    # Download image with timeout and size limits
                    client = await self.http_client
                    async with client.stream("GET", image_url) as response:
                        if response.status_code != 200:
                            raise ValueError(f"Failed to download image: {response.status_code}")
                        
                        # Stream image data with size limit
                        image_bytes = bytearray()
                        async for chunk in response.aiter_bytes():
                            image_bytes.extend(chunk)
                            if len(image_bytes) > settings.MAX_IMAGE_SIZE_BYTES:
                                raise ValueError("Image too large")
                    
                    # Optimized image processing
                    input_image = await self._process_image_optimized(image_bytes, max_size)
                    
                    # Remove background
                    output_image = remove(input_image, session=self.rembg_session)
                    has_alpha = output_image.mode == "RGBA" and np.array(output_image)[..., 3].max() > 0
                    
                    # Save processed image
                    buffered = io.BytesIO()
                    
                    if has_alpha:
                        output_image.save(buffered, format="PNG", optimize=True)
                        mime = "image/png"
                        ext = ".png"
                    else:
                        output_image = output_image.convert("RGB")
                        output_image.save(buffered, format="JPEG", quality=settings.JPEG_QUALITY, optimize=True)
                        mime = "image/jpeg"
                        ext = ".jpg"
                    
                    # Save to storage
                    buffered.seek(0)
                    filename = await self.storage_service.save_file(
                        file=buffered,
                        original_filename=f"processed{ext}"
                    )
                    processed_url = self.storage_service.get_file_url(filename)
                    
                    result = {
                        "type": "background_removal",
                        "original_url": image_url,
                        "processed_url": processed_url,
                        "mime": mime,
                        "processing_time": time.time() - start_time,
                        "file_size": len(buffered.getvalue())
                    }
                    
                    # Cache the result
                    await self._set_cached_result(cache_key, result, ttl=86400)  # 24 hours
                    
                    return result
                    
                except Exception as e:
                    logger.exception("Error removing background")
                    raise
    
    async def _process_image_optimized(self, image_bytes: bytes, max_size: int) -> Image.Image:
        """Optimized image processing with better error handling."""
        try:
            # Try pyvips first (faster for large images)
            img_vips = pyvips.Image.new_from_buffer(image_bytes, "", access="sequential")
            scale = min(max_size / img_vips.width, max_size / img_vips.height, 1.0)
            if scale < 1.0:
                img_vips = img_vips.resize(scale)
            png_bytes = img_vips.write_to_buffer(".png")
            return Image.open(io.BytesIO(png_bytes))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        except Exception as e:
            logger.warning(f"pyvips failed: {e}, falling back to PIL")
            try:
                # Use PIL directly for smaller images
                img = Image.open(io.BytesIO(image_bytes))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                if img.size[0] > max_size or img.size[1] > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                return img
            except Exception as e2:
                logger.warning(f"PIL failed: {e2}, trying OpenCV")
                # Final fallback to OpenCV
                arr = np.frombuffer(image_bytes, np.uint8)
                img_cv = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if img_cv is None:
                    raise ValueError("Failed to decode image with all methods")
                
                h, w = img_cv.shape[:2]
                scale = min(max_size / h, max_size / w, 1.0)
                if scale < 1.0:
                    img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)))
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                return Image.fromarray(img_rgb)
    
    async def batch_generate_ads(
        self,
        prompts: List[str],
        type: str,
        user_id: int,
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple ads concurrently with rate limiting.
        
        Args:
            prompts: List of prompts to generate ads from
            type: The type of content to generate
            user_id: User ID for rate limiting
            max_concurrent: Maximum concurrent generations
            
        Returns:
            List of generated content
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single(prompt: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.generate_ads(prompt, type, user_id)
        
        tasks = [generate_single(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to generate ad for prompt {i}: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def cleanup(self) -> Any:
        """Cleanup resources."""
        if self._http_client:
            await self._http_client.aclose()
        if self._redis_client:
            await self._redis_client.close()

class OptimizedAdService:
    """Optimized service layer for Ad business logic and persistence."""

    def __init__(self) -> Any:
        self._redis_client = None
    
    @property
    async def redis_client(self) -> Any:
        """Lazy initialization of Redis client."""
        if self._redis_client is None:
            self._redis_client = await aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis_client

    async def create_ad(self, data: AdCreate) -> Ad:
        """Create a new Ad with caching."""
        # TODO: Implement DB insert with connection pooling
        raise NotImplementedError

    async def get_ad(self, ad_id: UUID) -> Optional[Ad]:
        """Retrieve an Ad by ID with caching."""
        # Check cache first
        cache_key = f"ad:{ad_id}"
        redis = await self.redis_client
        cached = await redis.get(cache_key)
        if cached:
            return Ad.from_dict(json.loads(cached))
        
        # TODO: Implement DB fetch with connection pooling
        raise NotImplementedError

    async def list_ads(self, skip: int = 0, limit: int = 100) -> List[Ad]:
        """List Ads with pagination and caching."""
        # TODO: Implement DB query with connection pooling
        raise NotImplementedError

    async def update_ad(self, ad_id: UUID, data: AdCreate) -> Optional[Ad]:
        """Update an existing Ad with cache invalidation."""
        # TODO: Implement DB update with connection pooling
        # Invalidate cache
        cache_key = f"ad:{ad_id}"
        redis = await self.redis_client
        await redis.delete(cache_key)
        raise NotImplementedError

    async def delete_ad(self, ad_id: UUID) -> bool:
        """Delete an Ad by ID with cache invalidation."""
        # TODO: Implement DB delete with connection pooling
        # Invalidate cache
        cache_key = f"ad:{ad_id}"
        redis = await self.redis_client
        await redis.delete(cache_key)
        raise NotImplementedError 