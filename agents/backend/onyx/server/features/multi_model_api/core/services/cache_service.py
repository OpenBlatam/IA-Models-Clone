"""
Cache service for multi-model API
Handles all cache-related operations
"""

import logging
from typing import Optional
from fastapi import BackgroundTasks
from ...api.schemas import MultiModelRequest, MultiModelResponse
from ...api.helpers import get_enabled_models
from ..cache import EnhancedCache

logger = logging.getLogger(__name__)


class CacheService:
    """Service for cache operations"""
    
    def __init__(self, cache: EnhancedCache):
        """
        Initialize cache service
        
        Args:
            cache: EnhancedCache instance
        """
        self.cache = cache
    
    def _generate_cache_key(self, request: MultiModelRequest) -> str:
        """Generate cache key for request - optimized"""
        # Early return if cache disabled
        if not request.cache_enabled:
            return ""
        
        # Use helper for consistency
        enabled_models = get_enabled_models(request.models)
        if not enabled_models:
            return ""
        
        # Extract and sort for consistent cache keys
        model_types = [m.model_type.value for m in enabled_models]
        model_types.sort()  # In-place sort for consistency
        model_types_str = ",".join(model_types)
        
        return self.cache._generate_key(
            "multi_model",
            request.prompt,
            model_types_str,
            request.strategy,
            request.consensus_method or "majority"
        )
    
    async def get_cached_response(
        self,
        request: MultiModelRequest,
        retry_on_failure: bool = True
    ) -> Optional[MultiModelResponse]:
        """
        Get cached response if available
        
        Args:
            request: Multi-model request
            retry_on_failure: Whether to retry on cache failure
            
        Returns:
            Cached response or None
        """
        if not request.cache_enabled:
            return None
        
        cache_key = self._generate_cache_key(request)
        
        async def _get_from_cache():
            try:
                cached_data = await self.cache.get(cache_key)
                
                if cached_data:
                    logger.debug(
                        f"Cache hit for key: {cache_key[:50]}...",
                        extra={"cache_key_prefix": cache_key[:50]}
                    )
                    return MultiModelResponse(**cached_data, cache_hit=True)
                
                return None
            except Exception as e:
                logger.error(
                    f"Error getting from cache: {e}",
                    extra={"cache_key_prefix": cache_key[:50]},
                    exc_info=True
                )
                raise
        
        try:
            if retry_on_failure:
                from .retry_service import RetryService, RetryConfig
                retry_service = RetryService(
                    RetryConfig(
                        max_attempts=2,
                        initial_delay=0.1,
                        retryable_exceptions=(Exception,)
                    )
                )
                return await retry_service.execute_with_retry(
                    _get_from_cache,
                    operation_name="cache_get",
                    context={"cache_key_prefix": cache_key[:50]}
                )
            else:
                return await _get_from_cache()
        except Exception:
            # Don't fail request if cache fails
            logger.warning(
                f"Cache get failed after retries, continuing without cache",
                extra={"cache_key_prefix": cache_key[:50]}
            )
            return None
    
    async def cache_response(
        self,
        request: MultiModelRequest,
        response: MultiModelResponse,
        background_task: Optional[BackgroundTasks] = None
    ) -> None:
        """
        Cache response
        
        Args:
            request: Original request
            response: Response to cache
            background_task: Optional background task function
        """
        if not request.cache_enabled:
            return
        
        try:
            cache_key = self._generate_cache_key(request)
            cache_ttl = request.cache_ttl or 3600
            
            # Convert response to dict for caching - use model_dump() if available
            _dict_method = getattr(response, 'model_dump', None)
            response_dict = _dict_method() if _dict_method else response.dict()
            
            # Use background task if available, otherwise await
            if background_task:
                background_task.add_task(
                    self.cache.set,
                    cache_key,
                    response_dict,
                    ttl=cache_ttl
                )
            else:
                await self.cache.set(cache_key, response_dict, ttl=cache_ttl)
            
            logger.debug(f"Cached response with key: {cache_key[:50]}...")
        except Exception as e:
            logger.error(f"Error caching response: {e}", exc_info=True)
            # Don't fail request if cache fails

