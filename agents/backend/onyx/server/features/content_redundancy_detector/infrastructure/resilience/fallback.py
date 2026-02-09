"""
Advanced Fallback Mechanism - Graceful degradation and failover
Production-ready fallback system with multiple strategies
"""

import asyncio
import logging
from typing import Any, Callable, Optional, List, Dict, Union
from dataclasses import dataclass
from enum import Enum
import functools

logger = logging.getLogger(__name__)

class FallbackStrategy(Enum):
    """Fallback strategies"""
    DEFAULT_VALUE = "default_value"
    ALTERNATIVE_FUNC = "alternative_func"
    CACHED_VALUE = "cached_value"
    NULL_RESULT = "null_result"
    MULTIPLE_ATTEMPTS = "multiple_attempts"

@dataclass
class FallbackConfig:
    """Fallback configuration"""
    strategy: FallbackStrategy = FallbackStrategy.DEFAULT_VALUE
    default_value: Any = None
    alternative_func: Optional[Callable] = None
    cache_key: Optional[str] = None
    cache_service: Any = None
    attempt_order: List[Callable] = None
    on_fallback: Optional[Callable[[Exception], None]] = None

class FallbackHandler:
    """Advanced fallback handler with multiple strategies"""
    
    def __init__(self, config: FallbackConfig = None):
        self.config = config or FallbackConfig()
        self.fallback_count = 0
        self.success_count = 0

    async def execute_with_fallback(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with fallback handling"""
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self.success_count += 1
            return result
            
        except Exception as e:
            logger.warning(f"Primary function failed, using fallback: {e}")
            self.fallback_count += 1
            
            # Call on_fallback callback
            if self.config.on_fallback:
                try:
                    self.config.on_fallback(e)
                except Exception as callback_error:
                    logger.warning(f"Fallback callback error: {callback_error}")
            
            # Execute fallback strategy
            return await self._execute_fallback(e, *args, **kwargs)

    async def _execute_fallback(self, error: Exception, *args, **kwargs) -> Any:
        """Execute fallback strategy"""
        strategy = self.config.strategy
        
        if strategy == FallbackStrategy.DEFAULT_VALUE:
            return self.config.default_value
        
        elif strategy == FallbackStrategy.ALTERNATIVE_FUNC:
            if not self.config.alternative_func:
                raise ValueError("alternative_func not provided")
            
            alt_func = self.config.alternative_func
            if asyncio.iscoroutinefunction(alt_func):
                return await alt_func(*args, **kwargs)
            else:
                return alt_func(*args, **kwargs)
        
        elif strategy == FallbackStrategy.CACHED_VALUE:
            if not self.config.cache_key or not self.config.cache_service:
                raise ValueError("cache_key and cache_service required")
            
            cached = await self.config.cache_service.get(self.config.cache_key)
            if cached:
                logger.info(f"Using cached value for {self.config.cache_key}")
                return cached
            else:
                return self.config.default_value
        
        elif strategy == FallbackStrategy.NULL_RESULT:
            return None
        
        elif strategy == FallbackStrategy.MULTIPLE_ATTEMPTS:
            if not self.config.attempt_order:
                raise ValueError("attempt_order required for MULTIPLE_ATTEMPTS")
            
            for attempt_func in self.config.attempt_order:
                try:
                    if asyncio.iscoroutinefunction(attempt_func):
                        return await attempt_func(*args, **kwargs)
                    else:
                        return attempt_func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Fallback attempt failed: {e}")
                    continue
            
            # All attempts failed
            return self.config.default_value
        
        else:
            return self.config.default_value

    def get_stats(self) -> Dict[str, Any]:
        """Get fallback statistics"""
        total = self.fallback_count + self.success_count
        fallback_rate = self.fallback_count / max(total, 1)
        
        return {
            "fallback_count": self.fallback_count,
            "success_count": self.success_count,
            "total": total,
            "fallback_rate": fallback_rate
        }

def with_fallback(config: FallbackConfig):
    """Decorator for automatic fallback handling"""
    def decorator(func: Callable) -> Callable:
        handler = FallbackHandler(config)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await handler.execute_with_fallback(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to handle differently
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Primary function failed, using fallback: {e}")
                handler.fallback_count += 1
                
                if config.strategy == FallbackStrategy.DEFAULT_VALUE:
                    return config.default_value
                elif config.strategy == FallbackStrategy.ALTERNATIVE_FUNC:
                    if config.alternative_func:
                        return config.alternative_func(*args, **kwargs)
                elif config.strategy == FallbackStrategy.NULL_RESULT:
                    return None
                
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator






