"""
Graceful Degradation
Fallback strategies when services are unavailable
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """Degradation level"""
    NONE = "none"
    PARTIAL = "partial"
    FULL = "full"


class FallbackStrategy:
    """Fallback strategy configuration"""
    
    def __init__(
        self,
        fallback_func: Optional[Callable] = None,
        cache_fallback: bool = True,
        default_value: Any = None,
        degradation_level: DegradationLevel = DegradationLevel.PARTIAL
    ):
        self.fallback_func = fallback_func
        self.cache_fallback = cache_fallback
        self.default_value = default_value
        self.degradation_level = degradation_level


class GracefulDegradation:
    """
    Graceful degradation manager
    
    Features:
    - Fallback strategies
    - Cache fallback
    - Default values
    - Degradation levels
    """
    
    def __init__(self):
        self._strategies: Dict[str, FallbackStrategy] = {}
        self._cache: Dict[str, Any] = {}
    
    def register_fallback(
        self,
        service_name: str,
        strategy: FallbackStrategy
    ) -> None:
        """Register fallback strategy for service"""
        self._strategies[service_name] = strategy
        logger.info(f"Registered fallback for: {service_name}")
    
    async def execute_with_fallback(
        self,
        service_name: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation with fallback
        
        Args:
            service_name: Service name
            operation: Operation to execute
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result or fallback value
        """
        strategy = self._strategies.get(service_name)
        
        try:
            # Try primary operation
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            
            # Cache successful result
            if strategy and strategy.cache_fallback:
                cache_key = f"{service_name}:{args}:{kwargs}"
                self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.warning(f"Operation failed for {service_name}: {str(e)}")
            
            # Try fallback
            if strategy:
                if strategy.fallback_func:
                    try:
                        if asyncio.iscoroutinefunction(strategy.fallback_func):
                            return await strategy.fallback_func(*args, **kwargs)
                        else:
                            return strategy.fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"Fallback failed: {str(fallback_error)}")
                
                # Try cache fallback
                if strategy.cache_fallback:
                    cache_key = f"{service_name}:{args}:{kwargs}"
                    if cache_key in self._cache:
                        logger.info(f"Using cached result for {service_name}")
                        return self._cache[cache_key]
                
                # Return default value
                if strategy.default_value is not None:
                    logger.info(f"Using default value for {service_name}")
                    return strategy.default_value
            
            # Re-raise if no fallback
            raise
    
    def get_degradation_level(self, service_name: str) -> DegradationLevel:
        """Get current degradation level for service"""
        strategy = self._strategies.get(service_name)
        if strategy:
            return strategy.degradation_level
        return DegradationLevel.NONE


# Global degradation manager
_degradation: Optional[GracefulDegradation] = None


def get_graceful_degradation() -> GracefulDegradation:
    """Get global graceful degradation manager"""
    global _degradation
    if _degradation is None:
        _degradation = GracefulDegradation()
    return _degradation

