"""
Cache Helper for Contador AI
============================

Refactored with:
- CacheStrategy pattern for different caching approaches
- CacheResult dataclass for typed returns
- CacheConfig for configuration
- Observer hooks for cache events
"""

import logging
from typing import Dict, Any, Optional, Callable, Awaitable, Protocol, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class CacheEvent(Enum):
    """Cache events."""
    HIT = "hit"
    MISS = "miss"
    STORE = "store"
    EVICT = "evict"


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""
    enabled: bool = True
    default_ttl: int = 3600  # 1 hour
    service_ttls: Dict[str, int] = field(default_factory=dict)
    
    def get_ttl(self, service_name: str) -> int:
        """Get TTL for a service."""
        return self.service_ttls.get(service_name, self.default_ttl)


@dataclass
class CacheResult:
    """Result from cache operation."""
    found: bool
    data: Optional[Dict[str, Any]] = None
    from_cache: bool = False
    
    @classmethod
    def hit(cls, data: Dict[str, Any]) -> "CacheResult":
        """Create a cache hit result."""
        result_data = data.copy()
        result_data["from_cache"] = True
        return cls(found=True, data=result_data, from_cache=True)
    
    @classmethod
    def miss(cls) -> "CacheResult":
        """Create a cache miss result."""
        return cls(found=False)


class CacheObserver(Protocol):
    """Protocol for cache observers."""
    
    def on_cache_event(self, event: CacheEvent, service_name: str, params: Dict[str, Any]) -> None:
        """Called when a cache event occurs."""
        ...


class CacheStrategy(ABC):
    """Abstract base class for cache strategies."""
    
    @abstractmethod
    def get(
        self, 
        cache: Any, 
        service_name: str, 
        params: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> CacheResult:
        """Get from cache."""
        pass
    
    @abstractmethod
    def store(
        self,
        cache: Any,
        service_name: str,
        params: Dict[str, Any],
        data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Store in cache."""
        pass


class DefaultCacheStrategy(CacheStrategy):
    """Default cache strategy."""
    
    def get(
        self,
        cache: Any,
        service_name: str,
        params: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> CacheResult:
        """Get from cache using default method."""
        try:
            if ttl:
                cached = cache.get(service_name, params, ttl=ttl)
            else:
                cached = cache.get(service_name, params)
            
            if cached:
                return CacheResult.hit(cached)
        except Exception as e:
            logger.debug(f"Cache get error for {service_name}: {e}")
        
        return CacheResult.miss()
    
    def store(
        self,
        cache: Any,
        service_name: str,
        params: Dict[str, Any],
        data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Store in cache."""
        try:
            if ttl:
                cache.put(service_name, params, data, ttl=ttl)
            else:
                cache.put(service_name, params, data)
            return True
        except Exception as e:
            logger.debug(f"Cache store error for {service_name}: {e}")
            return False


class CacheObserverRegistry:
    """Registry for cache observers."""
    
    def __init__(self):
        self._observers: List[CacheObserver] = []
    
    def register(self, observer: CacheObserver):
        """Register an observer."""
        self._observers.append(observer)
    
    def notify(self, event: CacheEvent, service_name: str, params: Dict[str, Any]):
        """Notify all observers."""
        for observer in self._observers:
            try:
                observer.on_cache_event(event, service_name, params)
            except Exception as e:
                logger.warning(f"Cache observer error: {e}")


class CacheHelper:
    """
    Helper class for cache operations in ContadorAI service methods.
    
    Refactored with:
    - Strategy pattern for cache operations
    - Configuration object for settings
    - Observer pattern for cache events
    - Dataclass results for type safety
    
    Responsibilities:
    - Check cache for existing results
    - Store results in cache
    - Handle cache TTL configuration
    """
    
    _strategy = DefaultCacheStrategy()
    _observers = CacheObserverRegistry()
    _config = CacheConfig()
    
    @classmethod
    def configure(cls, config: CacheConfig):
        """Configure the cache helper."""
        cls._config = config
    
    @classmethod
    def set_strategy(cls, strategy: CacheStrategy):
        """Set the cache strategy."""
        cls._strategy = strategy
    
    @classmethod
    def add_observer(cls, observer: CacheObserver):
        """Add a cache observer."""
        cls._observers.register(observer)
    
    @staticmethod
    def get_cached_result(
        cache: Optional[Any],
        service_name: str,
        cache_params: Dict[str, Any],
        use_cache: bool = True,
        ttl: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result if available.
        
        Args:
            cache: Cache instance (can be None)
            service_name: Name of the service
            cache_params: Parameters to use as cache key
            use_cache: Whether to use cache
            ttl: Optional TTL override
            
        Returns:
            Cached result if found, None otherwise
        """
        if not use_cache or not cache or not CacheHelper._config.enabled:
            return None
        
        effective_ttl = ttl or CacheHelper._config.get_ttl(service_name)
        result = CacheHelper._strategy.get(cache, service_name, cache_params, effective_ttl)
        
        if result.found:
            CacheHelper._observers.notify(CacheEvent.HIT, service_name, cache_params)
            return result.data
        
        CacheHelper._observers.notify(CacheEvent.MISS, service_name, cache_params)
        return None
    
    @staticmethod
    def store_result(
        cache: Optional[Any],
        service_name: str,
        cache_params: Dict[str, Any],
        result: Dict[str, Any],
        use_cache: bool = True,
        ttl: Optional[int] = None
    ) -> None:
        """
        Store result in cache if enabled and result is successful.
        
        Args:
            cache: Cache instance (can be None)
            service_name: Name of the service
            cache_params: Parameters to use as cache key
            result: Result dictionary to cache
            use_cache: Whether to use cache
            ttl: Optional TTL override
        """
        if not use_cache or not cache or not CacheHelper._config.enabled:
            return
        
        if not result.get("success"):
            return
        
        effective_ttl = ttl or CacheHelper._config.get_ttl(service_name)
        success = CacheHelper._strategy.store(
            cache, service_name, cache_params, result, effective_ttl
        )
        
        if success:
            CacheHelper._observers.notify(CacheEvent.STORE, service_name, cache_params)
    
    @staticmethod
    async def cached_service_execution(
        cache: Optional[Any],
        service_name: str,
        cache_params: Dict[str, Any],
        service_func: Callable[[], Awaitable[Dict[str, Any]]],
        use_cache: bool = True,
        cache_ttl: Optional[int] = None,
        store_ttl: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute a service with automatic cache checking and storing.
        
        This method encapsulates the common pattern:
        1. Check cache
        2. If cached, return cached result
        3. Otherwise, execute service
        4. Store result in cache
        
        Args:
            cache: Cache instance (can be None)
            service_name: Name of the service
            cache_params: Parameters to use as cache key
            service_func: Async function that executes the service
            use_cache: Whether to use cache
            cache_ttl: Optional TTL for cache retrieval
            store_ttl: Optional TTL for cache storage
            
        Returns:
            Service result dictionary (from cache or fresh execution)
        """
        # Check cache first
        cached = CacheHelper.get_cached_result(
            cache, service_name, cache_params, use_cache, ttl=cache_ttl
        )
        if cached:
            return cached
        
        # Execute service
        result = await service_func()
        
        # Store result in cache
        CacheHelper.store_result(
            cache, service_name, cache_params, result, use_cache, ttl=store_ttl or cache_ttl
        )
        
        return result


# === Service TTL Defaults ===

DEFAULT_SERVICE_TTLS = {
    "calcular_impuestos": 3600,      # 1 hour
    "asesoria_fiscal": 1800,          # 30 minutes
    "guia_fiscal": 7200,              # 2 hours
    "tramite_sat": 14400,             # 4 hours
    "ayuda_declaracion": 3600,        # 1 hour
    "comparar_regimenes": 3600,       # 1 hour
}


def configure_default_ttls():
    """Configure default TTLs for services."""
    CacheHelper.configure(CacheConfig(
        enabled=True,
        default_ttl=3600,
        service_ttls=DEFAULT_SERVICE_TTLS,
    ))
