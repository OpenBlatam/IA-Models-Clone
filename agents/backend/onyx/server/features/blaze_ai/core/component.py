import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from .enums import ComponentStatus, PerformanceLevel
from .health import ComponentType
from .settings import ComponentConfig, PerformanceMetrics, CACHE_TTL

class BlazeComponent(ABC):
    """Base abstract class for all Blaze AI components."""
    
    def __init__(self, config: ComponentConfig):
        self.config = config
        self.status = ComponentStatus.INITIALIZING
        self.performance_metrics: List[PerformanceMetrics] = []
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.created_at = time.time()
        self.last_activity = time.time()
        self._lock = asyncio.Lock()
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        
        # Performance monitoring
        self._operation_count = 0
        self._total_duration = 0.0
        self._peak_memory = 0.0
        
        # Initialize logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{config.name}")
    
    async def initialize(self) -> bool:
        """Initialize the component."""
        try:
            async with self._lock:
                if self.status == ComponentStatus.INITIALIZING:
                    self.logger.info(f"Initializing {self.config.name}")
                    success = await self._initialize_impl()
                    if success:
                        self.status = ComponentStatus.ACTIVE
                        self.logger.info(f"{self.config.name} initialized successfully")
                    else:
                        self.status = ComponentStatus.ERROR
                        self.logger.error(f"{self.config.name} initialization failed")
                    return success
                return self.status == ComponentStatus.ACTIVE
        except Exception as e:
            self.status = ComponentStatus.ERROR
            self.last_error = str(e)
            self.logger.error(f"Initialization error: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the component."""
        try:
            async with self._lock:
                if self.status != ComponentStatus.SHUTDOWN:
                    self.logger.info(f"Shutting down {self.config.name}")
                    success = await self._shutdown_impl()
                    self.status = ComponentStatus.SHUTDOWN
                    self.logger.info(f"{self.config.name} shutdown completed")
                    return success
                return True
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check component health."""
        try:
            health_data = {
                "status": self.status.name,
                "error_count": self.error_count,
                "last_error": self.last_error,
                "uptime": time.time() - self.created_at,
                "last_activity": time.time() - self.last_activity,
                "operation_count": self._operation_count,
                "avg_duration": self._total_duration / max(self._operation_count, 1),
                "peak_memory": self._peak_memory,
                "cache_size": len(self._cache),
                "cache_hit_rate": self._get_cache_hit_rate()
            }
            
            # Add component-specific health data
            component_health = await self._health_check_impl()
            health_data.update(component_health)
            
            return health_data
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        return {
            "name": self.config.name,
            "type": self.config.component_type.name,
            "status": self.status.name,
            "performance_level": self.config.performance_level.value,
            "created_at": self.created_at,
            "uptime": time.time() - self.created_at,
            "operation_count": self._operation_count,
            "error_count": self.error_count,
            "cache_stats": self._get_cache_stats()
        }
    
    @abstractmethod
    async def _initialize_impl(self) -> bool:
        """Component-specific initialization."""
        pass
    
    @abstractmethod
    async def _shutdown_impl(self) -> bool:
        """Component-specific shutdown."""
        pass
    
    async def _health_check_impl(self) -> Dict[str, Any]:
        """Component-specific health check."""
        return {}
    
    def _get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if not hasattr(self, '_cache_hits'):
            return 0.0
        total_requests = getattr(self, '_cache_requests', 0)
        return self._cache_hits / max(total_requests, 1)
    
    def _get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "hit_rate": self._get_cache_hit_rate(),
            "ttl": self.config.cache_ttl
        }
    
    async def _execute_with_monitoring(self, operation_name: str, operation_func, *args, **kwargs):
        """Execute operation with performance monitoring."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        try:
            result = await operation_func(*args, **kwargs)
            success = True
            error_message = None
        except Exception as e:
            result = None
            success = False
            error_message = str(e)
            self.error_count += 1
            self.last_error = error_message
            self.logger.error(f"Operation {operation_name} failed: {e}")
            raise
        
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            # Record metrics
            metrics = PerformanceMetrics(
                start_time=start_time,
                end_time=end_time,
                memory_usage=end_memory - start_memory,
                success=success,
                error_message=error_message
            )
            
            self.performance_metrics.append(metrics)
            self._operation_count += 1
            self._total_duration += metrics.duration
            self._peak_memory = max(self._peak_memory, end_memory)
            self.last_activity = time.time()
            
            # Update cache
            if self.config.enable_caching and success:
                await self._update_cache(operation_name, result)
        
        return result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
    
    async def _update_cache(self, key: str, value: Any):
        """Update component cache."""
        if not self.config.enable_caching:
            return
        
        current_time = time.time()
        self._cache[key] = value
        self._cache_timestamps[key] = current_time
        
        # Clean expired cache entries
        expired_keys = [
            k for k, ts in self._cache_timestamps.items()
            if current_time - ts > self.config.cache_ttl
        ]
        
        for expired_key in expired_keys:
            del self._cache[expired_key]
            del self._cache_timestamps[expired_key]
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.config.enable_caching:
            return None
        
        current_time = time.time()
        if key in self._cache:
            timestamp = self._cache_timestamps.get(key, 0)
            if current_time - timestamp <= self.config.cache_ttl:
                if not hasattr(self, '_cache_hits'):
                    self._cache_hits = 0
                self._cache_hits += 1
                return self._cache[key]
            else:
                # Expired, remove
                del self._cache[key]
                del self._cache_timestamps[key]
        
        if not hasattr(self, '_cache_requests'):
            self._cache_requests = 0
        self._cache_requests += 1
        return None
