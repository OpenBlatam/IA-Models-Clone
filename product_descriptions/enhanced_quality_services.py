from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import hashlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Generic, TypeVar
from enum import Enum
import structlog
from redis.asyncio import Redis, ConnectionPool
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import QueuePool
from .enhanced_quality_config import config
from .enhanced_quality_schemas import (
                    import gzip
from typing import Any, List, Dict, Optional
import logging
"""
Enhanced Quality Services Module
===============================

Enterprise-grade services with advanced patterns, comprehensive error handling,
monitoring, observability, and production-ready architecture.
"""



    EnhancedProductCreateRequest,
    EnhancedProductResponse,
    ProductStatus,
    Money
)

logger = structlog.get_logger(__name__)

# Generic types for better type safety
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


# =============================================================================
# ERROR HANDLING - Custom exceptions
# =============================================================================

class ServiceError(Exception):
    """Base service error."""
    
    def __init__(self, message: str, code: str = None, details: Dict[str, Any] = None):
        
    """__init__ function."""
self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(ServiceError):
    """Validation error with field-level details."""
    pass


class NotFoundError(ServiceError):
    """Resource not found error."""
    pass


class ConflictError(ServiceError):
    """Resource conflict error."""
    pass


class ExternalServiceError(ServiceError):
    """External service communication error."""
    pass


class RateLimitError(ServiceError):
    """Rate limit exceeded error."""
    pass


# =============================================================================
# RESULT PATTERN - Better error handling
# =============================================================================

@dataclass(frozen=True)
class Result(Generic[T]):
    """Result pattern for better error handling."""
    
    value: Optional[T] = None
    error: Optional[ServiceError] = None
    success: bool = True
    
    @classmethod
    def ok(cls, value: T) -> 'Result[T]':
        """Create successful result."""
        return cls(value=value, success=True)
    
    @classmethod
    def fail(cls, error: ServiceError) -> 'Result[T]':
        """Create failed result."""
        return cls(error=error, success=False)
    
    def unwrap(self) -> T:
        """Get value or raise error."""
        if self.success and self.value is not None:
            return self.value
        raise self.error or ServiceError("Result has no value")
    
    def unwrap_or(self, default: T) -> T:
        """Get value or return default."""
        return self.value if self.success else default


# =============================================================================
# INTERFACES - Clean Architecture
# =============================================================================

class ICache(ABC):
    """Cache service interface with advanced operations."""
    
    @abstractmethod
    async def get(self, key: str) -> Result[Optional[Any]]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> Result[bool]:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> Result[bool]:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    async def get_many(self, keys: List[str]) -> Result[Dict[str, Any]]:
        """Get multiple values from cache."""
        pass
    
    @abstractmethod
    async def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None) -> Result[bool]:
        """Set multiple values in cache."""
        pass
    
    @abstractmethod
    async def invalidate_pattern(self, pattern: str) -> Result[int]:
        """Invalidate keys matching pattern."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Result[Dict[str, Any]]:
        """Check cache health."""
        pass


class IRepository(ABC, Generic[T]):
    """Generic repository interface."""
    
    @abstractmethod
    async def create(self, entity: T) -> Result[T]:
        """Create new entity."""
        pass
    
    @abstractmethod
    async def get_by_id(self, entity_id: str) -> Result[Optional[T]]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    async def update(self, entity_id: str, updates: Dict[str, Any]) -> Result[Optional[T]]:
        """Update entity."""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: str) -> Result[bool]:
        """Delete entity."""
        pass
    
    @abstractmethod
    async def find(self, filters: Dict[str, Any]) -> Result[Tuple[List[T], int]]:
        """Find entities with filters."""
        pass


class IEventPublisher(ABC):
    """Event publisher interface for domain events."""
    
    @abstractmethod
    async def publish(self, event_type: str, data: Dict[str, Any]) -> Result[bool]:
        """Publish domain event."""
        pass


class IMetrics(ABC):
    """Metrics collection interface."""
    
    @abstractmethod
    def increment_counter(self, name: str, tags: Dict[str, str] = None):
        """Increment counter metric."""
        pass
    
    @abstractmethod
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record histogram value."""
        pass
    
    @abstractmethod
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set gauge value."""
        pass


# =============================================================================
# CACHE SERVICE - Production-ready Redis implementation
# =============================================================================

class EnhancedRedisCache(ICache):
    """Production-ready Redis cache with advanced features."""
    
    def __init__(self, redis_url: str, max_connections: int = 50):
        
    """__init__ function."""
self.redis_url = redis_url
        self.max_connections = max_connections
        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[Redis] = None
        
        # Metrics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0,
            'operations': 0
        }
        
        # Circuit breaker state
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout=30,
            expected_exception=Exception
        )
    
    async def initialize(self) -> Result[bool]:
        """Initialize Redis connection with retry logic."""
        try:
            self.pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                retry_on_timeout=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                health_check_interval=30
            )
            
            self.client = Redis(connection_pool=self.pool)
            
            # Test connection
            await self.client.ping()
            
            logger.info("✅ Enhanced Redis cache initialized", 
                       url=self.redis_url,
                       max_connections=self.max_connections)
            
            return Result.ok(True)
            
        except Exception as e:
            logger.error("❌ Redis initialization failed", error=str(e))
            return Result.fail(ExternalServiceError(f"Redis initialization failed: {e}"))
    
    @asyncio.timeout(5.0)  # Timeout decorator
    async def get(self, key: str) -> Result[Optional[Any]]:
        """Get value from cache with circuit breaker."""
        if not self.client:
            return Result.fail(ExternalServiceError("Redis client not initialized"))
        
        try:
            async with self.circuit_breaker:
                self.stats['operations'] += 1
                
                value = await self.client.get(key)
                
                if value:
                    self.stats['hits'] += 1
                    try:
                        deserialized = json.loads(value)
                        logger.debug("Cache hit", key=key)
                        return Result.ok(deserialized)
                    except json.JSONDecodeError:
                        # Return raw value if not JSON
                        return Result.ok(value.decode('utf-8'))
                else:
                    self.stats['misses'] += 1
                    logger.debug("Cache miss", key=key)
                    return Result.ok(None)
                    
        except Exception as e:
            self.stats['errors'] += 1
            logger.error("Cache get error", key=key, error=str(e))
            return Result.fail(ExternalServiceError(f"Cache get failed: {e}"))
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> Result[bool]:
        """Set value in cache with compression for large values."""
        if not self.client:
            return Result.fail(ExternalServiceError("Redis client not initialized"))
        
        try:
            async with self.circuit_breaker:
                self.stats['operations'] += 1
                
                # Serialize value
                if isinstance(value, (dict, list)):
                    serialized_value = json.dumps(value, default=str)
                else:
                    serialized_value = str(value)
                
                # Use compression for large values
                if len(serialized_value) > 1024:  # 1KB threshold
                    serialized_value = gzip.compress(serialized_value.encode())
                    key = f"compressed:{key}"
                
                ttl = ttl or config.redis.default_ttl
                
                if isinstance(serialized_value, bytes):
                    await self.client.setex(key, ttl, serialized_value)
                else:
                    await self.client.setex(key, ttl, serialized_value)
                
                logger.debug("Cache set", key=key, ttl=ttl)
                return Result.ok(True)
                
        except Exception as e:
            self.stats['errors'] += 1
            logger.error("Cache set error", key=key, error=str(e))
            return Result.fail(ExternalServiceError(f"Cache set failed: {e}"))
    
    async def get_many(self, keys: List[str]) -> Result[Dict[str, Any]]:
        """Get multiple values efficiently."""
        if not self.client or not keys:
            return Result.ok({})
        
        try:
            async with self.circuit_breaker:
                self.stats['operations'] += len(keys)
                
                values = await self.client.mget(keys)
                result = {}
                
                for key, value in zip(keys, values):
                    if value:
                        try:
                            result[key] = json.loads(value)
                            self.stats['hits'] += 1
                        except json.JSONDecodeError:
                            result[key] = value.decode('utf-8')
                            self.stats['hits'] += 1
                    else:
                        self.stats['misses'] += 1
                
                return Result.ok(result)
                
        except Exception as e:
            self.stats['errors'] += len(keys)
            logger.error("Cache get_many error", keys=keys, error=str(e))
            return Result.fail(ExternalServiceError(f"Cache get_many failed: {e}"))
    
    async def invalidate_pattern(self, pattern: str) -> Result[int]:
        """Invalidate keys matching pattern."""
        if not self.client:
            return Result.fail(ExternalServiceError("Redis client not initialized"))
        
        try:
            async with self.circuit_breaker:
                keys = []
                async for key in self.client.scan_iter(match=pattern):
                    keys.append(key)
                
                if keys:
                    count = await self.client.delete(*keys)
                    logger.info("Cache invalidation", pattern=pattern, count=count)
                    return Result.ok(count)
                else:
                    return Result.ok(0)
                    
        except Exception as e:
            logger.error("Cache invalidation error", pattern=pattern, error=str(e))
            return Result.fail(ExternalServiceError(f"Cache invalidation failed: {e}"))
    
    async def health_check(self) -> Result[Dict[str, Any]]:
        """Comprehensive health check."""
        if not self.client:
            return Result.fail(ExternalServiceError("Redis client not initialized"))
        
        try:
            start_time = time.time()
            await self.client.ping()
            response_time = (time.time() - start_time) * 1000
            
            # Get Redis info
            info = await self.client.info()
            
            health_data = {
                "status": "healthy",
                "response_time_ms": response_time,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown"),
                "stats": self.stats.copy(),
                "hit_ratio": self.hit_ratio
            }
            
            return Result.ok(health_data)
            
        except Exception as e:
            return Result.fail(ExternalServiceError(f"Redis health check failed: {e}"))
    
    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total = self.stats['hits'] + self.stats['misses']
        return (self.stats['hits'] / total) if total > 0 else 0.0
    
    async def close(self) -> None:
        """Close Redis connections gracefully."""
        if self.client:
            await self.client.close()
        if self.pool:
            await self.pool.disconnect()
        logger.info("🔌 Enhanced Redis cache closed")


# =============================================================================
# CIRCUIT BREAKER PATTERN
# =============================================================================

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for external service calls."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60, expected_exception: type = Exception):
        
    """__init__ function."""
self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
    
    async def __aenter__(self) -> Any:
        """Context manager entry."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise ExternalServiceError("Circuit breaker is OPEN")
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Context manager exit."""
        if exc_type and issubclass(exc_type, self.expected_exception):
            self._record_failure()
        else:
            self._record_success()
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.timeout
        )
    
    def _record_failure(self) -> Any:
        """Record a failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker opened", failure_count=self.failure_count)
    
    def _record_success(self) -> Any:
        """Record a success."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED


# =============================================================================
# DOMAIN SERVICE - Business logic with advanced patterns
# =============================================================================

class EnhancedProductService:
    """Enhanced product service with enterprise patterns."""
    
    def __init__(
        self,
        repository: IRepository,
        cache: ICache,
        event_publisher: IEventPublisher,
        metrics: IMetrics
    ):
        
    """__init__ function."""
self.repository = repository
        self.cache = cache
        self.event_publisher = event_publisher
        self.metrics = metrics
        
        # Business rules
        self.max_concurrent_operations = 50
        self.cache_ttl_default = 3600
        self.cache_ttl_search = 300
    
    async def create_product(self, request: EnhancedProductCreateRequest) -> Result[EnhancedProductResponse]:
        """Create product with comprehensive business logic."""
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info("Product creation started", 
                       operation_id=operation_id,
                       sku=request.sku.value)
            
            # Validation phase
            validation_result = await self._validate_product_creation(request)
            if not validation_result.success:
                self.metrics.increment_counter("product.creation.validation_failed")
                return validation_result
            
            # Business logic phase
            product_data = await self._prepare_product_data(request)
            
            # Persistence phase
            creation_result = await self.repository.create(product_data)
            if not creation_result.success:
                self.metrics.increment_counter("product.creation.persistence_failed")
                return Result.fail(creation_result.error)
            
            created_product = creation_result.value
            
            # Post-creation tasks
            await self._handle_post_creation(created_product, operation_id)
            
            # Metrics and response
            duration = (time.time() - start_time) * 1000
            self.metrics.record_histogram("product.creation.duration", duration)
            self.metrics.increment_counter("product.creation.success")
            
            logger.info("Product created successfully",
                       operation_id=operation_id,
                       product_id=created_product.id,
                       duration_ms=duration)
            
            return Result.ok(created_product)
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.metrics.record_histogram("product.creation.duration", duration, {"status": "error"})
            self.metrics.increment_counter("product.creation.error")
            
            logger.error("Product creation failed",
                        operation_id=operation_id,
                        error=str(e),
                        duration_ms=duration)
            
            return Result.fail(ServiceError(f"Product creation failed: {e}"))
    
    async def _validate_product_creation(self, request: EnhancedProductCreateRequest) -> Result[bool]:
        """Comprehensive product validation."""
        
        # SKU uniqueness check
        existing_product = await self.repository.get_by_sku(request.sku.value)
        if existing_product.success and existing_product.value:
            return Result.fail(ConflictError(
                f"SKU '{request.sku.value}' already exists",
                code="SKU_DUPLICATE",
                details={"sku": request.sku.value}
            ))
        
        # Business rule validations
        if request.pricing.base_price.amount <= 0:
            return Result.fail(ValidationError(
                "Base price must be positive",
                code="INVALID_PRICE",
                details={"price": request.pricing.base_price.amount}
            ))
        
        # Inventory validation for physical products
        if request.product_type.value == "physical" and request.inventory.quantity < 0:
            return Result.fail(ValidationError(
                "Physical products cannot have negative inventory",
                code="INVALID_INVENTORY"
            ))
        
        return Result.ok(True)
    
    async def _prepare_product_data(self, request: EnhancedProductCreateRequest) -> Dict[str, Any]:
        """Prepare product data with calculated fields."""
        now = datetime.utcnow()
        
        # Generate product ID
        product_id = f"prod_{uuid.uuid4().hex[:12]}"
        
        # Prepare base data
        product_data = {
            "id": product_id,
            "created_at": now,
            "updated_at": now,
            "status": ProductStatus.DRAFT,
            **request.dict()
        }
        
        # Calculate derived fields
        product_data.update(self._calculate_pricing_fields(request.pricing))
        product_data.update(self._calculate_inventory_fields(request.inventory))
        product_data.update(self._calculate_seo_fields(request))
        
        return product_data
    
    def _calculate_pricing_fields(self, pricing) -> Dict[str, Any]:
        """Calculate pricing-related fields."""
        return {
            "effective_price": pricing.effective_price.amount,
            "is_on_sale": pricing.is_on_sale,
            "discount_percentage": pricing.discount_percentage,
            "profit_margin": pricing.profit_margin
        }
    
    def _calculate_inventory_fields(self, inventory) -> Dict[str, Any]:
        """Calculate inventory-related fields."""
        return {
            "is_in_stock": inventory.is_in_stock,
            "is_low_stock": inventory.is_low_stock,
            "available_quantity": inventory.available_quantity
        }
    
    def _calculate_seo_fields(self, request) -> Dict[str, Any]:
        """Calculate SEO-related fields."""
        # Auto-generate meta title if not provided
        meta_title = request.seo.meta_title or f"{request.name} - Premium Quality"
        
        # Auto-generate meta description
        meta_description = request.seo.meta_description or request.short_description or request.description[:160]
        
        return {
            "computed_meta_title": meta_title,
            "computed_meta_description": meta_description,
            "search_keywords": " ".join(request.seo.keywords + request.seo.tags)
        }
    
    async def _handle_post_creation(self, product: EnhancedProductResponse, operation_id: str):
        """Handle post-creation tasks asynchronously."""
        
        # Cache the new product
        cache_key = f"product:{product.id}"
        await self.cache.set(cache_key, product.dict(), ttl=self.cache_ttl_default)
        
        # Publish domain event
        await self.event_publisher.publish("product.created", {
            "product_id": product.id,
            "sku": product.sku.value,
            "name": product.name,
            "operation_id": operation_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Invalidate related caches
        await self.cache.invalidate_pattern("search:*")
        await self.cache.invalidate_pattern("category:*")
        
        logger.debug("Post-creation tasks completed", 
                    product_id=product.id,
                    operation_id=operation_id)


# =============================================================================
# SERVICE CONTAINER - Advanced dependency injection
# =============================================================================

class EnhancedServiceContainer:
    """Enhanced service container with lifecycle management."""
    
    def __init__(self) -> Any:
        self._instances = {}
        self._initializing = set()
        self._health_checks = {}
    
    async def get_cache_service(self) -> ICache:
        """Get cache service with lazy initialization."""
        if 'cache' not in self._instances:
            if 'cache' in self._initializing:
                # Wait for ongoing initialization
                while 'cache' in self._initializing:
                    await asyncio.sleep(0.1)
            else:
                self._initializing.add('cache')
                try:
                    cache = EnhancedRedisCache(
                        redis_url=config.redis.url,
                        max_connections=config.redis.max_connections
                    )
                    result = await cache.initialize()
                    if result.success:
                        self._instances['cache'] = cache
                    else:
                        raise result.error
                finally:
                    self._initializing.discard('cache')
        
        return self._instances['cache']
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Comprehensive health check of all services."""
        health_results = {}
        
        for service_name, instance in self._instances.items():
            if hasattr(instance, 'health_check'):
                try:
                    health_result = await instance.health_check()
                    health_results[service_name] = health_result.value if health_result.success else {"status": "unhealthy", "error": str(health_result.error)}
                except Exception as e:
                    health_results[service_name] = {"status": "error", "error": str(e)}
            else:
                health_results[service_name] = {"status": "no_health_check"}
        
        overall_status = "healthy" if all(
            result.get("status") == "healthy" 
            for result in health_results.values()
        ) else "degraded"
        
        return {
            "overall_status": overall_status,
            "services": health_results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def cleanup(self) -> None:
        """Cleanup all services gracefully."""
        for service_name, instance in self._instances.items():
            if hasattr(instance, 'close'):
                try:
                    await instance.close()
                    logger.info(f"Service {service_name} closed successfully")
                except Exception as e:
                    logger.error(f"Error closing service {service_name}: {e}")
        
        self._instances.clear()


# Global enhanced service container
enhanced_services = EnhancedServiceContainer() 