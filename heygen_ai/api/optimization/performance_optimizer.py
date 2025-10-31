from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import functools
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, TypeVar, Generic
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import structlog
from fastapi import Request, Response, Depends
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy import text, select, update, delete
import redis.asyncio as redis
from pydantic import BaseModel
import json
import hashlib
import pickle
from dataclasses import dataclass, asdict
from enum import Enum
import weakref
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Performance Optimizer for HeyGen AI API
Comprehensive performance optimization with caching, database optimization, and monitoring.
"""


logger = structlog.get_logger()

# =============================================================================
# Performance Types
# =============================================================================

class CacheStrategy(Enum):
    """Cache strategy enumeration."""
    NONE = "none"
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"

class OptimizationLevel(Enum):
    """Optimization level enumeration."""
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    operation: str
    duration_ms: float
    cache_hits: int = 0
    cache_misses: int = 0
    database_queries: int = 0
    database_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self) -> Any:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

# =============================================================================
# Cache Management
# =============================================================================

class CacheManager:
    """Comprehensive cache management system."""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        memory_cache_size: int = 1000,
        default_ttl: int = 300,
        compression_enabled: bool = True
    ):
        
    """__init__ function."""
self.redis_url = redis_url
        self.memory_cache_size = memory_cache_size
        self.default_ttl = default_ttl
        self.compression_enabled = compression_enabled
        
        # Memory cache (LRU)
        self.memory_cache: Dict[str, Any] = {}
        self.cache_order: List[str] = []
        
        # Redis connection
        self.redis_client: Optional[redis.Redis] = None
        if redis_url:
            self.redis_client = redis.from_url(redis_url)
        
        # Cache statistics
        self.stats = {
            "memory_hits": 0,
            "memory_misses": 0,
            "redis_hits": 0,
            "redis_misses": 0,
            "evictions": 0
        }
    
    async def get(self, key: str, strategy: CacheStrategy = CacheStrategy.HYBRID) -> Optional[Any]:
        """Get value from cache."""
        if strategy == CacheStrategy.NONE:
            return None
        
        # Try memory cache first
        if strategy in [CacheStrategy.MEMORY, CacheStrategy.HYBRID]:
            value = self._get_from_memory(key)
            if value is not None:
                self.stats["memory_hits"] += 1
                return value
            else:
                self.stats["memory_misses"] += 1
        
        # Try Redis cache
        if strategy in [CacheStrategy.REDIS, CacheStrategy.HYBRID] and self.redis_client:
            try:
                value = await self._get_from_redis(key)
                if value is not None:
                    self.stats["redis_hits"] += 1
                    # Store in memory cache for future access
                    if strategy == CacheStrategy.HYBRID:
                        self._set_in_memory(key, value)
                    return value
                else:
                    self.stats["redis_misses"] += 1
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        strategy: CacheStrategy = CacheStrategy.HYBRID
    ):
        """Set value in cache."""
        if strategy == CacheStrategy.NONE:
            return
        
        ttl = ttl or self.default_ttl
        
        # Set in memory cache
        if strategy in [CacheStrategy.MEMORY, CacheStrategy.HYBRID]:
            self._set_in_memory(key, value, ttl)
        
        # Set in Redis cache
        if strategy in [CacheStrategy.REDIS, CacheStrategy.HYBRID] and self.redis_client:
            try:
                await self._set_in_redis(key, value, ttl)
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
    
    async def delete(self, key: str, strategy: CacheStrategy = CacheStrategy.HYBRID):
        """Delete value from cache."""
        if strategy == CacheStrategy.NONE:
            return
        
        # Delete from memory cache
        if strategy in [CacheStrategy.MEMORY, CacheStrategy.HYBRID]:
            self._delete_from_memory(key)
        
        # Delete from Redis cache
        if strategy in [CacheStrategy.REDIS, CacheStrategy.HYBRID] and self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
    
    async def clear(self, strategy: CacheStrategy = CacheStrategy.HYBRID):
        """Clear all cache."""
        if strategy == CacheStrategy.NONE:
            return
        
        # Clear memory cache
        if strategy in [CacheStrategy.MEMORY, CacheStrategy.HYBRID]:
            self.memory_cache.clear()
            self.cache_order.clear()
        
        # Clear Redis cache
        if strategy in [CacheStrategy.REDIS, CacheStrategy.HYBRID] and self.redis_client:
            try:
                await self.redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
    
    def _get_from_memory(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        if key in self.memory_cache:
            # Move to end (most recently used)
            self.cache_order.remove(key)
            self.cache_order.append(key)
            return self.memory_cache[key]
        return None
    
    def _set_in_memory(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in memory cache."""
        # Check if cache is full
        if len(self.memory_cache) >= self.memory_cache_size:
            # Remove least recently used item
            lru_key = self.cache_order.pop(0)
            del self.memory_cache[lru_key]
            self.stats["evictions"] += 1
        
        # Add new item
        self.memory_cache[key] = value
        self.cache_order.append(key)
    
    def _delete_from_memory(self, key: str):
        """Delete value from memory cache."""
        if key in self.memory_cache:
            del self.memory_cache[key]
            self.cache_order.remove(key)
    
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self.redis_client:
            return None
        
        data = await self.redis_client.get(key)
        if data:
            return self._deserialize(data)
        return None
    
    async def _set_in_redis(self, key: str, value: Any, ttl: int):
        """Set value in Redis cache."""
        if not self.redis_client:
            return
        
        data = self._serialize(value)
        await self.redis_client.setex(key, ttl, data)
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        if self.compression_enabled:
            return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            return json.dumps(value, default=str).encode()
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            if self.compression_enabled:
                return pickle.loads(data)
            else:
                return json.loads(data.decode())
        except Exception as e:
            logger.warning(f"Deserialization error: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            **self.stats,
            "memory_cache_size": len(self.memory_cache),
            "memory_cache_capacity": self.memory_cache_size,
            "memory_hit_rate": (
                self.stats["memory_hits"] / (self.stats["memory_hits"] + self.stats["memory_misses"])
                if (self.stats["memory_hits"] + self.stats["memory_misses"]) > 0 else 0
            ),
            "redis_hit_rate": (
                self.stats["redis_hits"] / (self.stats["redis_hits"] + self.stats["redis_misses"])
                if (self.stats["redis_hits"] + self.stats["redis_misses"]) > 0 else 0
            )
        }

# =============================================================================
# Database Optimization
# =============================================================================

class DatabaseOptimizer:
    """Database optimization and connection pooling."""
    
    def __init__(
        self,
        database_url: str,
        pool_size: int = 20,
        max_overflow: int = 30,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False
    ):
        
    """__init__ function."""
self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.echo = echo
        
        # Create engine with optimized settings
        self.engine = create_async_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            echo=echo,
            # Additional optimizations
            pool_pre_ping=True,
            pool_reset_on_return='commit',
            # Query optimization
            connect_args={
                "server_settings": {
                    "jit": "off",  # Disable JIT for better performance
                    "statement_timeout": "30000",  # 30 seconds
                    "lock_timeout": "10000",  # 10 seconds
                }
            }
        )
        
        # Session factory
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Query statistics
        self.stats = {
            "total_queries": 0,
            "slow_queries": 0,
            "total_time_ms": 0.0,
            "average_time_ms": 0.0
        }
    
    @asynccontextmanager
    async def get_session(self) -> Optional[Dict[str, Any]]:
        """Get database session with automatic cleanup."""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Execute optimized query."""
        start_time = time.time()
        
        try:
            async with self.get_session() as session:
                # Set query timeout
                if timeout:
                    await session.execute(text(f"SET statement_timeout = {timeout * 1000}"))
                
                # Execute query
                result = await session.execute(text(query), params or {})
                
                # Convert to list of dictionaries
                rows = []
                for row in result:
                    rows.append(dict(row._mapping))
                
                # Update statistics
                duration_ms = (time.time() - start_time) * 1000
                self._update_stats(duration_ms)
                
                return rows
                
        except Exception as e:
            logger.error(f"Database query error: {e}")
            raise
    
    async def execute_batch(
        self,
        queries: List[str],
        params: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 1000
    ) -> List[List[Dict[str, Any]]]:
        """Execute batch of queries efficiently."""
        results = []
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_params = params[i:i + batch_size] if params else [{}] * len(batch_queries)
            
            # Execute batch in parallel
            tasks = [
                self.execute_query(query, param)
                for query, param in zip(batch_queries, batch_params)
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(batch_results)
        
        return results
    
    def _update_stats(self, duration_ms: float):
        """Update query statistics."""
        self.stats["total_queries"] += 1
        self.stats["total_time_ms"] += duration_ms
        self.stats["average_time_ms"] = self.stats["total_time_ms"] / self.stats["total_queries"]
        
        if duration_ms > 1000:  # Slow query threshold
            self.stats["slow_queries"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            **self.stats,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "active_connections": self.engine.pool.size(),
            "checked_in_connections": self.engine.pool.checkedin(),
            "checked_out_connections": self.engine.pool.checkedout(),
            "overflow_connections": self.engine.pool.overflow()
        }
    
    async def optimize_tables(self) -> Any:
        """Optimize database tables."""
        async with self.get_session() as session:
            # Analyze tables for better query planning
            await session.execute(text("ANALYZE"))
            await session.commit()
    
    async def vacuum_database(self) -> Any:
        """Vacuum database to reclaim space and update statistics."""
        async with self.get_session() as session:
            await session.execute(text("VACUUM ANALYZE"))
            await session.commit()

# =============================================================================
# Query Optimization
# =============================================================================

class QueryOptimizer:
    """Query optimization and caching."""
    
    def __init__(self, cache_manager: CacheManager):
        
    """__init__ function."""
self.cache_manager = cache_manager
        self.query_cache: Dict[str, Any] = {}
        self.query_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "optimized_queries": 0
        }
    
    def optimize_query(self, query: str) -> str:
        """Optimize SQL query for better performance."""
        # Remove unnecessary whitespace
        query = " ".join(query.split())
        
        # Add query hints for better performance
        if "SELECT" in query.upper() and "LIMIT" not in query.upper():
            # Add LIMIT for large result sets
            if "ORDER BY" in query.upper():
                query += " LIMIT 1000"
        
        # Add index hints for common patterns
        if "WHERE" in query.upper() and "user_id" in query:
            query = query.replace("WHERE", "WHERE /*+ INDEX(users idx_user_id) */")
        
        return query
    
    def generate_cache_key(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for query."""
        key_data = {
            "query": query,
            "params": params or {}
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def execute_cached_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        ttl: int = 300,
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """Execute query with caching."""
        cache_key = self.generate_cache_key(query, params)
        
        # Try to get from cache
        if not force_refresh:
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result is not None:
                self.query_stats["cache_hits"] += 1
                return cached_result
        
        self.query_stats["cache_misses"] += 1
        
        # Execute query
        optimized_query = self.optimize_query(query)
        result = await self.execute_query(optimized_query, params)
        
        # Cache result
        await self.cache_manager.set(cache_key, result, ttl)
        
        return result
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute query (to be implemented with database connection)."""
        # This would be implemented with actual database connection
        # For now, return empty result
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get query optimization statistics."""
        return {
            **self.query_stats,
            "cache_hit_rate": (
                self.query_stats["cache_hits"] / (self.query_stats["cache_hits"] + self.query_stats["cache_misses"])
                if (self.query_stats["cache_hits"] + self.query_stats["cache_misses"]) > 0 else 0
            )
        }

# =============================================================================
# Performance Monitoring
# =============================================================================

class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self, max_metrics: int = 10000):
        
    """__init__ function."""
self.max_metrics = max_metrics
        self.metrics: List[PerformanceMetrics] = []
        self.slow_query_threshold_ms = 1000
        self.memory_threshold_mb = 100
    
    def record_operation(
        self,
        operation: str,
        duration_ms: float,
        cache_hits: int = 0,
        cache_misses: int = 0,
        database_queries: int = 0,
        database_time_ms: float = 0.0,
        memory_usage_mb: float = 0.0
    ):
        """Record performance metrics."""
        metric = PerformanceMetrics(
            operation=operation,
            duration_ms=duration_ms,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            database_queries=database_queries,
            database_time_ms=database_time_ms,
            memory_usage_mb=memory_usage_mb
        )
        
        self.metrics.append(metric)
        
        # Trim metrics if too many
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]
        
        # Log slow operations
        if duration_ms > self.slow_query_threshold_ms:
            logger.warning(
                "Slow operation detected",
                operation=operation,
                duration_ms=duration_ms,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                database_queries=database_queries
            )
        
        # Log high memory usage
        if memory_usage_mb > self.memory_threshold_mb:
            logger.warning(
                "High memory usage detected",
                operation=operation,
                memory_usage_mb=memory_usage_mb
            )
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for specific operation."""
        operation_metrics = [m for m in self.metrics if m.operation == operation]
        
        if not operation_metrics:
            return {}
        
        durations = [m.duration_ms for m in operation_metrics]
        cache_hits = sum(m.cache_hits for m in operation_metrics)
        cache_misses = sum(m.cache_misses for m in operation_metrics)
        database_queries = sum(m.database_queries for m in operation_metrics)
        
        return {
            "operation": operation,
            "count": len(operation_metrics),
            "average_duration_ms": sum(durations) / len(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "total_cache_hits": cache_hits,
            "total_cache_misses": cache_misses,
            "cache_hit_rate": cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0,
            "total_database_queries": database_queries,
            "slow_operations": len([d for d in durations if d > self.slow_query_threshold_ms])
        }
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics."""
        if not self.metrics:
            return {}
        
        durations = [m.duration_ms for m in self.metrics]
        cache_hits = sum(m.cache_hits for m in self.metrics)
        cache_misses = sum(m.cache_misses for m in self.metrics)
        database_queries = sum(m.database_queries for m in self.metrics)
        
        return {
            "total_operations": len(self.metrics),
            "average_duration_ms": sum(durations) / len(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "total_cache_hits": cache_hits,
            "total_cache_misses": cache_misses,
            "overall_cache_hit_rate": cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0,
            "total_database_queries": database_queries,
            "slow_operations": len([d for d in durations if d > self.slow_query_threshold_ms]),
            "operations_by_type": self._get_operations_by_type()
        }
    
    def _get_operations_by_type(self) -> Dict[str, int]:
        """Get count of operations by type."""
        operation_counts = {}
        for metric in self.metrics:
            operation_counts[metric.operation] = operation_counts.get(metric.operation, 0) + 1
        return operation_counts
    
    def clear_metrics(self) -> Any:
        """Clear all metrics."""
        self.metrics.clear()

# =============================================================================
# Performance Decorators
# =============================================================================

def cache_result(
    ttl: int = 300,
    strategy: CacheStrategy = CacheStrategy.HYBRID,
    key_generator: Optional[Callable] = None
):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                # Default key generation
                key_data = {
                    "func": func.__name__,
                    "args": str(args),
                    "kwargs": str(sorted(kwargs.items()))
                }
                cache_key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
            
            # Try to get from cache
            cache_manager = getattr(wrapper, '_cache_manager', None)
            if cache_manager:
                cached_result = await cache_manager.get(cache_key, strategy)
                if cached_result is not None:
                    return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            if cache_manager:
                await cache_manager.set(cache_key, result, ttl, strategy)
            
            return result
        
        return wrapper
    return decorator

def monitor_performance(operation_name: Optional[str] = None):
    """Decorator to monitor function performance."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record metrics
                duration_ms = (time.time() - start_time) * 1000
                monitor = getattr(wrapper, '_monitor', None)
                if monitor:
                    monitor.record_operation(
                        operation=operation_name or func.__name__,
                        duration_ms=duration_ms
                    )
                
                return result
                
            except Exception as e:
                # Record error metrics
                duration_ms = (time.time() - start_time) * 1000
                monitor = getattr(wrapper, '_monitor', None)
                if monitor:
                    monitor.record_operation(
                        operation=f"{operation_name or func.__name__}_error",
                        duration_ms=duration_ms
                    )
                raise
        
        return wrapper
    return decorator

def optimize_database_query(timeout: Optional[int] = None):
    """Decorator to optimize database queries."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Add query optimization logic here
            # This would integrate with the QueryOptimizer
            
            result = await func(*args, **kwargs)
            return result
        
        return wrapper
    return decorator

# =============================================================================
# Performance Optimizer
# =============================================================================

class PerformanceOptimizer:
    """Main performance optimizer class."""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        database_url: Optional[str] = None,
        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
        memory_cache_size: int = 1000,
        database_pool_size: int = 20,
        max_metrics: int = 10000
    ):
        
    """__init__ function."""
self.optimization_level = optimization_level
        
        # Initialize components
        self.cache_manager = CacheManager(
            redis_url=redis_url,
            memory_cache_size=memory_cache_size
        )
        
        self.database_optimizer = None
        if database_url:
            self.database_optimizer = DatabaseOptimizer(
                database_url=database_url,
                pool_size=database_pool_size
            )
        
        self.query_optimizer = QueryOptimizer(self.cache_manager)
        self.performance_monitor = PerformanceMonitor(max_metrics=max_metrics)
        
        # Configure optimization level
        self._configure_optimization_level()
    
    def _configure_optimization_level(self) -> Any:
        """Configure optimization based on level."""
        if self.optimization_level == OptimizationLevel.BASIC:
            # Basic optimizations
            self.cache_manager.default_ttl = 60
            self.performance_monitor.slow_query_threshold_ms = 2000
        
        elif self.optimization_level == OptimizationLevel.STANDARD:
            # Standard optimizations
            self.cache_manager.default_ttl = 300
            self.performance_monitor.slow_query_threshold_ms = 1000
        
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            # Aggressive optimizations
            self.cache_manager.default_ttl = 600
            self.performance_monitor.slow_query_threshold_ms = 500
            self.cache_manager.memory_cache_size *= 2
        
        elif self.optimization_level == OptimizationLevel.CUSTOM:
            # Custom optimizations (configure manually)
            pass
    
    async def optimize_system(self) -> Any:
        """Run system-wide optimizations."""
        logger.info("Starting system optimization")
        
        # Optimize database
        if self.database_optimizer:
            await self.database_optimizer.optimize_tables()
            await self.database_optimizer.vacuum_database()
        
        # Clear old cache entries
        await self.cache_manager.clear()
        
        # Clear old metrics
        self.performance_monitor.clear_metrics()
        
        logger.info("System optimization completed")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "optimization_level": self.optimization_level.value,
            "cache_stats": self.cache_manager.get_stats(),
            "database_stats": self.database_optimizer.get_stats() if self.database_optimizer else {},
            "query_stats": self.query_optimizer.get_stats(),
            "performance_stats": self.performance_monitor.get_overall_stats()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of optimization components."""
        health_status = {
            "cache_manager": "healthy",
            "database_optimizer": "healthy" if self.database_optimizer else "not_configured",
            "query_optimizer": "healthy",
            "performance_monitor": "healthy"
        }
        
        # Check cache manager
        try:
            await self.cache_manager.set("health_check", "test", 60)
            test_value = await self.cache_manager.get("health_check")
            if test_value != "test":
                health_status["cache_manager"] = "unhealthy"
        except Exception as e:
            health_status["cache_manager"] = f"error: {str(e)}"
        
        # Check database optimizer
        if self.database_optimizer:
            try:
                async with self.database_optimizer.get_session() as session:
                    await session.execute(text("SELECT 1"))
            except Exception as e:
                health_status["database_optimizer"] = f"error: {str(e)}"
        
        return health_status

# =============================================================================
# FastAPI Integration
# =============================================================================

def get_performance_optimizer() -> PerformanceOptimizer:
    """Dependency to get performance optimizer instance."""
    # This would be configured in your FastAPI app
    return PerformanceOptimizer()

def optimize_endpoint(
    cache_ttl: int = 300,
    cache_strategy: CacheStrategy = CacheStrategy.HYBRID,
    monitor: bool = True
):
    """Decorator to optimize FastAPI endpoints."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(
            request: Request,
            *args,
            optimizer: PerformanceOptimizer = Depends(get_performance_optimizer),
            **kwargs
        ):
            start_time = time.time()
            
            # Generate cache key based on request
            cache_key = f"{request.url.path}:{request.query_params}"
            
            # Try to get from cache
            if cache_strategy != CacheStrategy.NONE:
                cached_result = await optimizer.cache_manager.get(cache_key, cache_strategy)
                if cached_result is not None:
                    return cached_result
            
            # Execute endpoint
            result = await func(request, *args, optimizer=optimizer, **kwargs)
            
            # Cache result
            if cache_strategy != CacheStrategy.NONE:
                await optimizer.cache_manager.set(cache_key, result, cache_ttl, cache_strategy)
            
            # Monitor performance
            if monitor:
                duration_ms = (time.time() - start_time) * 1000
                optimizer.performance_monitor.record_operation(
                    operation=f"endpoint_{func.__name__}",
                    duration_ms=duration_ms
                )
            
            return result
        
        return wrapper
    return decorator

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "CacheStrategy",
    "OptimizationLevel",
    "PerformanceMetrics",
    "CacheManager",
    "DatabaseOptimizer",
    "QueryOptimizer",
    "PerformanceMonitor",
    "PerformanceOptimizer",
    "cache_result",
    "monitor_performance",
    "optimize_database_query",
    "optimize_endpoint",
    "get_performance_optimizer",
] 