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
import logging
import functools
import threading
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, TypeVar, Generic
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
import json
import sqlite3
from pathlib import Path
import weakref
import signal
import contextlib
import structlog
from pydantic import BaseModel, Field
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from fastapi.responses import JSONResponse
import redis.asyncio as redis
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import aiofiles
import aiohttp
from typing import Any, List, Dict, Optional
"""
🚀 Blocking Operations Limiter System
=====================================

Comprehensive system to limit blocking operations in routes with:
- Async operation enforcement
- Background task management
- Operation timeouts and cancellation
- Performance monitoring and alerting
- Database query optimization
- File I/O optimization
- External API call management
- Resource pool management
- Circuit breaker patterns
- Rate limiting for blocking operations
"""



logger = structlog.get_logger(__name__)

T = TypeVar('T')
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')

class OperationType(Enum):
    """Types of operations that can be blocking"""
    DATABASE_QUERY = "database_query"
    FILE_IO = "file_io"
    EXTERNAL_API = "external_api"
    COMPUTATION = "computation"
    NETWORK_IO = "network_io"
    MEMORY_OPERATION = "memory_operation"
    DISK_IO = "disk_io"
    CPU_INTENSIVE = "cpu_intensive"

class BlockingLevel(Enum):
    """Levels of blocking operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TimeoutStrategy(Enum):
    """Timeout handling strategies"""
    CANCEL = "cancel"
    BACKGROUND = "background"
    CACHE_FALLBACK = "cache_fallback"
    ERROR_RESPONSE = "error_response"

@dataclass
class OperationConfig:
    """Configuration for operation handling"""
    operation_type: OperationType
    blocking_level: BlockingLevel
    timeout_seconds: float = 30.0
    timeout_strategy: TimeoutStrategy = TimeoutStrategy.CANCEL
    max_retries: int = 3
    retry_delay: float = 1.0
    background_fallback: bool = True
    cache_enabled: bool = False
    cache_ttl: int = 300
    circuit_breaker_enabled: bool = True
    rate_limit_enabled: bool = False
    max_concurrent: int = 10

@dataclass
class OperationMetrics:
    """Metrics for operation performance"""
    operation_id: str
    operation_type: OperationType
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = False
    timeout_occurred: bool = False
    retry_count: int = 0
    error_message: Optional[str] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)

class CircuitBreaker:
    """Circuit breaker pattern for blocking operations"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        
    """__init__ function."""
self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self) -> Any:
        """Handle successful operation"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self) -> Any:
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class RateLimiter:
    """Rate limiter for blocking operations"""
    
    def __init__(self, max_requests: int, time_window: float):
        
    """__init__ function."""
self.max_requests = max_requests
        self.time_window = time_window
        self.requests: deque = deque()
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        """Try to acquire a rate limit token"""
        with self.lock:
            now = time.time()
            
            # Remove expired requests
            while self.requests and self.requests[0] <= now - self.time_window:
                self.requests.popleft()
            
            # Check if we can make a new request
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    def wait_for_token(self, timeout: float = None) -> bool:
        """Wait for a rate limit token"""
        start_time = time.time()
        
        while True:
            if self.acquire():
                return True
            
            if timeout and time.time() - start_time > timeout:
                return False
            
            time.sleep(0.1)

class ResourcePool:
    """Resource pool for managing concurrent operations"""
    
    def __init__(self, max_workers: int = 10, pool_type: str = "thread"):
        
    """__init__ function."""
self.max_workers = max_workers
        self.pool_type = pool_type
        
        if pool_type == "thread":
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        elif pool_type == "process":
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            raise ValueError(f"Unsupported pool type: {pool_type}")
        
        self.active_operations = 0
        self.lock = threading.Lock()
    
    async def submit(self, func: Callable, *args, **kwargs):
        """Submit a function to the resource pool"""
        with self.lock:
            if self.active_operations >= self.max_workers:
                raise Exception("Resource pool is full")
            
            self.active_operations += 1
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, func, *args, **kwargs)
            return result
        finally:
            with self.lock:
                self.active_operations -= 1
    
    def shutdown(self) -> Any:
        """Shutdown the resource pool"""
        self.executor.shutdown(wait=True)

class BlockingOperationLimiter:
    """Main system for limiting blocking operations"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        
    """__init__ function."""
self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        
        # Operation tracking
        self.operation_metrics: Dict[str, OperationMetrics] = {}
        self.operation_metrics_lock = threading.Lock()
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Rate limiters
        self.rate_limiters: Dict[str, RateLimiter] = {}
        
        # Resource pools
        self.resource_pools: Dict[str, ResourcePool] = {}
        
        # Configuration
        self.default_config = OperationConfig(
            operation_type=OperationType.COMPUTATION,
            blocking_level=BlockingLevel.MEDIUM,
            timeout_seconds=30.0,
            timeout_strategy=TimeoutStrategy.CANCEL,
            max_retries=3,
            retry_delay=1.0,
            background_fallback=True,
            cache_enabled=False,
            cache_ttl=300,
            circuit_breaker_enabled=True,
            rate_limit_enabled=False,
            max_concurrent=10
        )
        
        # Performance monitoring
        self.performance_history: deque = deque(maxlen=10000)
        self.start_time = time.time()
        
        logger.info("Blocking Operation Limiter initialized")
    
    async def initialize(self) -> Any:
        """Initialize the limiter"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Redis connection established for blocking operations limiter")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory storage only.")
            self.redis_client = None
    
    def get_circuit_breaker(self, operation_name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for an operation"""
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreaker()
        return self.circuit_breakers[operation_name]
    
    def get_rate_limiter(self, operation_name: str, max_requests: int = 100, time_window: float = 60.0) -> RateLimiter:
        """Get or create a rate limiter for an operation"""
        key = f"{operation_name}_{max_requests}_{time_window}"
        if key not in self.rate_limiters:
            self.rate_limiters[key] = RateLimiter(max_requests, time_window)
        return self.rate_limiters[key]
    
    def get_resource_pool(self, pool_name: str, max_workers: int = 10, pool_type: str = "thread") -> ResourcePool:
        """Get or create a resource pool"""
        if pool_name not in self.resource_pools:
            self.resource_pools[pool_name] = ResourcePool(max_workers, pool_type)
        return self.resource_pools[pool_name]
    
    async def execute_operation(self, 
                               operation_name: str,
                               operation_func: Callable,
                               config: Optional[OperationConfig] = None,
                               *args, **kwargs) -> Any:
        """Execute an operation with blocking operation limits"""
        
        # Use default config if none provided
        if config is None:
            config = self.default_config
        
        # Generate operation ID
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        # Create metrics
        metrics = OperationMetrics(
            operation_id=operation_id,
            operation_type=config.operation_type,
            start_time=time.time()
        )
        
        # Check rate limiting
        if config.rate_limit_enabled:
            rate_limiter = self.get_rate_limiter(operation_name)
            if not rate_limiter.acquire():
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded for operation"
                )
        
        # Check circuit breaker
        if config.circuit_breaker_enabled:
            circuit_breaker = self.get_circuit_breaker(operation_name)
        
        try:
            # Execute operation with timeout
            if config.timeout_strategy == TimeoutStrategy.CANCEL:
                result = await self._execute_with_timeout(
                    operation_func, config.timeout_seconds, *args, **kwargs
                )
            elif config.timeout_strategy == TimeoutStrategy.BACKGROUND:
                result = await self._execute_in_background(
                    operation_func, config.timeout_seconds, *args, **kwargs
                )
            else:
                result = await self._execute_standard(
                    operation_func, *args, **kwargs
                )
            
            # Record success
            metrics.end_time = time.time()
            metrics.duration = metrics.end_time - metrics.start_time
            metrics.success = True
            
            # Store metrics
            self._store_metrics(metrics)
            
            return result
            
        except asyncio.TimeoutError:
            # Handle timeout
            metrics.timeout_occurred = True
            metrics.end_time = time.time()
            metrics.duration = metrics.end_time - metrics.start_time
            metrics.error_message = "Operation timed out"
            
            self._store_metrics(metrics)
            
            if config.background_fallback:
                # Move to background
                asyncio.create_task(self._execute_in_background(
                    operation_func, config.timeout_seconds, *args, **kwargs
                ))
                raise HTTPException(
                    status_code=status.HTTP_202_ACCEPTED,
                    detail="Operation moved to background due to timeout"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_408_REQUEST_TIMEOUT,
                    detail="Operation timed out"
                )
                
        except Exception as e:
            # Handle other errors
            metrics.end_time = time.time()
            metrics.duration = metrics.end_time - metrics.start_time
            metrics.error_message = str(e)
            
            self._store_metrics(metrics)
            
            # Retry logic
            if metrics.retry_count < config.max_retries:
                metrics.retry_count += 1
                await asyncio.sleep(config.retry_delay)
                return await self.execute_operation(
                    operation_name, operation_func, config, *args, **kwargs
                )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Operation failed after {config.max_retries} retries: {str(e)}"
            )
    
    async def _execute_with_timeout(self, func: Callable, timeout: float, *args, **kwargs):
        """Execute function with timeout"""
        if asyncio.iscoroutinefunction(func):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        else:
            # For sync functions, run in executor
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, func, *args, **kwargs),
                timeout=timeout
            )
    
    async def _execute_in_background(self, func: Callable, timeout: float, *args, **kwargs):
        """Execute function in background"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # For sync functions, run in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)
    
    async def _execute_standard(self, func: Callable, *args, **kwargs):
        """Execute function without timeout"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # For sync functions, run in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)
    
    def _store_metrics(self, metrics: OperationMetrics):
        """Store operation metrics"""
        with self.operation_metrics_lock:
            self.operation_metrics[metrics.operation_id] = metrics
            self.performance_history.append({
                "timestamp": time.time(),
                "operation_id": metrics.operation_id,
                "operation_type": metrics.operation_type.value,
                "duration": metrics.duration,
                "success": metrics.success,
                "timeout_occurred": metrics.timeout_occurred
            })
    
    def get_operation_metrics(self, operation_name: str = None) -> Dict[str, Any]:
        """Get operation metrics"""
        with self.operation_metrics_lock:
            if operation_name:
                filtered_metrics = {
                    k: v for k, v in self.operation_metrics.items()
                    if operation_name in k
                }
            else:
                filtered_metrics = self.operation_metrics.copy()
        
        # Calculate statistics
        durations = [m.duration for m in filtered_metrics.values() if m.duration is not None]
        success_count = sum(1 for m in filtered_metrics.values() if m.success)
        timeout_count = sum(1 for m in filtered_metrics.values() if m.timeout_occurred)
        
        return {
            "total_operations": len(filtered_metrics),
            "successful_operations": success_count,
            "failed_operations": len(filtered_metrics) - success_count,
            "timeout_operations": timeout_count,
            "average_duration": np.mean(durations) if durations else 0.0,
            "min_duration": np.min(durations) if durations else 0.0,
            "max_duration": np.max(durations) if durations else 0.0,
            "p95_duration": np.percentile(durations, 95) if durations else 0.0,
            "success_rate": success_count / len(filtered_metrics) if filtered_metrics else 0.0
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        all_metrics = self.get_operation_metrics()
        
        # Group by operation type
        operation_types = defaultdict(list)
        for metrics in self.operation_metrics.values():
            operation_types[metrics.operation_type.value].append(metrics)
        
        type_summary = {}
        for op_type, metrics_list in operation_types.items():
            durations = [m.duration for m in metrics_list if m.duration is not None]
            success_count = sum(1 for m in metrics_list if m.success)
            
            type_summary[op_type] = {
                "total_operations": len(metrics_list),
                "successful_operations": success_count,
                "success_rate": success_count / len(metrics_list) if metrics_list else 0.0,
                "average_duration": np.mean(durations) if durations else 0.0,
                "p95_duration": np.percentile(durations, 95) if durations else 0.0
            }
        
        return {
            "overall": all_metrics,
            "by_operation_type": type_summary,
            "circuit_breakers": {
                name: {
                    "state": cb.state,
                    "failure_count": cb.failure_count
                }
                for name, cb in self.circuit_breakers.items()
            },
            "resource_pools": {
                name: {
                    "active_operations": pool.active_operations,
                    "max_workers": pool.max_workers
                }
                for name, pool in self.resource_pools.items()
            },
            "uptime_seconds": time.time() - self.start_time
        }

# Decorators for limiting blocking operations

def limit_blocking_operations(operation_name: str = None, config: Optional[OperationConfig] = None):
    """Decorator to limit blocking operations in routes"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Get limiter instance
            limiter = await get_blocking_limiter()
            
            # Determine operation name
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # Execute with limits
            return await limiter.execute_operation(op_name, func, config, *args, **kwargs)
        
        return wrapper
    return decorator

def async_operation(operation_type: OperationType = OperationType.COMPUTATION, 
                   timeout_seconds: float = 30.0):
    """Decorator to ensure operations are async"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            if not asyncio.iscoroutinefunction(func):
                # Convert sync function to async
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, *args, **kwargs)
            else:
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def background_task(operation_name: str = None):
    """Decorator to move operations to background tasks"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(background_tasks: BackgroundTasks, *args, **kwargs):
            
    """wrapper function."""
# Add to background tasks
            background_tasks.add_task(func, *args, **kwargs)
            
            return {
                "message": "Operation queued for background processing",
                "operation_name": operation_name or func.__name__,
                "status": "queued"
            }
        
        return wrapper
    return decorator

def timeout_operation(timeout_seconds: float = 30.0, 
                     strategy: TimeoutStrategy = TimeoutStrategy.CANCEL):
    """Decorator to add timeout to operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                if asyncio.iscoroutinefunction(func):
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
                else:
                    loop = asyncio.get_event_loop()
                    return await asyncio.wait_for(
                        loop.run_in_executor(None, func, *args, **kwargs),
                        timeout=timeout_seconds
                    )
            except asyncio.TimeoutError:
                if strategy == TimeoutStrategy.ERROR_RESPONSE:
                    raise HTTPException(
                        status_code=status.HTTP_408_REQUEST_TIMEOUT,
                        detail=f"Operation timed out after {timeout_seconds} seconds"
                    )
                elif strategy == TimeoutStrategy.BACKGROUND:
                    # Move to background
                    asyncio.create_task(func(*args, **kwargs))
                    raise HTTPException(
                        status_code=status.HTTP_202_ACCEPTED,
                        detail="Operation moved to background due to timeout"
                    )
                else:
                    raise
        
        return wrapper
    return decorator

# Global limiter instance
_limiter: Optional[BlockingOperationLimiter] = None

async def get_blocking_limiter() -> BlockingOperationLimiter:
    """Get the global blocking operation limiter instance"""
    global _limiter
    if _limiter is None:
        _limiter = BlockingOperationLimiter()
        await _limiter.initialize()
    return _limiter

# Example usage functions

async def example_database_operation(user_id: str) -> Dict[str, Any]:
    """Example database operation that could be blocking"""
    # Simulate database query
    await asyncio.sleep(2.0)
    return {"user_id": user_id, "data": "user_data"}

async def example_file_operation(file_path: str) -> str:
    """Example file operation that could be blocking"""
    # Simulate file I/O
    await asyncio.sleep(1.0)
    return f"File content from {file_path}"

async async def example_external_api_call(url: str) -> Dict[str, Any]:
    """Example external API call that could be blocking"""
    # Simulate external API call
    await asyncio.sleep(3.0)
    return {"url": url, "response": "api_response"}

async def example_computation_operation(data: List[int]) -> int:
    """Example computation operation that could be blocking"""
    # Simulate CPU-intensive computation
    await asyncio.sleep(5.0)
    return sum(data)

async def example_usage():
    """Example usage of the blocking operations limiter"""
    
    # Get limiter
    limiter = await get_blocking_limiter()
    
    # Configure different operation types
    db_config = OperationConfig(
        operation_type=OperationType.DATABASE_QUERY,
        blocking_level=BlockingLevel.HIGH,
        timeout_seconds=10.0,
        timeout_strategy=TimeoutStrategy.BACKGROUND,
        max_retries=3
    )
    
    file_config = OperationConfig(
        operation_type=OperationType.FILE_IO,
        blocking_level=BlockingLevel.MEDIUM,
        timeout_seconds=5.0,
        timeout_strategy=TimeoutStrategy.CANCEL,
        max_retries=2
    )
    
    api_config = OperationConfig(
        operation_type=OperationType.EXTERNAL_API,
        blocking_level=BlockingLevel.CRITICAL,
        timeout_seconds=15.0,
        timeout_strategy=TimeoutStrategy.CACHE_FALLBACK,
        max_retries=5,
        circuit_breaker_enabled=True
    )
    
    # Execute operations with limits
    try:
        # Database operation
        db_result = await limiter.execute_operation(
            "get_user_data",
            example_database_operation,
            db_config,
            "user123"
        )
        print(f"Database result: {db_result}")
        
        # File operation
        file_result = await limiter.execute_operation(
            "read_file",
            example_file_operation,
            file_config,
            "/path/to/file.txt"
        )
        print(f"File result: {file_result}")
        
        # External API call
        api_result = await limiter.execute_operation(
            "external_api_call",
            example_external_api_call,
            api_config,
            "https://api.example.com/data"
        )
        print(f"API result: {api_result}")
        
    except HTTPException as e:
        print(f"Operation failed: {e.detail}")
    
    # Get performance summary
    summary = limiter.get_performance_summary()
    print(f"Performance Summary: {summary}")

match __name__:
    case "__main__":
    asyncio.run(example_usage()) 