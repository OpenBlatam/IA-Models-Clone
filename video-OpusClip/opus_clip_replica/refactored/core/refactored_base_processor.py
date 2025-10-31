"""
Refactored Base Processor for Final Ultimate AI System

Enhanced base processor with:
- Advanced error handling and recovery
- Performance optimization and caching
- Resource management and monitoring
- Async processing and concurrency
- Circuit breaker pattern
- Retry mechanisms
- Health checks and metrics
- Configuration management
- Logging and observability
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic
import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
import numpy as np
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import threading
from collections import defaultdict, deque
import weakref
import gc
import psutil
import tracemalloc

logger = structlog.get_logger("refactored_base_processor")

T = TypeVar('T')

class ProcessorState(Enum):
    """Processor state enumeration."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPING = "stopping"
    STOPPED = "stopped"

class CircuitState(Enum):
    """Circuit breaker state enumeration."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class ProcessorMetrics:
    """Processor metrics data structure."""
    processor_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_processing_time: float = 0.0
    last_processing_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 3
    timeout: int = 30

@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

class CircuitBreaker:
    """Circuit breaker implementation."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self._lock = threading.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return (datetime.now() - self.last_failure_time).total_seconds() > self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN

class RetryManager:
    """Retry manager with exponential backoff."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.config.max_attempts - 1:
                    break
                
                delay = self._calculate_delay(attempt)
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        if self.config.jitter:
            jitter = np.random.uniform(0.1, 0.3) * delay
            delay += jitter
        
        return delay

class ResourceMonitor:
    """Resource monitoring and management."""
    
    def __init__(self):
        self.memory_threshold = 0.8  # 80% memory usage threshold
        self.cpu_threshold = 0.8     # 80% CPU usage threshold
        self.monitoring_interval = 5  # seconds
        self._monitoring = False
        self._monitor_task = None
    
    async def start_monitoring(self):
        """Start resource monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_resources())
    
    async def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_resources(self):
        """Monitor system resources."""
        while self._monitoring:
            try:
                memory_usage = psutil.virtual_memory().percent / 100
                cpu_usage = psutil.cpu_percent() / 100
                
                if memory_usage > self.memory_threshold:
                    logger.warning(f"High memory usage: {memory_usage:.2%}")
                    await self._handle_high_memory()
                
                if cpu_usage > self.cpu_threshold:
                    logger.warning(f"High CPU usage: {cpu_usage:.2%}")
                    await self._handle_high_cpu()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _handle_high_memory(self):
        """Handle high memory usage."""
        # Force garbage collection
        gc.collect()
        
        # Log memory usage
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            logger.info(f"Memory usage - Current: {current / 1024 / 1024:.2f} MB, Peak: {peak / 1024 / 1024:.2f} MB")
    
    async def _handle_high_cpu(self):
        """Handle high CPU usage."""
        # Log CPU usage
        cpu_per_core = psutil.cpu_percent(percpu=True)
        logger.info(f"CPU usage per core: {cpu_per_core}")

class CacheManager:
    """Advanced caching system with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, datetime] = {}
        self._expiry_times: Dict[str, datetime] = {}
        self._lock = threading.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                return None
            
            # Check expiry
            if key in self._expiry_times:
                if datetime.now() > self._expiry_times[key]:
                    await self._remove(key)
                    return None
            
            # Update access time
            self._access_times[key] = datetime.now()
            return self._cache[key]['value']
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        with self._lock:
            # Remove expired entries
            await self._cleanup_expired()
            
            # Check size limit
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_lru()
            
            # Set value
            self._cache[key] = {'value': value, 'created': datetime.now()}
            self._access_times[key] = datetime.now()
            
            if ttl is None:
                ttl = self.default_ttl
            
            self._expiry_times[key] = datetime.now() + timedelta(seconds=ttl)
    
    async def _remove(self, key: str) -> None:
        """Remove key from cache."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._expiry_times.pop(key, None)
    
    async def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        now = datetime.now()
        expired_keys = [
            key for key, expiry in self._expiry_times.items()
            if now > expiry
        ]
        
        for key in expired_keys:
            await self._remove(key)
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        await self._remove(lru_key)
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._expiry_times.clear()

class HealthChecker:
    """Health check system for processors."""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, bool] = {}
    
    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function."""
        self.checks[name] = check_func
    
    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                result = await check_func()
                self.health_status[name] = True
                results[name] = {
                    "status": "healthy",
                    "result": result
                }
            except Exception as e:
                self.health_status[name] = False
                results[name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        return results
    
    def is_healthy(self) -> bool:
        """Check if all health checks pass."""
        return all(self.health_status.values())

class RefactoredBaseProcessor(ABC, Generic[T]):
    """Refactored base processor with advanced features."""
    
    def __init__(self, processor_id: str, config: Optional[Dict[str, Any]] = None):
        self.processor_id = processor_id
        self.config = config or {}
        self.state = ProcessorState.IDLE
        self.metrics = ProcessorMetrics(processor_id=processor_id)
        
        # Initialize components
        self.circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(**self.config.get('circuit_breaker', {}))
        )
        self.retry_manager = RetryManager(
            RetryConfig(**self.config.get('retry', {}))
        )
        self.resource_monitor = ResourceMonitor()
        self.cache_manager = CacheManager(
            max_size=self.config.get('cache_size', 1000),
            default_ttl=self.config.get('cache_ttl', 3600)
        )
        self.health_checker = HealthChecker()
        
        # Processing queue
        self._processing_queue = asyncio.Queue()
        self._processing_task = None
        self._shutdown_event = asyncio.Event()
        
        # Metrics tracking
        self._processing_times = deque(maxlen=1000)
        self._last_metrics_update = datetime.now()
        
        # Register default health checks
        self.health_checker.register_check("memory", self._check_memory)
        self.health_checker.register_check("cpu", self._check_cpu)
        self.health_checker.register_check("queue", self._check_queue)
    
    async def initialize(self) -> bool:
        """Initialize the processor."""
        try:
            # Start resource monitoring
            await self.resource_monitor.start_monitoring()
            
            # Start processing task
            self._processing_task = asyncio.create_task(self._processing_loop())
            
            # Initialize processor-specific components
            await self._initialize_processor()
            
            self.state = ProcessorState.IDLE
            self.logger.info(f"Processor {self.processor_id} initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Processor {self.processor_id} initialization failed: {e}")
            self.state = ProcessorState.ERROR
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the processor gracefully."""
        try:
            self.state = ProcessorState.STOPPING
            self._shutdown_event.set()
            
            # Wait for processing to complete
            if self._processing_task:
                await asyncio.wait_for(self._processing_task, timeout=30)
            
            # Stop resource monitoring
            await self.resource_monitor.stop_monitoring()
            
            # Shutdown processor-specific components
            await self._shutdown_processor()
            
            self.state = ProcessorState.STOPPED
            self.logger.info(f"Processor {self.processor_id} shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Processor {self.processor_id} shutdown error: {e}")
    
    async def process(self, input_data: T) -> Any:
        """Process input data with advanced features."""
        try:
            # Check if processor is healthy
            if not self.health_checker.is_healthy():
                raise Exception("Processor is not healthy")
            
            # Add to processing queue
            await self._processing_queue.put(input_data)
            
            # Wait for processing to complete
            result = await self._wait_for_result(input_data)
            
            # Update metrics
            await self._update_metrics(True, 0)
            
            return result
            
        except Exception as e:
            await self._update_metrics(False, 0)
            self.logger.error(f"Processing failed: {e}")
            raise e
    
    async def _processing_loop(self) -> None:
        """Main processing loop."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for input data
                input_data = await asyncio.wait_for(
                    self._processing_queue.get(),
                    timeout=1.0
                )
                
                # Process with circuit breaker and retry
                start_time = time.time()
                
                result = await self.circuit_breaker.call(
                    self.retry_manager.execute_with_retry,
                    self._process_internal,
                    input_data
                )
                
                processing_time = time.time() - start_time
                await self._update_metrics(True, processing_time)
                
                # Store result for retrieval
                await self._store_result(input_data, result)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                await self._update_metrics(False, 0)
                self.logger.error(f"Processing loop error: {e}")
    
    async def _process_internal(self, input_data: T) -> Any:
        """Internal processing method to be implemented by subclasses."""
        # Check cache first
        cache_key = self._generate_cache_key(input_data)
        cached_result = await self.cache_manager.get(cache_key)
        
        if cached_result is not None:
            return cached_result
        
        # Process data
        result = await self.process_data(input_data)
        
        # Cache result
        await self.cache_manager.set(cache_key, result)
        
        return result
    
    @abstractmethod
    async def process_data(self, input_data: T) -> Any:
        """Process data - to be implemented by subclasses."""
        pass
    
    async def _initialize_processor(self) -> None:
        """Initialize processor-specific components - to be implemented by subclasses."""
        pass
    
    async def _shutdown_processor(self) -> None:
        """Shutdown processor-specific components - to be implemented by subclasses."""
        pass
    
    def _generate_cache_key(self, input_data: T) -> str:
        """Generate cache key for input data."""
        if isinstance(input_data, dict):
            return f"{self.processor_id}:{hash(json.dumps(input_data, sort_keys=True))}"
        else:
            return f"{self.processor_id}:{hash(str(input_data))}"
    
    async def _wait_for_result(self, input_data: T) -> Any:
        """Wait for processing result."""
        # This is a simplified implementation
        # In practice, you'd use a more sophisticated result tracking system
        await asyncio.sleep(0.1)  # Placeholder
        return {"processed": True, "data": input_data}
    
    async def _store_result(self, input_data: T, result: Any) -> None:
        """Store processing result."""
        # This is a simplified implementation
        # In practice, you'd use a more sophisticated result storage system
        pass
    
    async def _update_metrics(self, success: bool, processing_time: float) -> None:
        """Update processor metrics."""
        now = datetime.now()
        
        self.metrics.total_requests += 1
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
        
        self.metrics.last_processing_time = processing_time
        self._processing_times.append(processing_time)
        
        # Update average processing time
        if self._processing_times:
            self.metrics.average_processing_time = sum(self._processing_times) / len(self._processing_times)
        
        # Update error rate
        self.metrics.error_rate = self.metrics.failed_requests / self.metrics.total_requests
        
        # Update throughput (requests per second)
        time_diff = (now - self._last_metrics_update).total_seconds()
        if time_diff > 0:
            self.metrics.throughput = 1 / time_diff
        
        # Update resource usage
        self.metrics.memory_usage = psutil.virtual_memory().percent / 100
        self.metrics.cpu_usage = psutil.cpu_percent() / 100
        
        self.metrics.last_updated = now
        self._last_metrics_update = now
    
    async def _check_memory(self) -> Dict[str, Any]:
        """Check memory health."""
        memory_usage = psutil.virtual_memory().percent / 100
        return {
            "usage": memory_usage,
            "threshold": self.resource_monitor.memory_threshold,
            "healthy": memory_usage < self.resource_monitor.memory_threshold
        }
    
    async def _check_cpu(self) -> Dict[str, Any]:
        """Check CPU health."""
        cpu_usage = psutil.cpu_percent() / 100
        return {
            "usage": cpu_usage,
            "threshold": self.resource_monitor.cpu_threshold,
            "healthy": cpu_usage < self.resource_monitor.cpu_threshold
        }
    
    async def _check_queue(self) -> Dict[str, Any]:
        """Check processing queue health."""
        queue_size = self._processing_queue.qsize()
        max_queue_size = self.config.get('max_queue_size', 1000)
        
        return {
            "size": queue_size,
            "max_size": max_queue_size,
            "healthy": queue_size < max_queue_size
        }
    
    async def get_metrics(self) -> ProcessorMetrics:
        """Get current processor metrics."""
        return self.metrics
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get processor health status."""
        health_checks = await self.health_checker.run_checks()
        
        return {
            "processor_id": self.processor_id,
            "state": self.state.value,
            "healthy": self.health_checker.is_healthy(),
            "checks": health_checks,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "success_rate": self.metrics.successful_requests / max(1, self.metrics.total_requests),
                "average_processing_time": self.metrics.average_processing_time,
                "error_rate": self.metrics.error_rate,
                "throughput": self.metrics.throughput
            }
        }
    
    async def clear_cache(self) -> None:
        """Clear processor cache."""
        await self.cache_manager.clear()
    
    @property
    def logger(self):
        """Get logger instance."""
        return structlog.get_logger(f"processor.{self.processor_id}")

# Example usage
class ExampleRefactoredProcessor(RefactoredBaseProcessor[Dict[str, Any]]):
    """Example refactored processor implementation."""
    
    async def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data."""
        # Simulate processing
        await asyncio.sleep(0.1)
        
        return {
            "processed": True,
            "input": input_data,
            "timestamp": datetime.now().isoformat(),
            "processor_id": self.processor_id
        }
    
    async def _initialize_processor(self) -> None:
        """Initialize processor-specific components."""
        self.logger.info("Example processor initialized")
    
    async def _shutdown_processor(self) -> None:
        """Shutdown processor-specific components."""
        self.logger.info("Example processor shutdown")

# Example usage
async def main():
    """Example usage of refactored base processor."""
    processor = ExampleRefactoredProcessor(
        processor_id="example_processor",
        config={
            "cache_size": 500,
            "cache_ttl": 1800,
            "max_queue_size": 100,
            "circuit_breaker": {
                "failure_threshold": 3,
                "recovery_timeout": 30
            },
            "retry": {
                "max_attempts": 2,
                "base_delay": 0.5
            }
        }
    )
    
    # Initialize processor
    success = await processor.initialize()
    if not success:
        print("Failed to initialize processor")
        return
    
    # Process data
    result = await processor.process({"test": "data"})
    print(f"Processing result: {result}")
    
    # Get health status
    health = await processor.get_health_status()
    print(f"Health status: {health}")
    
    # Get metrics
    metrics = await processor.get_metrics()
    print(f"Metrics: {metrics}")
    
    # Shutdown processor
    await processor.shutdown()

if __name__ == "__main__":
    asyncio.run(main())


