"""
Circuit Breaker Pattern Implementation
Provides resilient service communication with automatic failure handling
"""

import asyncio
import time
import logging
from typing import Any, Callable, Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
from functools import wraps
import statistics

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is back

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5          # Number of failures before opening
    recovery_timeout: float = 60.0      # Time to wait before trying again
    success_threshold: int = 3          # Successes needed to close circuit
    timeout: float = 30.0               # Request timeout
    max_retries: int = 3                # Maximum retry attempts
    retry_delay: float = 1.0            # Delay between retries
    exponential_backoff: bool = True    # Use exponential backoff
    max_backoff: float = 60.0           # Maximum backoff time

@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_opened_count: int = 0
    circuit_closed_count: int = 0
    average_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None

class CircuitBreakerError(Exception):
    """Circuit breaker specific error"""
    pass

class CircuitOpenError(CircuitBreakerError):
    """Circuit is open error"""
    pass

class CircuitTimeoutError(CircuitBreakerError):
    """Circuit timeout error"""
    pass

class CircuitBreaker:
    """
    Circuit Breaker implementation for resilient service communication
    
    Features:
    - Automatic failure detection
    - Exponential backoff retry
    - Metrics collection
    - Configurable thresholds
    - Async/await support
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: When circuit is open
            CircuitTimeoutError: When request times out
        """
        async with self._lock:
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker {self.name} moved to HALF_OPEN state")
                else:
                    raise CircuitOpenError(f"Circuit breaker {self.name} is OPEN")
            
            # Execute with retry logic
            return await self._execute_with_retry(func, *args, **kwargs)
    
    async def _execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()
                
                # Execute function with timeout
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.config.timeout
                    )
                else:
                    # For sync functions, run in thread pool
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, func, *args, **kwargs),
                        timeout=self.config.timeout
                    )
                
                # Record success
                response_time = time.time() - start_time
                await self._record_success(response_time)
                
                return result
                
            except asyncio.TimeoutError:
                last_exception = CircuitTimeoutError(f"Request timeout after {self.config.timeout}s")
                await self._record_failure()
                
            except Exception as e:
                last_exception = e
                await self._record_failure()
            
            # If not the last attempt, wait before retrying
            if attempt < self.config.max_retries:
                delay = self._calculate_retry_delay(attempt)
                logger.warning(f"Attempt {attempt + 1} failed for {self.name}, retrying in {delay}s")
                await asyncio.sleep(delay)
        
        # All retries failed
        raise last_exception
    
    async def _record_success(self, response_time: float):
        """Record successful request"""
        self.success_count += 1
        self.failure_count = 0
        self.metrics.successful_requests += 1
        self.metrics.last_success_time = time.time()
        
        # Update response time metrics
        self.metrics.response_times.append(response_time)
        if len(self.metrics.response_times) > 100:  # Keep only last 100
            self.metrics.response_times.pop(0)
        
        self.metrics.average_response_time = statistics.mean(self.metrics.response_times)
        
        # Check if we should close the circuit
        if self.state == CircuitState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.metrics.circuit_closed_count += 1
                logger.info(f"Circuit breaker {self.name} moved to CLOSED state")
    
    async def _record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.success_count = 0
        self.metrics.failed_requests += 1
        self.metrics.last_failure_time = time.time()
        
        # Check if we should open the circuit
        if self.failure_count >= self.config.failure_threshold:
            if self.state != CircuitState.OPEN:
                self.state = CircuitState.OPEN
                self.metrics.circuit_opened_count += 1
                logger.warning(f"Circuit breaker {self.name} moved to OPEN state")
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with optional exponential backoff"""
        if not self.config.exponential_backoff:
            return self.config.retry_delay
        
        delay = self.config.retry_delay * (2 ** attempt)
        return min(delay, self.config.max_backoff)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        total_requests = self.metrics.total_requests
        success_rate = (
            self.metrics.successful_requests / total_requests * 100
            if total_requests > 0 else 0
        )
        
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": success_rate,
            "average_response_time": self.metrics.average_response_time,
            "circuit_opened_count": self.metrics.circuit_opened_count,
            "circuit_closed_count": self.metrics.circuit_closed_count,
            "last_failure_time": self.metrics.last_failure_time,
            "last_success_time": self.metrics.last_success_time
        }
    
    def reset(self):
        """Reset circuit breaker to initial state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info(f"Circuit breaker {self.name} reset to CLOSED state")

class CircuitBreakerManager:
    """
    Manages multiple circuit breakers
    """
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
        return self.breakers[name]
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers"""
        return {
            name: breaker.get_metrics()
            for name, breaker in self.breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self.breakers.values():
            breaker.reset()

# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()

def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator for circuit breaker protection
    
    Args:
        name: Circuit breaker name
        config: Circuit breaker configuration
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            breaker = circuit_breaker_manager.get_breaker(name, config)
            return await breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator

class HTTPCircuitBreaker:
    """
    HTTP-specific circuit breaker with aiohttp integration
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.breaker = CircuitBreaker(name, config)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """
        Make HTTP request with circuit breaker protection
        
        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional aiohttp parameters
            
        Returns:
            aiohttp.ClientResponse
        """
        async def _make_request():
            if not self.session:
                raise RuntimeError("HTTPCircuitBreaker not properly initialized")
            
            return await self.session.request(method, url, **kwargs)
        
        return await self.breaker.call(_make_request)
    
    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """GET request with circuit breaker"""
        return await self.request('GET', url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """POST request with circuit breaker"""
        return await self.request('POST', url, **kwargs)
    
    async def put(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """PUT request with circuit breaker"""
        return await self.request('PUT', url, **kwargs)
    
    async def delete(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """DELETE request with circuit breaker"""
        return await self.request('DELETE', url, **kwargs)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        return self.breaker.get_metrics()






























