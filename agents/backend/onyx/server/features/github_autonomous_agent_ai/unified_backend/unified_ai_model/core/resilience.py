"""
Resilience Module
Provides retry with exponential backoff and circuit breaker patterns.
Ported from autonomous_long_term_agent/core/resilience.py
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Callable, Any, Optional, Dict, List, Type
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 30.0
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_TIMEOUT = 60.0


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = DEFAULT_MAX_RETRIES
    base_delay: float = DEFAULT_BASE_DELAY
    max_delay: float = DEFAULT_MAX_DELAY
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR
    timeout: float = DEFAULT_TIMEOUT
    retryable_errors: List[Type[Exception]] = field(
        default_factory=lambda: [ConnectionError, TimeoutError, OSError]
    )


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: int = 30


class RetryHandler:
    """Retry mechanism with exponential backoff."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.stats = {
            "total_attempts": 0,
            "successful_retries": 0,
            "failed_retries": 0
        }
    
    def _is_retryable(self, error: Exception) -> bool:
        """Check if error is retryable."""
        return any(isinstance(error, err_type) for err_type in self.config.retryable_errors)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff."""
        delay = self.config.base_delay * (self.config.backoff_factor ** attempt)
        return min(delay, self.config.max_delay)
    
    async def execute(
        self,
        func: Callable,
        *args,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """Execute function with retry logic."""
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            self.stats["total_attempts"] += 1
            
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt - 1)
                    logger.info(f"Retry attempt {attempt}/{self.config.max_retries} after {delay:.1f}s delay")
                    await asyncio.sleep(delay)
                
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                if attempt > 0:
                    self.stats["successful_retries"] += 1
                    logger.info(f"Retry successful on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_error = e
                
                if not self._is_retryable(e):
                    logger.warning(f"Non-retryable error: {e}")
                    break
                
                if attempt >= self.config.max_retries:
                    logger.error(f"Max retries ({self.config.max_retries}) reached")
                    break
                
                logger.warning(f"Retryable error on attempt {attempt + 1}: {e}")
        
        self.stats["failed_retries"] += 1
        raise last_error or Exception("Unknown error in retry handler")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        return self.stats.copy()


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        self.stats = {
            "total_calls": 0,
            "rejected_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0
        }
    
    def _should_attempt_call(self) -> bool:
        """Check if call should be attempted."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.config.timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                    return True
            return False
        
        # HALF_OPEN - allow limited calls
        return True
    
    def _on_success(self) -> None:
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info("Circuit breaker CLOSED after recovery")
        else:
            self.failure_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker re-OPENED after failure in HALF_OPEN")
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        self.stats["total_calls"] += 1
        
        if not self._should_attempt_call():
            self.stats["rejected_calls"] += 1
            raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
            
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self._on_success()
            self.stats["successful_calls"] += 1
            return result
            
        except Exception as e:
            self._on_failure()
            self.stats["failed_calls"] += 1
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "stats": self.stats.copy()
        }
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.last_failure_time = None
        logger.info("Circuit breaker reset")


class ResilienceManager:
    """Combined resilience manager with retry and circuit breaker."""
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        self.retry_handler = RetryHandler(retry_config)
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config)
    
    async def execute(
        self,
        func: Callable,
        *args,
        use_retry: bool = True,
        use_circuit_breaker: bool = True,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """Execute function with resilience patterns."""
        if use_circuit_breaker and use_retry:
            async def _wrapped():
                return await self.circuit_breaker.call(func, *args, **kwargs)
            return await self.retry_handler.execute(_wrapped, context=context)
        
        elif use_circuit_breaker:
            return await self.circuit_breaker.call(func, *args, **kwargs)
        
        elif use_retry:
            return await self.retry_handler.execute(func, *args, context=context, **kwargs)
        
        else:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined resilience statistics."""
        return {
            "retry": self.retry_handler.get_stats(),
            "circuit_breaker": self.circuit_breaker.get_state()
        }
