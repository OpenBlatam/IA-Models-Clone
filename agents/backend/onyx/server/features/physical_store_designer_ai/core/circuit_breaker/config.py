"""
Circuit Breaker Configuration

Defines configuration dataclass for circuit breaker.
"""

from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type = Exception
    success_threshold: int = 2  # Successes needed to close from half-open
    monitoring_window: float = 300.0  # Time window for failure counting (sliding window)
    call_timeout: Optional[float] = None  # Timeout for individual calls
    enable_adaptive_timeout: bool = False  # Enable adaptive recovery timeout
    min_timeout: float = 10.0  # Minimum recovery timeout
    max_timeout: float = 300.0  # Maximum recovery timeout
    timeout_multiplier: float = 2.0  # Multiplier for adaptive timeout
    # Retry configuration
    retry_enabled: bool = False  # Enable automatic retry
    max_retries: int = 3  # Maximum number of retries
    retry_backoff_base: float = 1.0  # Base delay for exponential backoff (seconds)
    retry_backoff_max: float = 60.0  # Maximum delay for backoff (seconds)
    retry_jitter: bool = True  # Add random jitter to backoff
    # Fallback configuration
    fallback_enabled: bool = False  # Enable fallback strategy
    fallback_func: Optional[Callable] = None  # Fallback function to call when circuit is open
    # Half-open rate limiting
    half_open_max_concurrent: int = 1  # Max concurrent requests in half-open state
    # Context manager configuration
    auto_reset_on_exit: bool = False  # Automatically reset circuit breaker on context exit
    # Health check configuration
    health_success_rate_threshold: float = 0.95  # Minimum success rate for healthy (95%)
    health_degraded_threshold: float = 0.80  # Success rate below which is considered degraded (80%)




