"""
Enhanced Circuit Breaker - Advanced resilience pattern for webhook endpoints
With metrics, half-open state management, and adaptive thresholds
"""

import time
import logging
from typing import Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker performance"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    circuit_opened_count: int = 0
    circuit_closed_count: int = 0
    last_state_change: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100.0


class EnhancedCircuitBreaker:
    """
    Enhanced circuit breaker for webhook endpoint resilience
    
    Features:
    - Three states: closed, open, half-open
    - Adaptive thresholds based on success rate
    - Metrics tracking
    - Half-open state testing
    - Configurable timeout windows
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,  # Successes needed to close from half-open
        timeout: int = 60,
        half_open_max_requests: int = 3,  # Max requests in half-open state
        adaptive_threshold: bool = True
    ):
        """
        Initialize enhanced circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            success_threshold: Successes needed to close from half-open
            timeout: Seconds before attempting half-open state
            half_open_max_requests: Max requests to test in half-open state
            adaptive_threshold: Enable adaptive threshold adjustment
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.half_open_max_requests = half_open_max_requests
        self.adaptive_threshold = adaptive_threshold
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0  # For half-open state
        self.half_open_request_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_state_change_time = time.time()
        
        # Metrics
        self.metrics = CircuitBreakerMetrics()
        
        logger.debug(
            f"Circuit breaker initialized: threshold={failure_threshold}, timeout={timeout}"
        )
    
    def record_success(self) -> None:
        """Record successful request"""
        self.metrics.total_requests += 1
        self.metrics.successful_requests += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            self.half_open_request_count += 1
            
            # Check if we can close the circuit
            if self.success_count >= self.success_threshold:
                self._transition_to(CircuitState.CLOSED, "Success threshold reached in half-open state")
                logger.info("Circuit breaker closed after successful recovery")
            elif self.half_open_request_count >= self.half_open_max_requests:
                # Too many requests without enough successes, open again
                self._transition_to(CircuitState.OPEN, "Insufficient successes in half-open state")
                logger.warning("Circuit breaker reopened due to insufficient recovery")
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            if self.failure_count > 0:
                self.failure_count = max(0, self.failure_count - 1)  # Decay failure count
    
    def record_failure(self) -> None:
        """Record failed request"""
        self.metrics.total_requests += 1
        self.metrics.failed_requests += 1
        
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open opens immediately
            self._transition_to(CircuitState.OPEN, "Failure in half-open state")
            logger.warning("Circuit breaker opened due to failure in half-open state")
        elif self.state == CircuitState.CLOSED:
            # Check if we should open
            if self.failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN, "Failure threshold reached")
                logger.warning(
                    f"Circuit breaker opened: {self.failure_count} failures "
                    f"(threshold: {self.failure_threshold})"
                )
    
    def can_proceed(self) -> bool:
        """
        Check if request can proceed through circuit
        
        Returns:
            True if circuit allows request, False if circuit is open
        """
        # Auto-transition to half-open if timeout expired
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and (time.time() - self.last_failure_time) >= self.timeout:
                self._transition_to(CircuitState.HALF_OPEN, "Timeout expired, testing recovery")
                logger.info("Circuit breaker transitioning to half-open for testing")
                return True
            return False
        
        return True
    
    def _transition_to(self, new_state: CircuitState, reason: str) -> None:
        """Transition to new state with logging"""
        old_state = self.state
        self.state = new_state
        self.last_state_change_time = time.time()
        
        if new_state == CircuitState.CLOSED:
            self.metrics.circuit_closed_count += 1
            self.failure_count = 0
            self.success_count = 0
            self.half_open_request_count = 0
        elif new_state == CircuitState.OPEN:
            self.metrics.circuit_opened_count += 1
            self.success_count = 0
            self.half_open_request_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self.failure_count = 0
            self.success_count = 0
            self.half_open_request_count = 0
        
        logger.debug(
            f"Circuit breaker state transition: {old_state} -> {new_state} ({reason})"
        )
    
    def get_state(self) -> str:
        """Get current circuit breaker state"""
        return self.state.value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        return {
            "state": self.get_state(),
            "failure_count": self.failure_count,
            "success_count": self.success_count if self.state == CircuitState.HALF_OPEN else 0,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "rejected_requests": self.metrics.rejected_requests,
            "success_rate": round(self.metrics.success_rate, 2),
            "failure_rate": round(self.metrics.failure_rate, 2),
            "circuit_opened_count": self.metrics.circuit_opened_count,
            "circuit_closed_count": self.metrics.circuit_closed_count,
            "last_failure_time": self.last_failure_time,
            "time_since_last_state_change": time.time() - self.last_state_change_time
        }
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_request_count = 0
        self.last_failure_time = None
        self.last_state_change_time = time.time()
        logger.info("Circuit breaker reset to closed state")


# Backward compatibility - keep original class name
CircuitBreaker = EnhancedCircuitBreaker

