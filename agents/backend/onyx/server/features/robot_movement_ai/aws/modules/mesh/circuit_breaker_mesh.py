"""
Circuit Breaker Mesh
===================

Circuit breaker for service mesh.
"""

import logging
import time
from typing import Dict, Any, Optional
from aws.modules.performance.async_optimizer import AsyncOptimizer

logger = logging.getLogger(__name__)


class CircuitBreakerMesh:
    """Circuit breaker for service mesh."""
    
    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures = 0
        self._last_failure_time: Optional[float] = None
        self._state = "closed"  # closed, open, half_open
    
    def can_call(self) -> bool:
        """Check if service can be called."""
        if self._state == "closed":
            return True
        
        if self._state == "open":
            if self._last_failure_time and time.time() - self._last_failure_time > self.recovery_timeout:
                self._state = "half_open"
                logger.info(f"Circuit breaker for {self.service_name} entering half-open state")
                return True
            return False
        
        # half_open
        return True
    
    def record_success(self):
        """Record successful call."""
        if self._state == "half_open":
            self._state = "closed"
            self._failures = 0
            logger.info(f"Circuit breaker for {self.service_name} closed")
    
    def record_failure(self):
        """Record failed call."""
        self._failures += 1
        self._last_failure_time = time.time()
        
        if self._failures >= self.failure_threshold:
            self._state = "open"
            logger.error(f"Circuit breaker for {self.service_name} opened after {self._failures} failures")
    
    def get_state(self) -> str:
        """Get circuit breaker state."""
        return self._state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "service": self.service_name,
            "state": self._state,
            "failures": self._failures,
            "last_failure_time": self._last_failure_time
        }















