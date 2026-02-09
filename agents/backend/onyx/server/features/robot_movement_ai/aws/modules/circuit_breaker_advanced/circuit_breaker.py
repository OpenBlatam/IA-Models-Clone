"""
Advanced Circuit Breaker
========================

Advanced circuit breaker implementation.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0
    half_open_timeout: float = 30.0


class AdvancedCircuitBreaker:
    """Advanced circuit breaker."""
    
    def __init__(self, name: str, config: Optional[CircuitConfig] = None):
        self.name = name
        self.config = config or CircuitConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._opened_at: Optional[datetime] = None
        self._stats: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "circuit_opens": 0
        }
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function through circuit breaker."""
        self._stats["total_requests"] += 1
        
        if self._state == CircuitState.OPEN:
            # Check if timeout has passed
            if self._opened_at:
                elapsed = (datetime.now() - self._opened_at).total_seconds()
                if elapsed >= self.config.half_open_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(f"Circuit {self.name} moved to HALF_OPEN")
                else:
                    raise Exception(f"Circuit {self.name} is OPEN")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = await asyncio.to_thread(func, *args, **kwargs)
            
            self._handle_success()
            return result
        
        except Exception as e:
            self._handle_failure()
            raise
    
    def _handle_success(self):
        """Handle successful call."""
        self._stats["successful_requests"] += 1
        
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            
            if self._success_count >= self.config.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                logger.info(f"Circuit {self.name} moved to CLOSED")
        
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0
    
    def _handle_failure(self):
        """Handle failed call."""
        self._stats["failed_requests"] += 1
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = datetime.now()
            logger.warning(f"Circuit {self.name} moved to OPEN from HALF_OPEN")
        
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = datetime.now()
                self._stats["circuit_opens"] += 1
                logger.warning(f"Circuit {self.name} moved to OPEN")
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "stats": self._stats.copy()
        }


# Import asyncio
import asyncio















