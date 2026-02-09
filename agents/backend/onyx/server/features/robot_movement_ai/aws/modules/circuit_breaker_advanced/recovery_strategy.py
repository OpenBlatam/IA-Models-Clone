"""
Recovery Strategy
=================

Circuit breaker recovery strategies.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RecoveryStrategyType(Enum):
    """Recovery strategy types."""
    IMMEDIATE = "immediate"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    CUSTOM = "custom"


@dataclass
class RecoveryConfig:
    """Recovery configuration."""
    strategy: RecoveryStrategyType
    initial_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    custom_handler: Optional[Callable] = None


class RecoveryStrategy:
    """Circuit breaker recovery strategy."""
    
    def __init__(self, config: RecoveryConfig):
        self.config = config
        self._attempts: Dict[str, int] = {}  # service -> attempts
    
    async def should_retry(self, service: str) -> bool:
        """Determine if should retry."""
        if service not in self._attempts:
            self._attempts[service] = 0
        
        self._attempts[service] += 1
        
        if self.config.strategy == RecoveryStrategyType.IMMEDIATE:
            return True
        
        elif self.config.strategy == RecoveryStrategyType.EXPONENTIAL_BACKOFF:
            delay = min(
                self.config.initial_delay * (self.config.multiplier ** (self._attempts[service] - 1)),
                self.config.max_delay
            )
            await asyncio.sleep(delay)
            return True
        
        elif self.config.strategy == RecoveryStrategyType.LINEAR_BACKOFF:
            delay = min(
                self.config.initial_delay * self._attempts[service],
                self.config.max_delay
            )
            await asyncio.sleep(delay)
            return True
        
        elif self.config.strategy == RecoveryStrategyType.CUSTOM:
            if self.config.custom_handler:
                return await self._execute_custom_handler(service)
        
        return False
    
    async def _execute_custom_handler(self, service: str) -> bool:
        """Execute custom recovery handler."""
        try:
            if asyncio.iscoroutinefunction(self.config.custom_handler):
                return await self.config.custom_handler(service, self._attempts[service])
            else:
                return self.config.custom_handler(service, self._attempts[service])
        except Exception as e:
            logger.error(f"Custom recovery handler failed: {e}")
            return False
    
    def reset_attempts(self, service: str):
        """Reset retry attempts for service."""
        self._attempts[service] = 0
        logger.info(f"Reset recovery attempts for {service}")
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        return {
            "strategy": self.config.strategy.value,
            "services": {
                service: {
                    "attempts": attempts
                }
                for service, attempts in self._attempts.items()
            }
        }















