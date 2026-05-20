"""
Error Recovery System
=====================

System for automatic error recovery and resilience.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategy."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    IGNORE = "ignore"


@dataclass
class RecoveryConfig:
    """Recovery configuration."""
    strategy: RecoveryStrategy
    max_attempts: int = 3
    timeout: float = 30.0
    fallback_func: Optional[Callable[[], Awaitable[Any]]] = None
    error_types: List[Type[Exception]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryResult:
    """Recovery result."""
    success: bool
    value: Any = None
    error: Optional[Exception] = None
    attempts: int = 0
    strategy_used: Optional[RecoveryStrategy] = None
    timestamp: datetime = field(default_factory=datetime.now)


class ErrorRecovery:
    """Error recovery handler."""
    
    def __init__(self, config: RecoveryConfig):
        """
        Initialize error recovery.
        
        Args:
            config: Recovery configuration
        """
        self.config = config
        self.attempts = 0
    
    async def execute(
        self,
        func: Callable[[], Awaitable[Any]]
    ) -> RecoveryResult:
        """
        Execute function with error recovery.
        
        Args:
            func: Function to execute
            
        Returns:
            Recovery result
        """
        self.attempts = 0
        
        if self.config.strategy == RecoveryStrategy.RETRY:
            return await self._retry_strategy(func)
        elif self.config.strategy == RecoveryStrategy.FALLBACK:
            return await self._fallback_strategy(func)
        elif self.config.strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return await self._circuit_breaker_strategy(func)
        elif self.config.strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return await self._graceful_degradation_strategy(func)
        else:
            return await self._ignore_strategy(func)
    
    async def _retry_strategy(self, func: Callable[[], Awaitable[Any]]) -> RecoveryResult:
        """Retry strategy."""
        last_error = None
        
        for attempt in range(self.config.max_attempts):
            self.attempts = attempt + 1
            try:
                value = await asyncio.wait_for(func(), timeout=self.config.timeout)
                return RecoveryResult(
                    success=True,
                    value=value,
                    attempts=self.attempts,
                    strategy_used=RecoveryStrategy.RETRY
                )
            except Exception as e:
                last_error = e
                if attempt < self.config.max_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Retry attempt {attempt + 1}/{self.config.max_attempts} after {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
        
        return RecoveryResult(
            success=False,
            error=last_error,
            attempts=self.attempts,
            strategy_used=RecoveryStrategy.RETRY
        )
    
    async def _fallback_strategy(self, func: Callable[[], Awaitable[Any]]) -> RecoveryResult:
        """Fallback strategy."""
        try:
            value = await asyncio.wait_for(func(), timeout=self.config.timeout)
            return RecoveryResult(
                success=True,
                value=value,
                attempts=1,
                strategy_used=RecoveryStrategy.FALLBACK
            )
        except Exception as e:
            if self.config.fallback_func:
                try:
                    fallback_value = await self.config.fallback_func()
                    logger.info(f"Using fallback for error: {e}")
                    return RecoveryResult(
                        success=True,
                        value=fallback_value,
                        attempts=1,
                        strategy_used=RecoveryStrategy.FALLBACK
                    )
                except Exception as fallback_error:
                    return RecoveryResult(
                        success=False,
                        error=fallback_error,
                        attempts=1,
                        strategy_used=RecoveryStrategy.FALLBACK
                    )
            else:
                return RecoveryResult(
                    success=False,
                    error=e,
                    attempts=1,
                    strategy_used=RecoveryStrategy.FALLBACK
                )
    
    async def _circuit_breaker_strategy(self, func: Callable[[], Awaitable[Any]]) -> RecoveryResult:
        """Circuit breaker strategy."""
        # Simplified circuit breaker
        # In production, use a proper circuit breaker implementation
        try:
            value = await asyncio.wait_for(func(), timeout=self.config.timeout)
            return RecoveryResult(
                success=True,
                value=value,
                attempts=1,
                strategy_used=RecoveryStrategy.CIRCUIT_BREAKER
            )
        except Exception as e:
            return RecoveryResult(
                success=False,
                error=e,
                attempts=1,
                strategy_used=RecoveryStrategy.CIRCUIT_BREAKER
            )
    
    async def _graceful_degradation_strategy(self, func: Callable[[], Awaitable[Any]]) -> RecoveryResult:
        """Graceful degradation strategy."""
        try:
            value = await asyncio.wait_for(func(), timeout=self.config.timeout)
            return RecoveryResult(
                success=True,
                value=value,
                attempts=1,
                strategy_used=RecoveryStrategy.GRACEFUL_DEGRADATION
            )
        except Exception as e:
            # Return a degraded result instead of failing
            logger.warning(f"Graceful degradation for error: {e}")
            return RecoveryResult(
                success=True,  # Still considered success with degraded value
                value=None,  # Degraded value
                error=e,
                attempts=1,
                strategy_used=RecoveryStrategy.GRACEFUL_DEGRADATION
            )
    
    async def _ignore_strategy(self, func: Callable[[], Awaitable[Any]]) -> RecoveryResult:
        """Ignore strategy."""
        try:
            value = await asyncio.wait_for(func(), timeout=self.config.timeout)
            return RecoveryResult(
                success=True,
                value=value,
                attempts=1,
                strategy_used=RecoveryStrategy.IGNORE
            )
        except Exception as e:
            logger.debug(f"Ignoring error: {e}")
            return RecoveryResult(
                success=False,
                error=e,
                attempts=1,
                strategy_used=RecoveryStrategy.IGNORE
            )


class RecoveryManager:
    """Manager for error recovery."""
    
    def __init__(self):
        """Initialize recovery manager."""
        self.recoveries: Dict[str, ErrorRecovery] = {}
        self.history: List[RecoveryResult] = []
        self.max_history = 1000
    
    def register(
        self,
        name: str,
        config: RecoveryConfig
    ):
        """
        Register a recovery handler.
        
        Args:
            name: Recovery name
            config: Recovery configuration
        """
        recovery = ErrorRecovery(config)
        self.recoveries[name] = recovery
        logger.debug(f"Registered recovery: {name}")
    
    async def execute(
        self,
        name: str,
        func: Callable[[], Awaitable[Any]]
    ) -> RecoveryResult:
        """
        Execute with recovery.
        
        Args:
            name: Recovery name
            func: Function to execute
            
        Returns:
            Recovery result
        """
        if name not in self.recoveries:
            # Default recovery
            config = RecoveryConfig(strategy=RecoveryStrategy.RETRY)
            recovery = ErrorRecovery(config)
        else:
            recovery = self.recoveries[name]
        
        result = await recovery.execute(func)
        
        # Add to history
        self.history.append(result)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        return result
    
    def get_history(self, limit: int = 100) -> List[RecoveryResult]:
        """Get recovery history."""
        return self.history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        if not self.history:
            return {}
        
        total = len(self.history)
        successful = len([r for r in self.history if r.success])
        
        return {
            "total_recoveries": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0.0
        }




