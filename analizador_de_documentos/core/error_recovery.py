"""
Error Recovery System for Document Analyzer
===========================================

Advanced error recovery with multiple strategies and automatic retry.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class RecoveryStrategy(Enum):
    """Recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    SKIP = "skip"
    ABORT = "abort"

@dataclass
class RecoveryConfig:
    """Recovery configuration"""
    strategy: RecoveryStrategy
    max_attempts: int = 3
    backoff_seconds: float = 1.0
    fallback_func: Optional[Callable] = None
    degrade_func: Optional[Callable] = None

class ErrorRecovery:
    """Advanced error recovery system"""
    
    def __init__(self):
        self.recovery_configs: Dict[str, RecoveryConfig] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        logger.info("ErrorRecovery initialized")
    
    def register_recovery(
        self,
        error_type: str,
        strategy: RecoveryStrategy,
        handler: Optional[Callable] = None,
        max_attempts: int = 3,
        backoff_seconds: float = 1.0,
        fallback_func: Optional[Callable] = None,
        degrade_func: Optional[Callable] = None
    ):
        """Register recovery strategy for error type"""
        config = RecoveryConfig(
            strategy=strategy,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            fallback_func=fallback_func,
            degrade_func=degrade_func
        )
        self.recovery_configs[error_type] = config
        logger.info(f"Registered recovery for {error_type}: {strategy.value}")
    
    async def recover(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        operation: Optional[Callable] = None
    ) -> Any:
        """Recover from error"""
        error_type = type(error).__name__
        context = context or {}
        
        if error_type not in self.recovery_configs:
            logger.warning(f"No recovery strategy for {error_type}")
            raise error
        
        config = self.recovery_configs[error_type]
        
        # Record recovery attempt
        recovery_record = {
            "error_type": error_type,
            "error_message": str(error),
            "strategy": config.strategy.value,
            "timestamp": datetime.now().isoformat(),
            "context": context
        }
        
        try:
            if config.strategy == RecoveryStrategy.RETRY:
                return await self._retry_strategy(operation, config, context)
            elif config.strategy == RecoveryStrategy.FALLBACK:
                return await self._fallback_strategy(config, context, error)
            elif config.strategy == RecoveryStrategy.DEGRADE:
                return await self._degrade_strategy(config, context, error)
            elif config.strategy == RecoveryStrategy.SKIP:
                return await self._skip_strategy(context)
            elif config.strategy == RecoveryStrategy.ABORT:
                raise error
        except Exception as recovery_error:
            recovery_record["recovery_failed"] = True
            recovery_record["recovery_error"] = str(recovery_error)
            self.recovery_history.append(recovery_record)
            raise recovery_error
        
        recovery_record["success"] = True
        self.recovery_history.append(recovery_record)
        
        if len(self.recovery_history) > self.max_history:
            self.recovery_history = self.recovery_history[-self.max_history:]
    
    async def _retry_strategy(
        self,
        operation: Optional[Callable],
        config: RecoveryConfig,
        context: Dict[str, Any]
    ) -> Any:
        """Retry strategy"""
        if not operation:
            raise ValueError("Operation required for retry strategy")
        
        for attempt in range(config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation(**context)
                else:
                    return operation(**context)
            except Exception as e:
                if attempt < config.max_attempts - 1:
                    wait_time = config.backoff_seconds * (2 ** attempt)
                    logger.warning(f"Retry attempt {attempt + 1}/{config.max_attempts} after {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    async def _fallback_strategy(
        self,
        config: RecoveryConfig,
        context: Dict[str, Any],
        error: Exception
    ) -> Any:
        """Fallback strategy"""
        if not config.fallback_func:
            raise error
        
        logger.info("Using fallback function")
        if asyncio.iscoroutinefunction(config.fallback_func):
            return await config.fallback_func(context, error)
        else:
            return config.fallback_func(context, error)
    
    async def _degrade_strategy(
        self,
        config: RecoveryConfig,
        context: Dict[str, Any],
        error: Exception
    ) -> Any:
        """Degrade strategy"""
        if not config.degrade_func:
            raise error
        
        logger.warning("Degrading service functionality")
        if asyncio.iscoroutinefunction(config.degrade_func):
            return await config.degrade_func(context, error)
        else:
            return config.degrade_func(context, error)
    
    async def _skip_strategy(self, context: Dict[str, Any]) -> Any:
        """Skip strategy"""
        logger.info("Skipping operation")
        return None
    
    def get_recovery_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recovery history"""
        return self.recovery_history[-limit:]

# Global instance
error_recovery = ErrorRecovery()
















