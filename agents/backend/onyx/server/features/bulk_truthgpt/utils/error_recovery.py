"""
Error Recovery
==============

Advanced error recovery with automatic retry and fallback strategies.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RecoveryStrategy(str, Enum):
    """Recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    DEGRADE = "degrade"
    SKIP = "skip"

@dataclass
class RecoveryAction:
    """Recovery action definition."""
    action_id: str
    error_type: str
    strategy: RecoveryStrategy
    handler: Callable
    max_attempts: int = 3
    backoff_seconds: float = 1.0
    fallback_value: Any = None

class ErrorRecoveryManager:
    """Advanced error recovery manager."""
    
    def __init__(self):
        self.recovery_actions: Dict[str, List[RecoveryAction]] = {}
        self.error_stats: Dict[str, int] = {}
        self.recovery_stats: Dict[str, int] = {}
    
    def register_recovery(
        self,
        error_type: str,
        strategy: RecoveryStrategy,
        handler: Callable,
        max_attempts: int = 3,
        backoff_seconds: float = 1.0,
        fallback_value: Any = None
    ) -> RecoveryAction:
        """Register a recovery action."""
        import uuid
        action_id = str(uuid.uuid4())
        
        action = RecoveryAction(
            action_id=action_id,
            error_type=error_type,
            strategy=strategy,
            handler=handler,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            fallback_value=fallback_value
        )
        
        if error_type not in self.recovery_actions:
            self.recovery_actions[error_type] = []
        
        self.recovery_actions[error_type].append(action)
        logger.info(f"Recovery action registered: {error_type} - {strategy.value}")
        
        return action
    
    async def recover(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Attempt to recover from an error."""
        error_type = type(error).__name__
        self.error_stats[error_type] = self.error_stats.get(error_type, 0) + 1
        
        if error_type not in self.recovery_actions:
            logger.warning(f"No recovery action for error type: {error_type}")
            raise error
        
        actions = self.recovery_actions[error_type]
        
        for action in actions:
            try:
                if action.strategy == RecoveryStrategy.RETRY:
                    return await self._retry_recovery(action, context)
                elif action.strategy == RecoveryStrategy.FALLBACK:
                    return await self._fallback_recovery(action, context)
                elif action.strategy == RecoveryStrategy.DEGRADE:
                    return await self._degrade_recovery(action, context)
                elif action.strategy == RecoveryStrategy.SKIP:
                    logger.info(f"Skipping operation due to error: {error_type}")
                    return action.fallback_value
                else:
                    continue
            except Exception as recovery_error:
                logger.error(f"Recovery action failed: {recovery_error}")
                continue
        
        # No recovery succeeded
        logger.error(f"All recovery actions failed for: {error_type}")
        raise error
    
    async def _retry_recovery(
        self,
        action: RecoveryAction,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Retry recovery strategy."""
        last_error = None
        
        for attempt in range(1, action.max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(action.handler):
                    result = await action.handler(context)
                else:
                    result = action.handler(context)
                
                self.recovery_stats["retry_success"] = self.recovery_stats.get("retry_success", 0) + 1
                logger.info(f"Recovery successful after {attempt} attempts")
                return result
                
            except Exception as e:
                last_error = e
                if attempt < action.max_attempts:
                    wait_time = action.backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(f"Retry attempt {attempt}/{action.max_attempts} failed, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
        
        self.recovery_stats["retry_failed"] = self.recovery_stats.get("retry_failed", 0) + 1
        raise last_error
    
    async def _fallback_recovery(
        self,
        action: RecoveryAction,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Fallback recovery strategy."""
        try:
            if asyncio.iscoroutinefunction(action.handler):
                result = await action.handler(context)
            else:
                result = action.handler(context)
            
            self.recovery_stats["fallback_success"] = self.recovery_stats.get("fallback_success", 0) + 1
            logger.info("Fallback recovery successful")
            return result
            
        except Exception as e:
            logger.error(f"Fallback recovery failed: {e}")
            if action.fallback_value is not None:
                logger.info("Using fallback value")
                return action.fallback_value
            raise
    
    async def _degrade_recovery(
        self,
        action: RecoveryAction,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Degrade recovery strategy."""
        try:
            if asyncio.iscoroutinefunction(action.handler):
                result = await action.handler(context)
            else:
                result = action.handler(context)
            
            self.recovery_stats["degrade_success"] = self.recovery_stats.get("degrade_success", 0) + 1
            logger.info("Degraded mode activated")
            return result
            
        except Exception as e:
            logger.error(f"Degrade recovery failed: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        return {
            "error_stats": self.error_stats.copy(),
            "recovery_stats": self.recovery_stats.copy(),
            "registered_actions": {
                error_type: len(actions)
                for error_type, actions in self.recovery_actions.items()
            }
        }

# Global instance
error_recovery = ErrorRecoveryManager()
















