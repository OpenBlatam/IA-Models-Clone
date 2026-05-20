"""
Error Recovery System for Color Grading AI
==========================================

Advanced error recovery and resilience mechanisms.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategies."""
    RETRY = "retry"  # Retry operation
    FALLBACK = "fallback"  # Use fallback method
    ROLLBACK = "rollback"  # Rollback changes
    SKIP = "skip"  # Skip operation
    NOTIFY = "notify"  # Notify and continue


@dataclass
class ErrorContext:
    """Error context."""
    error_type: str
    error_message: str
    operation: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Recovery action."""
    strategy: RecoveryStrategy
    action: Callable
    max_attempts: int = 3
    backoff_seconds: float = 1.0


@dataclass
class RecoveryResult:
    """Recovery result."""
    success: bool
    strategy: RecoveryStrategy
    attempts: int = 0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class ErrorRecoverySystem:
    """
    Error recovery system.
    
    Features:
    - Multiple recovery strategies
    - Automatic error classification
    - Fallback mechanisms
    - Rollback support
    - Error tracking
    - Recovery statistics
    """
    
    def __init__(self):
        """Initialize error recovery system."""
        self._recovery_actions: Dict[str, List[RecoveryAction]] = {}
        self._error_history: List[ErrorContext] = []
        self._recovery_history: List[RecoveryResult] = []
        self._max_history = 1000
    
    def register_recovery_action(
        self,
        error_type: str,
        strategy: RecoveryStrategy,
        action: Callable,
        max_attempts: int = 3,
        backoff_seconds: float = 1.0
    ):
        """
        Register recovery action for error type.
        
        Args:
            error_type: Error type (e.g., "TimeoutError", "ValueError")
            strategy: Recovery strategy
            action: Recovery action function
            max_attempts: Maximum recovery attempts
            backoff_seconds: Backoff delay in seconds
        """
        if error_type not in self._recovery_actions:
            self._recovery_actions[error_type] = []
        
        recovery_action = RecoveryAction(
            strategy=strategy,
            action=action,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds
        )
        
        self._recovery_actions[error_type].append(recovery_action)
        logger.info(f"Registered recovery action for {error_type}: {strategy.value}")
    
    async def recover(
        self,
        error: Exception,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RecoveryResult:
        """
        Attempt to recover from error.
        
        Args:
            error: Exception that occurred
            operation: Operation name
            context: Optional context
            
        Returns:
            Recovery result
        """
        error_type = type(error).__name__
        error_context = ErrorContext(
            error_type=error_type,
            error_message=str(error),
            operation=operation,
            metadata=context or {}
        )
        
        self._error_history.append(error_context)
        if len(self._error_history) > self._max_history:
            self._error_history.pop(0)
        
        # Get recovery actions for this error type
        actions = self._recovery_actions.get(error_type, [])
        if not actions:
            # Try generic error type
            actions = self._recovery_actions.get("Exception", [])
        
        if not actions:
            logger.warning(f"No recovery action found for {error_type}")
            return RecoveryResult(
                success=False,
                strategy=RecoveryStrategy.SKIP,
                error="No recovery action available"
            )
        
        # Try recovery actions in order
        for recovery_action in actions:
            try:
                result = await self._execute_recovery(
                    recovery_action,
                    error,
                    operation,
                    context
                )
                
                if result.success:
                    self._recovery_history.append(result)
                    return result
                
            except Exception as recovery_error:
                logger.error(f"Recovery action failed: {recovery_error}")
                continue
        
        # All recovery attempts failed
        return RecoveryResult(
            success=False,
            strategy=RecoveryStrategy.SKIP,
            error="All recovery attempts failed"
        )
    
    async def _execute_recovery(
        self,
        recovery_action: RecoveryAction,
        error: Exception,
        operation: str,
        context: Optional[Dict[str, Any]]
    ) -> RecoveryResult:
        """Execute a recovery action."""
        attempts = 0
        
        while attempts < recovery_action.max_attempts:
            attempts += 1
            
            try:
                if asyncio.iscoroutinefunction(recovery_action.action):
                    result = await recovery_action.action(error, operation, context)
                else:
                    result = recovery_action.action(error, operation, context)
                
                if result:
                    return RecoveryResult(
                        success=True,
                        strategy=recovery_action.strategy,
                        attempts=attempts
                    )
            
            except Exception as e:
                logger.debug(f"Recovery attempt {attempts} failed: {e}")
                if attempts < recovery_action.max_attempts:
                    await asyncio.sleep(recovery_action.backoff_seconds * attempts)
        
        return RecoveryResult(
            success=False,
            strategy=recovery_action.strategy,
            attempts=attempts,
            error="Max attempts reached"
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self._error_history:
            return {
                "total_errors": 0,
            }
        
        error_counts = {}
        for error_context in self._error_history:
            error_type = error_context.error_type
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self._error_history),
            "error_types": error_counts,
            "recovery_success_rate": (
                sum(1 for r in self._recovery_history if r.success) /
                len(self._recovery_history) if self._recovery_history else 0.0
            ),
        }


