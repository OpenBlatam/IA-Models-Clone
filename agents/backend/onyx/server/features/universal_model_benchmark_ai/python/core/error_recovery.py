"""
Error Recovery Module - Advanced error recovery strategies.

Provides:
- Automatic error recovery
- Recovery strategies
- Error classification
- Recovery policies
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(str, Enum):
    """Recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    IGNORE = "ignore"
    ESCALATE = "escalate"
    ROLLBACK = "rollback"


@dataclass
class ErrorContext:
    """Error context information."""
    error_type: str
    error_message: str
    severity: ErrorSeverity
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None


@dataclass
class RecoveryPolicy:
    """Recovery policy."""
    error_types: List[Type[Exception]]
    strategy: RecoveryStrategy
    max_attempts: int = 3
    fallback_func: Optional[Callable] = None
    escalate_on_failure: bool = False


class ErrorRecoveryManager:
    """Error recovery manager."""
    
    def __init__(self):
        """Initialize error recovery manager."""
        self.policies: List[RecoveryPolicy] = []
        self.error_history: List[ErrorContext] = []
        self.max_history = 1000
    
    def register_policy(self, policy: RecoveryPolicy) -> None:
        """
        Register recovery policy.
        
        Args:
            policy: Recovery policy
        """
        self.policies.append(policy)
        logger.info(f"Registered recovery policy for {len(policy.error_types)} error types")
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Handle error with recovery.
        
        Args:
            error: Exception to handle
            context: Optional context
            
        Returns:
            Recovery result or None
        """
        error_context = ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            severity=self._classify_error(error),
            context=context or {},
        )
        
        self.error_history.append(error_context)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Find matching policy
        policy = self._find_policy(error)
        if not policy:
            logger.warning(f"No recovery policy for {error_context.error_type}")
            return None
        
        # Execute recovery strategy
        return self._execute_recovery(policy, error, error_context)
    
    def _classify_error(self, error: Exception) -> ErrorSeverity:
        """Classify error severity."""
        error_name = type(error).__name__.lower()
        
        if "critical" in error_name or "fatal" in error_name:
            return ErrorSeverity.CRITICAL
        elif "timeout" in error_name or "connection" in error_name:
            return ErrorSeverity.HIGH
        elif "validation" in error_name or "value" in error_name:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _find_policy(self, error: Exception) -> Optional[RecoveryPolicy]:
        """Find matching recovery policy."""
        error_type = type(error)
        for policy in self.policies:
            if error_type in policy.error_types:
                return policy
        return None
    
    def _execute_recovery(
        self,
        policy: RecoveryPolicy,
        error: Exception,
        context: ErrorContext,
    ) -> Any:
        """Execute recovery strategy."""
        if policy.strategy == RecoveryStrategy.RETRY:
            # Retry logic would be handled by retry module
            logger.info(f"Retry strategy for {context.error_type}")
            return None
        
        elif policy.strategy == RecoveryStrategy.FALLBACK:
            if policy.fallback_func:
                try:
                    return policy.fallback_func(context)
                except Exception as e:
                    logger.error(f"Fallback function failed: {e}")
            return None
        
        elif policy.strategy == RecoveryStrategy.IGNORE:
            logger.warning(f"Ignoring error: {context.error_type}")
            return None
        
        elif policy.strategy == RecoveryStrategy.ESCALATE:
            logger.error(f"Escalating error: {context.error_type}")
            # Escalation logic (e.g., send alert, log to external system)
            return None
        
        elif policy.strategy == RecoveryStrategy.ROLLBACK:
            logger.warning(f"Rollback strategy for {context.error_type}")
            # Rollback logic
            return None
        
        return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self.error_history:
            return {}
        
        severity_counts = {}
        type_counts = {}
        
        for error in self.error_history:
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            error_type = error.error_type
            type_counts[error_type] = type_counts.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "severity_counts": severity_counts,
            "type_counts": type_counts,
            "recent_errors": [
                e.to_dict() for e in self.error_history[-10:]
            ],
        }
    
    def clear_history(self) -> None:
        """Clear error history."""
        self.error_history.clear()
        logger.info("Error history cleared")












