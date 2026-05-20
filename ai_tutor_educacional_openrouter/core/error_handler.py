"""
Advanced error handling and recovery system.
"""

import logging
import traceback
from typing import Any, Optional, Callable, Dict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorHandler:
    """
    Advanced error handling with recovery strategies.
    """
    
    def __init__(self):
        self.error_history: list = []
        self.recovery_strategies: Dict[str, Callable] = {}
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        retry: bool = False,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Handle error with context and recovery options.
        
        Args:
            error: Exception that occurred
            context: Additional context about the error
            severity: Error severity level
            retry: Whether to attempt retry
            max_retries: Maximum number of retries
        
        Returns:
            Error information dictionary
        """
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": severity.value,
            "timestamp": datetime.now().isoformat(),
            "context": context or {},
            "traceback": traceback.format_exc()
        }
        
        self.error_history.append(error_info)
        
        logger.error(
            f"Error [{severity.value}]: {error_info['error_type']} - {error_info['error_message']}",
            extra=error_info
        )
        
        if retry and max_retries > 0:
            recovery_result = self._attempt_recovery(error, context, max_retries)
            error_info["recovery_attempted"] = True
            error_info["recovery_result"] = recovery_result
        
        return error_info
    
    def _attempt_recovery(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]],
        max_retries: int
    ) -> Dict[str, Any]:
        """Attempt to recover from error."""
        error_type = type(error).__name__
        
        if error_type in self.recovery_strategies:
            strategy = self.recovery_strategies[error_type]
            try:
                result = strategy(error, context)
                return {
                    "success": True,
                    "strategy": error_type,
                    "result": result
                }
            except Exception as recovery_error:
                logger.warning(f"Recovery strategy failed: {recovery_error}")
                return {
                    "success": False,
                    "strategy": error_type,
                    "error": str(recovery_error)
                }
        
        return {
            "success": False,
            "reason": "No recovery strategy available"
        }
    
    def register_recovery_strategy(
        self,
        error_type: type,
        strategy: Callable
    ):
        """Register a recovery strategy for an error type."""
        self.recovery_strategies[error_type.__name__] = strategy
        logger.info(f"Registered recovery strategy for {error_type.__name__}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self.error_history:
            return {"total_errors": 0}
        
        severity_counts = {}
        for error in self.error_history:
            severity = error.get("severity", "unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "severity_breakdown": severity_counts,
            "recent_errors": self.error_history[-10:]
        }




