"""
Advanced Error Handler for Flux2 Clothing Changer
==================================================

Advanced error handling and recovery system.
"""

import time
import traceback
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Error:
    """Error information."""
    error_id: str
    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: float
    context: Dict[str, Any] = None
    stack_trace: Optional[str] = None
    resolved: bool = False
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


class ErrorHandler:
    """Advanced error handling system."""
    
    def __init__(
        self,
        enable_recovery: bool = True,
        max_error_history: int = 10000,
    ):
        """
        Initialize error handler.
        
        Args:
            enable_recovery: Enable automatic recovery
            max_error_history: Maximum error history size
        """
        self.enable_recovery = enable_recovery
        self.max_error_history = max_error_history
        
        self.errors: List[Error] = []
        self.error_handlers: Dict[str, Callable] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        self.error_counts: Dict[str, int] = {}
    
    def register_handler(
        self,
        error_type: str,
        handler: Callable[[Error], Any],
    ) -> None:
        """
        Register error handler.
        
        Args:
            error_type: Error type
            handler: Handler function
        """
        self.error_handlers[error_type] = handler
        logger.info(f"Registered handler for error type: {error_type}")
    
    def register_recovery(
        self,
        error_type: str,
        recovery: Callable[[Error], bool],
    ) -> None:
        """
        Register recovery strategy.
        
        Args:
            error_type: Error type
            recovery: Recovery function
        """
        self.recovery_strategies[error_type] = recovery
        logger.info(f"Registered recovery for error type: {error_type}")
    
    def handle_error(
        self,
        error_type: str,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Error:
        """
        Handle error.
        
        Args:
            error_type: Error type
            message: Error message
            severity: Error severity
            exception: Optional exception
            context: Optional context
            
        Returns:
            Created error
        """
        error_id = f"{error_type}_{int(time.time() * 1000)}"
        stack_trace = traceback.format_exc() if exception else None
        
        error = Error(
            error_id=error_id,
            error_type=error_type,
            message=message,
            severity=severity,
            timestamp=time.time(),
            context=context or {},
            stack_trace=stack_trace,
        )
        
        # Add to history
        self.errors.append(error)
        if len(self.errors) > self.max_error_history:
            self.errors.pop(0)
        
        # Update counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log error
        log_level = {
            ErrorSeverity.LOW: logging.DEBUG,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }.get(severity, logging.ERROR)
        
        logger.log(log_level, f"Error [{error_type}]: {message}")
        
        # Call registered handler
        if error_type in self.error_handlers:
            try:
                self.error_handlers[error_type](error)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
        
        # Attempt recovery
        if self.enable_recovery and error_type in self.recovery_strategies:
            try:
                recovered = self.recovery_strategies[error_type](error)
                if recovered:
                    error.resolved = True
                    logger.info(f"Error {error_id} recovered")
            except Exception as e:
                logger.error(f"Recovery failed: {e}")
        
        return error
    
    def get_errors(
        self,
        error_type: Optional[str] = None,
        severity: Optional[ErrorSeverity] = None,
        time_range: Optional[float] = None,
    ) -> List[Error]:
        """
        Get errors.
        
        Args:
            error_type: Optional error type filter
            severity: Optional severity filter
            time_range: Optional time range in seconds
            
        Returns:
            List of errors
        """
        errors = self.errors.copy()
        
        if error_type:
            errors = [e for e in errors if e.error_type == error_type]
        
        if severity:
            errors = [e for e in errors if e.severity == severity]
        
        if time_range:
            cutoff_time = time.time() - time_range
            errors = [e for e in errors if e.timestamp >= cutoff_time]
        
        return errors
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "total_errors": len(self.errors),
            "error_counts": dict(self.error_counts),
            "errors_by_severity": {
                severity.value: len([e for e in self.errors if e.severity == severity])
                for severity in ErrorSeverity
            },
            "resolved_errors": len([e for e in self.errors if e.resolved]),
        }


