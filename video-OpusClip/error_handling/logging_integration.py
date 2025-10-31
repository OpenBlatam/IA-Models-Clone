#!/usr/bin/env python3
"""
Logging Integration for Error Handling

Enhanced logging integration with:
- Structured error logging
- Performance metrics logging
- Security event logging
- Request/response logging
- Error correlation and tracking
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import time
import uuid
import structlog
from datetime import datetime
from contextlib import contextmanager

from .error_handling import (
    VideoProcessingError,
    ValidationError,
    SecurityError,
    ResourceError,
    DatabaseError,
    CacheError,
    MonitoringError
)

logger = structlog.get_logger("error_logging")

# =============================================================================
# ERROR LOGGING INTEGRATION
# =============================================================================

class ErrorLoggingIntegration:
    """Integration between error handling and logging systems."""
    
    def __init__(self):
        self.error_counts = {}
        self.error_timestamps = {}
        self.performance_metrics = {}
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None, 
                 request_id: Optional[str] = None, severity: str = "error"):
        """Log error with structured information."""
        
        # Generate request ID if not provided
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Determine error type and category
        error_type = type(error).__name__
        error_category = self._categorize_error(error)
        
        # Update error statistics
        self._update_error_stats(error_type, error_category)
        
        # Create structured log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "error_type": error_type,
            "error_category": error_category,
            "error_message": str(error),
            "severity": severity,
            "context": context or {},
            "stack_trace": self._get_stack_trace(error),
            "error_code": getattr(error, 'error_code', 'UNKNOWN_ERROR'),
            "field": getattr(error, 'field', None),
            "threat_type": getattr(error, 'threat_type', None),
            "original_error": str(getattr(error, 'original_error', None))
        }
        
        # Log based on severity
        if severity == "critical":
            logger.critical("Critical error occurred", **log_entry)
        elif severity == "error":
            logger.error("Error occurred", **log_entry)
        elif severity == "warning":
            logger.warning("Warning occurred", **log_entry)
        else:
            logger.info("Info message", **log_entry)
        
        return request_id
    
    def log_performance_error(self, operation: str, duration: float, 
                            error: Exception, context: Optional[Dict] = None):
        """Log performance-related errors."""
        
        request_id = str(uuid.uuid4())
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "operation": operation,
            "duration": duration,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "performance_impact": True
        }
        
        logger.error("Performance error occurred", **log_entry)
        
        # Update performance metrics
        self._update_performance_metrics(operation, duration, error=True)
        
        return request_id
    
    def log_security_error(self, error: SecurityError, context: Optional[Dict] = None,
                          client_ip: Optional[str] = None, user_agent: Optional[str] = None):
        """Log security-related errors."""
        
        request_id = str(uuid.uuid4())
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "error_type": "SecurityError",
            "threat_type": getattr(error, 'threat_type', 'unknown'),
            "error_message": str(error),
            "severity": "high",
            "client_ip": client_ip,
            "user_agent": user_agent,
            "context": context or {},
            "security_event": True
        }
        
        logger.error("Security error occurred", **log_entry)
        
        return request_id
    
    def log_validation_error(self, error: ValidationError, field: Optional[str] = None,
                           input_data: Optional[Dict] = None):
        """Log validation errors."""
        
        request_id = str(uuid.uuid4())
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "error_type": "ValidationError",
            "field": field or getattr(error, 'field', None),
            "error_message": str(error),
            "severity": "medium",
            "input_data": input_data or {},
            "validation_error": True
        }
        
        logger.warning("Validation error occurred", **log_entry)
        
        return request_id
    
    def _categorize_error(self, error: Exception) -> str:
        """Categorize error by type."""
        if isinstance(error, SecurityError):
            return "security"
        elif isinstance(error, ValidationError):
            return "validation"
        elif isinstance(error, ResourceError):
            return "resource"
        elif isinstance(error, DatabaseError):
            return "database"
        elif isinstance(error, CacheError):
            return "cache"
        elif isinstance(error, MonitoringError):
            return "monitoring"
        elif isinstance(error, VideoProcessingError):
            return "processing"
        else:
            return "unknown"
    
    def _get_stack_trace(self, error: Exception) -> Optional[str]:
        """Get stack trace for error."""
        import traceback
        try:
            return traceback.format_exc()
        except Exception:
            return None
    
    def _update_error_stats(self, error_type: str, error_category: str):
        """Update error statistics."""
        current_time = time.time()
        
        # Update error counts
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        # Update error timestamps
        if error_category not in self.error_timestamps:
            self.error_timestamps[error_category] = []
        self.error_timestamps[error_category].append(current_time)
        
        # Keep only recent timestamps (last 24 hours)
        cutoff_time = current_time - (24 * 60 * 60)
        self.error_timestamps[error_category] = [
            ts for ts in self.error_timestamps[error_category] if ts > cutoff_time
        ]
    
    def _update_performance_metrics(self, operation: str, duration: float, error: bool = False):
        """Update performance metrics."""
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = {
                'total_operations': 0,
                'total_duration': 0.0,
                'error_count': 0,
                'success_count': 0
            }
        
        metrics = self.performance_metrics[operation]
        metrics['total_operations'] += 1
        metrics['total_duration'] += duration
        
        if error:
            metrics['error_count'] += 1
        else:
            metrics['success_count'] += 1
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        current_time = time.time()
        
        # Calculate error rates for last 24 hours
        error_rates = {}
        for category, timestamps in self.error_timestamps.items():
            error_rates[category] = len(timestamps)
        
        return {
            'error_counts': self.error_counts.copy(),
            'error_rates_24h': error_rates,
            'performance_metrics': self.performance_metrics.copy(),
            'timestamp': current_time
        }

# =============================================================================
# ERROR CORRELATION
# =============================================================================

class ErrorCorrelation:
    """Error correlation and tracking system."""
    
    def __init__(self):
        self.error_correlations = {}
        self.error_patterns = {}
    
    def correlate_errors(self, error1: Exception, error2: Exception, 
                        correlation_type: str = "sequence"):
        """Correlate two errors."""
        
        error1_id = id(error1)
        error2_id = id(error2)
        
        if error1_id not in self.error_correlations:
            self.error_correlations[error1_id] = []
        
        self.error_correlations[error1_id].append({
            'error_id': error2_id,
            'correlation_type': correlation_type,
            'timestamp': time.time()
        })
    
    def detect_error_patterns(self, errors: List[Exception]) -> Dict[str, Any]:
        """Detect patterns in error sequences."""
        
        error_types = [type(error).__name__ for error in errors]
        error_categories = [self._categorize_error(error) for error in errors]
        
        # Detect repeating patterns
        patterns = {}
        for i in range(len(error_types) - 1):
            pattern = f"{error_types[i]} -> {error_types[i + 1]}"
            if pattern not in patterns:
                patterns[pattern] = 0
            patterns[pattern] += 1
        
        # Detect error clusters
        clusters = {}
        for category in set(error_categories):
            clusters[category] = error_categories.count(category)
        
        return {
            'error_types': error_types,
            'error_categories': error_categories,
            'patterns': patterns,
            'clusters': clusters,
            'total_errors': len(errors)
        }
    
    def _categorize_error(self, error: Exception) -> str:
        """Categorize error by type."""
        if isinstance(error, SecurityError):
            return "security"
        elif isinstance(error, ValidationError):
            return "validation"
        elif isinstance(error, ResourceError):
            return "resource"
        elif isinstance(error, DatabaseError):
            return "database"
        elif isinstance(error, CacheError):
            return "cache"
        elif isinstance(error, MonitoringError):
            return "monitoring"
        elif isinstance(error, VideoProcessingError):
            return "processing"
        else:
            return "unknown"

# =============================================================================
# ERROR RECOVERY LOGGING
# =============================================================================

class ErrorRecoveryLogging:
    """Logging for error recovery operations."""
    
    def __init__(self):
        self.recovery_attempts = {}
        self.recovery_successes = {}
    
    def log_recovery_attempt(self, error: Exception, recovery_strategy: str,
                           attempt_number: int, context: Optional[Dict] = None):
        """Log error recovery attempt."""
        
        request_id = str(uuid.uuid4())
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "error_type": type(error).__name__,
            "recovery_strategy": recovery_strategy,
            "attempt_number": attempt_number,
            "context": context or {},
            "recovery_attempt": True
        }
        
        logger.info("Error recovery attempt", **log_entry)
        
        # Update recovery statistics
        error_type = type(error).__name__
        if error_type not in self.recovery_attempts:
            self.recovery_attempts[error_type] = 0
        self.recovery_attempts[error_type] += 1
        
        return request_id
    
    def log_recovery_success(self, error: Exception, recovery_strategy: str,
                           attempt_number: int, recovery_time: float):
        """Log successful error recovery."""
        
        request_id = str(uuid.uuid4())
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "error_type": type(error).__name__,
            "recovery_strategy": recovery_strategy,
            "attempt_number": attempt_number,
            "recovery_time": recovery_time,
            "recovery_success": True
        }
        
        logger.info("Error recovery successful", **log_entry)
        
        # Update recovery statistics
        error_type = type(error).__name__
        if error_type not in self.recovery_successes:
            self.recovery_successes[error_type] = 0
        self.recovery_successes[error_type] += 1
        
        return request_id
    
    def log_recovery_failure(self, error: Exception, recovery_strategy: str,
                           attempt_number: int, failure_reason: str):
        """Log failed error recovery."""
        
        request_id = str(uuid.uuid4())
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "error_type": type(error).__name__,
            "recovery_strategy": recovery_strategy,
            "attempt_number": attempt_number,
            "failure_reason": failure_reason,
            "recovery_failure": True
        }
        
        logger.error("Error recovery failed", **log_entry)
        
        return request_id
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        return {
            'recovery_attempts': self.recovery_attempts.copy(),
            'recovery_successes': self.recovery_successes.copy(),
            'success_rate': self._calculate_success_rate()
        }
    
    def _calculate_success_rate(self) -> Dict[str, float]:
        """Calculate recovery success rate by error type."""
        success_rates = {}
        
        for error_type in self.recovery_attempts:
            attempts = self.recovery_attempts[error_type]
            successes = self.recovery_successes.get(error_type, 0)
            success_rates[error_type] = (successes / attempts * 100) if attempts > 0 else 0.0
        
        return success_rates

# =============================================================================
# CONTEXT MANAGER FOR ERROR LOGGING
# =============================================================================

@contextmanager
def error_logging_context(operation: str, context: Optional[Dict] = None):
    """Context manager for error logging."""
    
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Operation started",
            operation=operation,
            request_id=request_id,
            context=context or {}
        )
        
        yield request_id
        
        duration = time.time() - start_time
        logger.info(
            "Operation completed",
            operation=operation,
            request_id=request_id,
            duration=duration,
            success=True
        )
        
    except Exception as e:
        duration = time.time() - start_time
        
        # Log error with context
        error_logger = ErrorLoggingIntegration()
        error_logger.log_error(
            error=e,
            context={
                'operation': operation,
                'duration': duration,
                **(context or {})
            },
            request_id=request_id
        )
        
        raise

# =============================================================================
# GLOBAL ERROR LOGGING INSTANCE
# =============================================================================

# Global instances for error logging
error_logging = ErrorLoggingIntegration()
error_correlation = ErrorCorrelation()
error_recovery_logging = ErrorRecoveryLogging()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ErrorLoggingIntegration',
    'ErrorCorrelation',
    'ErrorRecoveryLogging',
    'error_logging_context',
    'error_logging',
    'error_correlation',
    'error_recovery_logging'
]