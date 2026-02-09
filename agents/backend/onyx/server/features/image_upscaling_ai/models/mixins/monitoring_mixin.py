"""
Monitoring Mixin

Contains monitoring and logging functionality.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class MonitoringMixin:
    """
    Mixin providing monitoring and logging functionality.
    
    This mixin contains:
    - Performance monitoring
    - Error tracking
    - Usage statistics
    - Health checks
    - Activity logging
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize monitoring mixin."""
        super().__init__(*args, **kwargs)
        if not hasattr(self, '_monitoring'):
            self._monitoring = {
                "operations": deque(maxlen=1000),
                "errors": deque(maxlen=100),
                "performance": {},
            }
    
    def log_operation(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log an operation.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            success: Whether operation succeeded
            metadata: Optional metadata
        """
        if not hasattr(self, '_monitoring'):
            self._monitoring = {
                "operations": deque(maxlen=1000),
                "errors": deque(maxlen=100),
                "performance": {},
            }
        
        operation_log = {
            "operation": operation,
            "duration": duration,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
        self._monitoring["operations"].append(operation_log)
        
        if not success:
            self._monitoring["errors"].append(operation_log)
        
        # Update performance stats
        if operation not in self._monitoring["performance"]:
            self._monitoring["performance"][operation] = {
                "count": 0,
                "total_time": 0.0,
                "success_count": 0,
                "error_count": 0,
            }
        
        perf = self._monitoring["performance"][operation]
        perf["count"] += 1
        perf["total_time"] += duration
        if success:
            perf["success_count"] += 1
        else:
            perf["error_count"] += 1
    
    def get_operation_stats(
        self,
        operation: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics for operations.
        
        Args:
            operation: Specific operation name or None for all
            
        Returns:
            Dictionary with statistics
        """
        if not hasattr(self, '_monitoring'):
            return {}
        
        if operation:
            if operation in self._monitoring["performance"]:
                perf = self._monitoring["performance"][operation]
                return {
                    "operation": operation,
                    "count": perf["count"],
                    "total_time": perf["total_time"],
                    "avg_time": perf["total_time"] / perf["count"] if perf["count"] > 0 else 0.0,
                    "success_count": perf["success_count"],
                    "error_count": perf["error_count"],
                    "success_rate": perf["success_count"] / perf["count"] if perf["count"] > 0 else 0.0,
                }
            return {}
        
        # All operations
        return {
            op: {
                "count": perf["count"],
                "total_time": perf["total_time"],
                "avg_time": perf["total_time"] / perf["count"] if perf["count"] > 0 else 0.0,
                "success_rate": perf["success_count"] / perf["count"] if perf["count"] > 0 else 0.0,
            }
            for op, perf in self._monitoring["performance"].items()
        }
    
    def get_recent_errors(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent errors.
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List of recent error logs
        """
        if not hasattr(self, '_monitoring'):
            return []
        
        return list(self._monitoring["errors"])[-limit:]
    
    def get_recent_operations(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent operations.
        
        Args:
            limit: Maximum number of operations to return
            
        Returns:
            List of recent operation logs
        """
        if not hasattr(self, '_monitoring'):
            return []
        
        return list(self._monitoring["operations"])[-limit:]
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the system.
        
        Returns:
            Dictionary with health information
        """
        if not hasattr(self, '_monitoring'):
            return {"status": "unknown", "message": "Monitoring not initialized"}
        
        # Check recent error rate
        recent_ops = list(self._monitoring["operations"])[-100:]
        if recent_ops:
            error_rate = sum(1 for op in recent_ops if not op["success"]) / len(recent_ops)
        else:
            error_rate = 0.0
        
        # Determine status
        if error_rate > 0.5:
            status = "unhealthy"
            message = "High error rate detected"
        elif error_rate > 0.2:
            status = "degraded"
            message = "Elevated error rate"
        else:
            status = "healthy"
            message = "System operating normally"
        
        return {
            "status": status,
            "message": message,
            "error_rate": error_rate,
            "total_operations": len(self._monitoring["operations"]),
            "total_errors": len(self._monitoring["errors"]),
            "performance_stats": self.get_operation_stats(),
        }
    
    def clear_monitoring_data(self):
        """Clear all monitoring data."""
        if hasattr(self, '_monitoring'):
            self._monitoring = {
                "operations": deque(maxlen=1000),
                "errors": deque(maxlen=100),
                "performance": {},
            }
            logger.info("Monitoring data cleared")


