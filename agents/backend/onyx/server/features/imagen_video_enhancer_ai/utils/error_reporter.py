"""
Error Reporter
==============

Advanced error reporting and tracking utilities.
"""

import logging
import traceback
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ErrorReporter:
    """
    Advanced error reporter with tracking and analysis.
    """
    
    def __init__(self, error_log_dir: str = "error_logs"):
        """
        Initialize error reporter.
        
        Args:
            error_log_dir: Directory for error logs
        """
        self.error_log_dir = Path(error_log_dir)
        self.error_log_dir.mkdir(parents=True, exist_ok=True)
        self.error_counts: Dict[str, int] = {}
        self.recent_errors: List[Dict[str, Any]] = []
        self.max_recent_errors = 100
    
    def report_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "error"
    ) -> str:
        """
        Report an error with context.
        
        Args:
            error: Exception object
            context: Additional context
            severity: Error severity (error, warning, critical)
            
        Returns:
            Error ID
        """
        error_id = f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        error_info = {
            "error_id": error_id,
            "timestamp": datetime.now().isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "severity": severity,
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        # Log error
        if severity == "critical":
            logger.critical(f"[{error_id}] {error}", exc_info=True)
        elif severity == "warning":
            logger.warning(f"[{error_id}] {error}", exc_info=True)
        else:
            logger.error(f"[{error_id}] {error}", exc_info=True)
        
        # Save to file
        self._save_error(error_info)
        
        # Track error
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Add to recent errors
        self.recent_errors.append(error_info)
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors.pop(0)
        
        return error_id
    
    def _save_error(self, error_info: Dict[str, Any]):
        """Save error to file."""
        error_file = self.error_log_dir / f"{error_info['error_id']}.json"
        
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(error_info, f, indent=2, ensure_ascii=False)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get error summary statistics.
        
        Returns:
            Error summary dictionary
        """
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_types": self.error_counts,
            "recent_errors_count": len(self.recent_errors),
            "recent_errors": self.recent_errors[-10:]  # Last 10 errors
        }
    
    def get_errors_by_type(self, error_type: str) -> List[Dict[str, Any]]:
        """
        Get all errors of a specific type.
        
        Args:
            error_type: Error type name
            
        Returns:
            List of error information
        """
        return [e for e in self.recent_errors if e["type"] == error_type]
    
    def clear_errors(self):
        """Clear error tracking."""
        self.error_counts.clear()
        self.recent_errors.clear()


def create_error_context(**kwargs) -> Dict[str, Any]:
    """
    Create error context dictionary.
    
    Args:
        **kwargs: Context key-value pairs
        
    Returns:
        Context dictionary
    """
    return {
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }




