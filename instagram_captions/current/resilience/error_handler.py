"""
Error Handler for Instagram Captions API v10.0

Optimized error handling and alerting.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ErrorHandler:
    """Optimized error handling system."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.error_history: deque = deque(maxlen=max_history)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.severity_counts: Dict[str, int] = defaultdict(int)
    
    def log_error(self, error: Exception, context: str = "", 
                  severity: str = "medium", user_id: Optional[str] = None) -> str:
        """Log an error with context."""
        error_id = f"err_{int(time.time() * 1000)}"
        
        error_record = {
            "error_id": error_id,
            "timestamp": time.time(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "severity": severity,
            "user_id": user_id
        }
        
        self.error_history.append(error_record)
        self.error_counts[type(error).__name__] += 1
        self.severity_counts[severity] += 1
        
        # Log to standard logger
        log_level = getattr(logging, severity.upper(), logging.INFO)
        logger.log(log_level, f"Error {error_id}: {error} in {context}")
        
        return error_id
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics."""
        current_time = time.time()
        one_hour_ago = current_time - 3600
        
        recent_errors = [e for e in self.error_history if e['timestamp'] > one_hour_ago]
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors_1h": len(recent_errors),
            "error_type_distribution": dict(self.error_counts),
            "severity_distribution": dict(self.severity_counts),
            "top_error_types": sorted(
                self.error_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }
    
    def clear_history(self):
        """Clear error history."""
        self.error_history.clear()
        self.error_counts.clear()
        self.severity_counts.clear()






