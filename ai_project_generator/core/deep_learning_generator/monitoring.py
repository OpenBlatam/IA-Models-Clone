"""
Monitoring Module

Monitoring and metrics for generator usage.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class GeneratorMetrics:
    """
    Tracks metrics for generator usage.
    """
    
    def __init__(self):
        self._metrics: Dict[str, Any] = defaultdict(int)
        self._timestamps: List[datetime] = []
        self._framework_usage: Dict[str, int] = defaultdict(int)
        self._model_type_usage: Dict[str, int] = defaultdict(int)
        self._errors: List[Dict[str, Any]] = []
    
    def record_creation(
        self,
        framework: str,
        model_type: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """Record a generator creation."""
        self._metrics["total_creations"] += 1
        self._timestamps.append(datetime.now())
        self._framework_usage[framework] += 1
        
        if model_type:
            self._model_type_usage[model_type] += 1
        
        if success:
            self._metrics["successful_creations"] += 1
        else:
            self._metrics["failed_creations"] += 1
            if error:
                self._errors.append({
                    "timestamp": datetime.now().isoformat(),
                    "framework": framework,
                    "model_type": model_type,
                    "error": error
                })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        total = self._metrics["total_creations"]
        successful = self._metrics["successful_creations"]
        failed = self._metrics["failed_creations"]
        
        success_rate = (successful / total * 100) if total > 0 else 0
        
        return {
            "total_creations": total,
            "successful_creations": successful,
            "failed_creations": failed,
            "success_rate": round(success_rate, 2),
            "framework_usage": dict(self._framework_usage),
            "model_type_usage": dict(self._model_type_usage),
            "error_count": len(self._errors),
            "first_creation": self._timestamps[0].isoformat() if self._timestamps else None,
            "last_creation": self._timestamps[-1].isoformat() if self._timestamps else None
        }
    
    def get_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors."""
        return self._errors[-limit:]
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._timestamps.clear()
        self._framework_usage.clear()
        self._model_type_usage.clear()
        self._errors.clear()


# Global metrics instance
_metrics = GeneratorMetrics()


def get_metrics() -> GeneratorMetrics:
    """Get the global metrics instance."""
    return _metrics


def record_generator_creation(
    framework: str,
    model_type: Optional[str] = None,
    success: bool = True,
    error: Optional[str] = None
) -> None:
    """Record a generator creation."""
    _metrics.record_creation(framework, model_type, success, error)















