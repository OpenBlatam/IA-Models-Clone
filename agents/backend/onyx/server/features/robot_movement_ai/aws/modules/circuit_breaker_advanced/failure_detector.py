"""
Failure Detector
================

Advanced failure detection.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class FailureEvent:
    """Failure event."""
    service: str
    error_type: str
    timestamp: datetime
    details: Dict[str, Any] = None


class FailureDetector:
    """Advanced failure detector."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._failures: Dict[str, deque] = {}  # service -> failure events
        self._patterns: Dict[str, Dict[str, int]] = {}  # service -> error_type -> count
    
    def record_failure(
        self,
        service: str,
        error_type: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Record failure event."""
        event = FailureEvent(
            service=service,
            error_type=error_type,
            timestamp=datetime.now(),
            details=details or {}
        )
        
        if service not in self._failures:
            self._failures[service] = deque(maxlen=self.window_size)
        
        self._failures[service].append(event)
        
        # Update patterns
        if service not in self._patterns:
            self._patterns[service] = {}
        
        self._patterns[service][error_type] = self._patterns[service].get(error_type, 0) + 1
        
        logger.warning(f"Recorded failure: {service} - {error_type}")
    
    def get_failure_rate(
        self,
        service: str,
        window_minutes: int = 5
    ) -> float:
        """Get failure rate for service."""
        if service not in self._failures:
            return 0.0
        
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent_failures = [
            f for f in self._failures[service]
            if f.timestamp > cutoff
        ]
        
        return len(recent_failures) / window_minutes if window_minutes > 0 else 0.0
    
    def detect_anomaly(self, service: str) -> bool:
        """Detect if service has anomaly."""
        failure_rate = self.get_failure_rate(service)
        
        # Anomaly if failure rate > 0.1 per minute
        return failure_rate > 0.1
    
    def get_failure_patterns(self, service: str) -> Dict[str, int]:
        """Get failure patterns for service."""
        return self._patterns.get(service, {}).copy()
    
    def get_failure_stats(self) -> Dict[str, Any]:
        """Get failure statistics."""
        return {
            "total_services": len(self._failures),
            "by_service": {
                service: {
                    "total_failures": len(failures),
                    "failure_rate": self.get_failure_rate(service),
                    "patterns": self.get_failure_patterns(service)
                }
                for service, failures in self._failures.items()
            }
        }















