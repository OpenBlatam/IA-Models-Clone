"""
Processor Metrics - Metrics tracking for continuous processor
=============================================================

Handles metrics collection and calculation for the continuous processor.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ProcessingMetrics:
    """Metrics for continuous processing."""
    start_time: datetime
    total_queries_processed: int = 0
    total_documents_generated: int = 0
    average_processing_time: float = 0.0
    success_rate: float = 0.0
    errors_count: int = 0
    last_activity: Optional[datetime] = None
    
    def update_from_stats(self, stats: dict) -> None:
        """Update metrics from processor stats."""
        total_requests = stats.get("total_requests", 0)
        completed_tasks = stats.get("completed_tasks", 0)
        failed_tasks = stats.get("total_documents_failed", 0)
        avg_time = stats.get("average_processing_time", 0.0)
        
        self.total_queries_processed = total_requests
        self.total_documents_generated = stats.get("total_documents_generated", 0)
        self.average_processing_time = avg_time
        
        if total_requests > 0:
            self.success_rate = (completed_tasks / total_requests) * 100
        
        self.errors_count = failed_tasks
    
    def record_activity(self) -> None:
        """Record current activity timestamp."""
        self.last_activity = datetime.now()
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "total_queries_processed": self.total_queries_processed,
            "total_documents_generated": self.total_documents_generated,
            "average_processing_time": self.average_processing_time,
            "success_rate": self.success_rate,
            "errors_count": self.errors_count,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None
        }






