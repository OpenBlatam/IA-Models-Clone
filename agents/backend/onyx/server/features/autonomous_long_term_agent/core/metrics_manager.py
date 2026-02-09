"""
Metrics Manager for Autonomous Agent
Handles collection and management of agent metrics
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tokens_used: int = 0
    uptime_seconds: float = 0.0
    last_activity: Optional[datetime] = None
    reasoning_calls: int = 0
    knowledge_retrievals: int = 0
    errors_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_tokens_used": self.total_tokens_used,
            "uptime_seconds": self.uptime_seconds,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "reasoning_calls": self.reasoning_calls,
            "knowledge_retrievals": self.knowledge_retrievals,
            "errors_count": self.errors_count,
        }
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_tokens_used = 0
        self.uptime_seconds = 0.0
        self.last_activity = None
        self.reasoning_calls = 0
        self.knowledge_retrievals = 0
        self.errors_count = 0


class MetricsManager:
    """
    Manages agent metrics collection and reporting
    """
    
    def __init__(self):
        self._metrics = AgentMetrics()
        self._start_time: Optional[datetime] = None
    
    def start_tracking(self) -> None:
        """Start tracking metrics"""
        self._start_time = datetime.utcnow()
        self._metrics.last_activity = datetime.utcnow()
        logger.debug("Metrics tracking started")
    
    def record_task_completed(self, tokens_used: int = 0) -> None:
        """Record a completed task"""
        self._metrics.tasks_completed += 1
        self._metrics.total_tokens_used += tokens_used
        self._metrics.last_activity = datetime.utcnow()
    
    def record_task_failed(self) -> None:
        """Record a failed task"""
        self._metrics.tasks_failed += 1
        self._metrics.errors_count += 1
        self._metrics.last_activity = datetime.utcnow()
    
    def record_reasoning_call(self) -> None:
        """Record a reasoning call"""
        self._metrics.reasoning_calls += 1
    
    def record_knowledge_retrieval(self) -> None:
        """Record a knowledge retrieval"""
        self._metrics.knowledge_retrievals += 1
    
    def record_error(self) -> None:
        """Record an error"""
        self._metrics.errors_count += 1
    
    def update_uptime(self) -> None:
        """Update uptime based on start time"""
        if self._start_time:
            self._metrics.uptime_seconds = (
                datetime.utcnow() - self._start_time
            ).total_seconds()
        self._metrics.last_activity = datetime.utcnow()
    
    def get_metrics(self) -> AgentMetrics:
        """Get current metrics"""
        self.update_uptime()
        return self._metrics
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary"""
        self.update_uptime()
        return self._metrics.to_dict()
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self._metrics.reset()
        self._start_time = datetime.utcnow()
        logger.info("Metrics reset")
    
    def get_success_rate(self) -> float:
        """Calculate success rate"""
        total = self._metrics.tasks_completed + self._metrics.tasks_failed
        if total == 0:
            return 0.0
        return self._metrics.tasks_completed / total
    
    def get_average_tokens_per_task(self) -> float:
        """Calculate average tokens per task"""
        if self._metrics.tasks_completed == 0:
            return 0.0
        return self._metrics.total_tokens_used / self._metrics.tasks_completed




