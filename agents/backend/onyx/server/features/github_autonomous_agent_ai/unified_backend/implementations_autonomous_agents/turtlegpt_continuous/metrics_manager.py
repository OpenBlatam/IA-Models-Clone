"""
Metrics Manager

Tracks and manages agent metrics.
"""

from typing import Dict, Any
import logging
from datetime import datetime

from .models import AgentMetrics

logger = logging.getLogger(__name__)


class MetricsManager:
    """Manages agent metrics."""
    
    def __init__(self):
        """Initialize metrics manager."""
        self.metrics = AgentMetrics()
    
    def record_llm_call(self, tokens_used: int, response_time: float) -> None:
        """
        Record an LLM call.
        
        Args:
            tokens_used: Tokens used in the call
            response_time: Response time in seconds
        """
        self.metrics.total_llm_calls += 1
        self.metrics.total_tokens_used += tokens_used
        self.metrics.last_activity = datetime.now()
        
        # Update average response time
        if self.metrics.total_llm_calls > 0:
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (self.metrics.total_llm_calls - 1) + response_time) /
                self.metrics.total_llm_calls
            )
    
    def record_task_processed(self) -> None:
        """Record a task being processed."""
        self.metrics.total_tasks_processed += 1
        self.metrics.last_activity = datetime.now()
    
    def record_task_completed(self) -> None:
        """Record a task completion."""
        self.metrics.total_tasks_completed += 1
        self.metrics.last_activity = datetime.now()
    
    def record_task_failed(self) -> None:
        """Record a task failure."""
        self.metrics.total_tasks_failed += 1
        self.metrics.last_activity = datetime.now()
    
    def record_error(self) -> None:
        """Record an error."""
        self.metrics.errors_count += 1
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.metrics.last_activity = datetime.now()
    
    def start_tracking(self) -> None:
        """Start tracking metrics."""
        self.metrics.start_time = datetime.now()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics as dictionary."""
        return self.metrics.to_dict()
    
    def print_final_metrics(self) -> None:
        """Print final metrics."""
        runtime = datetime.now() - self.metrics.start_time
        
        logger.info("=" * 60)
        logger.info("Final Agent Metrics")
        logger.info("=" * 60)
        logger.info(f"Total runtime: {runtime}")
        logger.info(f"Total tasks processed: {self.metrics.total_tasks_processed}")
        logger.info(f"Total tasks completed: {self.metrics.total_tasks_completed}")
        logger.info(f"Total tasks failed: {self.metrics.total_tasks_failed}")
        logger.info(f"Total LLM calls: {self.metrics.total_llm_calls}")
        logger.info(f"Total tokens used: {self.metrics.total_tokens_used}")
        logger.info(f"Average response time: {self.metrics.average_response_time:.2f}s")
        logger.info(f"Success rate: {self.metrics.to_dict()['success_rate']:.1f}%")
        logger.info(f"Errors: {self.metrics.errors_count}")
        logger.info("=" * 60)



