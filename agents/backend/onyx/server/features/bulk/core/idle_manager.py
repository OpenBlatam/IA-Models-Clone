"""
Idle Manager - Handles idle mode detection and management
========================================================

Manages idle mode detection and transitions.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Any

from .processor_config import ProcessorConfig

logger = logging.getLogger(__name__)


class IdleManager:
    """Manages idle mode detection and transitions."""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.idle_timeout = timedelta(minutes=config.idle_timeout_minutes)
    
    def should_enter_idle_mode(self, last_activity: Optional[datetime]) -> bool:
        """Check if should enter idle mode based on last activity."""
        if not last_activity:
            return False
        
        return datetime.now() - last_activity > self.idle_timeout
    
    def record_activity(self, metrics: Any) -> None:
        """Record current activity in metrics."""
        if hasattr(metrics, 'record_activity'):
            metrics.record_activity()
        elif hasattr(metrics, 'last_activity'):
            metrics.last_activity = datetime.now()

