"""
Processor Configuration - Configuration constants for continuous processor
=========================================================================

Centralized configuration constants for the continuous processor.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessorConfig:
    """Configuration constants for continuous processor."""
    idle_timeout_minutes: int = 30
    maintenance_sleep_seconds: int = 10
    loop_sleep_seconds: int = 1
    retry_sleep_seconds: int = 5
    task_monitor_sleep_seconds: float = 0.1
    max_completed_tasks_to_keep: int = 100


DEFAULT_CONFIG = ProcessorConfig()






