"""
Task Types and Definitions
==========================

Type definitions for background task processing.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from uuid import uuid4

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class TaskType(Enum):
    """Task type enumeration"""
    AGENT_EXECUTION = "agent_execution"
    WORKFLOW_EXECUTION = "workflow_execution"
    DOCUMENT_GENERATION = "document_generation"
    DATA_PROCESSING = "data_processing"
    NOTIFICATION = "notification"
    CLEANUP = "cleanup"
    REPORTING = "reporting"
    CUSTOM = "custom"

@dataclass
class Task:
    """Background task definition"""
    task_id: str
    task_type: TaskType
    name: str
    description: str
    function_name: str
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    timeout: int = 300  # seconds
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScheduledTask:
    """Scheduled task definition"""
    task_id: str
    task: Task
    schedule: str  # Cron expression or datetime
    next_run: datetime
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

class TaskCallback:
    """Task callback definition"""
    
    def __init__(self, callback: Callable, event: str = "completed"):
        self.callback = callback
        self.event = event  # completed, failed, retrying
    
    async def execute(self, task: Task, result: Optional[TaskResult] = None):
        """Execute the callback."""
        try:
            if asyncio.iscoroutinefunction(self.callback):
                await self.callback(task, result)
            else:
                self.callback(task, result)
        except Exception as e:
            logger.error(f"Task callback failed: {str(e)}")

# Task statistics
@dataclass
class TaskStatistics:
    """Task execution statistics"""
    total_tasks: int = 0
    pending_tasks: int = 0
    running_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    average_execution_time: float = 0.0
    success_rate: float = 0.0
    tasks_by_type: Dict[str, int] = field(default_factory=dict)
    tasks_by_priority: Dict[str, int] = field(default_factory=dict)
