"""
BUL Workflow Module
==================

Motor de flujos de trabajo para automatización de procesos empresariales.
"""

from .workflow_engine import (
    WorkflowEngine,
    Workflow,
    WorkflowTask,
    WorkflowTrigger,
    WorkflowStatus,
    TaskStatus,
    TaskType,
    TriggerType,
    TaskExecutor,
    DocumentGenerationExecutor,
    ContentReviewExecutor,
    ApprovalExecutor,
    NotificationExecutor,
    get_global_workflow_engine
)

__all__ = [
    "WorkflowEngine",
    "Workflow",
    "WorkflowTask",
    "WorkflowTrigger",
    "WorkflowStatus",
    "TaskStatus",
    "TaskType",
    "TriggerType",
    "TaskExecutor",
    "DocumentGenerationExecutor",
    "ContentReviewExecutor",
    "ApprovalExecutor",
    "NotificationExecutor",
    "get_global_workflow_engine"
]
























