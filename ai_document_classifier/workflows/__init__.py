"""
Workflows Package
=================

Advanced workflow automation system for document processing
with conditional logic, parallel execution, and integration capabilities.
"""

from .workflow_engine import (
    WorkflowEngine,
    WorkflowDefinition,
    WorkflowTask,
    WorkflowExecution,
    WorkflowCondition,
    WorkflowStatus,
    TaskStatus,
    ConditionType
)

__all__ = [
    "WorkflowEngine",
    "WorkflowDefinition",
    "WorkflowTask", 
    "WorkflowExecution",
    "WorkflowCondition",
    "WorkflowStatus",
    "TaskStatus",
    "ConditionType"
]



























