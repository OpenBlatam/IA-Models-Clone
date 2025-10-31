"""
Workflow Domain Events
=====================

Domain events for workflow-related operations.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from ..value_objects.workflow_id import WorkflowId


@dataclass(frozen=True)
class WorkflowCreated:
    """Event raised when a workflow is created"""
    workflow_id: WorkflowId
    name: str
    created_at: datetime
    event_type: str = "workflow.created"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "workflow_id": str(self.workflow_id),
            "name": self.name,
            "created_at": self.created_at.isoformat()
        }


@dataclass(frozen=True)
class WorkflowUpdated:
    """Event raised when a workflow is updated"""
    workflow_id: WorkflowId
    field: str
    old_value: Any
    new_value: Any
    updated_at: datetime
    event_type: str = "workflow.updated"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "workflow_id": str(self.workflow_id),
            "field": self.field,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "updated_at": self.updated_at.isoformat()
        }


@dataclass(frozen=True)
class WorkflowDeleted:
    """Event raised when a workflow is deleted"""
    workflow_id: WorkflowId
    deleted_at: datetime
    event_type: str = "workflow.deleted"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "workflow_id": str(self.workflow_id),
            "deleted_at": self.deleted_at.isoformat()
        }


@dataclass(frozen=True)
class WorkflowStatusChanged:
    """Event raised when workflow status changes"""
    workflow_id: WorkflowId
    old_status: str
    new_status: str
    changed_at: datetime
    event_type: str = "workflow.status_changed"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "workflow_id": str(self.workflow_id),
            "old_status": self.old_status,
            "new_status": self.new_status,
            "changed_at": self.changed_at.isoformat()
        }


@dataclass(frozen=True)
class WorkflowNodeAdded:
    """Event raised when a node is added to workflow"""
    workflow_id: WorkflowId
    node_id: str
    added_at: datetime
    event_type: str = "workflow.node_added"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "workflow_id": str(self.workflow_id),
            "node_id": str(self.node_id),
            "added_at": self.added_at.isoformat()
        }


@dataclass(frozen=True)
class WorkflowNodeRemoved:
    """Event raised when a node is removed from workflow"""
    workflow_id: WorkflowId
    node_id: str
    removed_at: datetime
    event_type: str = "workflow.node_removed"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "workflow_id": str(self.workflow_id),
            "node_id": str(self.node_id),
            "removed_at": self.removed_at.isoformat()
        }




