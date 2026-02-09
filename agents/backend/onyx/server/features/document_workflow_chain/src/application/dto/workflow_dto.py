"""
Workflow Data Transfer Objects
=============================

DTOs for workflow-related data transfer between layers.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from ...domain.value_objects.workflow_status import WorkflowStatus
from ...domain.value_objects.priority import Priority


class WorkflowStatusDto(str, Enum):
    """Workflow status DTO"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"
    DELETED = "deleted"


@dataclass
class WorkflowDto:
    """Workflow data transfer object"""
    workflow_id: str
    name: str
    description: str
    status: WorkflowStatusDto
    settings: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: int
    statistics: Dict[str, Any]
    
    @classmethod
    def from_domain(cls, workflow) -> WorkflowDto:
        """Create DTO from domain entity"""
        return cls(
            workflow_id=str(workflow.id),
            name=workflow.name,
            description=workflow.description,
            status=WorkflowStatusDto(workflow.status.value),
            settings=workflow.settings,
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
            version=workflow.version,
            statistics=workflow.get_statistics()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "settings": self.settings,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "statistics": self.statistics
        }


@dataclass
class CreateWorkflowDto:
    """Create workflow DTO"""
    name: str
    description: str
    settings: Optional[Dict[str, Any]] = None
    
    def to_domain_request(self):
        """Convert to domain request"""
        from .create_workflow_use_case import CreateWorkflowRequest
        return CreateWorkflowRequest(
            name=self.name,
            description=self.description,
            settings=self.settings
        )


@dataclass
class UpdateWorkflowDto:
    """Update workflow DTO"""
    workflow_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[WorkflowStatusDto] = None
    settings: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowListDto:
    """Workflow list DTO"""
    workflows: List[WorkflowDto]
    total: int
    limit: int
    offset: int
    has_more: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "workflows": [workflow.to_dict() for workflow in self.workflows],
            "total": self.total,
            "limit": self.limit,
            "offset": self.offset,
            "has_more": self.has_more
        }


@dataclass
class WorkflowStatisticsDto:
    """Workflow statistics DTO"""
    total_workflows: int
    active_workflows: int
    paused_workflows: int
    completed_workflows: int
    error_workflows: int
    total_nodes: int
    average_nodes_per_workflow: float
    most_used_tags: List[Dict[str, Any]]
    workflows_by_status: Dict[str, int]
    workflows_by_priority: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_workflows": self.total_workflows,
            "active_workflows": self.active_workflows,
            "paused_workflows": self.paused_workflows,
            "completed_workflows": self.completed_workflows,
            "error_workflows": self.error_workflows,
            "total_nodes": self.total_nodes,
            "average_nodes_per_workflow": self.average_nodes_per_workflow,
            "most_used_tags": self.most_used_tags,
            "workflows_by_status": self.workflows_by_status,
            "workflows_by_priority": self.workflows_by_priority
        }




