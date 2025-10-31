"""
Workflow Chain Domain Entity
===========================

Domain entity representing a workflow chain with business logic and invariants.
Follows Domain-Driven Design principles.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Set
from uuid import UUID, uuid4
import hashlib

from ..value_objects.workflow_id import WorkflowId
from ..value_objects.node_id import NodeId
from ..value_objects.workflow_status import WorkflowStatus
from ..value_objects.priority import Priority
from ..events.workflow_events import WorkflowCreated, WorkflowUpdated, WorkflowDeleted
from ..exceptions.workflow_exceptions import WorkflowDomainException


class WorkflowChain:
    """
    Workflow Chain Aggregate Root
    
    Represents a complete workflow chain with its nodes and business rules.
    This is the main aggregate root in the workflow domain.
    """
    
    def __init__(
        self,
        workflow_id: WorkflowId,
        name: str,
        description: str = "",
        status: WorkflowStatus = WorkflowStatus.DRAFT,
        settings: Optional[Dict[str, Any]] = None
    ):
        self._id = workflow_id
        self._name = name
        self._description = description
        self._status = status
        self._settings = settings or {}
        self._nodes: Dict[NodeId, 'WorkflowNode'] = {}
        self._created_at = datetime.utcnow()
        self._updated_at = datetime.utcnow()
        self._version = 1
        self._domain_events: List[Any] = []
        
        # Business invariants
        self._validate_name(name)
        self._validate_description(description)
        
        # Raise domain event
        self._add_domain_event(WorkflowCreated(
            workflow_id=self._id,
            name=self._name,
            created_at=self._created_at
        ))
    
    @property
    def id(self) -> WorkflowId:
        """Get workflow ID"""
        return self._id
    
    @property
    def name(self) -> str:
        """Get workflow name"""
        return self._name
    
    @property
    def description(self) -> str:
        """Get workflow description"""
        return self._description
    
    @property
    def status(self) -> WorkflowStatus:
        """Get workflow status"""
        return self._status
    
    @property
    def settings(self) -> Dict[str, Any]:
        """Get workflow settings"""
        return self._settings.copy()
    
    @property
    def nodes(self) -> Dict[NodeId, 'WorkflowNode']:
        """Get workflow nodes"""
        return self._nodes.copy()
    
    @property
    def created_at(self) -> datetime:
        """Get creation timestamp"""
        return self._created_at
    
    @property
    def updated_at(self) -> datetime:
        """Get last update timestamp"""
        return self._updated_at
    
    @property
    def version(self) -> int:
        """Get version for optimistic locking"""
        return self._version
    
    @property
    def domain_events(self) -> List[Any]:
        """Get domain events"""
        return self._domain_events.copy()
    
    def change_name(self, new_name: str) -> None:
        """Change workflow name with validation"""
        self._validate_name(new_name)
        old_name = self._name
        self._name = new_name
        self._updated_at = datetime.utcnow()
        self._version += 1
        
        self._add_domain_event(WorkflowUpdated(
            workflow_id=self._id,
            field="name",
            old_value=old_name,
            new_value=new_name,
            updated_at=self._updated_at
        ))
    
    def change_description(self, new_description: str) -> None:
        """Change workflow description with validation"""
        self._validate_description(new_description)
        old_description = self._description
        self._description = new_description
        self._updated_at = datetime.utcnow()
        self._version += 1
        
        self._add_domain_event(WorkflowUpdated(
            workflow_id=self._id,
            field="description",
            old_value=old_description,
            new_value=new_description,
            updated_at=self._updated_at
        ))
    
    def change_status(self, new_status: WorkflowStatus) -> None:
        """Change workflow status with validation"""
        self._validate_status_transition(self._status, new_status)
        old_status = self._status
        self._status = new_status
        self._updated_at = datetime.utcnow()
        self._version += 1
        
        self._add_domain_event(WorkflowUpdated(
            workflow_id=self._id,
            field="status",
            old_value=old_status.value,
            new_value=new_status.value,
            updated_at=self._updated_at
        ))
    
    def update_settings(self, new_settings: Dict[str, Any]) -> None:
        """Update workflow settings"""
        self._validate_settings(new_settings)
        old_settings = self._settings.copy()
        self._settings.update(new_settings)
        self._updated_at = datetime.utcnow()
        self._version += 1
        
        self._add_domain_event(WorkflowUpdated(
            workflow_id=self._id,
            field="settings",
            old_value=old_settings,
            new_value=self._settings.copy(),
            updated_at=self._updated_at
        ))
    
    def add_node(self, node: 'WorkflowNode') -> None:
        """Add a node to the workflow"""
        self._validate_node_addition(node)
        
        if node.id in self._nodes:
            raise WorkflowDomainException(f"Node {node.id} already exists in workflow {self._id}")
        
        self._nodes[node.id] = node
        self._updated_at = datetime.utcnow()
        self._version += 1
        
        # Update node relationships
        if node.parent_id:
            parent_node = self._nodes.get(node.parent_id)
            if parent_node:
                parent_node.add_child(node.id)
        
        self._add_domain_event(WorkflowUpdated(
            workflow_id=self._id,
            field="nodes",
            old_value=None,
            new_value=node.id,
            updated_at=self._updated_at
        ))
    
    def remove_node(self, node_id: NodeId) -> None:
        """Remove a node from the workflow"""
        if node_id not in self._nodes:
            raise WorkflowDomainException(f"Node {node_id} not found in workflow {self._id}")
        
        node = self._nodes[node_id]
        
        # Remove from parent's children
        if node.parent_id and node.parent_id in self._nodes:
            parent_node = self._nodes[node.parent_id]
            parent_node.remove_child(node_id)
        
        # Remove all children
        for child_id in node.children_ids:
            if child_id in self._nodes:
                child_node = self._nodes[child_id]
                child_node.parent_id = None
        
        del self._nodes[node_id]
        self._updated_at = datetime.utcnow()
        self._version += 1
        
        self._add_domain_event(WorkflowUpdated(
            workflow_id=self._id,
            field="nodes",
            old_value=node_id,
            new_value=None,
            updated_at=self._updated_at
        ))
    
    def get_node(self, node_id: NodeId) -> Optional['WorkflowNode']:
        """Get a node by ID"""
        return self._nodes.get(node_id)
    
    def get_nodes_by_title(self, title: str) -> List['WorkflowNode']:
        """Get nodes by title"""
        return [node for node in self._nodes.values() if node.title == title]
    
    def get_root_nodes(self) -> List['WorkflowNode']:
        """Get root nodes (nodes without parent)"""
        return [node for node in self._nodes.values() if node.parent_id is None]
    
    def get_leaf_nodes(self) -> List['WorkflowNode']:
        """Get leaf nodes (nodes without children)"""
        return [node for node in self._nodes.values() if not node.children_ids]
    
    def calculate_depth(self) -> int:
        """Calculate maximum depth of the workflow"""
        if not self._nodes:
            return 0
        
        def _calculate_node_depth(node_id: NodeId, visited: Set[NodeId] = None) -> int:
            if visited is None:
                visited = set()
            
            if node_id in visited:
                return 0  # Circular reference
            
            visited.add(node_id)
            node = self._nodes.get(node_id)
            
            if not node or not node.children_ids:
                return 0
            
            max_child_depth = 0
            for child_id in node.children_ids:
                child_depth = _calculate_node_depth(child_id, visited.copy())
                max_child_depth = max(max_child_depth, child_depth)
            
            return 1 + max_child_depth
        
        root_nodes = self.get_root_nodes()
        if not root_nodes:
            return 0
        
        max_depth = 0
        for root_node in root_nodes:
            depth = _calculate_node_depth(root_node.id)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get workflow statistics"""
        total_nodes = len(self._nodes)
        total_words = sum(node.word_count for node in self._nodes.values())
        total_characters = sum(node.character_count for node in self._nodes.values())
        
        quality_scores = [node.quality_score for node in self._nodes.values() if node.quality_score is not None]
        average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return {
            "total_nodes": total_nodes,
            "total_words": total_words,
            "total_characters": total_characters,
            "average_quality": average_quality,
            "max_depth": self.calculate_depth(),
            "root_nodes": len(self.get_root_nodes()),
            "leaf_nodes": len(self.get_leaf_nodes()),
            "status": self._status.value,
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat(),
            "version": self._version
        }
    
    def delete(self) -> None:
        """Delete the workflow"""
        if self._status == WorkflowStatus.DELETED:
            raise WorkflowDomainException(f"Workflow {self._id} is already deleted")
        
        self._status = WorkflowStatus.DELETED
        self._updated_at = datetime.utcnow()
        self._version += 1
        
        self._add_domain_event(WorkflowDeleted(
            workflow_id=self._id,
            deleted_at=self._updated_at
        ))
    
    def clear_domain_events(self) -> None:
        """Clear domain events after they've been processed"""
        self._domain_events.clear()
    
    def _validate_name(self, name: str) -> None:
        """Validate workflow name"""
        if not name or not name.strip():
            raise WorkflowDomainException("Workflow name cannot be empty")
        
        if len(name) > 255:
            raise WorkflowDomainException("Workflow name cannot exceed 255 characters")
    
    def _validate_description(self, description: str) -> None:
        """Validate workflow description"""
        if len(description) > 1000:
            raise WorkflowDomainException("Workflow description cannot exceed 1000 characters")
    
    def _validate_status_transition(self, current_status: WorkflowStatus, new_status: WorkflowStatus) -> None:
        """Validate status transition"""
        valid_transitions = {
            WorkflowStatus.DRAFT: [WorkflowStatus.ACTIVE, WorkflowStatus.DELETED],
            WorkflowStatus.ACTIVE: [WorkflowStatus.PAUSED, WorkflowStatus.COMPLETED, WorkflowStatus.DELETED],
            WorkflowStatus.PAUSED: [WorkflowStatus.ACTIVE, WorkflowStatus.DELETED],
            WorkflowStatus.COMPLETED: [WorkflowStatus.ACTIVE, WorkflowStatus.DELETED],
            WorkflowStatus.ERROR: [WorkflowStatus.DRAFT, WorkflowStatus.ACTIVE, WorkflowStatus.DELETED],
            WorkflowStatus.DELETED: []  # Cannot transition from deleted
        }
        
        if new_status not in valid_transitions.get(current_status, []):
            raise WorkflowDomainException(
                f"Invalid status transition from {current_status.value} to {new_status.value}"
            )
    
    def _validate_settings(self, settings: Dict[str, Any]) -> None:
        """Validate workflow settings"""
        # Add specific validation rules for settings
        if "max_nodes" in settings and settings["max_nodes"] <= 0:
            raise WorkflowDomainException("max_nodes must be positive")
        
        if "timeout" in settings and settings["timeout"] <= 0:
            raise WorkflowDomainException("timeout must be positive")
    
    def _validate_node_addition(self, node: 'WorkflowNode') -> None:
        """Validate node addition"""
        # Check if adding this node would exceed max nodes limit
        max_nodes = self._settings.get("max_nodes", 1000)
        if len(self._nodes) >= max_nodes:
            raise WorkflowDomainException(f"Workflow cannot have more than {max_nodes} nodes")
        
        # Validate parent exists if specified
        if node.parent_id and node.parent_id not in self._nodes:
            raise WorkflowDomainException(f"Parent node {node.parent_id} not found")
    
    def _add_domain_event(self, event: Any) -> None:
        """Add domain event"""
        self._domain_events.append(event)
    
    def __eq__(self, other: object) -> bool:
        """Check equality"""
        if not isinstance(other, WorkflowChain):
            return False
        return self._id == other._id
    
    def __hash__(self) -> int:
        """Get hash"""
        return hash(self._id)
    
    def __repr__(self) -> str:
        """String representation"""
        return f"WorkflowChain(id={self._id}, name='{self._name}', status={self._status.value})"


# Import here to avoid circular imports
from .workflow_node import WorkflowNode




