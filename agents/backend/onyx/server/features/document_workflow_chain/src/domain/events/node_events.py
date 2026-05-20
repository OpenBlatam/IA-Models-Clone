"""
Node Domain Events
=================

Domain events for node-related operations.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..value_objects.node_id import NodeId


@dataclass(frozen=True)
class NodeCreated:
    """Event raised when a node is created"""
    node_id: NodeId
    title: str
    created_at: datetime
    event_type: str = "node.created"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "node_id": str(self.node_id),
            "title": self.title,
            "created_at": self.created_at.isoformat()
        }


@dataclass(frozen=True)
class NodeUpdated:
    """Event raised when a node is updated"""
    node_id: NodeId
    field: str
    old_value: Any
    new_value: Any
    updated_at: datetime
    event_type: str = "node.updated"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "node_id": str(self.node_id),
            "field": self.field,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "updated_at": self.updated_at.isoformat()
        }


@dataclass(frozen=True)
class NodeDeleted:
    """Event raised when a node is deleted"""
    node_id: NodeId
    deleted_at: datetime
    event_type: str = "node.deleted"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "node_id": str(self.node_id),
            "deleted_at": self.deleted_at.isoformat()
        }


@dataclass(frozen=True)
class NodeContentUpdated:
    """Event raised when node content is updated"""
    node_id: NodeId
    old_content: str
    new_content: str
    updated_at: datetime
    event_type: str = "node.content_updated"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "node_id": str(self.node_id),
            "old_content": self.old_content,
            "new_content": self.new_content,
            "updated_at": self.updated_at.isoformat()
        }


@dataclass(frozen=True)
class NodePriorityChanged:
    """Event raised when node priority changes"""
    node_id: NodeId
    old_priority: int
    new_priority: int
    changed_at: datetime
    event_type: str = "node.priority_changed"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "node_id": str(self.node_id),
            "old_priority": self.old_priority,
            "new_priority": self.new_priority,
            "changed_at": self.changed_at.isoformat()
        }


@dataclass(frozen=True)
class NodeTagAdded:
    """Event raised when a tag is added to node"""
    node_id: NodeId
    tag: str
    added_at: datetime
    event_type: str = "node.tag_added"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "node_id": str(self.node_id),
            "tag": self.tag,
            "added_at": self.added_at.isoformat()
        }


@dataclass(frozen=True)
class NodeTagRemoved:
    """Event raised when a tag is removed from node"""
    node_id: NodeId
    tag: str
    removed_at: datetime
    event_type: str = "node.tag_removed"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "node_id": str(self.node_id),
            "tag": self.tag,
            "removed_at": self.removed_at.isoformat()
        }




