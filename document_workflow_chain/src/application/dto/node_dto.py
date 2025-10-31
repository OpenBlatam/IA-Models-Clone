"""
Node Data Transfer Objects
=========================

DTOs for node-related data transfer between layers.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any

from ...domain.value_objects.priority import Priority


@dataclass
class NodeDto:
    """Node data transfer object"""
    node_id: str
    workflow_id: str
    title: str
    content: str
    prompt: str
    parent_id: Optional[str]
    children_ids: List[str]
    priority: int
    tags: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: int
    content_metrics: Dict[str, Any]
    quality_scores: Dict[str, Any]
    
    @classmethod
    def from_domain(cls, node, workflow_id: str) -> NodeDto:
        """Create DTO from domain entity"""
        return cls(
            node_id=str(node.id),
            workflow_id=workflow_id,
            title=node.title,
            content=node.content,
            prompt=node.prompt,
            parent_id=str(node.parent_id) if node.parent_id else None,
            children_ids=[str(child_id) for child_id in node.children_ids],
            priority=node.priority.value,
            tags=node.tags,
            metadata=node.metadata,
            created_at=node.created_at,
            updated_at=node.updated_at,
            version=node.version,
            content_metrics={
                "word_count": node.content_metrics.word_count,
                "character_count": node.content_metrics.character_count,
                "sentence_count": node.content_metrics.sentence_count,
                "paragraph_count": node.content_metrics.paragraph_count,
                "reading_time_minutes": node.content_metrics.reading_time_minutes
            },
            quality_scores={
                "overall_score": node.quality_scores.overall_score,
                "readability_score": node.quality_scores.readability_score,
                "sentiment_score": node.quality_scores.sentiment_score,
                "seo_score": node.quality_scores.seo_score,
                "grammar_score": node.quality_scores.grammar_score,
                "coherence_score": node.quality_scores.coherence_score,
                "average_score": node.quality_scores.get_average_score()
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "node_id": self.node_id,
            "workflow_id": self.workflow_id,
            "title": self.title,
            "content": self.content,
            "prompt": self.prompt,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "priority": self.priority,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "content_metrics": self.content_metrics,
            "quality_scores": self.quality_scores
        }


@dataclass
class CreateNodeDto:
    """Create node DTO"""
    workflow_id: str
    title: str
    content: str
    prompt: str
    parent_id: Optional[str] = None
    priority: int = Priority.NORMAL
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_domain_request(self):
        """Convert to domain request"""
        from .add_node_use_case import AddNodeRequest
        return AddNodeRequest(
            workflow_id=self.workflow_id,
            title=self.title,
            content=self.content,
            prompt=self.prompt,
            parent_id=self.parent_id,
            priority=self.priority,
            tags=self.tags,
            metadata=self.metadata
        )


@dataclass
class UpdateNodeDto:
    """Update node DTO"""
    node_id: str
    title: Optional[str] = None
    content: Optional[str] = None
    prompt: Optional[str] = None
    parent_id: Optional[str] = None
    priority: Optional[int] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class NodeListDto:
    """Node list DTO"""
    nodes: List[NodeDto]
    total: int
    limit: int
    offset: int
    has_more: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "total": self.total,
            "limit": self.limit,
            "offset": self.offset,
            "has_more": self.has_more
        }


@dataclass
class NodeStatisticsDto:
    """Node statistics DTO"""
    total_nodes: int
    nodes_by_priority: Dict[str, int]
    nodes_by_workflow: Dict[str, int]
    average_quality_score: float
    most_used_tags: List[Dict[str, Any]]
    content_statistics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_nodes": self.total_nodes,
            "nodes_by_priority": self.nodes_by_priority,
            "nodes_by_workflow": self.nodes_by_workflow,
            "average_quality_score": self.average_quality_score,
            "most_used_tags": self.most_used_tags,
            "content_statistics": self.content_statistics
        }


@dataclass
class NodeSearchDto:
    """Node search DTO"""
    query: str
    workflow_id: Optional[str] = None
    tags: Optional[List[str]] = None
    priority: Optional[int] = None
    min_quality_score: Optional[float] = None
    max_quality_score: Optional[float] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    limit: int = 20
    offset: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "workflow_id": self.workflow_id,
            "tags": self.tags,
            "priority": self.priority,
            "min_quality_score": self.min_quality_score,
            "max_quality_score": self.max_quality_score,
            "created_after": self.created_after.isoformat() if self.created_after else None,
            "created_before": self.created_before.isoformat() if self.created_before else None,
            "limit": self.limit,
            "offset": self.offset
        }




