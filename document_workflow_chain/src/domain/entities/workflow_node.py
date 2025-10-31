"""
Workflow Node Domain Entity
==========================

Domain entity representing a workflow node with business logic and invariants.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Set
import hashlib

from ..value_objects.node_id import NodeId
from ..value_objects.priority import Priority
from ..events.node_events import NodeCreated, NodeUpdated, NodeDeleted
from ..exceptions.node_exceptions import NodeDomainException


@dataclass
class ContentMetrics:
    """Value object for content metrics"""
    word_count: int = 0
    character_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    reading_time_minutes: float = 0.0
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.word_count > 0:
            # Average reading speed: 200 words per minute
            self.reading_time_minutes = self.word_count / 200.0


@dataclass
class QualityScores:
    """Value object for quality scores"""
    overall_score: Optional[float] = None
    readability_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    seo_score: Optional[float] = None
    grammar_score: Optional[float] = None
    coherence_score: Optional[float] = None
    
    def get_average_score(self) -> float:
        """Get average of all available scores"""
        scores = [
            score for score in [
                self.overall_score,
                self.readability_score,
                self.sentiment_score,
                self.seo_score,
                self.grammar_score,
                self.coherence_score
            ] if score is not None
        ]
        return sum(scores) / len(scores) if scores else 0.0


class WorkflowNode:
    """
    Workflow Node Entity
    
    Represents a single node in a workflow chain with content and metadata.
    """
    
    def __init__(
        self,
        node_id: NodeId,
        title: str,
        content: str,
        prompt: str,
        parent_id: Optional[NodeId] = None,
        priority: Priority = Priority.NORMAL,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self._id = node_id
        self._title = title
        self._content = content
        self._prompt = prompt
        self._parent_id = parent_id
        self._children_ids: List[NodeId] = []
        self._priority = priority
        self._tags = tags or []
        self._metadata = metadata or {}
        self._created_at = datetime.utcnow()
        self._updated_at = datetime.utcnow()
        self._version = 1
        self._domain_events: List[Any] = []
        
        # Calculate content metrics
        self._content_metrics = self._calculate_content_metrics()
        
        # Initialize quality scores
        self._quality_scores = QualityScores()
        
        # Business invariants
        self._validate_title(title)
        self._validate_content(content)
        self._validate_prompt(prompt)
        self._validate_tags(self._tags)
        
        # Raise domain event
        self._add_domain_event(NodeCreated(
            node_id=self._id,
            title=self._title,
            created_at=self._created_at
        ))
    
    @property
    def id(self) -> NodeId:
        """Get node ID"""
        return self._id
    
    @property
    def title(self) -> str:
        """Get node title"""
        return self._title
    
    @property
    def content(self) -> str:
        """Get node content"""
        return self._content
    
    @property
    def prompt(self) -> str:
        """Get node prompt"""
        return self._prompt
    
    @property
    def parent_id(self) -> Optional[NodeId]:
        """Get parent node ID"""
        return self._parent_id
    
    @property
    def children_ids(self) -> List[NodeId]:
        """Get children node IDs"""
        return self._children_ids.copy()
    
    @property
    def priority(self) -> Priority:
        """Get node priority"""
        return self._priority
    
    @property
    def tags(self) -> List[str]:
        """Get node tags"""
        return self._tags.copy()
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get node metadata"""
        return self._metadata.copy()
    
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
    def content_metrics(self) -> ContentMetrics:
        """Get content metrics"""
        return self._content_metrics
    
    @property
    def quality_scores(self) -> QualityScores:
        """Get quality scores"""
        return self._quality_scores
    
    @property
    def domain_events(self) -> List[Any]:
        """Get domain events"""
        return self._domain_events.copy()
    
    def change_title(self, new_title: str) -> None:
        """Change node title with validation"""
        self._validate_title(new_title)
        old_title = self._title
        self._title = new_title
        self._updated_at = datetime.utcnow()
        self._version += 1
        
        self._add_domain_event(NodeUpdated(
            node_id=self._id,
            field="title",
            old_value=old_title,
            new_value=new_title,
            updated_at=self._updated_at
        ))
    
    def change_content(self, new_content: str) -> None:
        """Change node content with validation"""
        self._validate_content(new_content)
        old_content = self._content
        self._content = new_content
        self._updated_at = datetime.utcnow()
        self._version += 1
        
        # Recalculate content metrics
        self._content_metrics = self._calculate_content_metrics()
        
        self._add_domain_event(NodeUpdated(
            node_id=self._id,
            field="content",
            old_value=old_content,
            new_value=new_content,
            updated_at=self._updated_at
        ))
    
    def change_prompt(self, new_prompt: str) -> None:
        """Change node prompt with validation"""
        self._validate_prompt(new_prompt)
        old_prompt = self._prompt
        self._prompt = new_prompt
        self._updated_at = datetime.utcnow()
        self._version += 1
        
        self._add_domain_event(NodeUpdated(
            node_id=self._id,
            field="prompt",
            old_value=old_prompt,
            new_value=new_prompt,
            updated_at=self._updated_at
        ))
    
    def change_priority(self, new_priority: Priority) -> None:
        """Change node priority"""
        old_priority = self._priority
        self._priority = new_priority
        self._updated_at = datetime.utcnow()
        self._version += 1
        
        self._add_domain_event(NodeUpdated(
            node_id=self._id,
            field="priority",
            old_value=old_priority.value,
            new_value=new_priority.value,
            updated_at=self._updated_at
        ))
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the node"""
        if not tag or not tag.strip():
            raise NodeDomainException("Tag cannot be empty")
        
        tag = tag.strip().lower()
        if tag not in self._tags:
            self._tags.append(tag)
            self._updated_at = datetime.utcnow()
            self._version += 1
            
            self._add_domain_event(NodeUpdated(
                node_id=self._id,
                field="tags",
                old_value=None,
                new_value=tag,
                updated_at=self._updated_at
            ))
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the node"""
        tag = tag.strip().lower()
        if tag in self._tags:
            self._tags.remove(tag)
            self._updated_at = datetime.utcnow()
            self._version += 1
            
            self._add_domain_event(NodeUpdated(
                node_id=self._id,
                field="tags",
                old_value=tag,
                new_value=None,
                updated_at=self._updated_at
            ))
    
    def update_metadata(self, new_metadata: Dict[str, Any]) -> None:
        """Update node metadata"""
        old_metadata = self._metadata.copy()
        self._metadata.update(new_metadata)
        self._updated_at = datetime.utcnow()
        self._version += 1
        
        self._add_domain_event(NodeUpdated(
            node_id=self._id,
            field="metadata",
            old_value=old_metadata,
            new_value=self._metadata.copy(),
            updated_at=self._updated_at
        ))
    
    def set_parent(self, parent_id: Optional[NodeId]) -> None:
        """Set parent node"""
        old_parent_id = self._parent_id
        self._parent_id = parent_id
        self._updated_at = datetime.utcnow()
        self._version += 1
        
        self._add_domain_event(NodeUpdated(
            node_id=self._id,
            field="parent_id",
            old_value=old_parent_id,
            new_value=parent_id,
            updated_at=self._updated_at
        ))
    
    def add_child(self, child_id: NodeId) -> None:
        """Add a child node"""
        if child_id not in self._children_ids:
            self._children_ids.append(child_id)
            self._updated_at = datetime.utcnow()
            self._version += 1
            
            self._add_domain_event(NodeUpdated(
                node_id=self._id,
                field="children_ids",
                old_value=None,
                new_value=child_id,
                updated_at=self._updated_at
            ))
    
    def remove_child(self, child_id: NodeId) -> None:
        """Remove a child node"""
        if child_id in self._children_ids:
            self._children_ids.remove(child_id)
            self._updated_at = datetime.utcnow()
            self._version += 1
            
            self._add_domain_event(NodeUpdated(
                node_id=self._id,
                field="children_ids",
                old_value=child_id,
                new_value=None,
                updated_at=self._updated_at
            ))
    
    def update_quality_scores(self, scores: QualityScores) -> None:
        """Update quality scores"""
        old_scores = self._quality_scores
        self._quality_scores = scores
        self._updated_at = datetime.utcnow()
        self._version += 1
        
        self._add_domain_event(NodeUpdated(
            node_id=self._id,
            field="quality_scores",
            old_value=old_scores,
            new_value=scores,
            updated_at=self._updated_at
        ))
    
    def get_content_hash(self) -> str:
        """Get content hash for deduplication"""
        return hashlib.md5(self._content.encode('utf-8')).hexdigest()
    
    def is_root(self) -> bool:
        """Check if this is a root node"""
        return self._parent_id is None
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return len(self._children_ids) == 0
    
    def get_depth(self, node_map: Dict[NodeId, 'WorkflowNode']) -> int:
        """Calculate depth of this node in the tree"""
        if self.is_root():
            return 0
        
        if self._parent_id not in node_map:
            return 0
        
        parent = node_map[self._parent_id]
        return 1 + parent.get_depth(node_map)
    
    def get_ancestors(self, node_map: Dict[NodeId, 'WorkflowNode']) -> List['WorkflowNode']:
        """Get all ancestor nodes"""
        ancestors = []
        current = self
        
        while current._parent_id and current._parent_id in node_map:
            parent = node_map[current._parent_id]
            ancestors.append(parent)
            current = parent
        
        return ancestors
    
    def get_descendants(self, node_map: Dict[NodeId, 'WorkflowNode']) -> List['WorkflowNode']:
        """Get all descendant nodes"""
        descendants = []
        
        for child_id in self._children_ids:
            if child_id in node_map:
                child = node_map[child_id]
                descendants.append(child)
                descendants.extend(child.get_descendants(node_map))
        
        return descendants
    
    def delete(self) -> None:
        """Delete the node"""
        self._add_domain_event(NodeDeleted(
            node_id=self._id,
            deleted_at=datetime.utcnow()
        ))
    
    def clear_domain_events(self) -> None:
        """Clear domain events after they've been processed"""
        self._domain_events.clear()
    
    def _calculate_content_metrics(self) -> ContentMetrics:
        """Calculate content metrics"""
        content = self._content
        
        # Word count
        words = content.split()
        word_count = len(words)
        
        # Character count
        character_count = len(content)
        
        # Sentence count (simple heuristic)
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        sentence_count = len(sentences)
        
        # Paragraph count
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        return ContentMetrics(
            word_count=word_count,
            character_count=character_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count
        )
    
    def _validate_title(self, title: str) -> None:
        """Validate node title"""
        if not title or not title.strip():
            raise NodeDomainException("Node title cannot be empty")
        
        if len(title) > 255:
            raise NodeDomainException("Node title cannot exceed 255 characters")
    
    def _validate_content(self, content: str) -> None:
        """Validate node content"""
        if not content or not content.strip():
            raise NodeDomainException("Node content cannot be empty")
        
        if len(content) > 100000:  # 100KB limit
            raise NodeDomainException("Node content cannot exceed 100,000 characters")
    
    def _validate_prompt(self, prompt: str) -> None:
        """Validate node prompt"""
        if not prompt or not prompt.strip():
            raise NodeDomainException("Node prompt cannot be empty")
        
        if len(prompt) > 2000:
            raise NodeDomainException("Node prompt cannot exceed 2,000 characters")
    
    def _validate_tags(self, tags: List[str]) -> None:
        """Validate node tags"""
        if len(tags) > 20:
            raise NodeDomainException("Node cannot have more than 20 tags")
        
        for tag in tags:
            if not tag or not tag.strip():
                raise NodeDomainException("Tag cannot be empty")
            
            if len(tag) > 50:
                raise NodeDomainException("Tag cannot exceed 50 characters")
    
    def _add_domain_event(self, event: Any) -> None:
        """Add domain event"""
        self._domain_events.append(event)
    
    def __eq__(self, other: object) -> bool:
        """Check equality"""
        if not isinstance(other, WorkflowNode):
            return False
        return self._id == other._id
    
    def __hash__(self) -> int:
        """Get hash"""
        return hash(self._id)
    
    def __repr__(self) -> str:
        """String representation"""
        return f"WorkflowNode(id={self._id}, title='{self._title}', priority={self._priority.name})"


# Import here to avoid circular imports
from ..events.node_events import NodeCreated, NodeUpdated, NodeDeleted
from ..exceptions.node_exceptions import NodeDomainException




