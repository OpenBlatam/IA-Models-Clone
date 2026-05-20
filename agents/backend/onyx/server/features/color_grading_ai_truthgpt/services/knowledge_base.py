"""
Knowledge Base for Color Grading AI
====================================

Knowledge base system for storing and retrieving color grading knowledge.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class KnowledgeType(Enum):
    """Knowledge types."""
    RULE = "rule"
    PATTERN = "pattern"
    EXAMPLE = "example"
    BEST_PRACTICE = "best_practice"
    TROUBLESHOOTING = "troubleshooting"
    REFERENCE = "reference"


@dataclass
class KnowledgeEntry:
    """Knowledge base entry."""
    entry_id: str
    knowledge_type: KnowledgeType
    title: str
    content: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    relevance_score: float = 0.0


class KnowledgeBase:
    """
    Knowledge base system.
    
    Features:
    - Knowledge storage
    - Semantic search
    - Tag-based retrieval
    - Relevance scoring
    - Usage tracking
    - Knowledge updates
    """
    
    def __init__(self):
        """Initialize knowledge base."""
        self._entries: Dict[str, KnowledgeEntry] = {}
        self._index: Dict[str, List[str]] = {}  # tag -> entry_ids
        self._type_index: Dict[KnowledgeType, List[str]] = {}
    
    def add_entry(
        self,
        knowledge_type: KnowledgeType,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entry_id: Optional[str] = None
    ) -> str:
        """
        Add knowledge entry.
        
        Args:
            knowledge_type: Knowledge type
            title: Entry title
            content: Entry content
            tags: Optional tags
            metadata: Optional metadata
            entry_id: Optional entry ID
            
        Returns:
            Entry ID
        """
        import uuid
        
        if entry_id is None:
            entry_id = str(uuid.uuid4())
        
        entry = KnowledgeEntry(
            entry_id=entry_id,
            knowledge_type=knowledge_type,
            title=title,
            content=content,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        self._entries[entry_id] = entry
        
        # Index by tags
        for tag in entry.tags:
            if tag not in self._index:
                self._index[tag] = []
            self._index[tag].append(entry_id)
        
        # Index by type
        if knowledge_type not in self._type_index:
            self._type_index[knowledge_type] = []
        self._type_index[knowledge_type].append(entry_id)
        
        logger.info(f"Added knowledge entry: {entry_id} ({knowledge_type.value})")
        
        return entry_id
    
    def search(
        self,
        query: str,
        knowledge_type: Optional[KnowledgeType] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[KnowledgeEntry]:
        """
        Search knowledge base.
        
        Args:
            query: Search query
            knowledge_type: Optional knowledge type filter
            tags: Optional tags filter
            limit: Maximum results
            
        Returns:
            List of matching entries
        """
        candidates = []
        
        # Filter by type
        if knowledge_type:
            candidate_ids = self._type_index.get(knowledge_type, [])
        else:
            candidate_ids = list(self._entries.keys())
        
        # Filter by tags
        if tags:
            tag_ids = set()
            for tag in tags:
                tag_ids.update(self._index.get(tag, []))
            candidate_ids = [id for id in candidate_ids if id in tag_ids]
        
        # Simple text matching (in production, use semantic search)
        query_lower = query.lower()
        for entry_id in candidate_ids:
            entry = self._entries[entry_id]
            
            # Calculate relevance
            relevance = 0.0
            if query_lower in entry.title.lower():
                relevance += 0.5
            if query_lower in entry.content.lower():
                relevance += 0.3
            for tag in entry.tags:
                if query_lower in tag.lower():
                    relevance += 0.2
            
            if relevance > 0:
                entry.relevance_score = relevance
                candidates.append(entry)
        
        # Sort by relevance
        candidates.sort(key=lambda e: e.relevance_score, reverse=True)
        
        return candidates[:limit]
    
    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get knowledge entry by ID."""
        return self._entries.get(entry_id)
    
    def update_entry(
        self,
        entry_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Update knowledge entry."""
        entry = self._entries.get(entry_id)
        if not entry:
            return False
        
        if title:
            entry.title = title
        if content:
            entry.content = content
        if tags:
            entry.tags = tags
        
        entry.updated_at = datetime.now()
        
        logger.info(f"Updated knowledge entry: {entry_id}")
        
        return True
    
    def increment_usage(self, entry_id: str):
        """Increment entry usage count."""
        entry = self._entries.get(entry_id)
        if entry:
            entry.usage_count += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "total_entries": len(self._entries),
            "entries_by_type": {
                kt.value: len(ids)
                for kt, ids in self._type_index.items()
            },
            "tags_count": len(self._index),
            "most_used": sorted(
                self._entries.values(),
                key=lambda e: e.usage_count,
                reverse=True
            )[:5]
        }




