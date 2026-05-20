"""
Memory Systems for Autonomous Agents
=====================================

Implements different types of memory for agents:
- Episodic Memory: Stores specific events/experiences
- Semantic Memory: Stores general knowledge and facts
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class Memory:
    """Base memory class."""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary."""
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "metadata": self.metadata
        }


class EpisodicMemory:
    """
    Episodic Memory: Stores specific events and experiences.
    Based on human episodic memory system.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize episodic memory.
        
        Args:
            max_size: Maximum number of memories to store
        """
        self.memories: List[Memory] = []
        self.max_size = max_size
    
    def add(self, content: str, importance: float = 1.0, metadata: Optional[Dict] = None):
        """
        Add a memory.
        
        Args:
            content: Memory content
            importance: Importance score (0.0-1.0)
            metadata: Additional metadata
        """
        memory = Memory(
            content=content,
            importance=importance,
            metadata=metadata or {}
        )
        self.memories.append(memory)
        
        # Sort by importance and timestamp, keep most important
        self.memories.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
        
        if len(self.memories) > self.max_size:
            self.memories = self.memories[:self.max_size]
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Memory]:
        """
        Retrieve relevant memories based on query.
        
        Args:
            query: Search query
            top_k: Number of memories to retrieve
            
        Returns:
            List of relevant memories
        """
        # Simple keyword-based retrieval
        # In production, use embeddings/semantic search
        query_lower = query.lower()
        scored = []
        
        for memory in self.memories:
            score = memory.importance
            if query_lower in memory.content.lower():
                score += 0.5
            
            scored.append((score, memory))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored[:top_k]]
    
    def get_recent(self, n: int = 10) -> List[Memory]:
        """Get most recent memories."""
        return sorted(self.memories, key=lambda m: m.timestamp, reverse=True)[:n]
    
    def clear(self):
        """Clear all memories."""
        self.memories = []


class SemanticMemory:
    """
    Semantic Memory: Stores general knowledge and facts.
    Based on human semantic memory system.
    """
    
    def __init__(self):
        """Initialize semantic memory."""
        self.facts: Dict[str, Any] = {}
        self.relationships: Dict[str, List[str]] = {}
    
    def add_fact(self, key: str, value: Any, relationships: Optional[List[str]] = None):
        """
        Add a fact to semantic memory.
        
        Args:
            key: Fact key/identifier
            value: Fact value
            relationships: Related fact keys
        """
        self.facts[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        
        if relationships:
            self.relationships[key] = relationships
    
    def get_fact(self, key: str) -> Optional[Any]:
        """Get a fact by key."""
        fact = self.facts.get(key)
        return fact["value"] if fact else None
    
    def get_related(self, key: str) -> List[str]:
        """Get related facts."""
        return self.relationships.get(key, [])
    
    def query(self, pattern: str) -> Dict[str, Any]:
        """
        Query facts by pattern.
        
        Args:
            pattern: Pattern to match (simple substring match)
            
        Returns:
            Dictionary of matching facts
        """
        results = {}
        pattern_lower = pattern.lower()
        
        for key, fact in self.facts.items():
            if pattern_lower in key.lower() or pattern_lower in str(fact["value"]).lower():
                results[key] = fact["value"]
        
        return results
    
    def clear(self):
        """Clear all facts."""
        self.facts = {}
        self.relationships = {}



