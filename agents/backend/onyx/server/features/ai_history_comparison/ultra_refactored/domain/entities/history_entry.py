"""
History Entry Entity
===================

Represents a single AI history entry with all its data and business rules.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import uuid
import hashlib

from ..value_objects.content_metrics import ContentMetrics


@dataclass
class HistoryEntry:
    """
    Represents a single AI history entry.
    
    Single Responsibility: Manage a single history entry and its business rules.
    """
    id: str
    content: str
    content_hash: str
    model_version: str
    timestamp: datetime
    metrics: ContentMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_feedback: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and normalize data after initialization."""
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if isinstance(self.metrics, dict):
            self.metrics = ContentMetrics.from_dict(self.metrics)
    
    @classmethod
    def create(
        cls,
        content: str,
        model_version: str,
        metrics: ContentMetrics,
        metadata: Optional[Dict[str, Any]] = None,
        entry_id: Optional[str] = None
    ) -> 'HistoryEntry':
        """
        Factory method to create a new history entry.
        
        Args:
            content: The content text
            model_version: Version of the AI model
            metrics: Content analysis metrics
            metadata: Optional metadata
            entry_id: Optional custom ID
            
        Returns:
            New HistoryEntry instance
        """
        if entry_id is None:
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            entry_id = f"{model_version}_{timestamp}_{content_hash}"
        
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        return cls(
            id=entry_id,
            content=content,
            content_hash=content_hash,
            model_version=model_version,
            timestamp=datetime.utcnow(),
            metrics=metrics,
            metadata=metadata or {}
        )
    
    def calculate_quality_score(self) -> float:
        """
        Calculate overall quality score from metrics.
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        scores = [
            self.metrics.readability_score,
            self.metrics.coherence_score,
            self.metrics.relevance_score,
            self.metrics.consistency_score
        ]
        valid_scores = [s for s in scores if s is not None]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    
    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """
        Check if entry meets quality threshold.
        
        Args:
            threshold: Quality threshold (default: 0.7)
            
        Returns:
            True if quality meets threshold
        """
        return self.calculate_quality_score() >= threshold
    
    def is_recent(self, days: int = 7) -> bool:
        """
        Check if entry is within specified days.
        
        Args:
            days: Number of days to check
            
        Returns:
            True if entry is recent
        """
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        return self.timestamp >= cutoff
    
    def get_content_length(self) -> int:
        """Get content length in characters."""
        return len(self.content)
    
    def get_word_count(self) -> int:
        """Get word count from metrics."""
        return self.metrics.word_count or 0
    
    def has_metadata(self, key: str) -> bool:
        """
        Check if metadata contains specific key.
        
        Args:
            key: Metadata key to check
            
        Returns:
            True if key exists
        """
        return key in self.metadata
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata value by key.
        
        Args:
            key: Metadata key
            default: Default value if key not found
            
        Returns:
            Metadata value or default
        """
        return self.metadata.get(key, default)
    
    def update_metadata(self, key: str, value: Any) -> None:
        """
        Update metadata value.
        
        Args:
            key: Metadata key
            value: New value
        """
        self.metadata[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entity to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "content": self.content,
            "content_hash": self.content_hash,
            "model_version": self.model_version,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics.to_dict(),
            "metadata": self.metadata,
            "user_feedback": self.user_feedback,
            "quality_score": self.calculate_quality_score()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HistoryEntry':
        """
        Create entity from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            HistoryEntry instance
        """
        return cls(
            id=data["id"],
            content=data["content"],
            content_hash=data["content_hash"],
            model_version=data["model_version"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metrics=ContentMetrics.from_dict(data["metrics"]),
            metadata=data.get("metadata", {}),
            user_feedback=data.get("user_feedback")
        )
    
    def __str__(self) -> str:
        """String representation."""
        return f"HistoryEntry(id={self.id}, model={self.model_version}, quality={self.calculate_quality_score():.2f})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"HistoryEntry(id='{self.id}', model_version='{self.model_version}', timestamp={self.timestamp}, quality={self.calculate_quality_score():.2f})"




