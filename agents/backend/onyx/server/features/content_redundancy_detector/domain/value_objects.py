"""
Value Objects - Immutable objects with value equality
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass(frozen=True)
class AnalysisResult:
    """
    Immutable value object for analysis results
    """
    content_hash: str
    word_count: int
    character_count: int
    unique_words: int
    redundancy_score: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "content_hash": self.content_hash,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "unique_words": self.unique_words,
            "redundancy_score": self.redundancy_score,
            "timestamp": self.timestamp
        }


@dataclass(frozen=True)
class SimilarityResult:
    """
    Immutable value object for similarity results
    """
    similarity_score: float
    is_similar: bool
    common_words: List[str]
    differences: List[str]
    threshold: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "similarity_score": self.similarity_score,
            "is_similar": self.is_similar,
            "common_words": self.common_words,
            "differences": self.differences,
            "threshold": self.threshold,
            "timestamp": self.timestamp
        }


@dataclass(frozen=True)
class QualityResult:
    """
    Immutable value object for quality assessment results
    """
    quality_score: float
    readability_score: float
    sentiment_score: Optional[float]
    language: str
    suggestions: List[str]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "quality_score": self.quality_score,
            "readability_score": self.readability_score,
            "sentiment_score": self.sentiment_score,
            "language": self.language,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp
        }


@dataclass(frozen=True)
class BatchResult:
    """
    Value object for batch processing results
    """
    total_items: int
    processed_items: int
    failed_items: int
    results: List[AnalysisResult]
    errors: List[Dict[str, Any]]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "failed_items": self.failed_items,
            "results": [r.to_dict() for r in self.results],
            "errors": self.errors,
            "timestamp": self.timestamp
        }
