"""
Domain Entities - Business objects with behavior
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import hashlib


@dataclass
class ContentAnalysis:
    """
    Entity representing a content analysis
    Contains business logic methods
    """
    content: str
    content_hash: str
    word_count: int
    character_count: int
    unique_words: int
    redundancy_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Calculate hash if not provided"""
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()
    
    def is_redundant(self, threshold: float = 0.8) -> bool:
        """
        Business logic: Check if content is redundant
        """
        return self.redundancy_score >= threshold
    
    def get_uniqueness_ratio(self) -> float:
        """
        Business logic: Calculate uniqueness ratio
        """
        if self.word_count == 0:
            return 0.0
        return self.unique_words / self.word_count
    
    def get_redundancy_category(self) -> str:
        """
        Business logic: Categorize redundancy level
        """
        if self.redundancy_score >= 0.9:
            return "very_high"
        elif self.redundancy_score >= 0.7:
            return "high"
        elif self.redundancy_score >= 0.5:
            return "medium"
        elif self.redundancy_score >= 0.3:
            return "low"
        else:
            return "very_low"


@dataclass
class SimilarityAnalysis:
    """
    Entity representing similarity analysis between two texts
    """
    text1: str
    text2: str
    similarity_score: float
    common_words: List[str]
    differences: List[str]
    threshold: float
    is_similar: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Calculate similarity status"""
        if not hasattr(self, 'is_similar'):
            self.is_similar = self.similarity_score >= self.threshold
    
    def is_plagiarized(self, plagiarism_threshold: float = 0.9) -> bool:
        """
        Business logic: Determine if content is plagiarized
        """
        return self.similarity_score >= plagiarism_threshold
    
    def get_similarity_category(self) -> str:
        """
        Business logic: Categorize similarity level
        """
        if self.similarity_score >= 0.9:
            return "nearly_identical"
        elif self.similarity_score >= 0.7:
            return "very_similar"
        elif self.similarity_score >= 0.5:
            return "moderately_similar"
        elif self.similarity_score >= 0.3:
            return "somewhat_similar"
        else:
            return "different"


@dataclass
class QualityAnalysis:
    """
    Entity representing content quality assessment
    """
    content: str
    readability_score: float
    quality_score: float
    sentiment_score: float
    language: str
    suggestions: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """
        Business logic: Check if content is high quality
        """
        return self.quality_score >= threshold
    
    def needs_improvement(self) -> bool:
        """
        Business logic: Determine if content needs improvement
        """
        return self.quality_score < 0.6 or len(self.suggestions) > 0
    
    def get_quality_grade(self) -> str:
        """
        Business logic: Get quality grade
        """
        if self.quality_score >= 0.9:
            return "A"
        elif self.quality_score >= 0.8:
            return "B"
        elif self.quality_score >= 0.7:
            return "C"
        elif self.quality_score >= 0.6:
            return "D"
        else:
            return "F"
