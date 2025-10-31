"""
Content Metrics Value Object
============================

Immutable value object representing content analysis metrics.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class ContentMetrics:
    """
    Immutable value object representing content analysis metrics.
    
    Single Responsibility: Hold and validate content metrics data.
    """
    readability_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    word_count: Optional[int] = None
    sentence_count: Optional[int] = None
    avg_word_length: Optional[float] = None
    complexity_score: Optional[float] = None
    topic_diversity: Optional[float] = None
    consistency_score: Optional[float] = None
    quality_score: Optional[float] = None
    coherence_score: Optional[float] = None
    relevance_score: Optional[float] = None
    creativity_score: Optional[float] = None
    
    def __post_init__(self):
        """Validate metrics after initialization."""
        self._validate_scores()
        self._validate_counts()
    
    def _validate_scores(self) -> None:
        """Validate score values are between 0.0 and 1.0."""
        scores = [
            self.readability_score,
            self.sentiment_score,
            self.complexity_score,
            self.topic_diversity,
            self.consistency_score,
            self.quality_score,
            self.coherence_score,
            self.relevance_score,
            self.creativity_score
        ]
        
        for score in scores:
            if score is not None and not (0.0 <= score <= 1.0):
                raise ValueError(f"Score must be between 0.0 and 1.0, got {score}")
    
    def _validate_counts(self) -> None:
        """Validate count values are non-negative."""
        counts = [self.word_count, self.sentence_count]
        
        for count in counts:
            if count is not None and count < 0:
                raise ValueError(f"Count must be non-negative, got {count}")
    
    def get_available_metrics(self) -> Dict[str, Any]:
        """
        Get dictionary of available (non-None) metrics.
        
        Returns:
            Dictionary of available metrics
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    def get_metric_count(self) -> int:
        """
        Get count of available metrics.
        
        Returns:
            Number of non-None metrics
        """
        return len(self.get_available_metrics())
    
    def has_metric(self, metric_name: str) -> bool:
        """
        Check if specific metric is available.
        
        Args:
            metric_name: Name of metric to check
            
        Returns:
            True if metric is available (not None)
        """
        return hasattr(self, metric_name) and getattr(self, metric_name) is not None
    
    def get_metric(self, metric_name: str, default: Any = None) -> Any:
        """
        Get metric value by name.
        
        Args:
            metric_name: Name of metric
            default: Default value if metric not available
            
        Returns:
            Metric value or default
        """
        return getattr(self, metric_name, default) if self.has_metric(metric_name) else default
    
    def calculate_average_score(self) -> float:
        """
        Calculate average of all available scores.
        
        Returns:
            Average score or 0.0 if no scores available
        """
        scores = [
            self.readability_score,
            self.sentiment_score,
            self.complexity_score,
            self.topic_diversity,
            self.consistency_score,
            self.quality_score,
            self.coherence_score,
            self.relevance_score,
            self.creativity_score
        ]
        
        valid_scores = [s for s in scores if s is not None]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    
    def is_complete(self) -> bool:
        """
        Check if all metrics are available.
        
        Returns:
            True if all metrics are non-None
        """
        return all(
            getattr(self, field.name) is not None
            for field in self.__dataclass_fields__.values()
        )
    
    def get_completion_percentage(self) -> float:
        """
        Get percentage of metrics that are available.
        
        Returns:
            Completion percentage (0.0 to 1.0)
        """
        total_fields = len(self.__dataclass_fields__)
        available_fields = self.get_metric_count()
        return available_fields / total_fields if total_fields > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentMetrics':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            ContentMetrics instance
        """
        # Filter to only include valid field names
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered_data)
    
    @classmethod
    def empty(cls) -> 'ContentMetrics':
        """
        Create empty metrics instance.
        
        Returns:
            Empty ContentMetrics instance
        """
        return cls()
    
    @classmethod
    def with_scores(
        cls,
        readability: float,
        sentiment: float,
        complexity: float,
        quality: float
    ) -> 'ContentMetrics':
        """
        Create metrics with basic scores.
        
        Args:
            readability: Readability score
            sentiment: Sentiment score
            complexity: Complexity score
            quality: Quality score
            
        Returns:
            ContentMetrics instance with specified scores
        """
        return cls(
            readability_score=readability,
            sentiment_score=sentiment,
            complexity_score=complexity,
            quality_score=quality
        )
    
    def __str__(self) -> str:
        """String representation."""
        available = self.get_metric_count()
        total = len(self.__dataclass_fields__)
        return f"ContentMetrics({available}/{total} metrics available)"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ContentMetrics({self.get_available_metrics()})"




