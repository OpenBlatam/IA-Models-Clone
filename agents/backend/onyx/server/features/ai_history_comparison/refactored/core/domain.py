"""
Domain Models and Entities
=========================

This module defines the core domain entities and value objects for the AI History Comparison system.
These models represent the business concepts and rules of the domain.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import hashlib
from abc import ABC, abstractmethod


class ModelType(Enum):
    """Types of AI models"""
    TEXT_GENERATION = "text_generation"
    IMAGE_GENERATION = "image_generation"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"


class PerformanceMetric(Enum):
    """Performance metrics for AI models"""
    QUALITY_SCORE = "quality_score"
    RESPONSE_TIME = "response_time"
    TOKEN_EFFICIENCY = "token_efficiency"
    COST_EFFICIENCY = "cost_efficiency"
    ACCURACY = "accuracy"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    CREATIVITY = "creativity"
    READABILITY = "readability"
    SENTIMENT = "sentiment"
    COMPLEXITY = "complexity"


class AnalysisStatus(Enum):
    """Status of analysis operations"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrendDirection(Enum):
    """Direction of trends"""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class ModelDefinition:
    """Definition of an AI model"""
    name: str
    provider: str
    model_type: ModelType
    version: str
    context_length: int
    parameters: str
    release_date: datetime
    description: str
    capabilities: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    cost_per_1k_tokens: float = 0.0
    max_requests_per_minute: int = 0
    is_active: bool = True
    
    def __post_init__(self):
        if isinstance(self.release_date, str):
            self.release_date = datetime.fromisoformat(self.release_date)


@dataclass
class ContentMetrics:
    """Metrics for content analysis"""
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentMetrics':
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class HistoryEntry:
    """Core domain entity for AI history entries"""
    id: str
    content: str
    content_hash: str
    model_version: str
    timestamp: datetime
    metrics: ContentMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_feedback: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
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
        """Factory method to create a new history entry"""
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
        """Calculate overall quality score from metrics"""
        scores = [
            self.metrics.readability_score,
            self.metrics.coherence_score,
            self.metrics.relevance_score,
            self.metrics.consistency_score
        ]
        valid_scores = [s for s in scores if s is not None]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    
    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """Check if entry meets quality threshold"""
        return self.calculate_quality_score() >= threshold


@dataclass
class ComparisonResult:
    """Result of comparing two history entries"""
    id: str
    entry1_id: str
    entry2_id: str
    timestamp: datetime
    similarity_score: float
    quality_difference: Dict[str, float]
    trend_direction: TrendDirection
    significant_changes: List[str]
    recommendations: List[str]
    confidence_score: float
    comparison_type: str = "content_similarity"
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if isinstance(self.trend_direction, str):
            self.trend_direction = TrendDirection(self.trend_direction)
    
    @classmethod
    def create(
        cls,
        entry1_id: str,
        entry2_id: str,
        similarity_score: float,
        quality_difference: Dict[str, float],
        trend_direction: TrendDirection,
        significant_changes: List[str],
        recommendations: List[str],
        confidence_score: float,
        comparison_type: str = "content_similarity"
    ) -> 'ComparisonResult':
        """Factory method to create comparison result"""
        return cls(
            id=str(uuid.uuid4()),
            entry1_id=entry1_id,
            entry2_id=entry2_id,
            timestamp=datetime.utcnow(),
            similarity_score=similarity_score,
            quality_difference=quality_difference,
            trend_direction=trend_direction,
            significant_changes=significant_changes,
            recommendations=recommendations,
            confidence_score=confidence_score,
            comparison_type=comparison_type
        )
    
    def is_significant_change(self, threshold: float = 0.1) -> bool:
        """Check if comparison shows significant change"""
        return abs(self.similarity_score - 1.0) > threshold


@dataclass
class TrendAnalysis:
    """Analysis of trends in model performance"""
    id: str
    model_name: str
    metric: PerformanceMetric
    trend_direction: TrendDirection
    trend_strength: float
    confidence: float
    timestamp: datetime
    forecast: List[tuple] = field(default_factory=list)  # (datetime, value) tuples
    anomalies: List[tuple] = field(default_factory=list)  # (datetime, value) tuples
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if isinstance(self.metric, str):
            self.metric = PerformanceMetric(self.metric)
        if isinstance(self.trend_direction, str):
            self.trend_direction = TrendDirection(self.trend_direction)
    
    @classmethod
    def create(
        cls,
        model_name: str,
        metric: PerformanceMetric,
        trend_direction: TrendDirection,
        trend_strength: float,
        confidence: float,
        forecast: Optional[List[tuple]] = None,
        anomalies: Optional[List[tuple]] = None
    ) -> 'TrendAnalysis':
        """Factory method to create trend analysis"""
        return cls(
            id=str(uuid.uuid4()),
            model_name=model_name,
            metric=metric,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            confidence=confidence,
            timestamp=datetime.utcnow(),
            forecast=forecast or [],
            anomalies=anomalies or []
        )
    
    def is_significant_trend(self, threshold: float = 0.7) -> bool:
        """Check if trend is statistically significant"""
        return self.confidence >= threshold and self.trend_strength >= 0.5


@dataclass
class QualityReport:
    """Quality assessment report"""
    id: str
    report_type: str
    generated_at: datetime
    summary: Dict[str, Any]
    average_metrics: Dict[str, float]
    trends: Dict[str, Any]
    outliers: List[Dict[str, Any]]
    recommendations: List[str]
    time_window_start: Optional[datetime] = None
    time_window_end: Optional[datetime] = None
    total_entries_analyzed: Optional[int] = None
    report_version: str = "1.0"
    
    def __post_init__(self):
        if isinstance(self.generated_at, str):
            self.generated_at = datetime.fromisoformat(self.generated_at)
        if self.time_window_start and isinstance(self.time_window_start, str):
            self.time_window_start = datetime.fromisoformat(self.time_window_start)
        if self.time_window_end and isinstance(self.time_window_end, str):
            self.time_window_end = datetime.fromisoformat(self.time_window_end)
    
    @classmethod
    def create(
        cls,
        report_type: str,
        summary: Dict[str, Any],
        average_metrics: Dict[str, float],
        trends: Dict[str, Any],
        outliers: List[Dict[str, Any]],
        recommendations: List[str],
        time_window_start: Optional[datetime] = None,
        time_window_end: Optional[datetime] = None,
        total_entries: Optional[int] = None
    ) -> 'QualityReport':
        """Factory method to create quality report"""
        return cls(
            id=str(uuid.uuid4()),
            report_type=report_type,
            generated_at=datetime.utcnow(),
            summary=summary,
            average_metrics=average_metrics,
            trends=trends,
            outliers=outliers,
            recommendations=recommendations,
            time_window_start=time_window_start,
            time_window_end=time_window_end,
            total_entries_analyzed=total_entries
        )


@dataclass
class AnalysisJob:
    """Background analysis job"""
    id: str
    job_type: str
    status: AnalysisStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    progress_percentage: float = 0.0
    total_items: Optional[int] = None
    processed_items: Optional[int] = None
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if self.started_at and isinstance(self.started_at, str):
            self.started_at = datetime.fromisoformat(self.started_at)
        if self.completed_at and isinstance(self.completed_at, str):
            self.completed_at = datetime.fromisoformat(self.completed_at)
        if isinstance(self.status, str):
            self.status = AnalysisStatus(self.status)
    
    @classmethod
    def create(
        cls,
        job_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        status: AnalysisStatus = AnalysisStatus.PENDING
    ) -> 'AnalysisJob':
        """Factory method to create analysis job"""
        return cls(
            id=str(uuid.uuid4()),
            job_type=job_type,
            status=status,
            created_at=datetime.utcnow(),
            parameters=parameters or {}
        )
    
    def start(self):
        """Mark job as started"""
        self.status = AnalysisStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()
    
    def complete(self, result: Optional[Dict[str, Any]] = None):
        """Mark job as completed"""
        self.status = AnalysisStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.progress_percentage = 100.0
        self.result = result
    
    def fail(self, error_message: str):
        """Mark job as failed"""
        self.status = AnalysisStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message


@dataclass
class UserFeedback:
    """User feedback on content"""
    id: str
    entry_id: str
    user_id: Optional[str]
    timestamp: datetime
    rating: Optional[int]  # 1-5 scale
    feedback_type: str
    feedback_text: Optional[str] = None
    feedback_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if self.rating is not None and (self.rating < 1 or self.rating > 5):
            raise ValueError("Rating must be between 1 and 5")
    
    @classmethod
    def create(
        cls,
        entry_id: str,
        feedback_type: str,
        rating: Optional[int] = None,
        feedback_text: Optional[str] = None,
        feedback_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> 'UserFeedback':
        """Factory method to create user feedback"""
        return cls(
            id=str(uuid.uuid4()),
            entry_id=entry_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            rating=rating,
            feedback_type=feedback_type,
            feedback_text=feedback_text,
            feedback_data=feedback_data
        )


# Domain Events
@dataclass
class DomainEvent(ABC):
    """Base class for domain events"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @abstractmethod
    def get_event_type(self) -> str:
        """Get the event type"""
        pass


@dataclass
class AnalysisCompletedEvent(DomainEvent):
    """Event fired when analysis is completed"""
    entry_id: str
    analysis_type: str
    results: Dict[str, Any]
    
    def get_event_type(self) -> str:
        return "analysis_completed"


@dataclass
class ModelComparisonEvent(DomainEvent):
    """Event fired when models are compared"""
    entry1_id: str
    entry2_id: str
    comparison_result: ComparisonResult
    
    def get_event_type(self) -> str:
        return "model_comparison"


@dataclass
class TrendDetectedEvent(DomainEvent):
    """Event fired when significant trend is detected"""
    model_name: str
    metric: PerformanceMetric
    trend_analysis: TrendAnalysis
    
    def get_event_type(self) -> str:
        return "trend_detected"


@dataclass
class QualityAlertEvent(DomainEvent):
    """Event fired when quality threshold is breached"""
    entry_id: str
    quality_score: float
    threshold: float
    alert_type: str
    
    def get_event_type(self) -> str:
        return "quality_alert"




