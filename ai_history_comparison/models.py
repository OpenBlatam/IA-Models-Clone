"""
AI History Comparison System - Database Models

This module defines the database models for storing AI history entries,
comparison results, and analytics data.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean, 
    JSON, ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid
import json
import hashlib  # OPTIMIZATION: Import hashlib for faster hashing

Base = declarative_base()

class HistoryEntry(Base):
    """Model for storing AI history entries"""
    __tablename__ = "history_entries"
    
    id = Column(String(64), primary_key=True, index=True)
    content = Column(Text, nullable=False)
    content_hash = Column(String(32), nullable=False, index=True)
    model_version = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Content metrics
    readability_score = Column(Float, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    word_count = Column(Integer, nullable=True)
    sentence_count = Column(Integer, nullable=True)
    avg_word_length = Column(Float, nullable=True)
    complexity_score = Column(Float, nullable=True)
    topic_diversity = Column(Float, nullable=True)
    consistency_score = Column(Float, nullable=True)
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    user_feedback = Column(JSON, nullable=True)
    
    # Relationships
    comparisons = relationship("ComparisonResult", back_populates="entry1", foreign_keys="ComparisonResult.entry1_id")
    comparisons2 = relationship("ComparisonResult", back_populates="entry2", foreign_keys="ComparisonResult.entry2_id")
    trend_data = relationship("TrendData", back_populates="entry")
    
    # Indexes
    __table_args__ = (
        Index('idx_model_timestamp', 'model_version', 'timestamp'),
        Index('idx_content_hash', 'content_hash'),
        Index('idx_timestamp', 'timestamp'),
    )

class ComparisonResult(Base):
    """Model for storing comparison results between entries"""
    __tablename__ = "comparison_results"
    
    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    entry1_id = Column(String(64), ForeignKey("history_entries.id"), nullable=False)
    entry2_id = Column(String(64), ForeignKey("history_entries.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Comparison metrics
    similarity_score = Column(Float, nullable=False)
    quality_difference = Column(JSON, nullable=True)
    trend_direction = Column(String(20), nullable=True)
    significant_changes = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Comparison type
    comparison_type = Column(String(50), nullable=False, default="content_similarity")
    
    # Relationships
    entry1 = relationship("HistoryEntry", foreign_keys=[entry1_id], back_populates="comparisons")
    entry2 = relationship("HistoryEntry", foreign_keys=[entry2_id], back_populates="comparisons2")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('entry1_id', 'entry2_id', 'comparison_type', name='unique_comparison'),
        Index('idx_comparison_timestamp', 'timestamp'),
        Index('idx_similarity_score', 'similarity_score'),
    )

class TrendData(Base):
    """Model for storing trend analysis data"""
    __tablename__ = "trend_data"
    
    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    entry_id = Column(String(64), ForeignKey("history_entries.id"), nullable=False)
    metric_name = Column(String(50), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Trend analysis
    trend_direction = Column(String(20), nullable=True)
    change_percentage = Column(Float, nullable=True)
    significance_level = Column(Float, nullable=True)
    
    # Relationships
    entry = relationship("HistoryEntry", back_populates="trend_data")
    
    # Indexes
    __table_args__ = (
        Index('idx_metric_timestamp', 'metric_name', 'timestamp'),
        Index('idx_trend_direction', 'trend_direction'),
    )

class QualityReport(Base):
    """Model for storing quality reports"""
    __tablename__ = "quality_reports"
    
    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    report_type = Column(String(50), nullable=False, index=True)
    generated_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Report data
    summary = Column(JSON, nullable=True)
    average_metrics = Column(JSON, nullable=True)
    trends = Column(JSON, nullable=True)
    outliers = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    
    # Time window
    time_window_start = Column(DateTime, nullable=True)
    time_window_end = Column(DateTime, nullable=True)
    
    # Report metadata
    total_entries_analyzed = Column(Integer, nullable=True)
    report_version = Column(String(20), nullable=True, default="1.0")
    
    # Indexes
    __table_args__ = (
        Index('idx_report_type_timestamp', 'report_type', 'generated_at'),
    )

class ClusterData(Base):
    """Model for storing clustering results"""
    __tablename__ = "cluster_data"
    
    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    cluster_id = Column(Integer, nullable=False, index=True)
    entry_id = Column(String(64), ForeignKey("history_entries.id"), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Clustering metadata
    algorithm = Column(String(50), nullable=False, default="kmeans")
    n_clusters = Column(Integer, nullable=False)
    cluster_center = Column(JSON, nullable=True)
    distance_to_center = Column(Float, nullable=True)
    
    # Relationships
    entry = relationship("HistoryEntry")
    
    # Indexes
    __table_args__ = (
        Index('idx_cluster_id', 'cluster_id'),
        Index('idx_cluster_algorithm', 'algorithm'),
    )

class SystemMetrics(Base):
    """Model for storing system performance metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Metric metadata
    metric_type = Column(String(50), nullable=True)  # counter, gauge, histogram
    labels = Column(JSON, nullable=True)
    unit = Column(String(20), nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_metric_name_timestamp', 'metric_name', 'timestamp'),
        Index('idx_metric_type', 'metric_type'),
    )

class UserFeedback(Base):
    """Model for storing user feedback on content"""
    __tablename__ = "user_feedback"
    
    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    entry_id = Column(String(64), ForeignKey("history_entries.id"), nullable=False)
    user_id = Column(String(100), nullable=True, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Feedback data
    rating = Column(Integer, nullable=True)  # 1-5 scale
    feedback_type = Column(String(50), nullable=False)  # quality, relevance, accuracy
    feedback_text = Column(Text, nullable=True)
    feedback_data = Column(JSON, nullable=True)
    
    # Relationships
    entry = relationship("HistoryEntry")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('rating >= 1 AND rating <= 5', name='valid_rating'),
        Index('idx_entry_feedback', 'entry_id', 'feedback_type'),
        Index('idx_user_feedback', 'user_id', 'timestamp'),
    )

class AnalysisJob(Base):
    """Model for tracking analysis jobs"""
    __tablename__ = "analysis_jobs"
    
    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_type = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False, default="pending", index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Job parameters
    parameters = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Progress tracking
    progress_percentage = Column(Float, nullable=True, default=0.0)
    total_items = Column(Integer, nullable=True)
    processed_items = Column(Integer, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_job_status', 'status'),
        Index('idx_job_type_status', 'job_type', 'status'),
    )

class ConfigurationSnapshot(Base):
    """Model for storing configuration snapshots"""
    __tablename__ = "configuration_snapshots"
    
    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    snapshot_name = Column(String(100), nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Configuration data
    config_data = Column(JSON, nullable=False)
    config_version = Column(String(20), nullable=True)
    environment = Column(String(50), nullable=True)
    
    # Metadata
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_snapshot_name', 'snapshot_name'),
        Index('idx_active_snapshot', 'is_active'),
    )

class AuditLog(Base):
    """Model for storing audit logs"""
    __tablename__ = "audit_logs"
    
    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50), nullable=False, index=True)
    resource_id = Column(String(100), nullable=True, index=True)
    user_id = Column(String(100), nullable=True, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Audit data
    old_values = Column(JSON, nullable=True)
    new_values = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_action_timestamp', 'action', 'timestamp'),
        Index('idx_resource_audit', 'resource_type', 'resource_id'),
        Index('idx_user_audit', 'user_id', 'timestamp'),
    )

# Utility functions for model operations
class ModelUtils:
    """Utility functions for database model operations"""
    
    @staticmethod
    def create_history_entry(
        content: str,
        model_version: str,
        metrics: Dict[str, Any],
        metadata: Optional[Dict] = None,
        entry_id: Optional[str] = None
    ) -> HistoryEntry:
        """Create a new history entry"""
        if entry_id is None:
            import hashlib
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            entry_id = f"{model_version}_{timestamp}_{content_hash}"
        
        return HistoryEntry(
            id=entry_id,
            content=content,
            content_hash=hashlib.md5(content.encode()).hexdigest(),
            model_version=model_version,
            timestamp=datetime.utcnow(),
            readability_score=metrics.get('readability_score'),
            sentiment_score=metrics.get('sentiment_score'),
            word_count=metrics.get('word_count'),
            sentence_count=metrics.get('sentence_count'),
            avg_word_length=metrics.get('avg_word_length'),
            complexity_score=metrics.get('complexity_score'),
            topic_diversity=metrics.get('topic_diversity'),
            consistency_score=metrics.get('consistency_score'),
            metadata=metadata
        )
    
    @staticmethod
    def create_comparison_result(
        entry1_id: str,
        entry2_id: str,
        similarity_score: float,
        quality_difference: Dict[str, float],
        trend_direction: str,
        significant_changes: List[str],
        recommendations: List[str],
        confidence_score: float,
        comparison_type: str = "content_similarity"
    ) -> ComparisonResult:
        """Create a new comparison result"""
        return ComparisonResult(
            entry1_id=entry1_id,
            entry2_id=entry2_id,
            similarity_score=similarity_score,
            quality_difference=quality_difference,
            trend_direction=trend_direction,
            significant_changes=significant_changes,
            recommendations=recommendations,
            confidence_score=confidence_score,
            comparison_type=comparison_type,
            timestamp=datetime.utcnow()
        )
    
    @staticmethod
    def create_trend_data(
        entry_id: str,
        metric_name: str,
        metric_value: float,
        trend_direction: Optional[str] = None,
        change_percentage: Optional[float] = None,
        significance_level: Optional[float] = None
    ) -> TrendData:
        """Create new trend data entry"""
        return TrendData(
            entry_id=entry_id,
            metric_name=metric_name,
            metric_value=metric_value,
            trend_direction=trend_direction,
            change_percentage=change_percentage,
            significance_level=significance_level,
            timestamp=datetime.utcnow()
        )
    
    @staticmethod
    def create_quality_report(
        report_type: str,
        summary: Dict[str, Any],
        average_metrics: Dict[str, Any],
        trends: Dict[str, Any],
        outliers: List[Dict[str, Any]],
        recommendations: List[str],
        time_window_start: Optional[datetime] = None,
        time_window_end: Optional[datetime] = None,
        total_entries: Optional[int] = None
    ) -> QualityReport:
        """Create a new quality report"""
        return QualityReport(
            report_type=report_type,
            summary=summary,
            average_metrics=average_metrics,
            trends=trends,
            outliers=outliers,
            recommendations=recommendations,
            time_window_start=time_window_start,
            time_window_end=time_window_end,
            total_entries_analyzed=total_entries,
            generated_at=datetime.utcnow()
        )
    
    @staticmethod
    def create_cluster_data(
        cluster_id: int,
        entry_id: str,
        algorithm: str = "kmeans",
        n_clusters: int = 5,
        cluster_center: Optional[Dict[str, Any]] = None,
        distance_to_center: Optional[float] = None
    ) -> ClusterData:
        """Create new cluster data entry"""
        return ClusterData(
            cluster_id=cluster_id,
            entry_id=entry_id,
            algorithm=algorithm,
            n_clusters=n_clusters,
            cluster_center=cluster_center,
            distance_to_center=distance_to_center,
            created_at=datetime.utcnow()
        )
    
    @staticmethod
    def create_system_metric(
        metric_name: str,
        metric_value: float,
        metric_type: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None
    ) -> SystemMetrics:
        """Create new system metric entry"""
        return SystemMetrics(
            metric_name=metric_name,
            metric_value=metric_value,
            metric_type=metric_type,
            labels=labels,
            unit=unit,
            timestamp=datetime.utcnow()
        )
    
    @staticmethod
    def create_user_feedback(
        entry_id: str,
        feedback_type: str,
        rating: Optional[int] = None,
        feedback_text: Optional[str] = None,
        feedback_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> UserFeedback:
        """Create new user feedback entry"""
        return UserFeedback(
            entry_id=entry_id,
            user_id=user_id,
            feedback_type=feedback_type,
            rating=rating,
            feedback_text=feedback_text,
            feedback_data=feedback_data,
            timestamp=datetime.utcnow()
        )
    
    @staticmethod
    def create_analysis_job(
        job_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        status: str = "pending"
    ) -> AnalysisJob:
        """Create new analysis job"""
        return AnalysisJob(
            job_type=job_type,
            parameters=parameters,
            status=status,
            created_at=datetime.utcnow()
        )
    
    @staticmethod
    def create_audit_log(
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> AuditLog:
        """Create new audit log entry"""
        return AuditLog(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            old_values=old_values,
            new_values=new_values,
            metadata=metadata,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow()
        )

# Database session management
def create_session_factory(engine):
    """Create a session factory for the database"""
    return sessionmaker(bind=engine)

def get_session(session_factory):
    """Get a database session"""
    return session_factory()

# Model serialization utilities
class ModelSerializer:
    """Utility class for serializing models to dictionaries"""
    
    @staticmethod
    def serialize_history_entry(entry: HistoryEntry) -> Dict[str, Any]:
        """Serialize history entry to dictionary"""
        return {
            "id": entry.id,
            "content": entry.content,
            "content_hash": entry.content_hash,
            "model_version": entry.model_version,
            "timestamp": entry.timestamp.isoformat(),
            "readability_score": entry.readability_score,
            "sentiment_score": entry.sentiment_score,
            "word_count": entry.word_count,
            "sentence_count": entry.sentence_count,
            "avg_word_length": entry.avg_word_length,
            "complexity_score": entry.complexity_score,
            "topic_diversity": entry.topic_diversity,
            "consistency_score": entry.consistency_score,
            "metadata": entry.metadata,
            "user_feedback": entry.user_feedback
        }
    
    @staticmethod
    def serialize_comparison_result(result: ComparisonResult) -> Dict[str, Any]:
        """Serialize comparison result to dictionary"""
        return {
            "id": result.id,
            "entry1_id": result.entry1_id,
            "entry2_id": result.entry2_id,
            "timestamp": result.timestamp.isoformat(),
            "similarity_score": result.similarity_score,
            "quality_difference": result.quality_difference,
            "trend_direction": result.trend_direction,
            "significant_changes": result.significant_changes,
            "recommendations": result.recommendations,
            "confidence_score": result.confidence_score,
            "comparison_type": result.comparison_type
        }
    
    @staticmethod
    def serialize_quality_report(report: QualityReport) -> Dict[str, Any]:
        """Serialize quality report to dictionary"""
        return {
            "id": report.id,
            "report_type": report.report_type,
            "generated_at": report.generated_at.isoformat(),
            "summary": report.summary,
            "average_metrics": report.average_metrics,
            "trends": report.trends,
            "outliers": report.outliers,
            "recommendations": report.recommendations,
            "time_window_start": report.time_window_start.isoformat() if report.time_window_start else None,
            "time_window_end": report.time_window_end.isoformat() if report.time_window_end else None,
            "total_entries_analyzed": report.total_entries_analyzed,
            "report_version": report.report_version
        }










