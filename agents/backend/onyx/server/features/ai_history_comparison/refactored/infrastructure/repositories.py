"""
Repository Pattern Implementation
================================

This module implements the repository pattern for data access, providing
a clean abstraction over database operations for domain entities.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func
import logging

from ..core.domain import (
    HistoryEntry, ComparisonResult, TrendAnalysis, QualityReport,
    AnalysisJob, UserFeedback, ContentMetrics, PerformanceMetric,
    TrendDirection, AnalysisStatus
)

logger = logging.getLogger(__name__)


class BaseRepository(ABC):
    """Base repository class with common functionality"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def save(self, entity):
        """Save entity to database"""
        try:
            self.session.add(entity)
            self.session.flush()
            return entity
        except Exception as e:
            logger.error(f"Error saving entity: {e}")
            raise
    
    def delete(self, entity):
        """Delete entity from database"""
        try:
            self.session.delete(entity)
        except Exception as e:
            logger.error(f"Error deleting entity: {e}")
            raise
    
    def commit(self):
        """Commit transaction"""
        try:
            self.session.commit()
        except Exception as e:
            logger.error(f"Error committing transaction: {e}")
            raise


class HistoryRepository(BaseRepository):
    """Repository for HistoryEntry entities"""
    
    def find_by_id(self, entry_id: str) -> Optional[HistoryEntry]:
        """Find history entry by ID"""
        try:
            return self.session.query(HistoryEntry).filter(
                HistoryEntry.id == entry_id
            ).first()
        except Exception as e:
            logger.error(f"Error finding entry by ID {entry_id}: {e}")
            raise
    
    def find_by_model_version(self, model_version: str, 
                            limit: Optional[int] = None) -> List[HistoryEntry]:
        """Find entries by model version"""
        try:
            query = self.session.query(HistoryEntry).filter(
                HistoryEntry.model_version == model_version
            ).order_by(desc(HistoryEntry.timestamp))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            logger.error(f"Error finding entries by model version {model_version}: {e}")
            raise
    
    def find_by_time_range(self, start_time: datetime, end_time: datetime,
                          model_version: Optional[str] = None) -> List[HistoryEntry]:
        """Find entries within time range"""
        try:
            query = self.session.query(HistoryEntry).filter(
                and_(
                    HistoryEntry.timestamp >= start_time,
                    HistoryEntry.timestamp <= end_time
                )
            )
            
            if model_version:
                query = query.filter(HistoryEntry.model_version == model_version)
            
            return query.order_by(asc(HistoryEntry.timestamp)).all()
        except Exception as e:
            logger.error(f"Error finding entries in time range: {e}")
            raise
    
    def find_by_content_hash(self, content_hash: str) -> List[HistoryEntry]:
        """Find entries by content hash"""
        try:
            return self.session.query(HistoryEntry).filter(
                HistoryEntry.content_hash == content_hash
            ).order_by(desc(HistoryEntry.timestamp)).all()
        except Exception as e:
            logger.error(f"Error finding entries by content hash: {e}")
            raise
    
    def find_recent_entries(self, days: int = 7, 
                          model_version: Optional[str] = None) -> List[HistoryEntry]:
        """Find recent entries"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            return self.find_by_time_range(cutoff_time, datetime.utcnow(), model_version)
        except Exception as e:
            logger.error(f"Error finding recent entries: {e}")
            raise
    
    def find_high_quality_entries(self, threshold: float = 0.7,
                                limit: Optional[int] = None) -> List[HistoryEntry]:
        """Find high quality entries"""
        try:
            # This would need to be implemented with a custom query
            # since quality_score is calculated, not stored
            all_entries = self.session.query(HistoryEntry).all()
            high_quality = [e for e in all_entries if e.calculate_quality_score() >= threshold]
            
            if limit:
                high_quality = high_quality[:limit]
            
            return high_quality
        except Exception as e:
            logger.error(f"Error finding high quality entries: {e}")
            raise
    
    def count_entries(self, model_version: Optional[str] = None) -> int:
        """Count total entries"""
        try:
            query = self.session.query(HistoryEntry)
            if model_version:
                query = query.filter(HistoryEntry.model_version == model_version)
            return query.count()
        except Exception as e:
            logger.error(f"Error counting entries: {e}")
            raise
    
    def get_model_versions(self) -> List[str]:
        """Get all unique model versions"""
        try:
            results = self.session.query(HistoryEntry.model_version).distinct().all()
            return [result[0] for result in results]
        except Exception as e:
            logger.error(f"Error getting model versions: {e}")
            raise
    
    def get_entries_with_metrics(self, metric_name: str, 
                               min_value: Optional[float] = None,
                               max_value: Optional[float] = None) -> List[HistoryEntry]:
        """Find entries with specific metric values"""
        try:
            # This would need custom implementation based on metric storage
            # For now, return all entries and filter in memory
            all_entries = self.session.query(HistoryEntry).all()
            
            filtered_entries = []
            for entry in all_entries:
                metric_value = getattr(entry.metrics, metric_name, None)
                if metric_value is not None:
                    if min_value is not None and metric_value < min_value:
                        continue
                    if max_value is not None and metric_value > max_value:
                        continue
                    filtered_entries.append(entry)
            
            return filtered_entries
        except Exception as e:
            logger.error(f"Error finding entries with metrics: {e}")
            raise


class ComparisonRepository(BaseRepository):
    """Repository for ComparisonResult entities"""
    
    def find_by_id(self, comparison_id: str) -> Optional[ComparisonResult]:
        """Find comparison by ID"""
        try:
            return self.session.query(ComparisonResult).filter(
                ComparisonResult.id == comparison_id
            ).first()
        except Exception as e:
            logger.error(f"Error finding comparison by ID {comparison_id}: {e}")
            raise
    
    def find_by_entries(self, entry1_id: str, entry2_id: str) -> List[ComparisonResult]:
        """Find comparisons between specific entries"""
        try:
            return self.session.query(ComparisonResult).filter(
                or_(
                    and_(ComparisonResult.entry1_id == entry1_id,
                         ComparisonResult.entry2_id == entry2_id),
                    and_(ComparisonResult.entry1_id == entry2_id,
                         ComparisonResult.entry2_id == entry1_id)
                )
            ).order_by(desc(ComparisonResult.timestamp)).all()
        except Exception as e:
            logger.error(f"Error finding comparisons between entries: {e}")
            raise
    
    def find_by_entry(self, entry_id: str) -> List[ComparisonResult]:
        """Find all comparisons involving an entry"""
        try:
            return self.session.query(ComparisonResult).filter(
                or_(
                    ComparisonResult.entry1_id == entry_id,
                    ComparisonResult.entry2_id == entry_id
                )
            ).order_by(desc(ComparisonResult.timestamp)).all()
        except Exception as e:
            logger.error(f"Error finding comparisons for entry {entry_id}: {e}")
            raise
    
    def find_by_time_range(self, start_time: datetime, end_time: datetime) -> List[ComparisonResult]:
        """Find comparisons within time range"""
        try:
            return self.session.query(ComparisonResult).filter(
                and_(
                    ComparisonResult.timestamp >= start_time,
                    ComparisonResult.timestamp <= end_time
                )
            ).order_by(desc(ComparisonResult.timestamp)).all()
        except Exception as e:
            logger.error(f"Error finding comparisons in time range: {e}")
            raise
    
    def find_recent_comparisons(self, days: int = 7) -> List[ComparisonResult]:
        """Find recent comparisons"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            return self.find_by_time_range(cutoff_time, datetime.utcnow())
        except Exception as e:
            logger.error(f"Error finding recent comparisons: {e}")
            raise
    
    def find_by_trend_direction(self, trend_direction: TrendDirection) -> List[ComparisonResult]:
        """Find comparisons by trend direction"""
        try:
            return self.session.query(ComparisonResult).filter(
                ComparisonResult.trend_direction == trend_direction
            ).order_by(desc(ComparisonResult.timestamp)).all()
        except Exception as e:
            logger.error(f"Error finding comparisons by trend direction: {e}")
            raise
    
    def find_significant_changes(self, threshold: float = 0.1) -> List[ComparisonResult]:
        """Find comparisons with significant changes"""
        try:
            return self.session.query(ComparisonResult).filter(
                ComparisonResult.similarity_score < (1.0 - threshold)
            ).order_by(desc(ComparisonResult.timestamp)).all()
        except Exception as e:
            logger.error(f"Error finding significant changes: {e}")
            raise
    
    def count_comparisons(self) -> int:
        """Count total comparisons"""
        try:
            return self.session.query(ComparisonResult).count()
        except Exception as e:
            logger.error(f"Error counting comparisons: {e}")
            raise


class ReportRepository(BaseRepository):
    """Repository for QualityReport entities"""
    
    def find_by_id(self, report_id: str) -> Optional[QualityReport]:
        """Find report by ID"""
        try:
            return self.session.query(QualityReport).filter(
                QualityReport.id == report_id
            ).first()
        except Exception as e:
            logger.error(f"Error finding report by ID {report_id}: {e}")
            raise
    
    def find_by_type(self, report_type: str, 
                    limit: Optional[int] = None) -> List[QualityReport]:
        """Find reports by type"""
        try:
            query = self.session.query(QualityReport).filter(
                QualityReport.report_type == report_type
            ).order_by(desc(QualityReport.generated_at))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            logger.error(f"Error finding reports by type {report_type}: {e}")
            raise
    
    def find_by_time_range(self, start_time: datetime, end_time: datetime) -> List[QualityReport]:
        """Find reports within time range"""
        try:
            return self.session.query(QualityReport).filter(
                and_(
                    QualityReport.generated_at >= start_time,
                    QualityReport.generated_at <= end_time
                )
            ).order_by(desc(QualityReport.generated_at)).all()
        except Exception as e:
            logger.error(f"Error finding reports in time range: {e}")
            raise
    
    def find_latest_report(self, report_type: str) -> Optional[QualityReport]:
        """Find latest report of specific type"""
        try:
            return self.session.query(QualityReport).filter(
                QualityReport.report_type == report_type
            ).order_by(desc(QualityReport.generated_at)).first()
        except Exception as e:
            logger.error(f"Error finding latest report: {e}")
            raise
    
    def find_recent_reports(self, days: int = 30) -> List[QualityReport]:
        """Find recent reports"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            return self.find_by_time_range(cutoff_time, datetime.utcnow())
        except Exception as e:
            logger.error(f"Error finding recent reports: {e}")
            raise
    
    def count_reports(self, report_type: Optional[str] = None) -> int:
        """Count total reports"""
        try:
            query = self.session.query(QualityReport)
            if report_type:
                query = query.filter(QualityReport.report_type == report_type)
            return query.count()
        except Exception as e:
            logger.error(f"Error counting reports: {e}")
            raise


class AnalysisJobRepository(BaseRepository):
    """Repository for AnalysisJob entities"""
    
    def find_by_id(self, job_id: str) -> Optional[AnalysisJob]:
        """Find job by ID"""
        try:
            return self.session.query(AnalysisJob).filter(
                AnalysisJob.id == job_id
            ).first()
        except Exception as e:
            logger.error(f"Error finding job by ID {job_id}: {e}")
            raise
    
    def find_by_status(self, status: AnalysisStatus) -> List[AnalysisJob]:
        """Find jobs by status"""
        try:
            return self.session.query(AnalysisJob).filter(
                AnalysisJob.status == status
            ).order_by(desc(AnalysisJob.created_at)).all()
        except Exception as e:
            logger.error(f"Error finding jobs by status: {e}")
            raise
    
    def find_by_type(self, job_type: str) -> List[AnalysisJob]:
        """Find jobs by type"""
        try:
            return self.session.query(AnalysisJob).filter(
                AnalysisJob.job_type == job_type
            ).order_by(desc(AnalysisJob.created_at)).all()
        except Exception as e:
            logger.error(f"Error finding jobs by type: {e}")
            raise
    
    def find_pending_jobs(self) -> List[AnalysisJob]:
        """Find pending jobs"""
        try:
            return self.find_by_status(AnalysisStatus.PENDING)
        except Exception as e:
            logger.error(f"Error finding pending jobs: {e}")
            raise
    
    def find_running_jobs(self) -> List[AnalysisJob]:
        """Find running jobs"""
        try:
            return self.find_by_status(AnalysisStatus.IN_PROGRESS)
        except Exception as e:
            logger.error(f"Error finding running jobs: {e}")
            raise
    
    def find_completed_jobs(self, limit: Optional[int] = None) -> List[AnalysisJob]:
        """Find completed jobs"""
        try:
            query = self.session.query(AnalysisJob).filter(
                AnalysisJob.status == AnalysisStatus.COMPLETED
            ).order_by(desc(AnalysisJob.completed_at))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            logger.error(f"Error finding completed jobs: {e}")
            raise
    
    def find_failed_jobs(self) -> List[AnalysisJob]:
        """Find failed jobs"""
        try:
            return self.find_by_status(AnalysisStatus.FAILED)
        except Exception as e:
            logger.error(f"Error finding failed jobs: {e}")
            raise
    
    def find_by_time_range(self, start_time: datetime, end_time: datetime) -> List[AnalysisJob]:
        """Find jobs within time range"""
        try:
            return self.session.query(AnalysisJob).filter(
                and_(
                    AnalysisJob.created_at >= start_time,
                    AnalysisJob.created_at <= end_time
                )
            ).order_by(desc(AnalysisJob.created_at)).all()
        except Exception as e:
            logger.error(f"Error finding jobs in time range: {e}")
            raise
    
    def count_jobs_by_status(self) -> Dict[str, int]:
        """Count jobs by status"""
        try:
            results = self.session.query(
                AnalysisJob.status,
                func.count(AnalysisJob.id)
            ).group_by(AnalysisJob.status).all()
            
            return {status.value: count for status, count in results}
        except Exception as e:
            logger.error(f"Error counting jobs by status: {e}")
            raise


class UserFeedbackRepository(BaseRepository):
    """Repository for UserFeedback entities"""
    
    def find_by_id(self, feedback_id: str) -> Optional[UserFeedback]:
        """Find feedback by ID"""
        try:
            return self.session.query(UserFeedback).filter(
                UserFeedback.id == feedback_id
            ).first()
        except Exception as e:
            logger.error(f"Error finding feedback by ID {feedback_id}: {e}")
            raise
    
    def find_by_entry(self, entry_id: str) -> List[UserFeedback]:
        """Find feedback for specific entry"""
        try:
            return self.session.query(UserFeedback).filter(
                UserFeedback.entry_id == entry_id
            ).order_by(desc(UserFeedback.timestamp)).all()
        except Exception as e:
            logger.error(f"Error finding feedback for entry {entry_id}: {e}")
            raise
    
    def find_by_user(self, user_id: str) -> List[UserFeedback]:
        """Find feedback by user"""
        try:
            return self.session.query(UserFeedback).filter(
                UserFeedback.user_id == user_id
            ).order_by(desc(UserFeedback.timestamp)).all()
        except Exception as e:
            logger.error(f"Error finding feedback by user {user_id}: {e}")
            raise
    
    def find_by_type(self, feedback_type: str) -> List[UserFeedback]:
        """Find feedback by type"""
        try:
            return self.session.query(UserFeedback).filter(
                UserFeedback.feedback_type == feedback_type
            ).order_by(desc(UserFeedback.timestamp)).all()
        except Exception as e:
            logger.error(f"Error finding feedback by type: {e}")
            raise
    
    def find_high_rated_feedback(self, min_rating: int = 4) -> List[UserFeedback]:
        """Find high-rated feedback"""
        try:
            return self.session.query(UserFeedback).filter(
                UserFeedback.rating >= min_rating
            ).order_by(desc(UserFeedback.timestamp)).all()
        except Exception as e:
            logger.error(f"Error finding high-rated feedback: {e}")
            raise
    
    def find_low_rated_feedback(self, max_rating: int = 2) -> List[UserFeedback]:
        """Find low-rated feedback"""
        try:
            return self.session.query(UserFeedback).filter(
                UserFeedback.rating <= max_rating
            ).order_by(desc(UserFeedback.timestamp)).all()
        except Exception as e:
            logger.error(f"Error finding low-rated feedback: {e}")
            raise
    
    def get_average_rating(self, entry_id: Optional[str] = None) -> float:
        """Get average rating"""
        try:
            query = self.session.query(func.avg(UserFeedback.rating))
            
            if entry_id:
                query = query.filter(UserFeedback.entry_id == entry_id)
            
            result = query.scalar()
            return float(result) if result is not None else 0.0
        except Exception as e:
            logger.error(f"Error getting average rating: {e}")
            raise
    
    def count_feedback(self, entry_id: Optional[str] = None) -> int:
        """Count feedback entries"""
        try:
            query = self.session.query(UserFeedback)
            if entry_id:
                query = query.filter(UserFeedback.entry_id == entry_id)
            return query.count()
        except Exception as e:
            logger.error(f"Error counting feedback: {e}")
            raise




