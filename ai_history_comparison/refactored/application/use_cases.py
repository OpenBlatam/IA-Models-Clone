"""
Use Cases
=========

This module contains the use cases that implement the business workflows
of the AI History Comparison system. Use cases orchestrate domain services
and coordinate with infrastructure components.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from ..core.domain import (
    HistoryEntry, ComparisonResult, TrendAnalysis, QualityReport,
    AnalysisJob, ContentMetrics, PerformanceMetric, TrendDirection,
    AnalysisStatus, AnalysisCompletedEvent, ModelComparisonEvent
)
from ..core.services import (
    ContentAnalysisService, ModelComparisonService,
    TrendAnalysisService, QualityAssessmentService
)
from ..infrastructure.repositories import (
    HistoryRepository, ComparisonRepository, ReportRepository,
    AnalysisJobRepository, UserFeedbackRepository
)
from .dto import (
    AnalyzeContentRequest, AnalyzeContentResponse,
    CompareModelsRequest, CompareModelsResponse,
    GenerateReportRequest, GenerateReportResponse,
    TrackTrendsRequest, TrackTrendsResponse
)

logger = logging.getLogger(__name__)


class BaseUseCase(ABC):
    """Base use case class"""
    
    def __init__(self, 
                 history_repo: HistoryRepository,
                 comparison_repo: ComparisonRepository,
                 report_repo: ReportRepository,
                 job_repo: AnalysisJobRepository):
        self.history_repo = history_repo
        self.comparison_repo = comparison_repo
        self.report_repo = report_repo
        self.job_repo = job_repo


class AnalyzeContentUseCase(BaseUseCase):
    """Use case for analyzing content and creating history entries"""
    
    def __init__(self, 
                 history_repo: HistoryRepository,
                 comparison_repo: ComparisonRepository,
                 report_repo: ReportRepository,
                 job_repo: AnalysisJobRepository,
                 content_analysis_service: ContentAnalysisService):
        super().__init__(history_repo, comparison_repo, report_repo, job_repo)
        self.content_analysis_service = content_analysis_service
    
    def execute(self, request: AnalyzeContentRequest) -> AnalyzeContentResponse:
        """Execute content analysis"""
        try:
            # Analyze content
            metrics = self.content_analysis_service.analyze_content(request.content)
            
            # Create history entry
            entry = HistoryEntry.create(
                content=request.content,
                model_version=request.model_version,
                metrics=metrics,
                metadata=request.metadata
            )
            
            # Save to repository
            saved_entry = self.history_repo.save(entry)
            self.history_repo.commit()
            
            # Create analysis job record
            job = AnalysisJob.create(
                job_type="content_analysis",
                parameters={
                    "content_length": len(request.content),
                    "model_version": request.model_version
                }
            )
            job.complete({"entry_id": saved_entry.id})
            self.job_repo.save(job)
            self.job_repo.commit()
            
            logger.info(f"Content analysis completed for entry {saved_entry.id}")
            
            return AnalyzeContentResponse(
                entry_id=saved_entry.id,
                metrics=metrics.to_dict(),
                quality_score=saved_entry.calculate_quality_score(),
                timestamp=saved_entry.timestamp,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in content analysis: {e}")
            return AnalyzeContentResponse(
                entry_id=None,
                metrics={},
                quality_score=0.0,
                timestamp=datetime.utcnow(),
                success=False,
                error_message=str(e)
            )


class CompareModelsUseCase(BaseUseCase):
    """Use case for comparing AI models and their outputs"""
    
    def __init__(self, 
                 history_repo: HistoryRepository,
                 comparison_repo: ComparisonRepository,
                 report_repo: ReportRepository,
                 job_repo: AnalysisJobRepository,
                 model_comparison_service: ModelComparisonService):
        super().__init__(history_repo, comparison_repo, report_repo, job_repo)
        self.model_comparison_service = model_comparison_service
    
    def execute(self, request: CompareModelsRequest) -> CompareModelsResponse:
        """Execute model comparison"""
        try:
            # Get entries to compare
            entry1 = self.history_repo.find_by_id(request.entry1_id)
            entry2 = self.history_repo.find_by_id(request.entry2_id)
            
            if not entry1 or not entry2:
                raise ValueError("One or both entries not found")
            
            # Perform comparison
            comparison = self.model_comparison_service.compare_entries(entry1, entry2)
            
            # Save comparison result
            saved_comparison = self.comparison_repo.save(comparison)
            self.comparison_repo.commit()
            
            # Create analysis job record
            job = AnalysisJob.create(
                job_type="model_comparison",
                parameters={
                    "entry1_id": request.entry1_id,
                    "entry2_id": request.entry2_id,
                    "comparison_type": request.comparison_type
                }
            )
            job.complete({"comparison_id": saved_comparison.id})
            self.job_repo.save(job)
            self.job_repo.commit()
            
            logger.info(f"Model comparison completed: {saved_comparison.id}")
            
            return CompareModelsResponse(
                comparison_id=saved_comparison.id,
                similarity_score=saved_comparison.similarity_score,
                quality_difference=saved_comparison.quality_difference,
                trend_direction=saved_comparison.trend_direction.value,
                significant_changes=saved_comparison.significant_changes,
                recommendations=saved_comparison.recommendations,
                confidence_score=saved_comparison.confidence_score,
                timestamp=saved_comparison.timestamp,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in model comparison: {e}")
            return CompareModelsResponse(
                comparison_id=None,
                similarity_score=0.0,
                quality_difference={},
                trend_direction="stable",
                significant_changes=[],
                recommendations=[],
                confidence_score=0.0,
                timestamp=datetime.utcnow(),
                success=False,
                error_message=str(e)
            )
    
    def compare_recent_entries(self, model_version: str, days: int = 7) -> List[CompareModelsResponse]:
        """Compare recent entries for a model"""
        try:
            # Get recent entries
            recent_entries = self.history_repo.find_recent_entries(days, model_version)
            
            if len(recent_entries) < 2:
                return []
            
            # Compare consecutive entries
            comparisons = []
            for i in range(len(recent_entries) - 1):
                request = CompareModelsRequest(
                    entry1_id=recent_entries[i].id,
                    entry2_id=recent_entries[i + 1].id,
                    comparison_type="temporal_comparison"
                )
                comparison = self.execute(request)
                comparisons.append(comparison)
            
            return comparisons
            
        except Exception as e:
            logger.error(f"Error comparing recent entries: {e}")
            return []


class GenerateReportUseCase(BaseUseCase):
    """Use case for generating quality reports"""
    
    def __init__(self, 
                 history_repo: HistoryRepository,
                 comparison_repo: ComparisonRepository,
                 report_repo: ReportRepository,
                 job_repo: AnalysisJobRepository,
                 quality_assessment_service: QualityAssessmentService):
        super().__init__(history_repo, comparison_repo, report_repo, job_repo)
        self.quality_assessment_service = quality_assessment_service
    
    def execute(self, request: GenerateReportRequest) -> GenerateReportResponse:
        """Execute report generation"""
        try:
            # Get entries for report
            if request.entry_ids:
                entries = []
                for entry_id in request.entry_ids:
                    entry = self.history_repo.find_by_id(entry_id)
                    if entry:
                        entries.append(entry)
            elif request.model_version:
                entries = self.history_repo.find_by_model_version(request.model_version)
            elif request.time_range:
                start_time, end_time = request.time_range
                entries = self.history_repo.find_by_time_range(start_time, end_time)
            else:
                # Default to recent entries
                entries = self.history_repo.find_recent_entries(request.days or 30)
            
            if not entries:
                raise ValueError("No entries found for report generation")
            
            # Generate report
            report = self.quality_assessment_service.generate_quality_report(
                entries, request.report_type
            )
            
            # Save report
            saved_report = self.report_repo.save(report)
            self.report_repo.commit()
            
            # Create analysis job record
            job = AnalysisJob.create(
                job_type="report_generation",
                parameters={
                    "report_type": request.report_type,
                    "entries_count": len(entries),
                    "time_range": request.time_range
                }
            )
            job.complete({"report_id": saved_report.id})
            self.job_repo.save(job)
            self.job_repo.commit()
            
            logger.info(f"Quality report generated: {saved_report.id}")
            
            return GenerateReportResponse(
                report_id=saved_report.id,
                report_type=saved_report.report_type,
                summary=saved_report.summary,
                average_metrics=saved_report.average_metrics,
                trends=saved_report.trends,
                outliers=saved_report.outliers,
                recommendations=saved_report.recommendations,
                total_entries=saved_report.total_entries_analyzed,
                generated_at=saved_report.generated_at,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return GenerateReportResponse(
                report_id=None,
                report_type=request.report_type,
                summary={},
                average_metrics={},
                trends={},
                outliers=[],
                recommendations=[],
                total_entries=0,
                generated_at=datetime.utcnow(),
                success=False,
                error_message=str(e)
            )


class TrackTrendsUseCase(BaseUseCase):
    """Use case for tracking trends in model performance"""
    
    def __init__(self, 
                 history_repo: HistoryRepository,
                 comparison_repo: ComparisonRepository,
                 report_repo: ReportRepository,
                 job_repo: AnalysisJobRepository,
                 trend_analysis_service: TrendAnalysisService):
        super().__init__(history_repo, comparison_repo, report_repo, job_repo)
        self.trend_analysis_service = trend_analysis_service
    
    def execute(self, request: TrackTrendsRequest) -> TrackTrendsResponse:
        """Execute trend analysis"""
        try:
            # Get entries for trend analysis
            if request.entry_ids:
                entries = []
                for entry_id in request.entry_ids:
                    entry = self.history_repo.find_by_id(entry_id)
                    if entry:
                        entries.append(entry)
            elif request.model_version:
                entries = self.history_repo.find_by_model_version(request.model_version)
            elif request.time_range:
                start_time, end_time = request.time_range
                entries = self.history_repo.find_by_time_range(start_time, end_time)
            else:
                # Default to recent entries
                entries = self.history_repo.find_recent_entries(request.days or 30)
            
            if len(entries) < 2:
                raise ValueError("Need at least 2 entries for trend analysis")
            
            # Perform trend analysis
            trend_analysis = self.trend_analysis_service.analyze_trends(
                entries, request.metric
            )
            
            # Create analysis job record
            job = AnalysisJob.create(
                job_type="trend_analysis",
                parameters={
                    "metric": request.metric.value,
                    "model_version": request.model_version,
                    "entries_count": len(entries)
                }
            )
            job.complete({"trend_analysis_id": trend_analysis.id})
            self.job_repo.save(job)
            self.job_repo.commit()
            
            logger.info(f"Trend analysis completed: {trend_analysis.id}")
            
            return TrackTrendsResponse(
                trend_analysis_id=trend_analysis.id,
                model_name=trend_analysis.model_name,
                metric=trend_analysis.metric.value,
                trend_direction=trend_analysis.trend_direction.value,
                trend_strength=trend_analysis.trend_strength,
                confidence=trend_analysis.confidence,
                forecast=trend_analysis.forecast,
                anomalies=trend_analysis.anomalies,
                timestamp=trend_analysis.timestamp,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return TrackTrendsResponse(
                trend_analysis_id=None,
                model_name=request.model_version or "unknown",
                metric=request.metric.value,
                trend_direction="stable",
                trend_strength=0.0,
                confidence=0.0,
                forecast=[],
                anomalies=[],
                timestamp=datetime.utcnow(),
                success=False,
                error_message=str(e)
            )
    
    def track_all_metrics(self, model_version: str, days: int = 30) -> List[TrackTrendsResponse]:
        """Track trends for all metrics"""
        try:
            # Get recent entries
            entries = self.history_repo.find_recent_entries(days, model_version)
            
            if len(entries) < 2:
                return []
            
            # Track trends for each metric
            trend_responses = []
            for metric in PerformanceMetric:
                request = TrackTrendsRequest(
                    model_version=model_version,
                    metric=metric,
                    days=days
                )
                trend_response = self.execute(request)
                trend_responses.append(trend_response)
            
            return trend_responses
            
        except Exception as e:
            logger.error(f"Error tracking all metrics: {e}")
            return []


class ManageAnalysisJobUseCase(BaseUseCase):
    """Use case for managing analysis jobs"""
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        try:
            job = self.job_repo.find_by_id(job_id)
            if not job:
                return None
            
            return {
                "id": job.id,
                "job_type": job.job_type,
                "status": job.status.value,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "progress_percentage": job.progress_percentage,
                "total_items": job.total_items,
                "processed_items": job.processed_items,
                "error_message": job.error_message,
                "result": job.result
            }
            
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return None
    
    def get_pending_jobs(self) -> List[Dict[str, Any]]:
        """Get all pending jobs"""
        try:
            jobs = self.job_repo.find_pending_jobs()
            return [self.get_job_status(job.id) for job in jobs if self.get_job_status(job.id)]
        except Exception as e:
            logger.error(f"Error getting pending jobs: {e}")
            return []
    
    def get_running_jobs(self) -> List[Dict[str, Any]]:
        """Get all running jobs"""
        try:
            jobs = self.job_repo.find_running_jobs()
            return [self.get_job_status(job.id) for job in jobs if self.get_job_status(job.id)]
        except Exception as e:
            logger.error(f"Error getting running jobs: {e}")
            return []
    
    def get_job_statistics(self) -> Dict[str, Any]:
        """Get job statistics"""
        try:
            status_counts = self.job_repo.count_jobs_by_status()
            return {
                "total_jobs": sum(status_counts.values()),
                "by_status": status_counts,
                "pending": status_counts.get("pending", 0),
                "in_progress": status_counts.get("in_progress", 0),
                "completed": status_counts.get("completed", 0),
                "failed": status_counts.get("failed", 0)
            }
        except Exception as e:
            logger.error(f"Error getting job statistics: {e}")
            return {}
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        try:
            job = self.job_repo.find_by_id(job_id)
            if not job:
                return False
            
            if job.status in [AnalysisStatus.PENDING, AnalysisStatus.IN_PROGRESS]:
                job.status = AnalysisStatus.CANCELLED
                job.completed_at = datetime.utcnow()
                self.job_repo.save(job)
                self.job_repo.commit()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling job: {e}")
            return False




