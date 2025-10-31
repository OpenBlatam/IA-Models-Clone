"""
API Endpoints
=============

This module defines the REST API endpoints for the AI History Comparison system.
Each endpoint class handles a specific domain of functionality.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from ..application.dto import (
    AnalyzeContentRequest, AnalyzeContentResponse,
    CompareModelsRequest, CompareModelsResponse,
    GenerateReportRequest, GenerateReportResponse,
    TrackTrendsRequest, TrackTrendsResponse,
    BulkAnalysisRequest, BulkAnalysisResponse,
    SearchEntriesRequest, SearchEntriesResponse,
    SystemSummary, ModelSummary, QualitySummary,
    ErrorResponse, PaginationRequest, PaginationResponse
)
from ..application.use_cases import (
    AnalyzeContentUseCase, CompareModelsUseCase,
    GenerateReportUseCase, TrackTrendsUseCase, ManageAnalysisJobUseCase
)
from ..infrastructure.repositories import (
    HistoryRepository, ComparisonRepository, ReportRepository,
    AnalysisJobRepository, UserFeedbackRepository
)

logger = logging.getLogger(__name__)


class AnalysisEndpoints:
    """Endpoints for content analysis operations"""
    
    router = APIRouter()
    
    @staticmethod
    def get_use_case(request: Request) -> AnalyzeContentUseCase:
        """Get analyze content use case from request state"""
        return request.app.state.analyze_content_use_case
    
    @staticmethod
    def get_history_repo(request: Request) -> HistoryRepository:
        """Get history repository from request state"""
        return request.app.state.history_repo
    
    @router.post("/analyze", response_model=AnalyzeContentResponse)
    async def analyze_content(
        request: AnalyzeContentRequest,
        use_case: AnalyzeContentUseCase = Depends(get_use_case)
    ):
        """Analyze content and create history entry"""
        try:
            response = use_case.execute(request)
            
            if not response.success:
                raise HTTPException(
                    status_code=400,
                    detail=response.error_message or "Analysis failed"
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in analyze_content endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/bulk-analyze", response_model=BulkAnalysisResponse)
    async def bulk_analyze_content(
        request: BulkAnalysisRequest,
        use_case: AnalyzeContentUseCase = Depends(get_use_case)
    ):
        """Analyze multiple content pieces"""
        try:
            start_time = datetime.utcnow()
            results = []
            successful = 0
            failed = 0
            
            for content in request.contents:
                analyze_request = AnalyzeContentRequest(
                    content=content,
                    model_version=request.model_version,
                    metadata=request.metadata
                )
                
                response = use_case.execute(analyze_request)
                results.append(response)
                
                if response.success:
                    successful += 1
                else:
                    failed += 1
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return BulkAnalysisResponse(
                results=results,
                total_processed=len(request.contents),
                successful=successful,
                failed=failed,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in bulk_analyze_content endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/entries/{entry_id}")
    async def get_entry(
        entry_id: str = Path(..., description="Entry ID"),
        repo: HistoryRepository = Depends(get_history_repo)
    ):
        """Get history entry by ID"""
        try:
            entry = repo.find_by_id(entry_id)
            if not entry:
                raise HTTPException(status_code=404, detail="Entry not found")
            
            return {
                "id": entry.id,
                "content": entry.content,
                "model_version": entry.model_version,
                "timestamp": entry.timestamp.isoformat(),
                "metrics": entry.metrics.to_dict(),
                "quality_score": entry.calculate_quality_score(),
                "metadata": entry.metadata
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting entry {entry_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/entries")
    async def search_entries(
        model_version: Optional[str] = Query(None, description="Filter by model version"),
        days: Optional[int] = Query(7, description="Number of recent days"),
        limit: Optional[int] = Query(100, description="Maximum number of entries"),
        repo: HistoryRepository = Depends(get_history_repo)
    ):
        """Search and filter history entries"""
        try:
            if days:
                entries = repo.find_recent_entries(days, model_version)
            elif model_version:
                entries = repo.find_by_model_version(model_version, limit)
            else:
                entries = repo.find_recent_entries(7, limit=limit)
            
            return {
                "entries": [
                    {
                        "id": entry.id,
                        "model_version": entry.model_version,
                        "timestamp": entry.timestamp.isoformat(),
                        "quality_score": entry.calculate_quality_score(),
                        "word_count": entry.metrics.word_count,
                        "readability_score": entry.metrics.readability_score
                    }
                    for entry in entries
                ],
                "total_count": len(entries)
            }
            
        except Exception as e:
            logger.error(f"Error searching entries: {e}")
            raise HTTPException(status_code=500, detail=str(e))


class ComparisonEndpoints:
    """Endpoints for model comparison operations"""
    
    router = APIRouter()
    
    @staticmethod
    def get_use_case(request: Request) -> CompareModelsUseCase:
        """Get compare models use case from request state"""
        return request.app.state.compare_models_use_case
    
    @router.post("/compare", response_model=CompareModelsResponse)
    async def compare_models(
        request: CompareModelsRequest,
        use_case: CompareModelsUseCase = Depends(get_use_case)
    ):
        """Compare two model entries"""
        try:
            response = use_case.execute(request)
            
            if not response.success:
                raise HTTPException(
                    status_code=400,
                    detail=response.error_message or "Comparison failed"
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in compare_models endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/compare-recent/{model_version}")
    async def compare_recent_entries(
        model_version: str = Path(..., description="Model version"),
        days: int = Query(7, description="Number of recent days"),
        use_case: CompareModelsUseCase = Depends(get_use_case)
    ):
        """Compare recent entries for a model"""
        try:
            comparisons = use_case.compare_recent_entries(model_version, days)
            
            return {
                "model_version": model_version,
                "days": days,
                "comparisons": [
                    {
                        "comparison_id": comp.comparison_id,
                        "similarity_score": comp.similarity_score,
                        "trend_direction": comp.trend_direction,
                        "confidence_score": comp.confidence_score,
                        "timestamp": comp.timestamp.isoformat()
                    }
                    for comp in comparisons
                ],
                "total_comparisons": len(comparisons)
            }
            
        except Exception as e:
            logger.error(f"Error comparing recent entries: {e}")
            raise HTTPException(status_code=500, detail=str(e))


class ReportEndpoints:
    """Endpoints for report generation operations"""
    
    router = APIRouter()
    
    @staticmethod
    def get_use_case(request: Request) -> GenerateReportUseCase:
        """Get generate report use case from request state"""
        return request.app.state.generate_report_use_case
    
    @router.post("/generate", response_model=GenerateReportResponse)
    async def generate_report(
        request: GenerateReportRequest,
        use_case: GenerateReportUseCase = Depends(get_use_case)
    ):
        """Generate quality report"""
        try:
            response = use_case.execute(request)
            
            if not response.success:
                raise HTTPException(
                    status_code=400,
                    detail=response.error_message or "Report generation failed"
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in generate_report endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/reports")
    async def list_reports(
        report_type: Optional[str] = Query(None, description="Filter by report type"),
        limit: Optional[int] = Query(10, description="Maximum number of reports"),
        request: Request = None
    ):
        """List available reports"""
        try:
            repo = request.app.state.report_repo
            
            if report_type:
                reports = repo.find_by_type(report_type, limit)
            else:
                reports = repo.find_recent_reports(30)[:limit]
            
            return {
                "reports": [
                    {
                        "id": report.id,
                        "report_type": report.report_type,
                        "generated_at": report.generated_at.isoformat(),
                        "total_entries": report.total_entries_analyzed,
                        "summary": report.summary
                    }
                    for report in reports
                ],
                "total_count": len(reports)
            }
            
        except Exception as e:
            logger.error(f"Error listing reports: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/reports/{report_id}")
    async def get_report(
        report_id: str = Path(..., description="Report ID"),
        request: Request = None
    ):
        """Get specific report by ID"""
        try:
            repo = request.app.state.report_repo
            report = repo.find_by_id(report_id)
            
            if not report:
                raise HTTPException(status_code=404, detail="Report not found")
            
            return {
                "id": report.id,
                "report_type": report.report_type,
                "generated_at": report.generated_at.isoformat(),
                "summary": report.summary,
                "average_metrics": report.average_metrics,
                "trends": report.trends,
                "outliers": report.outliers,
                "recommendations": report.recommendations,
                "total_entries": report.total_entries_analyzed
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting report {report_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))


class TrendEndpoints:
    """Endpoints for trend analysis operations"""
    
    router = APIRouter()
    
    @staticmethod
    def get_use_case(request: Request) -> TrackTrendsUseCase:
        """Get track trends use case from request state"""
        return request.app.state.track_trends_use_case
    
    @router.post("/analyze", response_model=TrackTrendsResponse)
    async def analyze_trends(
        request: TrackTrendsRequest,
        use_case: TrackTrendsUseCase = Depends(get_use_case)
    ):
        """Analyze trends for a model"""
        try:
            response = use_case.execute(request)
            
            if not response.success:
                raise HTTPException(
                    status_code=400,
                    detail=response.error_message or "Trend analysis failed"
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in analyze_trends endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/metrics/{model_version}")
    async def track_all_metrics(
        model_version: str = Path(..., description="Model version"),
        days: int = Query(30, description="Number of days to analyze"),
        use_case: TrackTrendsUseCase = Depends(get_use_case)
    ):
        """Track trends for all metrics of a model"""
        try:
            trend_responses = use_case.track_all_metrics(model_version, days)
            
            return {
                "model_version": model_version,
                "days": days,
                "trends": [
                    {
                        "metric": trend.metric,
                        "trend_direction": trend.trend_direction,
                        "trend_strength": trend.trend_strength,
                        "confidence": trend.confidence,
                        "anomalies_count": len(trend.anomalies)
                    }
                    for trend in trend_responses
                ],
                "total_metrics": len(trend_responses)
            }
            
        except Exception as e:
            logger.error(f"Error tracking all metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))


class SystemEndpoints:
    """Endpoints for system management and monitoring"""
    
    router = APIRouter()
    
    @staticmethod
    def get_use_case(request: Request) -> ManageAnalysisJobUseCase:
        """Get manage jobs use case from request state"""
        return request.app.state.manage_jobs_use_case
    
    @router.get("/summary")
    async def get_system_summary(request: Request):
        """Get system summary information"""
        try:
            history_repo = request.app.state.history_repo
            comparison_repo = request.app.state.comparison_repo
            report_repo = request.app.state.report_repo
            job_repo = request.app.state.job_repo
            
            # Get counts
            total_entries = history_repo.count_entries()
            total_comparisons = comparison_repo.count_comparisons()
            total_reports = report_repo.count_reports()
            
            # Get active jobs
            running_jobs = job_repo.find_running_jobs()
            active_jobs = len(running_jobs)
            
            # Get model versions
            model_versions = history_repo.get_model_versions()
            
            # Get last analysis
            recent_entries = history_repo.find_recent_entries(1)
            last_analysis = recent_entries[0].timestamp if recent_entries else None
            
            return SystemSummary(
                total_entries=total_entries,
                total_comparisons=total_comparisons,
                total_reports=total_reports,
                active_jobs=active_jobs,
                model_versions=model_versions,
                last_analysis=last_analysis,
                system_health="healthy"
            )
            
        except Exception as e:
            logger.error(f"Error getting system summary: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/models")
    async def get_model_summaries(request: Request):
        """Get summaries for all models"""
        try:
            history_repo = request.app.state.history_repo
            model_versions = history_repo.get_model_versions()
            
            summaries = []
            for model_version in model_versions:
                entries = history_repo.find_by_model_version(model_version)
                
                if entries:
                    quality_scores = [e.calculate_quality_score() for e in entries]
                    average_quality = sum(quality_scores) / len(quality_scores)
                    last_analysis = max(e.timestamp for e in entries)
                    
                    summaries.append(ModelSummary(
                        model_version=model_version,
                        total_entries=len(entries),
                        average_quality=average_quality,
                        last_analysis=last_analysis,
                        trend_direction="stable",  # Would need trend analysis
                        performance_metrics={}  # Would need to calculate
                    ))
            
            return {"models": summaries}
            
        except Exception as e:
            logger.error(f"Error getting model summaries: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/jobs")
    async def get_jobs(
        status: Optional[str] = Query(None, description="Filter by job status"),
        use_case: ManageAnalysisJobUseCase = Depends(get_use_case)
    ):
        """Get analysis jobs"""
        try:
            if status == "pending":
                jobs = use_case.get_pending_jobs()
            elif status == "running":
                jobs = use_case.get_running_jobs()
            else:
                # Get all jobs
                pending = use_case.get_pending_jobs()
                running = use_case.get_running_jobs()
                jobs = pending + running
            
            return {"jobs": jobs}
            
        except Exception as e:
            logger.error(f"Error getting jobs: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/jobs/{job_id}")
    async def get_job_status(
        job_id: str = Path(..., description="Job ID"),
        use_case: ManageAnalysisJobUseCase = Depends(get_use_case)
    ):
        """Get specific job status"""
        try:
            job_status = use_case.get_job_status(job_id)
            
            if not job_status:
                raise HTTPException(status_code=404, detail="Job not found")
            
            return job_status
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting job status {job_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/jobs/statistics")
    async def get_job_statistics(
        use_case: ManageAnalysisJobUseCase = Depends(get_use_case)
    ):
        """Get job statistics"""
        try:
            stats = use_case.get_job_statistics()
            return stats
            
        except Exception as e:
            logger.error(f"Error getting job statistics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.delete("/jobs/{job_id}")
    async def cancel_job(
        job_id: str = Path(..., description="Job ID"),
        use_case: ManageAnalysisJobUseCase = Depends(get_use_case)
    ):
        """Cancel a job"""
        try:
            success = use_case.cancel_job(job_id)
            
            if not success:
                raise HTTPException(status_code=400, detail="Job cannot be cancelled")
            
            return {"message": "Job cancelled successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))




