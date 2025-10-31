"""
Data Transfer Objects (DTOs)
===========================

This module defines the data transfer objects used for API communication
and data exchange between layers of the application.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from ..core.domain import PerformanceMetric, TrendDirection


class ReportType(Enum):
    """Types of reports"""
    COMPREHENSIVE = "comprehensive"
    QUALITY_SUMMARY = "quality_summary"
    TREND_ANALYSIS = "trend_analysis"
    MODEL_COMPARISON = "model_comparison"
    PERFORMANCE_METRICS = "performance_metrics"


class ComparisonType(Enum):
    """Types of comparisons"""
    CONTENT_SIMILARITY = "content_similarity"
    TEMPORAL_COMPARISON = "temporal_comparison"
    MODEL_VERSION = "model_version"
    QUALITY_ASSESSMENT = "quality_assessment"


# Request DTOs
@dataclass
class AnalyzeContentRequest:
    """Request for content analysis"""
    content: str
    model_version: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CompareModelsRequest:
    """Request for model comparison"""
    entry1_id: str
    entry2_id: str
    comparison_type: str = "content_similarity"


@dataclass
class GenerateReportRequest:
    """Request for report generation"""
    report_type: str = "comprehensive"
    entry_ids: Optional[List[str]] = None
    model_version: Optional[str] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    days: Optional[int] = None


@dataclass
class TrackTrendsRequest:
    """Request for trend analysis"""
    model_version: Optional[str] = None
    metric: PerformanceMetric = PerformanceMetric.QUALITY_SCORE
    entry_ids: Optional[List[str]] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    days: Optional[int] = None


@dataclass
class BulkAnalysisRequest:
    """Request for bulk analysis"""
    contents: List[str]
    model_version: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchEntriesRequest:
    """Request for searching entries"""
    query: Optional[str] = None
    model_version: Optional[str] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    min_quality_score: Optional[float] = None
    max_quality_score: Optional[float] = None
    limit: Optional[int] = None
    offset: Optional[int] = None


@dataclass
class CreateUserFeedbackRequest:
    """Request for creating user feedback"""
    entry_id: str
    feedback_type: str
    rating: Optional[int] = None
    feedback_text: Optional[str] = None
    feedback_data: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None


# Response DTOs
@dataclass
class AnalyzeContentResponse:
    """Response for content analysis"""
    entry_id: Optional[str]
    metrics: Dict[str, Any]
    quality_score: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None


@dataclass
class CompareModelsResponse:
    """Response for model comparison"""
    comparison_id: Optional[str]
    similarity_score: float
    quality_difference: Dict[str, float]
    trend_direction: str
    significant_changes: List[str]
    recommendations: List[str]
    confidence_score: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None


@dataclass
class GenerateReportResponse:
    """Response for report generation"""
    report_id: Optional[str]
    report_type: str
    summary: Dict[str, Any]
    average_metrics: Dict[str, float]
    trends: Dict[str, Any]
    outliers: List[Dict[str, Any]]
    recommendations: List[str]
    total_entries: Optional[int]
    generated_at: datetime
    success: bool
    error_message: Optional[str] = None


@dataclass
class TrackTrendsResponse:
    """Response for trend analysis"""
    trend_analysis_id: Optional[str]
    model_name: str
    metric: str
    trend_direction: str
    trend_strength: float
    confidence: float
    forecast: List[Tuple[datetime, float]]
    anomalies: List[Tuple[datetime, float]]
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None


@dataclass
class BulkAnalysisResponse:
    """Response for bulk analysis"""
    results: List[AnalyzeContentResponse]
    total_processed: int
    successful: int
    failed: int
    processing_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class SearchEntriesResponse:
    """Response for entry search"""
    entries: List[Dict[str, Any]]
    total_count: int
    limit: int
    offset: int
    has_more: bool
    success: bool
    error_message: Optional[str] = None


@dataclass
class UserFeedbackResponse:
    """Response for user feedback operations"""
    feedback_id: Optional[str]
    entry_id: str
    user_id: Optional[str]
    rating: Optional[int]
    feedback_type: str
    feedback_text: Optional[str]
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None


# Summary DTOs
@dataclass
class SystemSummary:
    """System summary information"""
    total_entries: int
    total_comparisons: int
    total_reports: int
    active_jobs: int
    model_versions: List[str]
    last_analysis: Optional[datetime]
    system_health: str


@dataclass
class ModelSummary:
    """Model summary information"""
    model_version: str
    total_entries: int
    average_quality: float
    last_analysis: Optional[datetime]
    trend_direction: str
    performance_metrics: Dict[str, float]


@dataclass
class QualitySummary:
    """Quality summary information"""
    average_quality: float
    high_quality_count: int
    low_quality_count: int
    quality_distribution: Dict[str, int]
    recent_trend: str
    recommendations: List[str]


# Error DTOs
@dataclass
class ErrorResponse:
    """Error response"""
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class ValidationError:
    """Validation error details"""
    field: str
    message: str
    value: Any = None


@dataclass
class ValidationErrorResponse(ErrorResponse):
    """Validation error response"""
    validation_errors: List[ValidationError]


# Pagination DTOs
@dataclass
class PaginationRequest:
    """Pagination request parameters"""
    page: int = 1
    page_size: int = 20
    sort_by: Optional[str] = None
    sort_order: str = "desc"  # "asc" or "desc"


@dataclass
class PaginationResponse:
    """Pagination response information"""
    page: int
    page_size: int
    total_count: int
    total_pages: int
    has_next: bool
    has_previous: bool


# Filter DTOs
@dataclass
class DateRangeFilter:
    """Date range filter"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


@dataclass
class QualityRangeFilter:
    """Quality range filter"""
    min_quality: Optional[float] = None
    max_quality: Optional[float] = None


@dataclass
class ModelFilter:
    """Model filter"""
    model_versions: Optional[List[str]] = None
    exclude_models: Optional[List[str]] = None


@dataclass
class ContentFilter:
    """Content filter"""
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    contains_text: Optional[str] = None
    exclude_text: Optional[str] = None


# Export DTOs
@dataclass
class ExportRequest:
    """Export request"""
    format: str  # "json", "csv", "excel"
    filters: Optional[Dict[str, Any]] = None
    include_metrics: bool = True
    include_content: bool = False
    compression: bool = False


@dataclass
class ExportResponse:
    """Export response"""
    export_id: str
    download_url: Optional[str] = None
    file_size: Optional[int] = None
    expires_at: Optional[datetime] = None
    success: bool
    error_message: Optional[str] = None


# Configuration DTOs
@dataclass
class SystemConfiguration:
    """System configuration"""
    analysis_settings: Dict[str, Any]
    quality_thresholds: Dict[str, float]
    alert_settings: Dict[str, Any]
    export_settings: Dict[str, Any]
    performance_settings: Dict[str, Any]


@dataclass
class ModelConfiguration:
    """Model configuration"""
    model_name: str
    provider: str
    version: str
    is_active: bool
    settings: Dict[str, Any]
    capabilities: List[str]
    limitations: List[str]


# Health Check DTOs
@dataclass
class HealthCheckResponse:
    """Health check response"""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    version: str
    components: Dict[str, Dict[str, Any]]
    uptime: Optional[float] = None
    memory_usage: Optional[Dict[str, Any]] = None
    database_status: Optional[Dict[str, Any]] = None


@dataclass
class ComponentHealth:
    """Component health information"""
    name: str
    status: str
    response_time: Optional[float] = None
    last_check: Optional[datetime] = None
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None




