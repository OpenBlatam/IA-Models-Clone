"""
Document Analytics Service
=========================

Advanced analytics and metrics service for professional documents.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum

from .models import ProfessionalDocument, DocumentType, ExportFormat
from .services import DocumentGenerationService

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""
    USAGE = "usage"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    CONTENT = "content"
    USER = "user"
    EXPORT = "export"


class TimeRange(str, Enum):
    """Time range for analytics."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


@dataclass
class DocumentMetric:
    """Document metric data structure."""
    metric_type: MetricType
    name: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class AnalyticsReport:
    """Analytics report data structure."""
    report_id: str
    title: str
    time_range: TimeRange
    generated_at: datetime
    metrics: List[DocumentMetric]
    insights: List[str]
    recommendations: List[str]
    summary: Dict[str, Any]


class DocumentAnalyticsService:
    """Advanced analytics service for professional documents."""
    
    def __init__(self, document_service: DocumentGenerationService):
        self.document_service = document_service
        self.metrics_cache = {}
        self.analytics_cache = {}
    
    async def generate_comprehensive_analytics(
        self,
        time_range: TimeRange = TimeRange.MONTH,
        user_id: Optional[str] = None,
        document_type: Optional[DocumentType] = None
    ) -> AnalyticsReport:
        """Generate comprehensive analytics report."""
        
        try:
            # Get documents for analysis
            documents = await self._get_documents_for_analysis(time_range, user_id, document_type)
            
            # Calculate various metrics
            usage_metrics = await self._calculate_usage_metrics(documents, time_range)
            performance_metrics = await self._calculate_performance_metrics(documents, time_range)
            quality_metrics = await self._calculate_quality_metrics(documents, time_range)
            content_metrics = await self._calculate_content_metrics(documents, time_range)
            export_metrics = await self._calculate_export_metrics(documents, time_range)
            
            # Combine all metrics
            all_metrics = usage_metrics + performance_metrics + quality_metrics + content_metrics + export_metrics
            
            # Generate insights
            insights = await self._generate_insights(all_metrics, documents)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(all_metrics, documents)
            
            # Create summary
            summary = await self._create_analytics_summary(all_metrics, documents)
            
            # Create report
            report = AnalyticsReport(
                report_id=f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                title=f"Document Analytics Report - {time_range.value.title()}",
                time_range=time_range,
                generated_at=datetime.now(),
                metrics=all_metrics,
                insights=insights,
                recommendations=recommendations,
                summary=summary
            )
            
            # Cache the report
            self.analytics_cache[report.report_id] = report
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating analytics: {str(e)}")
            raise
    
    async def _get_documents_for_analysis(
        self,
        time_range: TimeRange,
        user_id: Optional[str],
        document_type: Optional[DocumentType]
    ) -> List[ProfessionalDocument]:
        """Get documents for analysis based on criteria."""
        
        # Get all documents
        all_documents = self.document_service.list_documents(limit=1000, offset=0)
        
        # Filter by time range
        cutoff_date = self._get_cutoff_date(time_range)
        filtered_documents = [
            doc for doc in all_documents
            if doc.date_created >= cutoff_date
        ]
        
        # Filter by user if specified
        if user_id:
            filtered_documents = [
                doc for doc in filtered_documents
                if doc.metadata.get("user_id") == user_id
            ]
        
        # Filter by document type if specified
        if document_type:
            filtered_documents = [
                doc for doc in filtered_documents
                if doc.document_type == document_type
            ]
        
        return filtered_documents
    
    def _get_cutoff_date(self, time_range: TimeRange) -> datetime:
        """Get cutoff date for time range."""
        
        now = datetime.now()
        
        if time_range == TimeRange.HOUR:
            return now - timedelta(hours=1)
        elif time_range == TimeRange.DAY:
            return now - timedelta(days=1)
        elif time_range == TimeRange.WEEK:
            return now - timedelta(weeks=1)
        elif time_range == TimeRange.MONTH:
            return now - timedelta(days=30)
        elif time_range == TimeRange.QUARTER:
            return now - timedelta(days=90)
        elif time_range == TimeRange.YEAR:
            return now - timedelta(days=365)
        else:
            return now - timedelta(days=30)
    
    async def _calculate_usage_metrics(
        self,
        documents: List[ProfessionalDocument],
        time_range: TimeRange
    ) -> List[DocumentMetric]:
        """Calculate usage metrics."""
        
        metrics = []
        
        # Total documents created
        total_documents = len(documents)
        metrics.append(DocumentMetric(
            metric_type=MetricType.USAGE,
            name="total_documents_created",
            value=total_documents,
            unit="documents",
            timestamp=datetime.now(),
            metadata={"time_range": time_range.value}
        ))
        
        # Documents by type
        type_counts = Counter(doc.document_type.value for doc in documents)
        for doc_type, count in type_counts.items():
            metrics.append(DocumentMetric(
                metric_type=MetricType.USAGE,
                name=f"documents_by_type_{doc_type}",
                value=count,
                unit="documents",
                timestamp=datetime.now(),
                metadata={"document_type": doc_type, "time_range": time_range.value}
            ))
        
        # Documents by status
        status_counts = Counter(doc.status for doc in documents)
        for status, count in status_counts.items():
            metrics.append(DocumentMetric(
                metric_type=MetricType.USAGE,
                name=f"documents_by_status_{status}",
                value=count,
                unit="documents",
                timestamp=datetime.now(),
                metadata={"status": status, "time_range": time_range.value}
            ))
        
        # Average documents per day
        days_in_range = self._get_days_in_range(time_range)
        avg_documents_per_day = total_documents / days_in_range if days_in_range > 0 else 0
        metrics.append(DocumentMetric(
            metric_type=MetricType.USAGE,
            name="average_documents_per_day",
            value=avg_documents_per_day,
            unit="documents/day",
            timestamp=datetime.now(),
            metadata={"time_range": time_range.value}
        ))
        
        return metrics
    
    async def _calculate_performance_metrics(
        self,
        documents: List[ProfessionalDocument],
        time_range: TimeRange
    ) -> List[DocumentMetric]:
        """Calculate performance metrics."""
        
        metrics = []
        
        if not documents:
            return metrics
        
        # Average word count
        word_counts = [doc.word_count for doc in documents if doc.word_count > 0]
        if word_counts:
            avg_word_count = sum(word_counts) / len(word_counts)
            metrics.append(DocumentMetric(
                metric_type=MetricType.PERFORMANCE,
                name="average_word_count",
                value=avg_word_count,
                unit="words",
                timestamp=datetime.now(),
                metadata={"time_range": time_range.value}
            ))
        
        # Average page count
        page_counts = [doc.page_count for doc in documents if doc.page_count > 0]
        if page_counts:
            avg_page_count = sum(page_counts) / len(page_counts)
            metrics.append(DocumentMetric(
                metric_type=MetricType.PERFORMANCE,
                name="average_page_count",
                value=avg_page_count,
                unit="pages",
                timestamp=datetime.now(),
                metadata={"time_range": time_range.value}
            ))
        
        # Average sections per document
        section_counts = [len(doc.sections) for doc in documents if doc.sections]
        if section_counts:
            avg_sections = sum(section_counts) / len(section_counts)
            metrics.append(DocumentMetric(
                metric_type=MetricType.PERFORMANCE,
                name="average_sections_per_document",
                value=avg_sections,
                unit="sections",
                timestamp=datetime.now(),
                metadata={"time_range": time_range.value}
            ))
        
        # Total words generated
        total_words = sum(doc.word_count for doc in documents)
        metrics.append(DocumentMetric(
            metric_type=MetricType.PERFORMANCE,
            name="total_words_generated",
            value=total_words,
            unit="words",
            timestamp=datetime.now(),
            metadata={"time_range": time_range.value}
        ))
        
        return metrics
    
    async def _calculate_quality_metrics(
        self,
        documents: List[ProfessionalDocument],
        time_range: TimeRange
    ) -> List[DocumentMetric]:
        """Calculate quality metrics."""
        
        metrics = []
        
        if not documents:
            return metrics
        
        # Success rate
        successful_docs = len([doc for doc in documents if doc.status == "completed"])
        success_rate = (successful_docs / len(documents)) * 100 if documents else 0
        metrics.append(DocumentMetric(
            metric_type=MetricType.QUALITY,
            name="success_rate",
            value=success_rate,
            unit="percentage",
            timestamp=datetime.now(),
            metadata={"time_range": time_range.value}
        ))
        
        # Error rate
        error_docs = len([doc for doc in documents if doc.status == "error"])
        error_rate = (error_docs / len(documents)) * 100 if documents else 0
        metrics.append(DocumentMetric(
            metric_type=MetricType.QUALITY,
            name="error_rate",
            value=error_rate,
            unit="percentage",
            timestamp=datetime.now(),
            metadata={"time_range": time_range.value}
        ))
        
        # Average quality score (if available in metadata)
        quality_scores = []
        for doc in documents:
            if doc.metadata and "quality_score" in doc.metadata:
                quality_scores.append(doc.metadata["quality_score"])
        
        if quality_scores:
            avg_quality_score = sum(quality_scores) / len(quality_scores)
            metrics.append(DocumentMetric(
                metric_type=MetricType.QUALITY,
                name="average_quality_score",
                value=avg_quality_score,
                unit="score",
                timestamp=datetime.now(),
                metadata={"time_range": time_range.value}
            ))
        
        return metrics
    
    async def _calculate_content_metrics(
        self,
        documents: List[ProfessionalDocument],
        time_range: TimeRange
    ) -> List[DocumentMetric]:
        """Calculate content metrics."""
        
        metrics = []
        
        if not documents:
            return metrics
        
        # Most common document types
        type_counts = Counter(doc.document_type.value for doc in documents)
        most_common_type = type_counts.most_common(1)[0] if type_counts else None
        
        if most_common_type:
            metrics.append(DocumentMetric(
                metric_type=MetricType.CONTENT,
                name="most_common_document_type",
                value=most_common_type[1],
                unit="documents",
                timestamp=datetime.now(),
                metadata={"document_type": most_common_type[0], "time_range": time_range.value}
            ))
        
        # Average content length by type
        type_word_counts = defaultdict(list)
        for doc in documents:
            if doc.word_count > 0:
                type_word_counts[doc.document_type.value].append(doc.word_count)
        
        for doc_type, word_counts in type_word_counts.items():
            avg_words = sum(word_counts) / len(word_counts)
            metrics.append(DocumentMetric(
                metric_type=MetricType.CONTENT,
                name=f"average_words_by_type_{doc_type}",
                value=avg_words,
                unit="words",
                timestamp=datetime.now(),
                metadata={"document_type": doc_type, "time_range": time_range.value}
            ))
        
        # Content diversity (number of unique document types)
        unique_types = len(set(doc.document_type.value for doc in documents))
        metrics.append(DocumentMetric(
            metric_type=MetricType.CONTENT,
            name="content_diversity",
            value=unique_types,
            unit="types",
            timestamp=datetime.now(),
            metadata={"time_range": time_range.value}
        ))
        
        return metrics
    
    async def _calculate_export_metrics(
        self,
        documents: List[ProfessionalDocument],
        time_range: TimeRange
    ) -> List[DocumentMetric]:
        """Calculate export metrics."""
        
        metrics = []
        
        # This would typically come from export service logs
        # For now, we'll simulate some export metrics
        
        # Total exports (simulated)
        total_exports = len(documents) * 1.5  # Assume 1.5 exports per document on average
        metrics.append(DocumentMetric(
            metric_type=MetricType.EXPORT,
            name="total_exports",
            value=total_exports,
            unit="exports",
            timestamp=datetime.now(),
            metadata={"time_range": time_range.value}
        ))
        
        # Export format distribution (simulated)
        export_formats = ["pdf", "docx", "md", "html"]
        for format_type in export_formats:
            # Simulate distribution
            format_count = total_exports * (0.4 if format_type == "pdf" else 0.3 if format_type == "docx" else 0.2 if format_type == "md" else 0.1)
            metrics.append(DocumentMetric(
                metric_type=MetricType.EXPORT,
                name=f"exports_by_format_{format_type}",
                value=format_count,
                unit="exports",
                timestamp=datetime.now(),
                metadata={"export_format": format_type, "time_range": time_range.value}
            ))
        
        return metrics
    
    async def _generate_insights(
        self,
        metrics: List[DocumentMetric],
        documents: List[ProfessionalDocument]
    ) -> List[str]:
        """Generate insights from metrics."""
        
        insights = []
        
        # Find key metrics
        total_docs_metric = next((m for m in metrics if m.name == "total_documents_created"), None)
        success_rate_metric = next((m for m in metrics if m.name == "success_rate"), None)
        avg_word_count_metric = next((m for m in metrics if m.name == "average_word_count"), None)
        
        if total_docs_metric:
            insights.append(f"Generated {int(total_docs_metric.value)} documents in the selected time period")
        
        if success_rate_metric:
            if success_rate_metric.value >= 95:
                insights.append("Excellent document generation success rate - system is performing optimally")
            elif success_rate_metric.value >= 90:
                insights.append("Good document generation success rate with room for minor improvements")
            else:
                insights.append("Document generation success rate could be improved - consider investigating error patterns")
        
        if avg_word_count_metric:
            if avg_word_count_metric.value >= 1000:
                insights.append("Users are generating comprehensive, detailed documents")
            elif avg_word_count_metric.value >= 500:
                insights.append("Users are generating moderately detailed documents")
            else:
                insights.append("Users are generating concise documents - consider offering longer format options")
        
        # Content insights
        type_metrics = [m for m in metrics if m.name.startswith("documents_by_type_")]
        if type_metrics:
            most_used_type = max(type_metrics, key=lambda m: m.value)
            insights.append(f"Most popular document type: {most_used_type.metadata.get('document_type', 'unknown')}")
        
        # Performance insights
        total_words_metric = next((m for m in metrics if m.name == "total_words_generated"), None)
        if total_words_metric:
            insights.append(f"Generated {int(total_words_metric.value):,} total words across all documents")
        
        return insights
    
    async def _generate_recommendations(
        self,
        metrics: List[DocumentMetric],
        documents: List[ProfessionalDocument]
    ) -> List[str]:
        """Generate recommendations based on metrics."""
        
        recommendations = []
        
        # Find key metrics for recommendations
        success_rate_metric = next((m for m in metrics if m.name == "success_rate"), None)
        error_rate_metric = next((m for m in metrics if m.name == "error_rate"), None)
        avg_docs_per_day_metric = next((m for m in metrics if m.name == "average_documents_per_day"), None)
        
        if success_rate_metric and success_rate_metric.value < 95:
            recommendations.append("Investigate and resolve document generation errors to improve success rate")
        
        if error_rate_metric and error_rate_metric.value > 5:
            recommendations.append("High error rate detected - review error logs and improve error handling")
        
        if avg_docs_per_day_metric and avg_docs_per_day_metric.value < 1:
            recommendations.append("Low document generation activity - consider promoting the feature to increase usage")
        
        # Content recommendations
        type_metrics = [m for m in metrics if m.name.startswith("documents_by_type_")]
        if type_metrics:
            least_used_type = min(type_metrics, key=lambda m: m.value)
            if least_used_type.value == 0:
                recommendations.append(f"Consider promoting {least_used_type.metadata.get('document_type', 'this')} document type - no usage detected")
        
        # Performance recommendations
        avg_word_count_metric = next((m for m in metrics if m.name == "average_word_count"), None)
        if avg_word_count_metric and avg_word_count_metric.value < 200:
            recommendations.append("Consider offering templates for longer, more comprehensive documents")
        
        # General recommendations
        recommendations.append("Monitor document quality scores and user feedback to identify improvement opportunities")
        recommendations.append("Analyze export patterns to optimize export functionality")
        recommendations.append("Consider implementing document templates for underutilized document types")
        
        return recommendations
    
    async def _create_analytics_summary(
        self,
        metrics: List[DocumentMetric],
        documents: List[ProfessionalDocument]
    ) -> Dict[str, Any]:
        """Create analytics summary."""
        
        # Find key metrics
        total_docs_metric = next((m for m in metrics if m.name == "total_documents_created"), None)
        success_rate_metric = next((m for m in metrics if m.name == "success_rate"), None)
        total_words_metric = next((m for m in metrics if m.name == "total_words_generated"), None)
        
        summary = {
            "total_documents": int(total_docs_metric.value) if total_docs_metric else 0,
            "success_rate": round(success_rate_metric.value, 2) if success_rate_metric else 0,
            "total_words_generated": int(total_words_metric.value) if total_words_metric else 0,
            "document_types_used": len(set(doc.document_type.value for doc in documents)),
            "average_document_length": sum(doc.word_count for doc in documents) / len(documents) if documents else 0,
            "most_common_type": max(Counter(doc.document_type.value for doc in documents).items(), key=lambda x: x[1])[0] if documents else None,
            "time_period_analyzed": "last 30 days",  # This would be dynamic based on time_range
            "metrics_calculated": len(metrics),
            "insights_generated": 0,  # Will be updated when insights are generated
            "recommendations_provided": 0  # Will be updated when recommendations are generated
        }
        
        return summary
    
    def _get_days_in_range(self, time_range: TimeRange) -> int:
        """Get number of days in time range."""
        
        if time_range == TimeRange.HOUR:
            return 1/24
        elif time_range == TimeRange.DAY:
            return 1
        elif time_range == TimeRange.WEEK:
            return 7
        elif time_range == TimeRange.MONTH:
            return 30
        elif time_range == TimeRange.QUARTER:
            return 90
        elif time_range == TimeRange.YEAR:
            return 365
        else:
            return 30
    
    async def get_metric_trends(
        self,
        metric_name: str,
        time_range: TimeRange = TimeRange.MONTH,
        granularity: str = "day"
    ) -> List[Tuple[datetime, float]]:
        """Get metric trends over time."""
        
        # This would typically query a time-series database
        # For now, we'll simulate trend data
        
        trends = []
        cutoff_date = self._get_cutoff_date(time_range)
        
        if granularity == "day":
            days = int((datetime.now() - cutoff_date).days)
            for i in range(days):
                date = cutoff_date + timedelta(days=i)
                # Simulate some trend data
                value = 10 + (i * 0.5) + (i % 7) * 2  # Some variation
                trends.append((date, value))
        
        return trends
    
    async def get_user_analytics(self, user_id: str, time_range: TimeRange = TimeRange.MONTH) -> Dict[str, Any]:
        """Get analytics for a specific user."""
        
        documents = await self._get_documents_for_analysis(time_range, user_id, None)
        
        if not documents:
            return {
                "user_id": user_id,
                "time_range": time_range.value,
                "total_documents": 0,
                "message": "No documents found for this user in the specified time range"
            }
        
        # Calculate user-specific metrics
        total_documents = len(documents)
        total_words = sum(doc.word_count for doc in documents)
        avg_word_count = total_words / total_documents if total_documents > 0 else 0
        document_types = list(set(doc.document_type.value for doc in documents))
        success_rate = len([doc for doc in documents if doc.status == "completed"]) / total_documents * 100
        
        return {
            "user_id": user_id,
            "time_range": time_range.value,
            "total_documents": total_documents,
            "total_words_generated": total_words,
            "average_word_count": round(avg_word_count, 2),
            "document_types_used": document_types,
            "success_rate": round(success_rate, 2),
            "most_used_type": max(Counter(doc.document_type.value for doc in documents).items(), key=lambda x: x[1])[0] if documents else None,
            "first_document_date": min(doc.date_created for doc in documents).isoformat(),
            "last_document_date": max(doc.date_created for doc in documents).isoformat()
        }
    
    async def export_analytics_report(self, report: AnalyticsReport, format: str = "json") -> str:
        """Export analytics report in specified format."""
        
        if format == "json":
            return json.dumps({
                "report_id": report.report_id,
                "title": report.title,
                "time_range": report.time_range.value,
                "generated_at": report.generated_at.isoformat(),
                "metrics": [
                    {
                        "type": metric.metric_type.value,
                        "name": metric.name,
                        "value": metric.value,
                        "unit": metric.unit,
                        "timestamp": metric.timestamp.isoformat(),
                        "metadata": metric.metadata
                    }
                    for metric in report.metrics
                ],
                "insights": report.insights,
                "recommendations": report.recommendations,
                "summary": report.summary
            }, indent=2)
        
        elif format == "csv":
            # Convert metrics to CSV format
            csv_lines = ["metric_type,name,value,unit,timestamp,metadata"]
            for metric in report.metrics:
                csv_lines.append(f"{metric.metric_type.value},{metric.name},{metric.value},{metric.unit},{metric.timestamp.isoformat()},{json.dumps(metric.metadata)}")
            return "\n".join(csv_lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_cached_report(self, report_id: str) -> Optional[AnalyticsReport]:
        """Get cached analytics report."""
        return self.analytics_cache.get(report_id)
    
    def clear_analytics_cache(self):
        """Clear analytics cache."""
        self.analytics_cache.clear()
        self.metrics_cache.clear()



























