"""
Reporting System
================

System for generating various reports.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Report type."""
    SUMMARY = "summary"
    DETAILED = "detailed"
    PERFORMANCE = "performance"
    USAGE = "usage"
    ERROR = "error"
    CUSTOM = "custom"


@dataclass
class Report:
    """Report definition."""
    report_type: ReportType
    title: str
    content: Dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.now)
    period: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReportGenerator:
    """Report generator."""
    
    def __init__(self):
        """Initialize report generator."""
        self.reports: List[Report] = []
        self.max_reports = 1000
    
    def generate_summary_report(
        self,
        metrics: Dict[str, Any],
        period: Optional[timedelta] = None
    ) -> Report:
        """
        Generate summary report.
        
        Args:
            metrics: Metrics dictionary
            period: Optional time period
            
        Returns:
            Summary report
        """
        content = {
            "summary": {
                "total_tasks": metrics.get("total_tasks", 0),
                "successful_tasks": metrics.get("successful_tasks", 0),
                "failed_tasks": metrics.get("failed_tasks", 0),
                "success_rate": metrics.get("success_rate", 0.0),
                "avg_processing_time": metrics.get("avg_processing_time", 0.0)
            },
            "period": period.total_seconds() if period else None
        }
        
        report = Report(
            report_type=ReportType.SUMMARY,
            title="Summary Report",
            content=content,
            period=period
        )
        
        self._save_report(report)
        return report
    
    def generate_performance_report(
        self,
        performance_data: Dict[str, Any],
        period: Optional[timedelta] = None
    ) -> Report:
        """
        Generate performance report.
        
        Args:
            performance_data: Performance data dictionary
            period: Optional time period
            
        Returns:
            Performance report
        """
        content = {
            "performance": {
                "avg_response_time": performance_data.get("avg_response_time", 0.0),
                "p95_response_time": performance_data.get("p95_response_time", 0.0),
                "p99_response_time": performance_data.get("p99_response_time", 0.0),
                "throughput": performance_data.get("throughput", 0.0),
                "error_rate": performance_data.get("error_rate", 0.0)
            },
            "period": period.total_seconds() if period else None
        }
        
        report = Report(
            report_type=ReportType.PERFORMANCE,
            title="Performance Report",
            content=content,
            period=period
        )
        
        self._save_report(report)
        return report
    
    def generate_usage_report(
        self,
        usage_data: Dict[str, Any],
        period: Optional[timedelta] = None
    ) -> Report:
        """
        Generate usage report.
        
        Args:
            usage_data: Usage data dictionary
            period: Optional time period
            
        Returns:
            Usage report
        """
        content = {
            "usage": {
                "total_requests": usage_data.get("total_requests", 0),
                "unique_users": usage_data.get("unique_users", 0),
                "api_calls": usage_data.get("api_calls", 0),
                "cache_hits": usage_data.get("cache_hits", 0),
                "cache_misses": usage_data.get("cache_misses", 0)
            },
            "period": period.total_seconds() if period else None
        }
        
        report = Report(
            report_type=ReportType.USAGE,
            title="Usage Report",
            content=content,
            period=period
        )
        
        self._save_report(report)
        return report
    
    def generate_custom_report(
        self,
        title: str,
        content: Dict[str, Any],
        period: Optional[timedelta] = None
    ) -> Report:
        """
        Generate custom report.
        
        Args:
            title: Report title
            content: Report content
            period: Optional time period
            
        Returns:
            Custom report
        """
        report = Report(
            report_type=ReportType.CUSTOM,
            title=title,
            content=content,
            period=period
        )
        
        self._save_report(report)
        return report
    
    def _save_report(self, report: Report):
        """Save report to list."""
        self.reports.append(report)
        
        # Limit reports
        if len(self.reports) > self.max_reports:
            self.reports = self.reports[-self.max_reports:]
    
    def export_report(self, report: Report, output_path: Path, format: str = "json"):
        """
        Export report to file.
        
        Args:
            report: Report to export
            output_path: Output file path
            format: Export format (json, markdown)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "type": report.report_type.value,
                    "title": report.title,
                    "content": report.content,
                    "generated_at": report.generated_at.isoformat(),
                    "period": report.period.total_seconds() if report.period else None,
                    "metadata": report.metadata
                }, f, indent=2)
        elif format == "markdown":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# {report.title}\n\n")
                f.write(f"**Generated:** {report.generated_at.isoformat()}\n\n")
                f.write(f"**Type:** {report.report_type.value}\n\n")
                f.write("## Content\n\n")
                f.write(f"```json\n{json.dumps(report.content, indent=2)}\n```\n")
        
        logger.info(f"Report exported to {output_path}")
    
    def get_recent_reports(self, limit: int = 10) -> List[Report]:
        """Get recent reports."""
        return self.reports[-limit:]

