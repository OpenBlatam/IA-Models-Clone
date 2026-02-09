"""
Analytics Service for Color Grading AI
=======================================

Advanced analytics and reporting.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    Advanced analytics service.
    
    Features:
    - Usage analytics
    - Performance analytics
    - User behavior analytics
    - Custom reports
    - Export capabilities
    """
    
    def __init__(self, metrics_collector, history_manager):
        """
        Initialize analytics service.
        
        Args:
            metrics_collector: Metrics collector instance
            history_manager: History manager instance
        """
        self.metrics_collector = metrics_collector
        self.history_manager = history_manager
    
    def get_usage_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get usage report.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Usage report
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        # Get history in date range
        history = self.history_manager.search(limit=10000)
        filtered = [
            h for h in history
            if start_date <= h.timestamp <= end_date
        ]
        
        # Calculate statistics
        total_operations = len(filtered)
        successful = sum(1 for h in filtered if h.success)
        failed = total_operations - successful
        
        # Group by operation type
        by_operation = defaultdict(int)
        by_template = defaultdict(int)
        by_day = defaultdict(int)
        
        for entry in filtered:
            by_operation[entry.operation] += 1
            if entry.template_used:
                by_template[entry.template_used] += 1
            day = entry.timestamp.date()
            by_day[day] += 1
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "summary": {
                "total_operations": total_operations,
                "successful": successful,
                "failed": failed,
                "success_rate": successful / total_operations if total_operations > 0 else 0,
            },
            "by_operation": dict(by_operation),
            "by_template": dict(by_template),
            "by_day": {str(k): v for k, v in sorted(by_day.items())},
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report."""
        metrics = self.metrics_collector.get_stats()
        
        # Calculate averages
        operations = metrics.get("operations", {})
        avg_durations = {}
        total_durations = {}
        
        for op_name, op_data in operations.items():
            if isinstance(op_data, dict):
                avg_durations[op_name] = op_data.get("avg_duration", 0)
                total_durations[op_name] = op_data.get("total_duration", 0)
        
        return {
            "avg_durations": avg_durations,
            "total_durations": total_durations,
            "most_used_operations": sorted(
                operations.items(),
                key=lambda x: x[1].get("count", 0) if isinstance(x[1], dict) else 0,
                reverse=True
            )[:10],
        }
    
    def get_template_analytics(self) -> Dict[str, Any]:
        """Get template usage analytics."""
        template_stats = self.metrics_collector.get_template_stats()
        
        if not template_stats:
            return {}
        
        # Find most/least used
        sorted_templates = sorted(
            template_stats.items(),
            key=lambda x: x[1].get("count", 0),
            reverse=True
        )
        
        return {
            "most_used": sorted_templates[:5],
            "least_used": sorted_templates[-5:] if len(sorted_templates) > 5 else [],
            "total_templates": len(template_stats),
            "total_usage": sum(s.get("count", 0) for s in template_stats.values()),
        }
    
    def get_trend_analysis(self, days: int = 30) -> Dict[str, Any]:
        """
        Get trend analysis.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Trend analysis
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get daily counts
        history = self.history_manager.search(limit=10000)
        filtered = [h for h in history if start_date <= h.timestamp <= end_date]
        
        daily_counts = defaultdict(int)
        daily_success = defaultdict(int)
        
        for entry in filtered:
            day = entry.timestamp.date()
            daily_counts[day] += 1
            if entry.success:
                daily_success[day] += 1
        
        # Calculate trends
        days_list = sorted(daily_counts.keys())
        if len(days_list) < 2:
            return {"trend": "insufficient_data"}
        
        first_half = days_list[:len(days_list)//2]
        second_half = days_list[len(days_list)//2:]
        
        first_avg = sum(daily_counts[d] for d in first_half) / len(first_half) if first_half else 0
        second_avg = sum(daily_counts[d] for d in second_half) / len(second_half) if second_half else 0
        
        trend = "stable"
        if second_avg > first_avg * 1.1:
            trend = "increasing"
        elif second_avg < first_avg * 0.9:
            trend = "decreasing"
        
        return {
            "trend": trend,
            "first_half_avg": first_avg,
            "second_half_avg": second_avg,
            "change_percent": ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0,
            "daily_data": {
                str(day): {
                    "total": daily_counts[day],
                    "successful": daily_success[day],
                }
                for day in days_list
            },
        }
    
    def export_report(self, report_type: str, format: str = "json") -> str:
        """
        Export analytics report.
        
        Args:
            report_type: Type of report (usage, performance, templates, trends)
            format: Export format (json, csv)
            
        Returns:
            Exported report content
        """
        if report_type == "usage":
            data = self.get_usage_report()
        elif report_type == "performance":
            data = self.get_performance_report()
        elif report_type == "templates":
            data = self.get_template_analytics()
        elif report_type == "trends":
            data = self.get_trend_analysis()
        else:
            raise ValueError(f"Unknown report type: {report_type}")
        
        if format == "json":
            import json
            return json.dumps(data, indent=2, default=str)
        elif format == "csv":
            # Simple CSV export
            lines = []
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float, str)):
                        lines.append(f"{key},{value}")
            return "\n".join(lines)
        
        return str(data)




