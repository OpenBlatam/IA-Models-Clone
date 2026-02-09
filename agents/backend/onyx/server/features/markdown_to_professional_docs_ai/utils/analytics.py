"""Analytics and reporting utilities"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """Generate analytics and reports"""
    
    def __init__(self, analytics_dir: Optional[str] = None):
        """
        Initialize analytics engine
        
        Args:
            analytics_dir: Directory for analytics data
        """
        if analytics_dir is None:
            from config import settings
            analytics_dir = settings.temp_dir + "/analytics"
        
        self.analytics_dir = Path(analytics_dir)
        self.analytics_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate analytics report
        
        Args:
            start_date: Start date for report
            end_date: End date for report
            
        Returns:
            Report dictionary
        """
        from utils.metrics import get_metrics
        from utils.cache import get_cache
        
        metrics = get_metrics()
        cache = get_cache()
        
        # Get metrics data
        metrics_data = metrics.get_metrics()
        cache_stats = cache.get_stats()
        
        # Calculate time range
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        report = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "conversions": {
                "total": metrics_data.get("counters", {}).get("conversions.requested", 0),
                "successful": metrics_data.get("counters", {}).get("conversions.success", 0),
                "failed": metrics_data.get("counters", {}).get("conversions.error", 0),
                "cached": metrics_data.get("counters", {}).get("conversions.cache_hit", 0)
            },
            "formats": self._get_format_statistics(metrics_data),
            "performance": {
                "average_conversion_time_ms": self._calculate_avg_time(metrics_data),
                "cache_hit_rate": self._calculate_cache_hit_rate(metrics_data),
                "error_rate": self._calculate_error_rate(metrics_data)
            },
            "cache": cache_stats,
            "timings": metrics_data.get("timers", {}),
            "generated_at": datetime.now().isoformat()
        }
        
        return report
    
    def _get_format_statistics(self, metrics_data: Dict[str, Any]) -> Dict[str, int]:
        """Get statistics by format"""
        counters = metrics_data.get("counters", {})
        formats = {}
        
        for key, value in counters.items():
            if key.startswith("conversions.format."):
                format_name = key.replace("conversions.format.", "")
                formats[format_name] = value
        
        return formats
    
    def _calculate_avg_time(self, metrics_data: Dict[str, Any]) -> float:
        """Calculate average conversion time"""
        timers = metrics_data.get("timers", {})
        total_timer = timers.get("conversion.total", {})
        
        if total_timer and total_timer.get("count", 0) > 0:
            return total_timer.get("avg_ms", 0)
        
        return 0.0
    
    def _calculate_cache_hit_rate(self, metrics_data: Dict[str, Any]) -> float:
        """Calculate cache hit rate"""
        counters = metrics_data.get("counters", {})
        hits = counters.get("conversions.cache_hit", 0)
        misses = counters.get("conversions.cache_miss", 0)
        total = hits + misses
        
        if total > 0:
            return (hits / total) * 100
        
        return 0.0
    
    def _calculate_error_rate(self, metrics_data: Dict[str, Any]) -> float:
        """Calculate error rate"""
        counters = metrics_data.get("counters", {})
        errors = counters.get("conversions.error", 0)
        total = counters.get("conversions.requested", 0)
        
        if total > 0:
            return (errors / total) * 100
        
        return 0.0
    
    def generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily report"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        return self.generate_report(start_date, end_date)
    
    def generate_weekly_report(self) -> Dict[str, Any]:
        """Generate weekly report"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        return self.generate_report(start_date, end_date)
    
    def generate_monthly_report(self) -> Dict[str, Any]:
        """Generate monthly report"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        return self.generate_report(start_date, end_date)
    
    def save_report(
        self,
        report: Dict[str, Any],
        report_name: Optional[str] = None
    ) -> str:
        """
        Save report to file
        
        Args:
            report: Report dictionary
            report_name: Optional report name
            
        Returns:
            Path to saved report
        """
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"report_{timestamp}"
        
        report_file = self.analytics_dir / f"{report_name}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return str(report_file)
    
    def get_report_history(self) -> List[Dict[str, Any]]:
        """Get history of saved reports"""
        reports = []
        
        for report_file in self.analytics_dir.glob("report_*.json"):
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                    reports.append({
                        "file": str(report_file),
                        "generated_at": report.get("generated_at"),
                        "period": report.get("period", {})
                    })
            except Exception as e:
                logger.error(f"Error reading report {report_file}: {e}")
        
        reports.sort(key=lambda x: x.get("generated_at", ""), reverse=True)
        
        return reports


# Global analytics engine
_analytics_engine: Optional[AnalyticsEngine] = None


def get_analytics_engine() -> AnalyticsEngine:
    """Get global analytics engine"""
    global _analytics_engine
    if _analytics_engine is None:
        _analytics_engine = AnalyticsEngine()
    return _analytics_engine

