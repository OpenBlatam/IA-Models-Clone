"""
Analytics
Analytics and reporting utilities.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class AnalyticsCollector:
    """Collect analytics data."""
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.metrics: Dict[str, List[float]] = defaultdict(list)
    
    def log_event(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ):
        """Log an event."""
        event = {
            "type": event_type,
            "timestamp": (timestamp or datetime.now()).isoformat(),
            "data": data or {},
        }
        self.events.append(event)
    
    def record_metric(self, name: str, value: float):
        """Record a metric value."""
        self.metrics[name].append(value)
    
    def get_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get events with filters."""
        filtered = self.events
        
        if event_type:
            filtered = [e for e in filtered if e["type"] == event_type]
        
        if start_time:
            filtered = [
                e for e in filtered
                if datetime.fromisoformat(e["timestamp"]) >= start_time
            ]
        
        if end_time:
            filtered = [
                e for e in filtered
                if datetime.fromisoformat(e["timestamp"]) <= end_time
            ]
        
        return filtered
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        values = self.metrics.get(name, [])
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get analytics summary."""
        # Event counts by type
        event_counts = defaultdict(int)
        for event in self.events:
            event_counts[event["type"]] += 1
        
        # Metric summaries
        metric_summaries = {
            name: self.get_metric_stats(name)
            for name in self.metrics.keys()
        }
        
        return {
            "total_events": len(self.events),
            "event_counts": dict(event_counts),
            "metrics": metric_summaries,
        }


class ReportGenerator:
    """Generate analytics reports."""
    
    @staticmethod
    def generate_daily_report(
        collector: AnalyticsCollector,
        date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Generate daily report."""
        if date is None:
            date = datetime.now()
        
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        
        events = collector.get_events(start_time=start, end_time=end)
        
        return {
            "date": date.date().isoformat(),
            "total_events": len(events),
            "events_by_type": defaultdict(int, {
                e["type"]: sum(1 for ev in events if ev["type"] == e["type"])
                for e in events
            }),
            "metrics": {
                name: collector.get_metric_stats(name)
                for name in collector.metrics.keys()
            },
        }
    
    @staticmethod
    def generate_performance_report(
        collector: AnalyticsCollector,
        period_hours: int = 24,
    ) -> Dict[str, Any]:
        """Generate performance report."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=period_hours)
        
        events = collector.get_events(start_time=start_time, end_time=end_time)
        
        # Filter performance-related events
        perf_events = [e for e in events if "latency" in e["type"].lower() or "duration" in e["type"].lower()]
        
        return {
            "period_hours": period_hours,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_events": len(events),
            "performance_events": len(perf_events),
            "metrics": {
                name: collector.get_metric_stats(name)
                for name in collector.metrics.keys()
            },
        }



