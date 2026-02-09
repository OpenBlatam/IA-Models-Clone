"""
Business Metrics for Flux2 Clothing Changer
============================================

Business-level metrics and KPIs.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class BusinessMetric:
    """Business metric record."""
    metric_name: str
    value: float
    timestamp: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BusinessMetrics:
    """Business metrics and KPI tracking."""
    
    def __init__(
        self,
        history_size: int = 10000,
    ):
        """
        Initialize business metrics system.
        
        Args:
            history_size: Maximum number of metrics to keep
        """
        self.history_size = history_size
        self.metrics: deque = deque(maxlen=history_size)
        
        # Aggregated metrics
        self.daily_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.user_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # KPIs
        self.kpis = {
            "total_users": 0,
            "active_users": set(),
            "total_requests": 0,
            "revenue": 0.0,
            "conversion_rate": 0.0,
        }
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a business metric.
        
        Args:
            metric_name: Metric name
            value: Metric value
            user_id: Optional user ID
            session_id: Optional session ID
            metadata: Optional metadata
        """
        metric = BusinessMetric(
            metric_name=metric_name,
            value=value,
            timestamp=time.time(),
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
        )
        
        self.metrics.append(metric)
        
        # Update daily metrics
        date_key = datetime.fromtimestamp(metric.timestamp).strftime("%Y-%m-%d")
        if metric_name not in self.daily_metrics[date_key]:
            self.daily_metrics[date_key][metric_name] = []
        self.daily_metrics[date_key][metric_name].append(value)
        
        # Update user metrics
        if user_id:
            if metric_name not in self.user_metrics[user_id]:
                self.user_metrics[user_id][metric_name] = []
            self.user_metrics[user_id][metric_name].append(value)
            self.kpis["active_users"].add(user_id)
    
    def record_conversion(
        self,
        user_id: str,
        conversion_type: str,
        value: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a conversion event.
        
        Args:
            user_id: User ID
            conversion_type: Type of conversion
            value: Conversion value
            metadata: Optional metadata
        """
        self.record_metric(
            metric_name=f"conversion_{conversion_type}",
            value=value,
            user_id=user_id,
            metadata={"conversion_type": conversion_type, **(metadata or {})},
        )
        
        # Update conversion rate
        self._update_conversion_rate()
    
    def record_revenue(
        self,
        user_id: str,
        amount: float,
        currency: str = "USD",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record revenue.
        
        Args:
            user_id: User ID
            amount: Revenue amount
            currency: Currency code
            metadata: Optional metadata
        """
        self.record_metric(
            metric_name="revenue",
            value=amount,
            user_id=user_id,
            metadata={"currency": currency, **(metadata or {})},
        )
        
        self.kpis["revenue"] += amount
    
    def get_kpis(
        self,
        time_range: Optional[timedelta] = None,
    ) -> Dict[str, Any]:
        """
        Get key performance indicators.
        
        Args:
            time_range: Optional time range
            
        Returns:
            Dictionary of KPIs
        """
        cutoff_time = time.time() - time_range.total_seconds() if time_range else 0
        
        # Filter metrics by time range
        relevant_metrics = [
            m for m in self.metrics
            if m.timestamp >= cutoff_time
        ]
        
        # Calculate KPIs
        total_requests = len([m for m in relevant_metrics if m.metric_name == "request"])
        unique_users = len(set(m.user_id for m in relevant_metrics if m.user_id))
        
        revenue = sum(
            m.value for m in relevant_metrics
            if m.metric_name == "revenue"
        )
        
        conversions = len([
            m for m in relevant_metrics
            if m.metric_name.startswith("conversion_")
        ])
        
        conversion_rate = (
            conversions / total_requests
            if total_requests > 0 else 0.0
        )
        
        return {
            "total_requests": total_requests,
            "unique_users": unique_users,
            "revenue": revenue,
            "conversions": conversions,
            "conversion_rate": conversion_rate,
            "avg_revenue_per_user": (
                revenue / unique_users
                if unique_users > 0 else 0.0
            ),
        }
    
    def get_user_lifetime_value(self, user_id: str) -> Dict[str, Any]:
        """
        Calculate user lifetime value.
        
        Args:
            user_id: User ID
            
        Returns:
            LTV metrics
        """
        if user_id not in self.user_metrics:
            return {
                "user_id": user_id,
                "total_requests": 0,
                "total_revenue": 0.0,
                "conversions": 0,
                "ltv": 0.0,
            }
        
        user_metrics = self.user_metrics[user_id]
        
        total_requests = len(user_metrics.get("request", []))
        total_revenue = sum(user_metrics.get("revenue", []))
        conversions = len([
            m for m in self.metrics
            if m.user_id == user_id and m.metric_name.startswith("conversion_")
        ])
        
        return {
            "user_id": user_id,
            "total_requests": total_requests,
            "total_revenue": total_revenue,
            "conversions": conversions,
            "ltv": total_revenue,
            "avg_request_value": (
                total_revenue / total_requests
                if total_requests > 0 else 0.0
            ),
        }
    
    def get_daily_summary(
        self,
        date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get daily summary.
        
        Args:
            date: Date string (YYYY-MM-DD) or None for today
            
        Returns:
            Daily summary
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        if date not in self.daily_metrics:
            return {
                "date": date,
                "metrics": {},
            }
        
        daily = self.daily_metrics[date]
        
        summary = {
            "date": date,
            "metrics": {},
        }
        
        for metric_name, values in daily.items():
            if values:
                summary["metrics"][metric_name] = {
                    "count": len(values),
                    "sum": sum(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
        
        return summary
    
    def _update_conversion_rate(self) -> None:
        """Update conversion rate KPI."""
        total_requests = len([m for m in self.metrics if m.metric_name == "request"])
        conversions = len([
            m for m in self.metrics
            if m.metric_name.startswith("conversion_")
        ])
        
        self.kpis["conversion_rate"] = (
            conversions / total_requests
            if total_requests > 0 else 0.0
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get business metrics statistics."""
        return {
            "total_metrics": len(self.metrics),
            "total_users": len(self.user_metrics),
            "active_users": len(self.kpis["active_users"]),
            "kpis": {
                **self.kpis,
                "conversion_rate": self.kpis["conversion_rate"],
            },
            "daily_metrics_count": len(self.daily_metrics),
        }


