"""
Data Analyzer
=============

Advanced data analysis utilities for documents and metrics.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import Counter
import statistics

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Advanced data analyzer."""
    
    def __init__(self):
        self.analysis_cache: Dict[str, Any] = {}
    
    def analyze_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a collection of documents."""
        if not documents:
            return {}
        
        # Content analysis
        content_lengths = [len(doc.get("content", "")) for doc in documents]
        quality_scores = [doc.get("quality_score", 0.0) for doc in documents if "quality_score" in doc]
        
        # Model distribution
        models = [doc.get("model_used", "unknown") for doc in documents]
        model_distribution = dict(Counter(models))
        
        # Time analysis
        timestamps = []
        for doc in documents:
            if "timestamp" in doc:
                try:
                    timestamps.append(datetime.fromisoformat(doc["timestamp"].replace("Z", "+00:00")))
                except:
                    pass
        
        analysis = {
            "total_documents": len(documents),
            "content_stats": {
                "avg_length": statistics.mean(content_lengths) if content_lengths else 0,
                "min_length": min(content_lengths) if content_lengths else 0,
                "max_length": max(content_lengths) if content_lengths else 0,
                "median_length": statistics.median(content_lengths) if content_lengths else 0
            },
            "quality_stats": {
                "avg_score": statistics.mean(quality_scores) if quality_scores else 0,
                "min_score": min(quality_scores) if quality_scores else 0,
                "max_score": max(quality_scores) if quality_scores else 0,
                "median_score": statistics.median(quality_scores) if quality_scores else 0
            },
            "model_distribution": model_distribution,
            "time_span": {
                "start": min(timestamps).isoformat() if timestamps else None,
                "end": max(timestamps).isoformat() if timestamps else None,
                "duration_hours": (max(timestamps) - min(timestamps)).total_seconds() / 3600 if len(timestamps) > 1 else 0
            }
        }
        
        return analysis
    
    def analyze_metrics(self, metrics: List[float]) -> Dict[str, Any]:
        """Analyze metrics data."""
        if not metrics:
            return {}
        
        return {
            "count": len(metrics),
            "mean": statistics.mean(metrics),
            "median": statistics.median(metrics),
            "std_dev": statistics.stdev(metrics) if len(metrics) > 1 else 0.0,
            "min": min(metrics),
            "max": max(metrics),
            "p25": self._percentile(metrics, 0.25),
            "p75": self._percentile(metrics, 0.75),
            "p90": self._percentile(metrics, 0.90),
            "p95": self._percentile(metrics, 0.95),
            "p99": self._percentile(metrics, 0.99)
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def analyze_trends(
        self,
        data: List[Dict[str, Any]],
        time_field: str = "timestamp",
        value_field: str = "value"
    ) -> Dict[str, Any]:
        """Analyze trends in time series data."""
        if not data:
            return {}
        
        # Sort by time
        sorted_data = sorted(
            data,
            key=lambda x: x.get(time_field, "")
        )
        
        # Calculate trend
        if len(sorted_data) >= 2:
            first_half = sorted_data[:len(sorted_data)//2]
            second_half = sorted_data[len(sorted_data)//2:]
            
            first_avg = statistics.mean([d.get(value_field, 0) for d in first_half])
            second_avg = statistics.mean([d.get(value_field, 0) for d in second_half])
            
            trend = "increasing" if second_avg > first_avg * 1.05 else (
                "decreasing" if second_avg < first_avg * 0.95 else "stable"
            )
            
            change_percent = ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
        else:
            trend = "stable"
            change_percent = 0
        
        return {
            "trend": trend,
            "change_percent": round(change_percent, 2),
            "data_points": len(sorted_data),
            "first_value": sorted_data[0].get(value_field) if sorted_data else None,
            "last_value": sorted_data[-1].get(value_field) if sorted_data else None
        }
    
    def compare_datasets(
        self,
        dataset1: List[Dict[str, Any]],
        dataset2: List[Dict[str, Any]],
        key_field: str = "value"
    ) -> Dict[str, Any]:
        """Compare two datasets."""
        values1 = [d.get(key_field, 0) for d in dataset1]
        values2 = [d.get(key_field, 0) for d in dataset2]
        
        return {
            "dataset1": {
                "count": len(values1),
                "mean": statistics.mean(values1) if values1 else 0,
                "median": statistics.median(values1) if values1 else 0
            },
            "dataset2": {
                "count": len(values2),
                "mean": statistics.mean(values2) if values2 else 0,
                "median": statistics.median(values2) if values2 else 0
            },
            "comparison": {
                "mean_diff": (statistics.mean(values2) - statistics.mean(values1)) if values1 and values2 else 0,
                "mean_diff_percent": ((statistics.mean(values2) - statistics.mean(values1)) / statistics.mean(values1) * 100) if values1 and values2 and statistics.mean(values1) > 0 else 0
            }
        }

# Global instance
data_analyzer = DataAnalyzer()
















