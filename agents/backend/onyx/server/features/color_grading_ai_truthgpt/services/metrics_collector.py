"""
Metrics Collector for Color Grading AI
======================================

Collects metrics and analytics for color grading operations.
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetric:
    """Single processing metric."""
    timestamp: datetime
    operation: str  # grade_video, grade_image, analyze
    media_type: str  # video, image
    duration: float  # Processing time in seconds
    success: bool
    file_size: int  # Input file size in bytes
    output_size: Optional[int] = None
    template_used: Optional[str] = None
    error: Optional[str] = None


class MetricsCollector:
    """
    Collects and aggregates metrics.
    
    Features:
    - Track processing times
    - Track success/failure rates
    - Track template usage
    - Export metrics
    """
    
    def __init__(self, metrics_dir: str = "metrics"):
        """
        Initialize metrics collector.
        
        Args:
            metrics_dir: Directory for metrics storage
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self._metrics: List[ProcessingMetric] = []
        self._aggregates: Dict[str, Any] = defaultdict(lambda: {
            "count": 0,
            "success": 0,
            "failed": 0,
            "total_duration": 0.0,
            "total_size": 0,
        })
    
    def record(
        self,
        operation: str,
        media_type: str,
        duration: float,
        success: bool,
        file_size: int,
        output_size: Optional[int] = None,
        template_used: Optional[str] = None,
        error: Optional[str] = None
    ):
        """
        Record a processing metric.
        
        Args:
            operation: Operation type
            media_type: Media type
            duration: Processing duration
            success: Success status
            file_size: Input file size
            output_size: Output file size
            template_used: Template used
            error: Error message if failed
        """
        metric = ProcessingMetric(
            timestamp=datetime.now(),
            operation=operation,
            media_type=media_type,
            duration=duration,
            success=success,
            file_size=file_size,
            output_size=output_size,
            template_used=template_used,
            error=error
        )
        
        self._metrics.append(metric)
        
        # Update aggregates
        key = f"{operation}_{media_type}"
        self._aggregates[key]["count"] += 1
        if success:
            self._aggregates[key]["success"] += 1
        else:
            self._aggregates[key]["failed"] += 1
        self._aggregates[key]["total_duration"] += duration
        self._aggregates[key]["total_size"] += file_size
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        for key, agg in self._aggregates.items():
            if agg["count"] > 0:
                stats[key] = {
                    "count": agg["count"],
                    "success_rate": agg["success"] / agg["count"] * 100,
                    "failure_rate": agg["failed"] / agg["count"] * 100,
                    "avg_duration": agg["total_duration"] / agg["count"],
                    "total_duration": agg["total_duration"],
                    "total_size": agg["total_size"],
                    "avg_size": agg["total_size"] / agg["count"],
                }
        
        # Overall stats
        total_metrics = len(self._metrics)
        if total_metrics > 0:
            successful = sum(1 for m in self._metrics if m.success)
            stats["overall"] = {
                "total_operations": total_metrics,
                "successful": successful,
                "failed": total_metrics - successful,
                "success_rate": successful / total_metrics * 100,
            }
        
        return stats
    
    def get_template_stats(self) -> Dict[str, Any]:
        """
        Get statistics by template.
        
        Returns:
            Dictionary with template statistics
        """
        template_stats = defaultdict(lambda: {"count": 0, "success": 0, "failed": 0})
        
        for metric in self._metrics:
            if metric.template_used:
                template_stats[metric.template_used]["count"] += 1
                if metric.success:
                    template_stats[metric.template_used]["success"] += 1
                else:
                    template_stats[metric.template_used]["failed"] += 1
        
        return dict(template_stats)
    
    async def export(self, format: str = "json") -> str:
        """
        Export metrics to file.
        
        Args:
            format: Export format (json, csv)
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            file_path = self.metrics_dir / f"metrics_{timestamp}.json"
            data = {
                "exported_at": datetime.now().isoformat(),
                "stats": self.get_stats(),
                "template_stats": self.get_template_stats(),
                "metrics": [asdict(m) for m in self._metrics[-1000:]]  # Last 1000
            }
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(file_path)




