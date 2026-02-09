"""Metrics and monitoring utilities."""

from datetime import datetime
from typing import Dict, Optional
from collections import defaultdict
import json
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """Collects and stores application metrics."""
    
    def __init__(self, metrics_dir: str = "./storage/metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.counters: Dict[str, int] = defaultdict(int)
        self.timings: Dict[str, list] = defaultdict(list)
    
    def increment(self, metric_name: str, value: int = 1):
        """Increment a counter metric."""
        self.counters[metric_name] += value
        logger.debug(f"Metric {metric_name} incremented by {value}")
    
    def record_timing(self, metric_name: str, duration: float):
        """Record a timing metric."""
        self.timings[metric_name].append(duration)
        logger.debug(f"Timing {metric_name}: {duration}s")
    
    def get_counter(self, metric_name: str) -> int:
        """Get counter value."""
        return self.counters.get(metric_name, 0)
    
    def get_avg_timing(self, metric_name: str) -> Optional[float]:
        """Get average timing."""
        timings = self.timings.get(metric_name, [])
        if not timings:
            return None
        return sum(timings) / len(timings)
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of all metrics."""
        return {
            "counters": dict(self.counters),
            "timings": {
                name: {
                    "count": len(times),
                    "avg": sum(times) / len(times) if times else 0,
                    "min": min(times) if times else 0,
                    "max": max(times) if times else 0
                }
                for name, times in self.timings.items()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def save_metrics(self):
        """Save metrics to file."""
        try:
            metrics_file = self.metrics_dir / f"metrics_{datetime.utcnow().strftime('%Y%m%d')}.json"
            summary = self.get_metrics_summary()
            
            # Load existing metrics if file exists
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    existing = json.load(f)
                    # Merge counters
                    for key, value in existing.get("counters", {}).items():
                        summary["counters"][key] = summary["counters"].get(key, 0) + value
            
            with open(metrics_file, "w") as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Metrics saved to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")


# Global metrics collector instance
metrics_collector = MetricsCollector()

