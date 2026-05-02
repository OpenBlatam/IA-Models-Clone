"""
Advanced metrics export utilities.

Provides export capabilities for various monitoring systems.
"""
from __future__ import annotations

import logging
import json
import csv
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsExporter:
    """
    Metrics exporter for various formats.
    
    Exports cache metrics to different formats.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize metrics exporter.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
    
    def export_to_json(
        self,
        filepath: str,
        include_history: bool = False
    ) -> None:
        """
        Export metrics to JSON file.
        
        Args:
            filepath: Path to output file
            include_history: Whether to include history
        """
        stats = self.cache.get_stats(include_history=include_history)
        
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to JSON: {filepath}")
    
    def export_to_csv(
        self,
        filepath: str,
        metrics: Optional[List[str]] = None
    ) -> None:
        """
        Export metrics to CSV file.
        
        Args:
            filepath: Path to output file
            metrics: List of metric names to export (None = all)
        """
        stats = self.cache.get_stats()
        
        if metrics is None:
            metrics = list(stats.keys())
        
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value"])
            writer.writeheader()
            
            for metric in metrics:
                if metric in stats:
                    writer.writerow({
                        "metric": metric,
                        "value": stats[metric]
                    })
        
        logger.info(f"Metrics exported to CSV: {filepath}")
    
    def export_to_prometheus(
        self,
        filepath: str
    ) -> None:
        """
        Export metrics to Prometheus format.
        
        Args:
            filepath: Path to output file
        """
        stats = self.cache.get_stats()
        
        with open(filepath, "w") as f:
            f.write("# Cache Metrics (Prometheus format)\n")
            
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    # Sanitize metric name
                    metric_name = key.replace(".", "_").replace("-", "_")
                    f.write(f"cache_{metric_name} {value}\n")
        
        logger.info(f"Metrics exported to Prometheus format: {filepath}")
    
    def export_to_influxdb_line_protocol(
        self,
        filepath: str,
        measurement: str = "cache_metrics",
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Export metrics to InfluxDB line protocol.
        
        Args:
            filepath: Path to output file
            measurement: Measurement name
            tags: Optional tags
        """
        stats = self.cache.get_stats()
        
        tags_str = ""
        if tags:
            tag_pairs = [f"{k}={v}" for k, v in tags.items()]
            tags_str = "," + ",".join(tag_pairs)
        
        with open(filepath, "w") as f:
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    field_name = key.replace(".", "_").replace("-", "_")
                    f.write(f"{measurement}{tags_str} {field_name}={value}\n")
        
        logger.info(f"Metrics exported to InfluxDB format: {filepath}")
    
    def export_summary_report(
        self,
        filepath: str
    ) -> str:
        """
        Export summary report.
        
        Args:
            filepath: Path to output file
            
        Returns:
            Report content
        """
        stats = self.cache.get_stats()
        config = self.cache.config
        
        report = f"""
# Cache Metrics Summary Report

Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Configuration
- Max Tokens: {config.max_tokens}
- Strategy: {config.cache_strategy.value}
- Quantization: {config.use_quantization}
- Compression: {config.use_compression}

## Performance Metrics
- Hit Rate: {stats.get('hit_rate', 0.0):.2%}
- Total Hits: {stats.get('hits', 0)}
- Total Misses: {stats.get('misses', 0)}
- Evictions: {stats.get('evictions', 0)}

## Cache State
- Current Entries: {stats.get('num_entries', 0)}
- Memory Usage: {stats.get('storage_memory_mb', 0.0):.2f} MB
- Utilization: {(stats.get('num_entries', 0) / max(config.max_tokens, 1)) * 100:.2f}%
"""
        
        with open(filepath, "w") as f:
            f.write(report)
        
        logger.info(f"Summary report exported: {filepath}")
        
        return report

