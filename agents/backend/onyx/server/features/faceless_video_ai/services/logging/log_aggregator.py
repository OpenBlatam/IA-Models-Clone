"""
Log Aggregator
Aggregate and analyze logs
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LogAggregator:
    """Aggregate and analyze logs"""
    
    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else Path("/tmp/faceless_video/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse JSON log line"""
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None
    
    def aggregate_logs(
        self,
        time_range: timedelta = timedelta(hours=1),
        log_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Aggregate logs for time range
        
        Args:
            time_range: Time range to analyze
            log_file: Log file path (optional)
            
        Returns:
            Aggregated statistics
        """
        if log_file:
            log_path = Path(log_file)
        else:
            # Find latest log file
            log_files = sorted(self.log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            log_path = log_files[0] if log_files else None
        
        if not log_path or not log_path.exists():
            return {"error": "No log file found"}
        
        cutoff_time = datetime.utcnow() - time_range
        
        stats = {
            "total_logs": 0,
            "by_level": defaultdict(int),
            "by_event_type": defaultdict(int),
            "errors": [],
            "api_requests": [],
            "video_generations": [],
        }
        
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    log_entry = self.parse_log_line(line.strip())
                    if not log_entry:
                        continue
                    
                    timestamp_str = log_entry.get("timestamp")
                    if timestamp_str:
                        try:
                            log_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if log_time < cutoff_time:
                                continue
                        except ValueError:
                            pass
                    
                    stats["total_logs"] += 1
                    stats["by_level"][log_entry.get("level", "UNKNOWN")] += 1
                    stats["by_event_type"][log_entry.get("event_type", "unknown")] += 1
                    
                    if log_entry.get("level") == "ERROR":
                        stats["errors"].append(log_entry)
                    
                    if log_entry.get("event_type") == "api_request":
                        stats["api_requests"].append(log_entry)
                    
                    if log_entry.get("event_type") == "video_generation":
                        stats["video_generations"].append(log_entry)
        
        except Exception as e:
            logger.error(f"Log aggregation failed: {str(e)}")
            return {"error": str(e)}
        
        return stats
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary"""
        stats = self.aggregate_logs(time_range=timedelta(hours=hours))
        
        error_summary = {
            "total_errors": len(stats.get("errors", [])),
            "error_types": defaultdict(int),
            "recent_errors": stats.get("errors", [])[-10:],  # Last 10 errors
        }
        
        for error in stats.get("errors", []):
            error_type = error.get("error_type", "unknown")
            error_summary["error_types"][error_type] += 1
        
        return error_summary


_log_aggregator: Optional[LogAggregator] = None


def get_log_aggregator(log_dir: Optional[str] = None) -> LogAggregator:
    """Get log aggregator instance (singleton)"""
    global _log_aggregator
    if _log_aggregator is None:
        _log_aggregator = LogAggregator(log_dir=log_dir)
    return _log_aggregator

