"""
Telemetry Service for Color Grading AI
=======================================

Telemetry and observability service.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class TelemetryEvent:
    """Telemetry event."""
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class TelemetryService:
    """
    Telemetry and observability service.
    
    Features:
    - Event tracking
    - User behavior analytics
    - Performance telemetry
    - Error tracking
    - Custom metrics
    """
    
    def __init__(self, storage_dir: str = "telemetry"):
        """
        Initialize telemetry service.
        
        Args:
            storage_dir: Directory for telemetry storage
        """
        from pathlib import Path
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._events: List[TelemetryEvent] = []
        self._metrics: Dict[str, Any] = defaultdict(int)
        self._max_events_in_memory = 1000
    
    def track_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Track telemetry event.
        
        Args:
            event_type: Event type
            data: Event data
            user_id: Optional user ID
            session_id: Optional session ID
        """
        event = TelemetryEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            user_id=user_id,
            session_id=session_id
        )
        
        self._events.append(event)
        
        # Flush to disk if too many events
        if len(self._events) >= self._max_events_in_memory:
            self._flush_events()
        
        logger.debug(f"Tracked event: {event_type}")
    
    def track_metric(self, metric_name: str, value: float = 1.0):
        """
        Track custom metric.
        
        Args:
            metric_name: Metric name
            value: Metric value
        """
        self._metrics[metric_name] += value
    
    def track_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Track error.
        
        Args:
            error_type: Error type
            error_message: Error message
            context: Optional context
        """
        self.track_event(
            "error",
            {
                "error_type": error_type,
                "error_message": error_message,
                "context": context or {}
            }
        )
    
    def track_performance(
        self,
        operation: str,
        duration: float,
        success: bool = True
    ):
        """
        Track performance metric.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            success: Whether operation succeeded
        """
        self.track_event(
            "performance",
            {
                "operation": operation,
                "duration": duration,
                "success": success
            }
        )
        
        self.track_metric(f"operation.{operation}.duration", duration)
        self.track_metric(f"operation.{operation}.count", 1)
        if success:
            self.track_metric(f"operation.{operation}.success", 1)
        else:
            self.track_metric(f"operation.{operation}.failure", 1)
    
    def _flush_events(self):
        """Flush events to disk."""
        if not self._events:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        events_file = self.storage_dir / f"events_{timestamp}.json"
        
        try:
            events_data = [
                {
                    "event_type": e.event_type,
                    "timestamp": e.timestamp.isoformat(),
                    "data": e.data,
                    "user_id": e.user_id,
                    "session_id": e.session_id,
                }
                for e in self._events
            ]
            
            with open(events_file, "w") as f:
                json.dump(events_data, f, indent=2, default=str)
            
            self._events.clear()
            logger.debug(f"Flushed {len(events_data)} events to disk")
        except Exception as e:
            logger.error(f"Error flushing events: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return dict(self._metrics)
    
    def get_event_summary(
        self,
        event_type: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get event summary.
        
        Args:
            event_type: Optional event type filter
            hours: Number of hours to analyze
            
        Returns:
            Event summary
        """
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Load events from disk if needed
        all_events = self._load_recent_events(cutoff)
        
        if event_type:
            all_events = [e for e in all_events if e.event_type == event_type]
        
        return {
            "total_events": len(all_events),
            "event_types": self._count_by_type(all_events),
            "time_range": {
                "start": cutoff.isoformat(),
                "end": datetime.now().isoformat(),
            }
        }
    
    def _load_recent_events(self, cutoff: datetime) -> List[TelemetryEvent]:
        """Load recent events from disk."""
        events = []
        
        # Add in-memory events
        events.extend([e for e in self._events if e.timestamp > cutoff])
        
        # Load from disk
        for events_file in self.storage_dir.glob("events_*.json"):
            try:
                with open(events_file, "r") as f:
                    file_events = json.load(f)
                
                for event_data in file_events:
                    event_time = datetime.fromisoformat(event_data["timestamp"])
                    if event_time > cutoff:
                        events.append(TelemetryEvent(
                            event_type=event_data["event_type"],
                            timestamp=event_time,
                            data=event_data["data"],
                            user_id=event_data.get("user_id"),
                            session_id=event_data.get("session_id")
                        ))
            except Exception as e:
                logger.debug(f"Error loading events from {events_file}: {e}")
        
        return events
    
    def _count_by_type(self, events: List[TelemetryEvent]) -> Dict[str, int]:
        """Count events by type."""
        counts = defaultdict(int)
        for event in events:
            counts[event.event_type] += 1
        return dict(counts)
    
    def export_telemetry(
        self,
        output_path: str,
        format: str = "json"
    ) -> str:
        """
        Export telemetry data.
        
        Args:
            output_path: Output file path
            format: Export format (json, csv)
            
        Returns:
            Path to exported file
        """
        from pathlib import Path
        output_file = Path(output_path)
        
        if format == "json":
            data = {
                "metrics": self.get_metrics(),
                "events": self.get_event_summary(),
            }
            
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        
        return str(output_file)




