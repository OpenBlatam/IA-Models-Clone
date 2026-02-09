"""
UX Metrics System for Flux2 Clothing Changer
============================================

User experience metrics and analysis.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class UXEvent:
    """User experience event."""
    event_type: str
    user_id: Optional[str]
    session_id: Optional[str]
    timestamp: float
    duration: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class UXMetrics:
    """User experience metrics system."""
    
    def __init__(
        self,
        history_size: int = 10000,
    ):
        """
        Initialize UX metrics system.
        
        Args:
            history_size: Maximum number of events to keep
        """
        self.history_size = history_size
        self.events: deque = deque(maxlen=history_size)
        
        # Session tracking
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # User journey tracking
        self.user_journeys: Dict[str, List[str]] = defaultdict(list)
        
        # Performance metrics
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
    
    def track_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        duration: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UXEvent:
        """
        Track a UX event.
        
        Args:
            event_type: Event type
            user_id: Optional user ID
            session_id: Optional session ID
            duration: Event duration
            metadata: Optional metadata
            
        Returns:
            Created event
        """
        event = UXEvent(
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            timestamp=time.time(),
            duration=duration,
            metadata=metadata or {},
        )
        
        self.events.append(event)
        
        # Track in session
        if session_id:
            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    "user_id": user_id,
                    "start_time": event.timestamp,
                    "events": [],
                }
            self.sessions[session_id]["events"].append(event_type)
        
        # Track user journey
        if user_id:
            self.user_journeys[user_id].append(event_type)
        
        # Track performance
        if duration > 0:
            self.performance_metrics[event_type].append(duration)
        
        return event
    
    def track_page_view(
        self,
        page: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        duration: float = 0.0,
    ) -> None:
        """Track page view."""
        self.track_event(
            event_type="page_view",
            user_id=user_id,
            session_id=session_id,
            duration=duration,
            metadata={"page": page},
        )
    
    def track_interaction(
        self,
        interaction_type: str,
        element: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Track user interaction."""
        self.track_event(
            event_type="interaction",
            user_id=user_id,
            session_id=session_id,
            metadata={
                "interaction_type": interaction_type,
                "element": element,
            },
        )
    
    def track_error(
        self,
        error_type: str,
        error_message: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Track error event."""
        self.track_event(
            event_type="error",
            user_id=user_id,
            session_id=session_id,
            metadata={
                "error_type": error_type,
                "error_message": error_message,
            },
        )
    
    def get_session_metrics(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get metrics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session metrics
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        session_events = [
            e for e in self.events
            if e.session_id == session_id
        ]
        
        total_duration = (
            session_events[-1].timestamp - session["start_time"]
            if session_events else 0.0
        )
        
        return {
            "session_id": session_id,
            "user_id": session["user_id"],
            "start_time": session["start_time"],
            "duration": total_duration,
            "event_count": len(session_events),
            "events": session["events"],
            "page_views": len([e for e in session_events if e.event_type == "page_view"]),
            "errors": len([e for e in session_events if e.event_type == "error"]),
        }
    
    def get_user_journey(
        self,
        user_id: str,
    ) -> List[str]:
        """
        Get user journey.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of events in journey
        """
        return self.user_journeys.get(user_id, []).copy()
    
    def get_performance_metrics(
        self,
        event_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Args:
            event_type: Optional event type filter
            
        Returns:
            Performance metrics
        """
        if event_type:
            metrics = {event_type: self.performance_metrics.get(event_type, [])}
        else:
            metrics = dict(self.performance_metrics)
        
        result = {}
        for evt_type, durations in metrics.items():
            if durations:
                result[evt_type] = {
                    "count": len(durations),
                    "avg": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "p95": self._percentile(durations, 0.95),
                }
        
        return result
    
    def get_funnel_analysis(
        self,
        steps: List[str],
        time_range: Optional[timedelta] = None,
    ) -> Dict[str, Any]:
        """
        Analyze conversion funnel.
        
        Args:
            steps: List of funnel steps
            time_range: Time range to analyze
            
        Returns:
            Funnel analysis
        """
        cutoff_time = time.time() - time_range.total_seconds() if time_range else 0
        
        # Count users at each step
        step_counts = {}
        users_by_step = {}
        
        for step in steps:
            step_events = [
                e for e in self.events
                if e.event_type == step and e.timestamp >= cutoff_time
            ]
            step_counts[step] = len(step_events)
            users_by_step[step] = set(e.user_id for e in step_events if e.user_id)
        
        # Calculate conversion rates
        conversions = {}
        if step_counts:
            first_step_count = step_counts[steps[0]]
            for i, step in enumerate(steps[1:], 1):
                current_count = step_counts[step]
                conversions[step] = (
                    current_count / first_step_count
                    if first_step_count > 0 else 0.0
                )
        
        return {
            "steps": steps,
            "step_counts": step_counts,
            "conversions": conversions,
            "drop_offs": self._calculate_drop_offs(step_counts, steps),
        }
    
    def _calculate_drop_offs(
        self,
        step_counts: Dict[str, int],
        steps: List[str],
    ) -> Dict[str, float]:
        """Calculate drop-off rates."""
        drop_offs = {}
        
        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]
            
            current_count = step_counts.get(current_step, 0)
            next_count = step_counts.get(next_step, 0)
            
            if current_count > 0:
                drop_off = (current_count - next_count) / current_count
                drop_offs[f"{current_step}_to_{next_step}"] = drop_off
        
        return drop_offs
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get UX metrics statistics."""
        return {
            "total_events": len(self.events),
            "total_sessions": len(self.sessions),
            "total_users": len(self.user_journeys),
            "performance_metrics": self.get_performance_metrics(),
        }


