"""
Time Travel Service for Gamma App
=================================

Advanced service for Time Travel capabilities including temporal
manipulation, timeline management, and temporal paradox prevention.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import json
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class TimeTravelType(str, Enum):
    """Types of time travel."""
    PAST = "past"
    FUTURE = "future"
    PARALLEL = "parallel"
    LOOP = "loop"
    BRANCH = "branch"
    MERGE = "merge"
    FREEZE = "freeze"
    ACCELERATE = "accelerate"

class TimelineStability(str, Enum):
    """Timeline stability levels."""
    STABLE = "stable"
    UNSTABLE = "unstable"
    CRITICAL = "critical"
    PARADOX = "paradox"
    COLLAPSING = "collapsing"
    RESTORED = "restored"

class TemporalEventType(str, Enum):
    """Types of temporal events."""
    CREATION = "creation"
    MODIFICATION = "modification"
    DELETION = "deletion"
    DIVERGENCE = "divergence"
    CONVERGENCE = "convergence"
    PARADOX = "paradox"
    ANOMALY = "anomaly"
    CORRECTION = "correction"

@dataclass
class Timeline:
    """Timeline definition."""
    timeline_id: str
    name: str
    creation_date: datetime
    stability: TimelineStability
    divergence_point: Optional[datetime]
    key_events: List[Dict[str, Any]]
    paradox_count: int
    is_active: bool = True
    last_accessed: Optional[datetime] = None

@dataclass
class TimeTravelEvent:
    """Time travel event definition."""
    event_id: str
    timeline_id: str
    traveler_id: str
    travel_type: TimeTravelType
    departure_time: datetime
    arrival_time: datetime
    destination_timeline: str
    purpose: str
    changes_made: List[Dict[str, Any]]
    paradox_risk: float
    success: bool
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class TemporalAnomaly:
    """Temporal anomaly definition."""
    anomaly_id: str
    timeline_id: str
    anomaly_type: TemporalEventType
    location: Tuple[float, float, float]
    time_coordinates: datetime
    severity: float
    description: str
    detected_by: str
    resolution_status: str
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class TemporalParadox:
    """Temporal paradox definition."""
    paradox_id: str
    timeline_id: str
    paradox_type: str
    description: str
    severity: float
    affected_events: List[str]
    resolution_attempts: List[Dict[str, Any]]
    is_resolved: bool = False
    created_at: datetime = field(default_factory=datetime.now)

class TimeTravelService:
    """Service for Time Travel capabilities."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.timelines: Dict[str, Timeline] = {}
        self.time_travel_events: List[TimeTravelEvent] = []
        self.temporal_anomalies: List[TemporalAnomaly] = []
        self.temporal_paradoxes: List[TemporalParadox] = []
        self.active_travelers: Dict[str, str] = {}  # traveler_id -> timeline_id
        
        # Initialize primary timeline
        self._initialize_primary_timeline()
        
        logger.info("TimeTravelService initialized")
    
    async def create_timeline(self, timeline_info: Dict[str, Any]) -> str:
        """Create a new timeline."""
        try:
            timeline_id = str(uuid.uuid4())
            timeline = Timeline(
                timeline_id=timeline_id,
                name=timeline_info.get("name", "Unknown Timeline"),
                creation_date=datetime.now(),
                stability=TimelineStability.STABLE,
                divergence_point=timeline_info.get("divergence_point"),
                key_events=timeline_info.get("key_events", []),
                paradox_count=0
            )
            
            self.timelines[timeline_id] = timeline
            logger.info(f"Timeline created: {timeline_id}")
            return timeline_id
            
        except Exception as e:
            logger.error(f"Error creating timeline: {e}")
            raise
    
    async def initiate_time_travel(self, travel_info: Dict[str, Any]) -> str:
        """Initiate time travel."""
        try:
            event_id = str(uuid.uuid4())
            travel_event = TimeTravelEvent(
                event_id=event_id,
                timeline_id=travel_info.get("timeline_id", ""),
                traveler_id=travel_info.get("traveler_id", ""),
                travel_type=TimeTravelType(travel_info.get("travel_type", "past")),
                departure_time=datetime.now(),
                arrival_time=travel_info.get("arrival_time", datetime.now()),
                destination_timeline=travel_info.get("destination_timeline", ""),
                purpose=travel_info.get("purpose", ""),
                changes_made=[],
                paradox_risk=self._calculate_paradox_risk(travel_info),
                success=False
            )
            
            self.time_travel_events.append(travel_event)
            self.active_travelers[travel_event.traveler_id] = travel_event.timeline_id
            
            # Start time travel in background
            asyncio.create_task(self._execute_time_travel(event_id))
            
            logger.info(f"Time travel initiated: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Error initiating time travel: {e}")
            raise
    
    async def detect_temporal_anomaly(self, anomaly_info: Dict[str, Any]) -> str:
        """Detect a temporal anomaly."""
        try:
            anomaly_id = str(uuid.uuid4())
            anomaly = TemporalAnomaly(
                anomaly_id=anomaly_id,
                timeline_id=anomaly_info.get("timeline_id", ""),
                anomaly_type=TemporalEventType(anomaly_info.get("anomaly_type", "anomaly")),
                location=anomaly_info.get("location", (0.0, 0.0, 0.0)),
                time_coordinates=anomaly_info.get("time_coordinates", datetime.now()),
                severity=anomaly_info.get("severity", 0.5),
                description=anomaly_info.get("description", ""),
                detected_by=anomaly_info.get("detected_by", "system"),
                resolution_status="detected"
            )
            
            self.temporal_anomalies.append(anomaly)
            
            # Check for paradox creation
            if anomaly.severity > 0.8:
                await self._check_paradox_creation(anomaly)
            
            logger.info(f"Temporal anomaly detected: {anomaly_id}")
            return anomaly_id
            
        except Exception as e:
            logger.error(f"Error detecting temporal anomaly: {e}")
            raise
    
    async def resolve_temporal_paradox(self, paradox_id: str, resolution_info: Dict[str, Any]) -> bool:
        """Resolve a temporal paradox."""
        try:
            paradox = next((p for p in self.temporal_paradoxes if p.paradox_id == paradox_id), None)
            if not paradox:
                return False
            
            resolution_attempt = {
                "method": resolution_info.get("method", "unknown"),
                "description": resolution_info.get("description", ""),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            
            # Simulate paradox resolution
            success_probability = 0.7 - (paradox.severity * 0.3)
            resolution_attempt["success"] = np.random.random() < success_probability
            
            paradox.resolution_attempts.append(resolution_attempt)
            
            if resolution_attempt["success"]:
                paradox.is_resolved = True
                paradox.severity = 0.0
                
                # Update timeline stability
                timeline = self.timelines.get(paradox.timeline_id)
                if timeline:
                    timeline.stability = TimelineStability.RESTORED
                    timeline.paradox_count = max(0, timeline.paradox_count - 1)
            
            logger.info(f"Paradox resolution attempt: {paradox_id}, Success: {resolution_attempt['success']}")
            return resolution_attempt["success"]
            
        except Exception as e:
            logger.error(f"Error resolving temporal paradox: {e}")
            return False
    
    async def get_timeline_status(self, timeline_id: str) -> Optional[Dict[str, Any]]:
        """Get timeline status."""
        try:
            if timeline_id not in self.timelines:
                return None
            
            timeline = self.timelines[timeline_id]
            return {
                "timeline_id": timeline.timeline_id,
                "name": timeline.name,
                "creation_date": timeline.creation_date.isoformat(),
                "stability": timeline.stability.value,
                "divergence_point": timeline.divergence_point.isoformat() if timeline.divergence_point else None,
                "key_events": timeline.key_events,
                "paradox_count": timeline.paradox_count,
                "is_active": timeline.is_active,
                "last_accessed": timeline.last_accessed.isoformat() if timeline.last_accessed else None
            }
            
        except Exception as e:
            logger.error(f"Error getting timeline status: {e}")
            return None
    
    async def get_time_travel_history(self, traveler_id: str) -> List[Dict[str, Any]]:
        """Get time travel history for a traveler."""
        try:
            traveler_events = [e for e in self.time_travel_events if e.traveler_id == traveler_id]
            
            history = []
            for event in traveler_events:
                history.append({
                    "event_id": event.event_id,
                    "timeline_id": event.timeline_id,
                    "travel_type": event.travel_type.value,
                    "departure_time": event.departure_time.isoformat(),
                    "arrival_time": event.arrival_time.isoformat(),
                    "destination_timeline": event.destination_timeline,
                    "purpose": event.purpose,
                    "changes_made": event.changes_made,
                    "paradox_risk": event.paradox_risk,
                    "success": event.success,
                    "created_at": event.created_at.isoformat()
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting time travel history: {e}")
            return []
    
    async def get_temporal_statistics(self) -> Dict[str, Any]:
        """Get temporal service statistics."""
        try:
            total_timelines = len(self.timelines)
            stable_timelines = len([t for t in self.timelines.values() if t.stability == TimelineStability.STABLE])
            total_travel_events = len(self.time_travel_events)
            successful_travels = len([e for e in self.time_travel_events if e.success])
            total_anomalies = len(self.temporal_anomalies)
            resolved_anomalies = len([a for a in self.temporal_anomalies if a.resolution_status == "resolved"])
            total_paradoxes = len(self.temporal_paradoxes)
            resolved_paradoxes = len([p for p in self.temporal_paradoxes if p.is_resolved])
            active_travelers = len(self.active_travelers)
            
            # Travel type distribution
            travel_type_stats = {}
            for event in self.time_travel_events:
                travel_type = event.travel_type.value
                travel_type_stats[travel_type] = travel_type_stats.get(travel_type, 0) + 1
            
            # Timeline stability distribution
            stability_stats = {}
            for timeline in self.timelines.values():
                stability = timeline.stability.value
                stability_stats[stability] = stability_stats.get(stability, 0) + 1
            
            # Anomaly type distribution
            anomaly_type_stats = {}
            for anomaly in self.temporal_anomalies:
                anomaly_type = anomaly.anomaly_type.value
                anomaly_type_stats[anomaly_type] = anomaly_type_stats.get(anomaly_type, 0) + 1
            
            return {
                "total_timelines": total_timelines,
                "stable_timelines": stable_timelines,
                "timeline_stability_rate": (stable_timelines / total_timelines * 100) if total_timelines > 0 else 0,
                "total_travel_events": total_travel_events,
                "successful_travels": successful_travels,
                "travel_success_rate": (successful_travels / total_travel_events * 100) if total_travel_events > 0 else 0,
                "total_anomalies": total_anomalies,
                "resolved_anomalies": resolved_anomalies,
                "anomaly_resolution_rate": (resolved_anomalies / total_anomalies * 100) if total_anomalies > 0 else 0,
                "total_paradoxes": total_paradoxes,
                "resolved_paradoxes": resolved_paradoxes,
                "paradox_resolution_rate": (resolved_paradoxes / total_paradoxes * 100) if total_paradoxes > 0 else 0,
                "active_travelers": active_travelers,
                "travel_type_distribution": travel_type_stats,
                "timeline_stability_distribution": stability_stats,
                "anomaly_type_distribution": anomaly_type_stats,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting temporal statistics: {e}")
            return {}
    
    async def _execute_time_travel(self, event_id: str):
        """Execute time travel in background."""
        try:
            travel_event = next((e for e in self.time_travel_events if e.event_id == event_id), None)
            if not travel_event:
                return
            
            # Simulate time travel process
            await asyncio.sleep(2)  # Simulate travel time
            
            # Check for paradox creation
            if travel_event.paradox_risk > 0.7:
                await self._create_temporal_paradox(travel_event)
            
            # Simulate changes made during travel
            changes = self._simulate_temporal_changes(travel_event)
            travel_event.changes_made = changes
            
            # Determine success based on paradox risk
            success_probability = 1.0 - (travel_event.paradox_risk * 0.5)
            travel_event.success = np.random.random() < success_probability
            
            if travel_event.success:
                # Update timeline if travel was successful
                timeline = self.timelines.get(travel_event.timeline_id)
                if timeline:
                    timeline.last_accessed = datetime.now()
                    timeline.key_events.append({
                        "event_id": event_id,
                        "type": "time_travel",
                        "timestamp": datetime.now().isoformat(),
                        "description": f"Time travel to {travel_event.arrival_time.isoformat()}"
                    })
            
            logger.info(f"Time travel {event_id} completed. Success: {travel_event.success}")
            
        except Exception as e:
            logger.error(f"Error executing time travel {event_id}: {e}")
            travel_event = next((e for e in self.time_travel_events if e.event_id == event_id), None)
            if travel_event:
                travel_event.success = False
    
    async def _create_temporal_paradox(self, travel_event: TimeTravelEvent):
        """Create a temporal paradox from time travel."""
        try:
            paradox_id = str(uuid.uuid4())
            paradox = TemporalParadox(
                paradox_id=paradox_id,
                timeline_id=travel_event.timeline_id,
                paradox_type="causal_loop",
                description=f"Paradox created by time travel event {travel_event.event_id}",
                severity=travel_event.paradox_risk,
                affected_events=[travel_event.event_id],
                resolution_attempts=[]
            )
            
            self.temporal_paradoxes.append(paradox)
            
            # Update timeline stability
            timeline = self.timelines.get(travel_event.timeline_id)
            if timeline:
                timeline.paradox_count += 1
                if timeline.paradox_count > 3:
                    timeline.stability = TimelineStability.CRITICAL
                elif timeline.paradox_count > 1:
                    timeline.stability = TimelineStability.UNSTABLE
            
            logger.info(f"Temporal paradox created: {paradox_id}")
            
        except Exception as e:
            logger.error(f"Error creating temporal paradox: {e}")
    
    async def _check_paradox_creation(self, anomaly: TemporalAnomaly):
        """Check if anomaly creates a paradox."""
        try:
            if anomaly.severity > 0.9:
                paradox_id = str(uuid.uuid4())
                paradox = TemporalParadox(
                    paradox_id=paradox_id,
                    timeline_id=anomaly.timeline_id,
                    paradox_type="anomaly_induced",
                    description=f"Paradox created by temporal anomaly {anomaly.anomaly_id}",
                    severity=anomaly.severity,
                    affected_events=[anomaly.anomaly_id],
                    resolution_attempts=[]
                )
                
                self.temporal_paradoxes.append(paradox)
                
                # Update timeline stability
                timeline = self.timelines.get(anomaly.timeline_id)
                if timeline:
                    timeline.paradox_count += 1
                    timeline.stability = TimelineStability.PARADOX
                
                logger.info(f"Paradox created from anomaly: {paradox_id}")
                
        except Exception as e:
            logger.error(f"Error checking paradox creation: {e}")
    
    def _calculate_paradox_risk(self, travel_info: Dict[str, Any]) -> float:
        """Calculate paradox risk for time travel."""
        try:
            base_risk = 0.1
            
            # Increase risk based on travel type
            travel_type = travel_info.get("travel_type", "past")
            if travel_type == "past":
                base_risk += 0.3
            elif travel_type == "loop":
                base_risk += 0.5
            elif travel_type == "branch":
                base_risk += 0.2
            
            # Increase risk based on purpose
            purpose = travel_info.get("purpose", "").lower()
            if "change" in purpose or "prevent" in purpose:
                base_risk += 0.4
            elif "observe" in purpose:
                base_risk += 0.1
            
            # Add random factor
            base_risk += np.random.uniform(0, 0.2)
            
            return min(1.0, base_risk)
            
        except Exception as e:
            logger.error(f"Error calculating paradox risk: {e}")
            return 0.5
    
    def _simulate_temporal_changes(self, travel_event: TimeTravelEvent) -> List[Dict[str, Any]]:
        """Simulate changes made during time travel."""
        try:
            changes = []
            
            # Simulate different types of changes based on purpose
            purpose = travel_event.purpose.lower()
            
            if "observe" in purpose:
                changes.append({
                    "type": "observation",
                    "description": "Observed historical event",
                    "impact": "minimal"
                })
            elif "prevent" in purpose:
                changes.append({
                    "type": "prevention",
                    "description": "Prevented historical event",
                    "impact": "major"
                })
            elif "change" in purpose:
                changes.append({
                    "type": "modification",
                    "description": "Modified historical event",
                    "impact": "moderate"
                })
            else:
                changes.append({
                    "type": "interaction",
                    "description": "Interacted with historical figures",
                    "impact": "minor"
                })
            
            return changes
            
        except Exception as e:
            logger.error(f"Error simulating temporal changes: {e}")
            return []
    
    def _initialize_primary_timeline(self):
        """Initialize the primary timeline."""
        try:
            primary_timeline = Timeline(
                timeline_id="primary_timeline",
                name="Primary Timeline",
                creation_date=datetime.now(),
                stability=TimelineStability.STABLE,
                divergence_point=None,
                key_events=[
                    {
                        "event_id": "big_bang",
                        "type": "cosmic",
                        "timestamp": "13.8 billion years ago",
                        "description": "The Big Bang"
                    },
                    {
                        "event_id": "earth_formation",
                        "type": "planetary",
                        "timestamp": "4.5 billion years ago",
                        "description": "Earth formation"
                    },
                    {
                        "event_id": "life_origin",
                        "type": "biological",
                        "timestamp": "3.8 billion years ago",
                        "description": "Origin of life"
                    }
                ],
                paradox_count=0
            )
            
            self.timelines["primary_timeline"] = primary_timeline
            logger.info("Primary timeline initialized")
            
        except Exception as e:
            logger.error(f"Error initializing primary timeline: {e}")


