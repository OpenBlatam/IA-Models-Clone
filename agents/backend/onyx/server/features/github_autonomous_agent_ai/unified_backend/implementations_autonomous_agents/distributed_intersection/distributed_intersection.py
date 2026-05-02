"""
A Distributed Approach to Autonomous Intersection Management
=============================================================

Paper: "A Distributed Approach to Autonomous Intersection"

Key concepts:
- Distributed intersection management
- Vehicle-to-infrastructure communication
- Reservation-based intersection control
- Conflict resolution
- Real-time coordination
"""

from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import math

from ..common.agent_base import BaseAgent, AgentState, AgentStatus


class IntersectionState(Enum):
    """Intersection states."""
    FREE = "free"
    RESERVED = "reserved"
    OCCUPIED = "occupied"
    CONFLICT = "conflict"


class ReservationStatus(Enum):
    """Reservation status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ACTIVE = "active"
    COMPLETED = "completed"


@dataclass
class IntersectionZone:
    """A zone within an intersection."""
    zone_id: str
    position: Tuple[float, float]
    radius: float
    state: IntersectionState = IntersectionState.FREE
    reserved_by: Optional[str] = None
    reservation_time: Optional[datetime] = None


@dataclass
class ReservationRequest:
    """A reservation request for intersection access."""
    request_id: str
    vehicle_id: str
    zones: List[str]  # Zone IDs to reserve
    entry_time: datetime
    exit_time: datetime
    status: ReservationStatus = ReservationStatus.PENDING
    priority: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)


class DistributedIntersectionManager(BaseAgent):
    """
    Distributed intersection manager for autonomous vehicles.
    
    Manages intersection access through distributed reservation system.
    """
    
    def __init__(
        self,
        name: str,
        intersection_id: str,
        zones: List[IntersectionZone],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize distributed intersection manager.
        
        Args:
            name: Agent name
            intersection_id: Intersection identifier
            zones: List of intersection zones
            config: Configuration parameters
        """
        super().__init__(name, config)
        self.intersection_id = intersection_id
        self.zones = {zone.zone_id: zone for zone in zones}
        
        # Reservation system
        self.reservations: Dict[str, ReservationRequest] = {}
        self.active_reservations: Set[str] = set()
        
        # Communication
        self.pending_requests: List[ReservationRequest] = []
        self.conflict_queue: List[ReservationRequest] = []
        
        # Parameters
        self.time_slot_duration = config.get("time_slot_duration", 1.0)  # seconds
        self.max_reservation_time = config.get("max_reservation_time", 10.0)  # seconds
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about intersection management task.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Thinking result
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Process pending requests
        processed_requests = self._process_pending_requests()
        
        # Resolve conflicts
        resolved_conflicts = self._resolve_conflicts()
        
        result = {
            "task": task,
            "processed_requests": len(processed_requests),
            "resolved_conflicts": len(resolved_conflicts),
            "active_reservations": len(self.active_reservations),
            "free_zones": len([z for z in self.zones.values() if z.state == IntersectionState.FREE])
        }
        
        self.state.add_step("thinking", result)
        return result
    
    def _process_pending_requests(self) -> List[ReservationRequest]:
        """Process pending reservation requests."""
        processed = []
        
        for request in self.pending_requests[:]:
            if self._can_approve_request(request):
                request.status = ReservationStatus.APPROVED
                self.reservations[request.request_id] = request
                self.active_reservations.add(request.request_id)
                
                # Reserve zones
                for zone_id in request.zones:
                    if zone_id in self.zones:
                        zone = self.zones[zone_id]
                        zone.state = IntersectionState.RESERVED
                        zone.reserved_by = request.vehicle_id
                        zone.reservation_time = request.entry_time
                
                processed.append(request)
                self.pending_requests.remove(request)
            else:
                # Check for conflicts
                if self._has_conflicts(request):
                    request.status = ReservationStatus.PENDING
                    self.conflict_queue.append(request)
                    self.pending_requests.remove(request)
        
        return processed
    
    def _can_approve_request(self, request: ReservationRequest) -> bool:
        """Check if request can be approved."""
        # Check if all requested zones are available
        for zone_id in request.zones:
            if zone_id not in self.zones:
                return False
            
            zone = self.zones[zone_id]
            if zone.state != IntersectionState.FREE:
                return False
        
        return True
    
    def _has_conflicts(self, request: ReservationRequest) -> bool:
        """Check if request has conflicts with existing reservations."""
        for zone_id in request.zones:
            if zone_id in self.zones:
                zone = self.zones[zone_id]
                if zone.state == IntersectionState.RESERVED:
                    # Check time overlap
                    existing_reservation = self._get_reservation_for_zone(zone_id)
                    if existing_reservation:
                        if self._time_overlap(
                            request.entry_time,
                            request.exit_time,
                            existing_reservation.entry_time,
                            existing_reservation.exit_time
                        ):
                            return True
        
        return False
    
    def _get_reservation_for_zone(self, zone_id: str) -> Optional[ReservationRequest]:
        """Get reservation for a zone."""
        for reservation in self.reservations.values():
            if zone_id in reservation.zones and reservation.status == ReservationStatus.ACTIVE:
                return reservation
        return None
    
    def _time_overlap(
        self,
        start1: datetime,
        end1: datetime,
        start2: datetime,
        end2: datetime
    ) -> bool:
        """Check if two time intervals overlap."""
        return not (end1 < start2 or end2 < start1)
    
    def _resolve_conflicts(self) -> List[ReservationRequest]:
        """Resolve conflicts in reservation queue."""
        resolved = []
        
        # Sort by priority
        self.conflict_queue.sort(key=lambda r: r.priority, reverse=True)
        
        for request in self.conflict_queue[:]:
            if self._can_approve_request(request):
                request.status = ReservationStatus.APPROVED
                self.reservations[request.request_id] = request
                self.active_reservations.add(request.request_id)
                
                # Reserve zones
                for zone_id in request.zones:
                    if zone_id in self.zones:
                        zone = self.zones[zone_id]
                        zone.state = IntersectionState.RESERVED
                        zone.reserved_by = request.vehicle_id
                        zone.reservation_time = request.entry_time
                
                resolved.append(request)
                self.conflict_queue.remove(request)
        
        return resolved
    
    def request_reservation(
        self,
        vehicle_id: str,
        zones: List[str],
        entry_time: datetime,
        exit_time: datetime,
        priority: float = 0.5
    ) -> ReservationRequest:
        """
        Request intersection reservation.
        
        Args:
            vehicle_id: Vehicle identifier
            zones: List of zone IDs to reserve
            entry_time: Requested entry time
            exit_time: Requested exit time
            priority: Request priority
            
        Returns:
            Reservation request
        """
        request = ReservationRequest(
            request_id=f"req_{datetime.now().timestamp()}",
            vehicle_id=vehicle_id,
            zones=zones,
            entry_time=entry_time,
            exit_time=exit_time,
            priority=priority
        )
        
        self.pending_requests.append(request)
        return request
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute intersection management action.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        action_type = action.get("type", "process")
        
        if action_type == "approve_reservation":
            request_id = action.get("request_id")
            if request_id in self.reservations:
                reservation = self.reservations[request_id]
                reservation.status = ReservationStatus.ACTIVE
                self.active_reservations.add(request_id)
        
        result = {
            "action": action,
            "status": "executed",
            "timestamp": datetime.now().isoformat()
        }
        
        self.state.add_step("action", result)
        return result
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation (vehicle updates, time progression, etc.).
        
        Args:
            observation: Observation data
            
        Returns:
            Processed observation
        """
        self.state.status = AgentStatus.OBSERVING
        
        if isinstance(observation, dict):
            current_time = observation.get("current_time", datetime.now())
            
            # Update active reservations
            self._update_reservations(current_time)
            
            # Update zone states
            self._update_zone_states(current_time)
        
        processed = {
            "observation": observation,
            "active_reservations": len(self.active_reservations),
            "timestamp": datetime.now().isoformat()
        }
        
        self.state.add_step("observation", processed)
        return processed
    
    def _update_reservations(self, current_time: datetime):
        """Update reservation states based on current time."""
        completed = []
        
        for request_id in list(self.active_reservations):
            reservation = self.reservations.get(request_id)
            if reservation:
                if current_time >= reservation.exit_time:
                    reservation.status = ReservationStatus.COMPLETED
                    completed.append(request_id)
                    self.active_reservations.remove(request_id)
        
        return completed
    
    def _update_zone_states(self, current_time: datetime):
        """Update zone states based on reservations."""
        for zone in self.zones.values():
            if zone.state == IntersectionState.RESERVED:
                reservation = self._get_reservation_for_zone(zone.zone_id)
                if reservation:
                    if current_time >= reservation.entry_time:
                        zone.state = IntersectionState.OCCUPIED
                    elif current_time >= reservation.exit_time:
                        zone.state = IntersectionState.FREE
                        zone.reserved_by = None
                        zone.reservation_time = None
    
    def get_intersection_status(self) -> Dict[str, Any]:
        """Get current intersection status."""
        return {
            "intersection_id": self.intersection_id,
            "total_zones": len(self.zones),
            "free_zones": len([z for z in self.zones.values() if z.state == IntersectionState.FREE]),
            "reserved_zones": len([z for z in self.zones.values() if z.state == IntersectionState.RESERVED]),
            "occupied_zones": len([z for z in self.zones.values() if z.state == IntersectionState.OCCUPIED]),
            "active_reservations": len(self.active_reservations),
            "pending_requests": len(self.pending_requests),
            "conflicts": len(self.conflict_queue)
        }
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run intersection management task.
        
        Args:
            task: Task description
            context: Optional context
            
        Returns:
            Final result
        """
        from ..common.agent_utils import standard_run_pattern
        
        # Prepare observation with current time
        if context is None:
            context = {}
        
        if "observation" not in context:
            context["observation"] = {"current_time": datetime.now()}
        
        # Use standard pattern
        result = standard_run_pattern(self, task, context)
        
        # Add intersection-specific information
        result["intersection_status"] = self.get_intersection_status()
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "intersection_id": self.intersection_id,
            "active_reservations": len(self.active_reservations),
            "pending_requests": len(self.pending_requests)
        })



