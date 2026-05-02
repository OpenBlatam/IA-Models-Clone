"""
Multi-Agent Connected Autonomous Driving
=========================================

Paper: "Multi-Agent Connected Autonomous Driving using"

Key concepts:
- Multiple autonomous vehicles coordination
- Vehicle-to-vehicle (V2V) communication
- Traffic management and optimization
- Collision avoidance
- Route planning and coordination
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import math

from ..common.agent_base import BaseAgent, AgentState, AgentStatus


class VehicleStatus(Enum):
    """Vehicle status."""
    IDLE = "idle"
    DRIVING = "driving"
    STOPPED = "stopped"
    PARKING = "parking"
    EMERGENCY = "emergency"


class ActionType(Enum):
    """Vehicle action types."""
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STOP = "stop"
    CHANGE_LANE = "change_lane"
    MAINTAIN_SPEED = "maintain_speed"


@dataclass
class Position:
    """Vehicle position."""
    x: float
    y: float
    heading: float  # in radians
    speed: float  # in m/s


@dataclass
class VehicleState:
    """State of a vehicle."""
    vehicle_id: str
    position: Position
    status: VehicleStatus
    destination: Optional[Tuple[float, float]] = None
    route: List[Tuple[float, float]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrafficMessage:
    """Message between vehicles."""
    message_id: str
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrafficEvent:
    """Traffic event."""
    event_id: str
    event_type: str  # collision, congestion, accident, etc.
    location: Tuple[float, float]
    severity: float  # 0.0 to 1.0
    affected_vehicles: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class AutonomousVehicleAgent(BaseAgent):
    """
    Autonomous vehicle agent for multi-agent connected driving.
    
    Coordinates with other vehicles and manages driving decisions.
    """
    
    def __init__(
        self,
        name: str,
        vehicle_id: str,
        initial_position: Position,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize autonomous vehicle agent.
        
        Args:
            name: Agent name
            vehicle_id: Unique vehicle identifier
            initial_position: Initial vehicle position
            config: Additional configuration
        """
        super().__init__(name, config)
        self.vehicle_id = vehicle_id
        self.vehicle_state = VehicleState(
            vehicle_id=vehicle_id,
            position=initial_position,
            status=VehicleStatus.IDLE
        )
        
        # Communication
        self.received_messages: List[TrafficMessage] = []
        self.sent_messages: List[TrafficMessage] = []
        
        # Neighbor vehicles
        self.neighbor_vehicles: Dict[str, VehicleState] = {}
        
        # Traffic events
        self.known_events: List[TrafficEvent] = []
        
        # Action history
        self.action_history: List[Dict[str, Any]] = []
        
        # Safety parameters
        self.safe_distance = config.get("safe_distance", 5.0)  # meters
        self.max_speed = config.get("max_speed", 30.0)  # m/s
        self.reaction_time = config.get("reaction_time", 0.5)  # seconds
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about driving task.
        
        Args:
            task: Task description
            context: Additional context (other vehicles, traffic, etc.)
            
        Returns:
            Thinking result
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Update neighbor information
        if context and "neighbors" in context:
            self._update_neighbors(context["neighbors"])
        
        # Analyze traffic situation
        traffic_analysis = self._analyze_traffic()
        
        # Plan route if destination set
        route_plan = None
        if self.vehicle_state.destination:
            route_plan = self._plan_route()
        
        result = {
            "task": task,
            "traffic_analysis": traffic_analysis,
            "route_plan": route_plan,
            "neighbors_count": len(self.neighbor_vehicles),
            "position": {
                "x": self.vehicle_state.position.x,
                "y": self.vehicle_state.position.y,
                "heading": self.vehicle_state.position.heading,
                "speed": self.vehicle_state.position.speed
            }
        }
        
        self.state.add_step("thinking", result)
        return result
    
    def _update_neighbors(self, neighbors: List[VehicleState]):
        """Update information about neighbor vehicles."""
        for neighbor in neighbors:
            self.neighbor_vehicles[neighbor.vehicle_id] = neighbor
    
    def _analyze_traffic(self) -> Dict[str, Any]:
        """Analyze current traffic situation."""
        # Check for nearby vehicles
        nearby_vehicles = self._get_nearby_vehicles()
        
        # Check for traffic events
        nearby_events = [
            event for event in self.known_events
            if self._distance_to_point(event.location) < 100.0
        ]
        
        # Calculate collision risk
        collision_risk = self._calculate_collision_risk(nearby_vehicles)
        
        return {
            "nearby_vehicles": len(nearby_vehicles),
            "traffic_events": len(nearby_events),
            "collision_risk": collision_risk,
            "safe_to_proceed": collision_risk < 0.3
        }
    
    def _get_nearby_vehicles(self, radius: float = 50.0) -> List[VehicleState]:
        """Get vehicles within radius."""
        nearby = []
        my_pos = self.vehicle_state.position
        
        for vehicle in self.neighbor_vehicles.values():
            distance = math.sqrt(
                (vehicle.position.x - my_pos.x) ** 2 +
                (vehicle.position.y - my_pos.y) ** 2
            )
            if distance <= radius:
                nearby.append(vehicle)
        
        return nearby
    
    def _distance_to_point(self, point: Tuple[float, float]) -> float:
        """Calculate distance to a point."""
        my_pos = self.vehicle_state.position
        return math.sqrt(
            (point[0] - my_pos.x) ** 2 +
            (point[1] - my_pos.y) ** 2
        )
    
    def _calculate_collision_risk(self, nearby_vehicles: List[VehicleState]) -> float:
        """Calculate collision risk with nearby vehicles."""
        if not nearby_vehicles:
            return 0.0
        
        max_risk = 0.0
        my_pos = self.vehicle_state.position
        
        for vehicle in nearby_vehicles:
            distance = math.sqrt(
                (vehicle.position.x - my_pos.x) ** 2 +
                (vehicle.position.y - my_pos.y) ** 2
            )
            
            # Risk increases as distance decreases
            if distance < self.safe_distance:
                risk = 1.0 - (distance / self.safe_distance)
            else:
                risk = 0.0
            
            # Consider relative speeds
            speed_diff = abs(vehicle.position.speed - my_pos.speed)
            if speed_diff > 10.0:  # Large speed difference
                risk *= 1.2
            
            max_risk = max(max_risk, risk)
        
        return min(max_risk, 1.0)
    
    def _plan_route(self) -> Dict[str, Any]:
        """Plan route to destination."""
        if not self.vehicle_state.destination:
            return {}
        
        # Simple route planning (in production, use A* or similar)
        start = (self.vehicle_state.position.x, self.vehicle_state.position.y)
        end = self.vehicle_state.destination
        
        # Calculate direct route
        route = [start, end]
        distance = math.sqrt(
            (end[0] - start[0]) ** 2 +
            (end[1] - start[1]) ** 2
        )
        
        return {
            "route": route,
            "distance": distance,
            "estimated_time": distance / self.max_speed if self.max_speed > 0 else 0
        }
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute driving action.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        action_type_str = action.get("type", "maintain_speed")
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            action_type = ActionType.MAINTAIN_SPEED
        
        # Execute action
        result = self._execute_driving_action(action_type, action.get("parameters", {}))
        
        # Broadcast action to neighbors
        self._broadcast_action(action_type, result)
        
        action_record = {
            "action_type": action_type.value,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        self.action_history.append(action_record)
        self.state.add_step("action", action_record)
        
        return result
    
    def _execute_driving_action(
        self,
        action_type: ActionType,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a driving action."""
        pos = self.vehicle_state.position
        
        if action_type == ActionType.ACCELERATE:
            acceleration = parameters.get("acceleration", 2.0)
            new_speed = min(pos.speed + acceleration, self.max_speed)
            pos.speed = new_speed
        
        elif action_type == ActionType.DECELERATE:
            deceleration = parameters.get("deceleration", 2.0)
            new_speed = max(pos.speed - deceleration, 0.0)
            pos.speed = new_speed
        
        elif action_type == ActionType.TURN_LEFT:
            turn_angle = parameters.get("angle", math.pi / 6)  # 30 degrees
            pos.heading = (pos.heading - turn_angle) % (2 * math.pi)
        
        elif action_type == ActionType.TURN_RIGHT:
            turn_angle = parameters.get("angle", math.pi / 6)
            pos.heading = (pos.heading + turn_angle) % (2 * math.pi)
        
        elif action_type == ActionType.STOP:
            pos.speed = 0.0
            self.vehicle_state.status = VehicleStatus.STOPPED
        
        elif action_type == ActionType.CHANGE_LANE:
            lane_change = parameters.get("direction", "left")
            if lane_change == "left":
                pos.y += 3.5  # Standard lane width
            else:
                pos.y -= 3.5
        
        # Update position based on speed and heading
        dt = parameters.get("dt", 0.1)  # time step
        pos.x += pos.speed * math.cos(pos.heading) * dt
        pos.y += pos.speed * math.sin(pos.heading) * dt
        
        if pos.speed > 0:
            self.vehicle_state.status = VehicleStatus.DRIVING
        
        return {
            "status": "executed",
            "new_position": {
                "x": pos.x,
                "y": pos.y,
                "heading": pos.heading,
                "speed": pos.speed
            },
            "vehicle_status": self.vehicle_state.status.value
        }
    
    def _broadcast_action(self, action_type: ActionType, result: Dict[str, Any]):
        """Broadcast action to neighbor vehicles."""
        message = TrafficMessage(
            message_id=f"msg_{datetime.now().timestamp()}",
            sender_id=self.vehicle_id,
            receiver_id=None,  # Broadcast
            message_type="action_update",
            content={
                "action": action_type.value,
                "position": result.get("new_position"),
                "vehicle_id": self.vehicle_id
            }
        )
        
        self.sent_messages.append(message)
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation from environment.
        
        Args:
            observation: Observation data (traffic, messages, events, etc.)
            
        Returns:
            Processed observation
        """
        self.state.status = AgentStatus.OBSERVING
        
        if isinstance(observation, dict):
            # Process traffic messages
            if "messages" in observation:
                for msg_data in observation["messages"]:
                    self._process_message(msg_data)
            
            # Process traffic events
            if "events" in observation:
                for event_data in observation["events"]:
                    self._process_traffic_event(event_data)
            
            # Update vehicle state
            if "position" in observation:
                self._update_position(observation["position"])
        
        processed = {
            "observation": observation,
            "messages_received": len(self.received_messages),
            "known_events": len(self.known_events),
            "timestamp": datetime.now().isoformat()
        }
        
        self.state.add_step("observation", processed)
        return processed
    
    def _process_message(self, msg_data: Dict[str, Any]):
        """Process received traffic message."""
        message = TrafficMessage(
            message_id=msg_data.get("message_id", ""),
            sender_id=msg_data.get("sender_id", ""),
            receiver_id=msg_data.get("receiver_id"),
            message_type=msg_data.get("message_type", "unknown"),
            content=msg_data.get("content", {})
        )
        
        self.received_messages.append(message)
        
        # Update neighbor information from message
        if message.message_type == "action_update":
            vehicle_id = message.content.get("vehicle_id")
            if vehicle_id and vehicle_id != self.vehicle_id:
                position_data = message.content.get("position", {})
                neighbor_state = VehicleState(
                    vehicle_id=vehicle_id,
                    position=Position(
                        x=position_data.get("x", 0),
                        y=position_data.get("y", 0),
                        heading=position_data.get("heading", 0),
                        speed=position_data.get("speed", 0)
                    ),
                    status=VehicleStatus.DRIVING
                )
                self.neighbor_vehicles[vehicle_id] = neighbor_state
    
    def _process_traffic_event(self, event_data: Dict[str, Any]):
        """Process traffic event."""
        event = TrafficEvent(
            event_id=event_data.get("event_id", ""),
            event_type=event_data.get("event_type", "unknown"),
            location=tuple(event_data.get("location", (0, 0))),
            severity=event_data.get("severity", 0.5),
            affected_vehicles=event_data.get("affected_vehicles", [])
        )
        
        self.known_events.append(event)
    
    def _update_position(self, position_data: Dict[str, Any]):
        """Update vehicle position."""
        self.vehicle_state.position.x = position_data.get("x", self.vehicle_state.position.x)
        self.vehicle_state.position.y = position_data.get("y", self.vehicle_state.position.y)
        self.vehicle_state.position.heading = position_data.get("heading", self.vehicle_state.position.heading)
        self.vehicle_state.position.speed = position_data.get("speed", self.vehicle_state.position.speed)
    
    def set_destination(self, destination: Tuple[float, float]):
        """
        Set destination for vehicle.
        
        Args:
            destination: Destination coordinates (x, y)
        """
        self.vehicle_state.destination = destination
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run driving task.
        
        Args:
            task: Task description
            context: Optional context
            
        Returns:
            Final result
        """
        # Think about task
        thinking = self.think(task, context)
        
        # Execute driving actions
        if "drive" in task.lower() and self.vehicle_state.destination:
            # Drive towards destination
            actions = self._generate_driving_actions()
            results = []
            
            for action in actions:
                result = self.act(action)
                results.append(result)
                
                # Observe after each action
                self.observe({"position": result.get("new_position", {})})
        
        self.state.status = AgentStatus.COMPLETED
        
        return {
            "task": task,
            "thinking": thinking,
            "final_position": {
                "x": self.vehicle_state.position.x,
                "y": self.vehicle_state.position.y,
                "speed": self.vehicle_state.position.speed
            },
            "status": self.vehicle_state.status.value
        }
    
    def _generate_driving_actions(self) -> List[Dict[str, Any]]:
        """Generate driving actions to reach destination."""
        if not self.vehicle_state.destination:
            return []
        
        actions = []
        my_pos = self.vehicle_state.position
        dest = self.vehicle_state.destination
        
        # Calculate direction to destination
        dx = dest[0] - my_pos.x
        dy = dest[1] - my_pos.y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        
        if distance > 1.0:  # Not at destination
            # Accelerate if not at max speed
            if my_pos.speed < self.max_speed:
                actions.append({
                    "type": ActionType.ACCELERATE.value,
                    "parameters": {"acceleration": 2.0}
                })
            
            # Turn towards destination
            target_heading = math.atan2(dy, dx)
            heading_diff = target_heading - my_pos.heading
            
            # Normalize angle difference
            while heading_diff > math.pi:
                heading_diff -= 2 * math.pi
            while heading_diff < -math.pi:
                heading_diff += 2 * math.pi
            
            if abs(heading_diff) > 0.1:  # Need to turn
                if heading_diff > 0:
                    actions.append({
                        "type": ActionType.TURN_RIGHT.value,
                        "parameters": {"angle": min(heading_diff, math.pi / 6)}
                    })
                else:
                    actions.append({
                        "type": ActionType.TURN_LEFT.value,
                        "parameters": {"angle": min(-heading_diff, math.pi / 6)}
                    })
        
        return actions
    
    def get_status(self) -> Dict[str, Any]:
        """Get current vehicle status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "vehicle_id": self.vehicle_id,
            "vehicle_status": self.vehicle_state.status.value,
            "position": {
                "x": self.vehicle_state.position.x,
                "y": self.vehicle_state.position.y,
                "speed": self.vehicle_state.position.speed
            },
            "neighbors": len(self.neighbor_vehicles),
            "messages_sent": len(self.sent_messages),
            "messages_received": len(self.received_messages)
        })



