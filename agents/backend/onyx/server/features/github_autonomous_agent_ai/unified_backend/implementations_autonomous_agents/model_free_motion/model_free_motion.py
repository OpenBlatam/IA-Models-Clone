"""
Model-free Motion Planning of Autonomous Vehicles
==================================================

Paper: "Model-free Motion Planning of Autonomous"

Key concepts:
- Motion planning without explicit environment model
- Learning-based path planning
- Real-time adaptation
- Obstacle avoidance
- Dynamic replanning
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import math

from ..common.agent_base import BaseAgent, AgentState, AgentStatus


class MotionType(Enum):
    """Types of motion."""
    STRAIGHT = "straight"
    CURVE = "curve"
    TURN = "turn"
    REVERSE = "reverse"
    STOP = "stop"


@dataclass
class MotionPlan:
    """A motion plan."""
    plan_id: str
    waypoints: List[Tuple[float, float]]
    motion_type: MotionType
    duration: float
    safety_score: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Obstacle:
    """An obstacle in the environment."""
    obstacle_id: str
    position: Tuple[float, float]
    radius: float
    velocity: Tuple[float, float]
    obstacle_type: str  # static, dynamic


class ModelFreeMotionPlanner(BaseAgent):
    """
    Model-free motion planner for autonomous vehicles.
    
    Plans motion without explicit environment model,
    using learning and real-time adaptation.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize model-free motion planner.
        
        Args:
            name: Agent name
            config: Configuration parameters
        """
        super().__init__(name, config)
        
        # Planning components
        self.current_plan: Optional[MotionPlan] = None
        self.plan_history: List[MotionPlan] = []
        
        # Environment knowledge (learned, not modeled)
        self.obstacles: List[Obstacle] = []
        self.learned_paths: List[List[Tuple[float, float]]] = []
        
        # Planning parameters
        self.lookahead_distance = config.get("lookahead_distance", 50.0)
        self.safety_margin = config.get("safety_margin", 5.0)
        self.replan_threshold = config.get("replan_threshold", 0.3)
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about motion planning task.
        
        Args:
            task: Task description
            context: Additional context (current position, goal, obstacles, etc.)
            
        Returns:
            Thinking result
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Extract planning information
        start_pos = None
        goal_pos = None
        if context:
            start_pos = tuple(context.get("start_position", (0, 0)))
            goal_pos = tuple(context.get("goal_position", (100, 100)))
            obstacles_data = context.get("obstacles", [])
            self._update_obstacles(obstacles_data)
        
        # Plan motion
        if start_pos and goal_pos:
            plan = self._plan_motion(start_pos, goal_pos)
            self.current_plan = plan
        else:
            plan = None
        
        result = {
            "task": task,
            "plan": plan.__dict__ if plan else None,
            "obstacles_detected": len(self.obstacles),
            "reasoning": f"Planning motion from {start_pos} to {goal_pos}"
        }
        
        self.state.add_step("thinking", result)
        return result
    
    def _update_obstacles(self, obstacles_data: List[Dict[str, Any]]):
        """Update obstacle information."""
        self.obstacles = []
        for obs_data in obstacles_data:
            obstacle = Obstacle(
                obstacle_id=obs_data.get("id", ""),
                position=tuple(obs_data.get("position", (0, 0))),
                radius=obs_data.get("radius", 2.0),
                velocity=tuple(obs_data.get("velocity", (0, 0))),
                obstacle_type=obs_data.get("type", "static")
            )
            self.obstacles.append(obstacle)
    
    def _plan_motion(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float]
    ) -> MotionPlan:
        """Plan motion from start to goal."""
        # Model-free planning: use learned heuristics and real-time obstacle avoidance
        
        # Generate waypoints
        waypoints = self._generate_waypoints(start, goal)
        
        # Determine motion type
        motion_type = self._classify_motion(waypoints)
        
        # Calculate duration
        duration = self._estimate_duration(waypoints)
        
        # Evaluate safety
        safety_score = self._evaluate_safety(waypoints)
        
        return MotionPlan(
            plan_id=f"plan_{datetime.now().timestamp()}",
            waypoints=waypoints,
            motion_type=motion_type,
            duration=duration,
            safety_score=safety_score
        )
    
    def _generate_waypoints(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        """Generate waypoints avoiding obstacles."""
        waypoints = [start]
        
        # Simple path: direct line with obstacle avoidance
        current = start
        distance = math.sqrt((goal[0] - start[0]) ** 2 + (goal[1] - start[1]) ** 2)
        num_waypoints = max(3, int(distance / self.lookahead_distance))
        
        for i in range(1, num_waypoints):
            t = i / num_waypoints
            waypoint = (
                start[0] + t * (goal[0] - start[0]),
                start[1] + t * (goal[1] - start[1])
            )
            
            # Adjust for obstacles
            waypoint = self._avoid_obstacles(waypoint, current)
            waypoints.append(waypoint)
            current = waypoint
        
        waypoints.append(goal)
        return waypoints
    
    def _avoid_obstacles(
        self,
        waypoint: Tuple[float, float],
        previous: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Adjust waypoint to avoid obstacles."""
        adjusted = waypoint
        
        for obstacle in self.obstacles:
            distance = math.sqrt(
                (waypoint[0] - obstacle.position[0]) ** 2 +
                (waypoint[1] - obstacle.position[1]) ** 2
            )
            
            if distance < obstacle.radius + self.safety_margin:
                # Push waypoint away from obstacle
                direction = (
                    (waypoint[0] - obstacle.position[0]) / max(distance, 0.1),
                    (waypoint[1] - obstacle.position[1]) / max(distance, 0.1)
                )
                push_distance = obstacle.radius + self.safety_margin - distance
                adjusted = (
                    waypoint[0] + direction[0] * push_distance,
                    waypoint[1] + direction[1] * push_distance
                )
        
        return adjusted
    
    def _classify_motion(self, waypoints: List[Tuple[float, float]]) -> MotionType:
        """Classify type of motion."""
        if len(waypoints) < 2:
            return MotionType.STOP
        
        # Calculate curvature
        total_angle_change = 0.0
        for i in range(1, len(waypoints) - 1):
            v1 = (
                waypoints[i][0] - waypoints[i-1][0],
                waypoints[i][1] - waypoints[i-1][1]
            )
            v2 = (
                waypoints[i+1][0] - waypoints[i][0],
                waypoints[i+1][1] - waypoints[i][1]
            )
            
            # Calculate angle between vectors
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
            
            if mag1 > 0 and mag2 > 0:
                angle = math.acos(max(-1, min(1, dot / (mag1 * mag2))))
                total_angle_change += angle
        
        if total_angle_change < 0.1:
            return MotionType.STRAIGHT
        elif total_angle_change < 1.0:
            return MotionType.CURVE
        else:
            return MotionType.TURN
    
    def _estimate_duration(self, waypoints: List[Tuple[float, float]]) -> float:
        """Estimate duration to traverse waypoints."""
        total_distance = 0.0
        for i in range(len(waypoints) - 1):
            distance = math.sqrt(
                (waypoints[i+1][0] - waypoints[i][0]) ** 2 +
                (waypoints[i+1][1] - waypoints[i][1]) ** 2
            )
            total_distance += distance
        
        # Assume average speed of 10 m/s
        return total_distance / 10.0
    
    def _evaluate_safety(self, waypoints: List[Tuple[float, float]]) -> float:
        """Evaluate safety of motion plan."""
        if not waypoints:
            return 0.0
        
        min_distance = float('inf')
        for waypoint in waypoints:
            for obstacle in self.obstacles:
                distance = math.sqrt(
                    (waypoint[0] - obstacle.position[0]) ** 2 +
                    (waypoint[1] - obstacle.position[1]) ** 2
                )
                min_distance = min(min_distance, distance)
        
        # Safety score: 1.0 if far from obstacles, 0.0 if too close
        if min_distance > self.safety_margin * 2:
            return 1.0
        elif min_distance > self.safety_margin:
            return 0.7
        else:
            return 0.3
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute motion action.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        result = {
            "action": action,
            "status": "executed",
            "current_plan": self.current_plan.plan_id if self.current_plan else None,
            "timestamp": datetime.now().isoformat()
        }
        
        self.state.add_step("action", result)
        return result
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation and adapt plan if needed.
        
        Args:
            observation: Observation data
            
        Returns:
            Processed observation
        """
        self.state.status = AgentStatus.OBSERVING
        
        # Check if replanning is needed
        if isinstance(observation, dict):
            new_obstacles = observation.get("obstacles", [])
            current_position = observation.get("position")
            
            if new_obstacles or current_position:
                # Update obstacles
                self._update_obstacles(new_obstacles)
                
                # Check if replanning needed
                if self._should_replan(current_position):
                    self._replan(current_position, observation.get("goal"))
        
        processed = {
            "observation": observation,
            "replanned": self._should_replan(None) if isinstance(observation, dict) else False,
            "timestamp": datetime.now().isoformat()
        }
        
        self.state.add_step("observation", processed)
        return processed
    
    def _should_replan(self, current_position: Optional[Tuple[float, float]]) -> bool:
        """Check if replanning is needed."""
        if not self.current_plan:
            return True
        
        # Check if current plan is still safe
        if self.current_plan.safety_score < self.replan_threshold:
            return True
        
        # Check if obstacles have moved into path
        for obstacle in self.obstacles:
            for waypoint in self.current_plan.waypoints:
                distance = math.sqrt(
                    (waypoint[0] - obstacle.position[0]) ** 2 +
                    (waypoint[1] - obstacle.position[1]) ** 2
                )
                if distance < obstacle.radius + self.safety_margin:
                    return True
        
        return False
    
    def _replan(
        self,
        current_position: Optional[Tuple[float, float]],
        goal: Optional[Tuple[float, float]]
    ):
        """Replan motion."""
        if current_position and goal:
            new_plan = self._plan_motion(current_position, goal)
            self.current_plan = new_plan
            self.plan_history.append(new_plan)
    
    def get_current_plan(self) -> Optional[MotionPlan]:
        """Get current motion plan."""
        return self.current_plan
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run motion planning task.
        
        Args:
            task: Task description
            context: Optional context
            
        Returns:
            Final result
        """
        from ..common.agent_utils import standard_run_pattern
        
        # Prepare context with default planning parameters
        if context is None:
            context = {}
        
        if "start_position" not in context:
            context["start_position"] = (0, 0)
        if "goal_position" not in context:
            context["goal_position"] = (100, 100)
        if "obstacles" not in context:
            context["obstacles"] = []
        
        # Use standard pattern
        result = standard_run_pattern(self, task, context)
        
        # Add plan information
        result["plan"] = self.current_plan.__dict__ if self.current_plan else None
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "current_plan": self.current_plan.plan_id if self.current_plan else None,
            "plans_generated": len(self.plan_history),
            "obstacles_tracked": len(self.obstacles)
        })



