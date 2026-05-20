"""
Finding Coordinated Paths for Multiple Holonomic Agents
========================================================

Paper: "Finding Coordinated Paths for Multiple Holonomic Agents"

Key concepts:
- Path planning for multiple agents
- Collision avoidance
- Coordination algorithms
- Holonomic agent movement
- Temporal-spatial coordination
"""

from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import math
from collections import defaultdict

from ..common.agent_base import BaseAgent, AgentState, AgentStatus


class PathStatus(Enum):
    """Path status."""
    PLANNED = "planned"
    EXECUTING = "executing"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass
class Point:
    """2D point."""
    x: float
    y: float
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate distance to another point."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class PathSegment:
    """A segment of a path."""
    start: Point
    end: Point
    duration: float  # Time to traverse
    timestamp: float  # When this segment starts


@dataclass
class AgentPath:
    """Path for a single agent."""
    agent_id: str
    segments: List[PathSegment]
    status: PathStatus = PathStatus.PLANNED
    current_segment: int = 0
    start_time: Optional[float] = None


@dataclass
class Conflict:
    """Conflict between agent paths."""
    agent1_id: str
    agent2_id: str
    location: Point
    time: float
    conflict_type: str  # collision, crossing, etc.


class CoordinatedPathPlanner:
    """
    Planner for finding coordinated paths for multiple holonomic agents.
    
    Ensures collision-free paths through coordination.
    """
    
    def __init__(
        self,
        agents: List[str],
        obstacles: Optional[List[Point]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize path planner.
        
        Args:
            agents: List of agent IDs
            obstacles: List of obstacle positions
            config: Configuration parameters
        """
        self.agents = agents
        self.obstacles = obstacles or []
        self.config = config or {}
        
        # Agent positions
        self.agent_positions: Dict[str, Point] = {}
        
        # Agent goals
        self.agent_goals: Dict[str, Point] = {}
        
        # Planned paths
        self.paths: Dict[str, AgentPath] = {}
        
        # Conflicts detected
        self.conflicts: List[Conflict] = []
        
        # Coordination parameters
        self.safety_radius = config.get("safety_radius", 1.0)
        self.max_speed = config.get("max_speed", 5.0)
        self.time_horizon = config.get("time_horizon", 10.0)
    
    def plan_paths(
        self,
        start_positions: Dict[str, Point],
        goal_positions: Dict[str, Point]
    ) -> Dict[str, AgentPath]:
        """
        Plan coordinated paths for all agents.
        
        Args:
            start_positions: Starting positions for each agent
            goal_positions: Goal positions for each agent
            
        Returns:
            Dictionary of planned paths
        """
        self.agent_positions = start_positions
        self.agent_goals = goal_positions
        
        # Plan initial paths (simple straight-line for now)
        paths = {}
        for agent_id in self.agents:
            if agent_id in start_positions and agent_id in goal_positions:
                path = self._plan_single_path(
                    agent_id,
                    start_positions[agent_id],
                    goal_positions[agent_id]
                )
                paths[agent_id] = path
        
        self.paths = paths
        
        # Detect and resolve conflicts
        self._detect_conflicts()
        self._resolve_conflicts()
        
        return self.paths
    
    def _plan_single_path(
        self,
        agent_id: str,
        start: Point,
        goal: Point
    ) -> AgentPath:
        """Plan path for a single agent."""
        # Simple straight-line path (in production, use A* or RRT)
        distance = start.distance_to(goal)
        duration = distance / self.max_speed
        
        segment = PathSegment(
            start=start,
            end=goal,
            duration=duration,
            timestamp=0.0
        )
        
        return AgentPath(
            agent_id=agent_id,
            segments=[segment],
            status=PathStatus.PLANNED
        )
    
    def _detect_conflicts(self):
        """Detect conflicts between agent paths."""
        self.conflicts = []
        
        for i, agent1_id in enumerate(self.agents):
            for agent2_id in self.agents[i+1:]:
                if agent1_id in self.paths and agent2_id in self.paths:
                    conflicts = self._check_path_conflicts(
                        self.paths[agent1_id],
                        self.paths[agent2_id]
                    )
                    self.conflicts.extend(conflicts)
    
    def _check_path_conflicts(
        self,
        path1: AgentPath,
        path2: AgentPath
    ) -> List[Conflict]:
        """Check for conflicts between two paths."""
        conflicts = []
        
        # Check each segment pair
        for seg1 in path1.segments:
            for seg2 in path2.segments:
                # Check if segments overlap in time
                time_overlap = self._check_time_overlap(seg1, seg2)
                
                if time_overlap:
                    # Check if paths are too close
                    conflict_point = self._find_conflict_point(seg1, seg2)
                    
                    if conflict_point:
                        min_distance = self._min_distance_between_segments(seg1, seg2)
                        
                        if min_distance < 2 * self.safety_radius:
                            conflict = Conflict(
                                agent1_id=path1.agent_id,
                                agent2_id=path2.agent_id,
                                location=conflict_point,
                                time=(seg1.timestamp + seg2.timestamp) / 2,
                                conflict_type="collision"
                            )
                            conflicts.append(conflict)
        
        return conflicts
    
    def _check_time_overlap(
        self,
        seg1: PathSegment,
        seg2: PathSegment
    ) -> bool:
        """Check if two segments overlap in time."""
        seg1_end = seg1.timestamp + seg1.duration
        seg2_end = seg2.timestamp + seg2.duration
        
        return not (seg1_end < seg2.timestamp or seg2_end < seg1.timestamp)
    
    def _find_conflict_point(
        self,
        seg1: PathSegment,
        seg2: PathSegment
    ) -> Optional[Point]:
        """Find potential conflict point between segments."""
        # Simplified: check if segments are close
        mid1 = Point(
            x=(seg1.start.x + seg1.end.x) / 2,
            y=(seg1.start.y + seg1.end.y) / 2
        )
        mid2 = Point(
            x=(seg2.start.x + seg2.end.x) / 2,
            y=(seg2.start.y + seg2.end.y) / 2
        )
        
        if mid1.distance_to(mid2) < 2 * self.safety_radius:
            return mid1
        
        return None
    
    def _min_distance_between_segments(
        self,
        seg1: PathSegment,
        seg2: PathSegment
    ) -> float:
        """Calculate minimum distance between two path segments."""
        # Simplified: distance between midpoints
        mid1 = Point(
            x=(seg1.start.x + seg1.end.x) / 2,
            y=(seg1.start.y + seg1.end.y) / 2
        )
        mid2 = Point(
            x=(seg2.start.x + seg2.end.x) / 2,
            y=(seg2.start.y + seg2.end.y) / 2
        )
        
        return mid1.distance_to(mid2)
    
    def _resolve_conflicts(self):
        """Resolve detected conflicts through coordination."""
        if not self.conflicts:
            return
        
        # Group conflicts by agent
        agent_conflicts: Dict[str, List[Conflict]] = defaultdict(list)
        for conflict in self.conflicts:
            agent_conflicts[conflict.agent1_id].append(conflict)
            agent_conflicts[conflict.agent2_id].append(conflict)
        
        # Resolve conflicts using priority-based approach
        # Agent with more conflicts gets priority
        priorities = sorted(
            agent_conflicts.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for agent_id, conflicts in priorities:
            if agent_id in self.paths:
                # Modify path to avoid conflicts
                self._modify_path_for_conflicts(agent_id, conflicts)
    
    def _modify_path_for_conflicts(
        self,
        agent_id: str,
        conflicts: List[Conflict]
    ):
        """Modify agent path to avoid conflicts."""
        if agent_id not in self.paths:
            return
        
        path = self.paths[agent_id]
        
        # Simple resolution: add wait time before conflicting segments
        for conflict in conflicts:
            if conflict.agent1_id == agent_id:
                # Delay this agent's path
                for segment in path.segments:
                    if segment.timestamp <= conflict.time:
                        segment.timestamp += 0.5  # Add delay
            elif conflict.agent2_id == agent_id:
                # Delay this agent's path
                for segment in path.segments:
                    if segment.timestamp <= conflict.time:
                        segment.timestamp += 0.5  # Add delay
    
    def execute_paths(self, current_time: float = 0.0) -> Dict[str, Point]:
        """
        Execute planned paths and return current positions.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Current positions of all agents
        """
        positions = {}
        
        for agent_id, path in self.paths.items():
            if path.status == PathStatus.EXECUTING or path.status == PathStatus.PLANNED:
                position = self._get_agent_position(path, current_time)
                positions[agent_id] = position
                
                # Update path status
                if path.current_segment >= len(path.segments):
                    path.status = PathStatus.COMPLETED
        
        return positions
    
    def _get_agent_position(
        self,
        path: AgentPath,
        current_time: float
    ) -> Point:
        """Get agent position at given time."""
        if not path.segments:
            return Point(0, 0)
        
        # Find current segment
        elapsed = current_time - (path.start_time or 0)
        
        for i, segment in enumerate(path.segments):
            if elapsed < segment.duration:
                # Interpolate position along segment
                t = elapsed / segment.duration if segment.duration > 0 else 0
                x = segment.start.x + t * (segment.end.x - segment.start.x)
                y = segment.start.y + t * (segment.end.y - segment.start.y)
                return Point(x, y)
            elapsed -= segment.duration
        
        # Past all segments, return end of last segment
        last_segment = path.segments[-1]
        return last_segment.end
    
    def get_conflicts(self) -> List[Conflict]:
        """Get detected conflicts."""
        return self.conflicts
    
    def is_coordination_successful(self) -> bool:
        """Check if coordination was successful (no remaining conflicts)."""
        return len(self.conflicts) == 0



