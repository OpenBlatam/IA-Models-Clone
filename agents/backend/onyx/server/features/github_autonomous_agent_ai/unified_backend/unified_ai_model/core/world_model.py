"""
Continual World Model
Implements concepts from EvoAgent paper: continual world model for long-horizon tasks
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class WorldState(str, Enum):
    """World state types"""
    STABLE = "stable"
    CHANGING = "changing"
    UNKNOWN = "unknown"


@dataclass
class WorldObservation:
    """Observation of the world state"""
    timestamp: datetime
    observation_type: str
    data: Dict[str, Any]
    confidence: float = 0.0


@dataclass
class WorldModelEntry:
    """Entry in the continual world model"""
    state_key: str
    state_value: Any
    last_observed: datetime
    observation_count: int = 0
    confidence: float = 0.0
    change_rate: float = 0.0
    observations: List[WorldObservation] = field(default_factory=list)


class ContinualWorldModel:
    """
    Continual World Model for Long-Horizon Tasks
    Implements EvoAgent concepts:
    - Continual world model that updates based on experiences
    - Self-planning based on world state
    - Self-control and self-reflection
    - Updates experiences without human intervention
    """
    
    def __init__(self):
        self._world_states: Dict[str, WorldModelEntry] = {}
        self._observation_history: List[WorldObservation] = []
        self._lock = asyncio.Lock()
        self._max_observations = 1000
        self._change_detection_threshold = 0.3
    
    async def observe(
        self,
        observation_type: str,
        data: Dict[str, Any],
        confidence: float = 0.5
    ) -> WorldObservation:
        """Record a world observation"""
        observation = WorldObservation(
            timestamp=datetime.utcnow(),
            observation_type=observation_type,
            data=data,
            confidence=confidence
        )
        
        async with self._lock:
            self._observation_history.append(observation)
            if len(self._observation_history) > self._max_observations:
                self._observation_history = self._observation_history[-self._max_observations:]
        
        # Update world model based on observation
        await self._update_world_model(observation)
        
        logger.debug(f"World observation recorded: {observation_type}")
        return observation
    
    async def _update_world_model(self, observation: WorldObservation) -> None:
        """Update world model based on new observation"""
        async with self._lock:
            # Extract state keys from observation
            for key, value in observation.data.items():
                if key in self._world_states:
                    entry = self._world_states[key]
                    # Detect change
                    if entry.state_value != value:
                        change_detected = True
                        entry.change_rate = self._calculate_change_rate(entry, value)
                    else:
                        change_detected = False
                    
                    # Update entry
                    entry.state_value = value
                    entry.last_observed = observation.timestamp
                    entry.observation_count += 1
                    entry.confidence = self._calculate_confidence(entry)
                    entry.observations.append(observation)
                    
                    # Keep only recent observations
                    if len(entry.observations) > 50:
                        entry.observations = entry.observations[-50:]
                    
                    if change_detected:
                        logger.info(f"World state changed: {key} (change_rate={entry.change_rate:.2f})")
                else:
                    # New state
                    self._world_states[key] = WorldModelEntry(
                        state_key=key,
                        state_value=value,
                        last_observed=observation.timestamp,
                        observation_count=1,
                        confidence=observation.confidence,
                        change_rate=0.0,
                        observations=[observation]
                    )
    
    def _calculate_change_rate(self, entry: WorldModelEntry, new_value: Any) -> float:
        """Calculate rate of change for a world state"""
        if entry.observation_count == 0:
            return 0.0
        
        # Simple change rate: frequency of changes
        recent_changes = sum(
            1 for obs in entry.observations[-10:]
            if obs.data.get(entry.state_key) != entry.state_value
        )
        return recent_changes / min(10, entry.observation_count)
    
    def _calculate_confidence(self, entry: WorldModelEntry) -> float:
        """Calculate confidence in world state"""
        if entry.observation_count == 0:
            return 0.0
        
        # Confidence increases with more observations
        base_confidence = min(entry.observation_count / 10.0, 1.0)
        
        # Reduce confidence if high change rate
        if entry.change_rate > self._change_detection_threshold:
            base_confidence *= 0.7
        
        return base_confidence
    
    async def get_world_state(self, state_key: str) -> Optional[WorldModelEntry]:
        """Get current world state for a key"""
        async with self._lock:
            return self._world_states.get(state_key)
    
    async def get_all_states(self) -> Dict[str, WorldModelEntry]:
        """Get all world states"""
        async with self._lock:
            return self._world_states.copy()
    
    async def detect_changes(self) -> List[Dict[str, Any]]:
        """Detect significant changes in world model"""
        changes = []
        
        async with self._lock:
            for key, entry in self._world_states.items():
                if entry.change_rate > self._change_detection_threshold:
                    changes.append({
                        "state_key": key,
                        "current_value": entry.state_value,
                        "change_rate": entry.change_rate,
                        "confidence": entry.confidence,
                        "last_observed": entry.last_observed.isoformat()
                    })
        
        return changes
    
    async def get_world_summary(self) -> Dict[str, Any]:
        """Get summary of world model state"""
        async with self._lock:
            total_states = len(self._world_states)
            changing_states = sum(
                1 for e in self._world_states.values()
                if e.change_rate > self._change_detection_threshold
            )
            stable_states = total_states - changing_states
            
            avg_confidence = sum(
                e.confidence for e in self._world_states.values()
            ) / total_states if total_states > 0 else 0.0
            
            return {
                "total_states": total_states,
                "stable_states": stable_states,
                "changing_states": changing_states,
                "average_confidence": avg_confidence,
                "total_observations": len(self._observation_history),
                "recent_observations": len([
                    o for o in self._observation_history
                    if (datetime.utcnow() - o.timestamp).total_seconds() < 3600
                ])
            }
    
    async def plan_based_on_world(self, goal: str) -> Dict[str, Any]:
        """Self-planning based on current world model state"""
        world_summary = await self.get_world_summary()
        changes = await self.detect_changes()
        
        plan = {
            "goal": goal,
            "world_state": world_summary,
            "detected_changes": changes,
            "planning_strategy": self._determine_planning_strategy(world_summary, changes),
            "recommended_actions": self._generate_recommended_actions(world_summary, changes)
        }
        
        logger.info(f"Generated plan for goal: {goal}")
        return plan
    
    def _determine_planning_strategy(
        self,
        world_summary: Dict[str, Any],
        changes: List[Dict[str, Any]]
    ) -> str:
        """Determine planning strategy based on world state"""
        if world_summary["changing_states"] > world_summary["stable_states"]:
            return "adaptive"  # World is changing, need adaptive planning
        elif world_summary["average_confidence"] < 0.5:
            return "exploratory"  # Low confidence, need exploration
        else:
            return "stable"  # Stable world, can use standard planning
    
    def _generate_recommended_actions(
        self,
        world_summary: Dict[str, Any],
        changes: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommended actions based on world state"""
        actions = []
        
        if changes:
            actions.append("monitor_changes")
            actions.append("adapt_strategy")
        
        if world_summary["average_confidence"] < 0.5:
            actions.append("gather_more_observations")
        
        if world_summary["changing_states"] > 0:
            actions.append("update_world_model")
        
        return actions
