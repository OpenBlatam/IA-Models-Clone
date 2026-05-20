"""
Generative AI Agents in Autonomous Machines
============================================

Paper: "Generative AI Agents in Autonomous Machines"

Key concepts:
- Generative AI agents in autonomous machines
- Machine-environment interaction
- Generative capabilities for decision-making
- Autonomous operation
- Real-world deployment
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus


class MachineType(Enum):
    """Types of autonomous machines."""
    ROBOT = "robot"
    DRONE = "drone"
    VEHICLE = "vehicle"
    MANIPULATOR = "manipulator"
    MOBILE_BASE = "mobile_base"


class OperationMode(Enum):
    """Operation modes."""
    AUTONOMOUS = "autonomous"
    SEMI_AUTONOMOUS = "semi_autonomous"
    TELEOPERATED = "teleoperated"
    LEARNING = "learning"


@dataclass
class MachineState:
    """State of an autonomous machine."""
    machine_id: str
    machine_type: MachineType
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # quaternion
    velocity: Tuple[float, float, float]
    battery_level: float
    status: str
    sensors: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ActionPlan:
    """Action plan for machine."""
    plan_id: str
    actions: List[Dict[str, Any]]
    estimated_duration: float
    safety_score: float
    priority: int
    timestamp: datetime = field(default_factory=datetime.now)


class GenerativeAIMachineAgent(BaseAgent):
    """
    Generative AI agent for autonomous machines.
    
    Uses generative capabilities for decision-making and operation.
    """
    
    def __init__(
        self,
        name: str,
        machine_type: MachineType,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize generative AI machine agent.
        
        Args:
            name: Agent name
            machine_type: Type of machine
            config: Configuration parameters
        """
        super().__init__(name, config)
        self.machine_type = machine_type
        self.machine_state: Optional[MachineState] = None
        self.operation_mode = OperationMode.AUTONOMOUS
        
        # Generative components
        self.action_generator = None  # Placeholder for generative model
        self.decision_history: List[Dict[str, Any]] = []
        self.action_plans: List[ActionPlan] = []
        
        # Machine-specific parameters
        self.max_speed = config.get("max_speed", 1.0)
        self.safety_margin = config.get("safety_margin", 0.5)
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about task using generative capabilities.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Thinking result with generated plan
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Update machine state from context
        if context and "machine_state" in context:
            self.machine_state = self._parse_machine_state(context["machine_state"])
        
        # Generate action plan
        plan = self._generate_action_plan(task, context)
        self.action_plans.append(plan)
        
        result = {
            "task": task,
            "machine_type": self.machine_type.value,
            "operation_mode": self.operation_mode.value,
            "generated_plan": {
                "plan_id": plan.plan_id,
                "actions": plan.actions,
                "estimated_duration": plan.estimated_duration,
                "safety_score": plan.safety_score
            },
            "reasoning": f"Generated action plan for {task} using generative AI"
        }
        
        self.state.add_step("thinking", result)
        return result
    
    def _parse_machine_state(self, state_data: Dict[str, Any]) -> MachineState:
        """Parse machine state from dictionary."""
        return MachineState(
            machine_id=state_data.get("machine_id", self.name),
            machine_type=MachineType(state_data.get("machine_type", self.machine_type.value)),
            position=tuple(state_data.get("position", (0, 0, 0))),
            orientation=tuple(state_data.get("orientation", (0, 0, 0, 1))),
            velocity=tuple(state_data.get("velocity", (0, 0, 0))),
            battery_level=state_data.get("battery_level", 100.0),
            status=state_data.get("status", "idle"),
            sensors=state_data.get("sensors", {})
        )
    
    def _generate_action_plan(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> ActionPlan:
        """Generate action plan using generative AI."""
        # Simplified plan generation
        # In production, this would use a generative model
        
        actions = []
        
        # Parse task and generate actions
        if "move" in task.lower():
            actions.append({
                "type": "move",
                "target": context.get("target", (1, 0, 0)) if context else (1, 0, 0),
                "speed": min(self.max_speed, 0.5)
            })
        elif "grasp" in task.lower() or "pick" in task.lower():
            actions.append({
                "type": "grasp",
                "object": context.get("object", "object") if context else "object",
                "force": 0.5
            })
        elif "navigate" in task.lower():
            actions.append({
                "type": "navigate",
                "destination": context.get("destination", (5, 5, 0)) if context else (5, 5, 0),
                "avoid_obstacles": True
            })
        else:
            # Default action
            actions.append({
                "type": "execute",
                "command": task
            })
        
        # Estimate duration
        estimated_duration = len(actions) * 2.0  # 2 seconds per action
        
        # Calculate safety score
        safety_score = self._calculate_safety_score(actions)
        
        return ActionPlan(
            plan_id=f"plan_{datetime.now().timestamp()}",
            actions=actions,
            estimated_duration=estimated_duration,
            safety_score=safety_score,
            priority=1
        )
    
    def _calculate_safety_score(self, actions: List[Dict[str, Any]]) -> float:
        """Calculate safety score for actions."""
        # Simplified safety calculation
        base_score = 1.0
        
        for action in actions:
            action_type = action.get("type", "")
            
            # Reduce score for risky actions
            if action_type in ["move", "navigate"]:
                speed = action.get("speed", 0.0)
                if speed > self.max_speed * 0.8:
                    base_score -= 0.1
            
            if action_type == "grasp":
                force = action.get("force", 0.0)
                if force > 0.7:
                    base_score -= 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action on machine.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        action_type = action.get("type", "execute")
        
        # Execute action (simplified)
        result = {
            "action": action,
            "action_type": action_type,
            "status": "executed",
            "machine_state": self.machine_state.__dict__ if self.machine_state else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Record decision
        self.decision_history.append({
            "action": action,
            "timestamp": datetime.now(),
            "result": result
        })
        
        self.state.add_step("action", result)
        return result
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation from machine sensors.
        
        Args:
            observation: Observation data
            
        Returns:
            Processed observation
        """
        self.state.status = AgentStatus.OBSERVING
        
        if isinstance(observation, dict):
            # Update machine state
            if "machine_state" in observation:
                self.machine_state = self._parse_machine_state(observation["machine_state"])
            
            # Process sensor data
            sensor_data = observation.get("sensors", {})
            
            processed = {
                "observation": observation,
                "machine_state": self.machine_state.__dict__ if self.machine_state else None,
                "sensor_data": sensor_data,
                "battery_level": self.machine_state.battery_level if self.machine_state else 100.0,
                "timestamp": datetime.now().isoformat()
            }
        else:
            processed = {
                "observation": observation,
                "timestamp": datetime.now().isoformat()
            }
        
        self.state.add_step("observation", processed)
        return processed
    
    def set_operation_mode(self, mode: OperationMode):
        """Set operation mode."""
        self.operation_mode = mode
    
    def get_current_plan(self) -> Optional[ActionPlan]:
        """Get current action plan."""
        return self.action_plans[-1] if self.action_plans else None
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run task on machine.
        
        Args:
            task: Task description
            context: Optional context
            
        Returns:
            Final result
        """
        from ..common.agent_utils import standard_run_pattern
        
        # Prepare machine context
        if context is None:
            context = {}
        
        # Prepare observation with machine state
        if "observation" not in context:
            context["observation"] = {
                "machine_state": self.machine_state.__dict__ if self.machine_state else {},
                "sensors": {}
            }
        
        # Use standard pattern but customize action
        thinking = self.think(task, context)
        
        # Execute first action from plan
        plan = self.get_current_plan()
        if plan and plan.actions:
            action = plan.actions[0]
        else:
            action = {"type": "execute", "command": task}
        
        action_result = self.act(action)
        observation_result = self.observe(context.get("observation", {}))
        
        self.state.status = AgentStatus.COMPLETED
        
        return {
            "task": task,
            "thinking": thinking,
            "action": action_result,
            "observation": observation_result
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "machine_type": self.machine_type.value,
            "operation_mode": self.operation_mode.value,
            "battery_level": self.machine_state.battery_level if self.machine_state else 100.0,
            "plans_generated": len(self.action_plans),
            "decisions_made": len(self.decision_history)
        })



