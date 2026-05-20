"""
Augmenting the Action Space with Conventions
=============================================

Paper: "Augmenting the action space with conventions to..."

Key concepts:
- Action space augmentation
- Conventions for agent coordination
- Shared action protocols
- Convention-based communication
- Standardized action patterns
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus
from ..common.memory import EpisodicMemory, SemanticMemory


class ConventionType(Enum):
    """Types of conventions."""
    COMMUNICATION = "communication"
    COORDINATION = "coordination"
    PROTOCOL = "protocol"
    STANDARD = "standard"
    CUSTOM = "custom"


class ActionProtocol(Enum):
    """Action protocols."""
    REQUEST_RESPONSE = "request_response"
    BROADCAST = "broadcast"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONSENSUS = "consensus"


@dataclass
class Convention:
    """A convention for action coordination."""
    convention_id: str
    convention_type: ConventionType
    name: str
    description: str
    protocol: ActionProtocol
    rules: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ActionConvention:
    """Action augmented with convention."""
    action_id: str
    base_action: Dict[str, Any]
    convention_applied: Convention
    augmented_action: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class ActionConventionsAgent(BaseAgent):
    """
    Agent that uses conventions to augment action space.
    
    Implements shared protocols and conventions for
    coordinated multi-agent actions.
    """
    
    def __init__(
        self,
        name: str,
        conventions: Optional[List[Convention]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize action conventions agent.
        
        Args:
            name: Agent name
            conventions: List of conventions to follow
            config: Additional configuration
        """
        super().__init__(name, config)
        
        # Convention management
        self.conventions: List[Convention] = conventions or []
        self.action_conventions: List[ActionConvention] = []
        
        # Metrics
        self.conventions_applied = 0
        self.coordinated_actions = 0
        self.protocol_followed = 0
        
        # Memory systems
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        
        # Protocol state
        self.protocol_state: Dict[str, Any] = {}
        self.coordination_history: List[Dict[str, Any]] = []
        
        # Initialize default conventions
        self._initialize_default_conventions()
    
    def _initialize_default_conventions(self):
        """Initialize default conventions."""
        if not self.conventions:
            self.conventions = [
                Convention(
                    convention_id="default_communication",
                    convention_type=ConventionType.COMMUNICATION,
                    name="Standard Communication",
                    description="Standard communication protocol",
                    protocol=ActionProtocol.REQUEST_RESPONSE,
                    rules={"timeout": 30, "retry": 3}
                ),
                Convention(
                    convention_id="default_coordination",
                    convention_type=ConventionType.COORDINATION,
                    name="Basic Coordination",
                    description="Basic coordination protocol",
                    protocol=ActionProtocol.SEQUENTIAL,
                    rules={"wait_for_ack": True}
                )
            ]
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about task considering conventions.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Thinking result with convention analysis
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Determine applicable conventions
        applicable_conventions = self._find_applicable_conventions(task, context)
        
        # Plan action with conventions
        convention_plan = self._plan_with_conventions(task, applicable_conventions)
        
        result = {
            "task": task,
            "applicable_conventions": [
                {
                    "id": c.convention_id,
                    "type": c.convention_type.value,
                    "name": c.name
                }
                for c in applicable_conventions
            ],
            "convention_plan": convention_plan,
            "protocol": applicable_conventions[0].protocol.value if applicable_conventions else None
        }
        
        self.state.add_step("think", result)
        return result
    
    def _find_applicable_conventions(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> List[Convention]:
        """Find conventions applicable to task."""
        applicable = []
        
        for convention in self.conventions:
            if not convention.active:
                continue
            
            # Check if convention applies
            if convention.convention_type == ConventionType.COMMUNICATION:
                if "communicate" in task.lower() or "message" in task.lower():
                    applicable.append(convention)
            elif convention.convention_type == ConventionType.COORDINATION:
                if "coordinate" in task.lower() or "collaborate" in task.lower():
                    applicable.append(convention)
            else:
                # Default: apply if no specific match
                applicable.append(convention)
        
        return applicable[:1] if applicable else []  # Return first applicable
    
    def _plan_with_conventions(
        self,
        task: str,
        conventions: List[Convention]
    ) -> Dict[str, Any]:
        """Plan action using conventions."""
        if not conventions:
            return {"conventions_used": False}
        
        convention = conventions[0]
        
        plan = {
            "conventions_used": True,
            "convention_id": convention.convention_id,
            "protocol": convention.protocol.value,
            "steps": []
        }
        
        if convention.protocol == ActionProtocol.REQUEST_RESPONSE:
            plan["steps"] = ["Send request", "Wait for response", "Process response"]
        elif convention.protocol == ActionProtocol.SEQUENTIAL:
            plan["steps"] = ["Execute step 1", "Wait for completion", "Execute step 2"]
        elif convention.protocol == ActionProtocol.BROADCAST:
            plan["steps"] = ["Broadcast action", "Collect responses", "Synthesize"]
        
        return plan
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action with convention augmentation.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        # Find applicable convention
        applicable_convention = self._find_applicable_conventions(
            str(action),
            None
        )
        
        if applicable_convention:
            convention = applicable_convention[0]
            augmented_action = self._augment_with_convention(action, convention)
            self.conventions_applied += 1
            self.protocol_followed += 1
        else:
            augmented_action = action
            convention = None
        
        # Execute augmented action
        result = self._execute_augmented_action(augmented_action)
        
        # Track convention usage
        if convention:
            action_convention = ActionConvention(
                action_id=f"action_{datetime.now().timestamp()}",
                base_action=action,
                convention_applied=convention,
                augmented_action=augmented_action
            )
            self.action_conventions.append(action_convention)
        
        self.state.add_step("act", {
            "original_action": action,
            "augmented_action": augmented_action,
            "convention_applied": convention.convention_id if convention else None
        })
        
        return result
    
    def _augment_with_convention(
        self,
        action: Dict[str, Any],
        convention: Convention
    ) -> Dict[str, Any]:
        """Augment action with convention."""
        augmented = action.copy()
        
        # Add convention metadata
        augmented["convention_id"] = convention.convention_id
        augmented["protocol"] = convention.protocol.value
        
        # Apply convention rules
        if convention.rules:
            augmented["convention_rules"] = convention.rules
        
        # Protocol-specific augmentation
        if convention.protocol == ActionProtocol.REQUEST_RESPONSE:
            augmented["requires_response"] = True
            augmented["timeout"] = convention.rules.get("timeout", 30)
        elif convention.protocol == ActionProtocol.SEQUENTIAL:
            augmented["sequential"] = True
            augmented["wait_for_ack"] = convention.rules.get("wait_for_ack", True)
        elif convention.protocol == ActionProtocol.BROADCAST:
            augmented["broadcast"] = True
            augmented["collect_responses"] = True
        
        return augmented
    
    def _execute_augmented_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute augmented action."""
        return {
            "status": "executed",
            "action": action,
            "convention_applied": action.get("convention_id"),
            "timestamp": datetime.now().isoformat()
        }
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation and update convention state.
        
        Args:
            observation: Observation data
            
        Returns:
            Processed observation
        """
        from ..common.agent_utils import standard_observe_pattern
        
        self.state.status = AgentStatus.OBSERVING
        
        # Check for coordination signals
        if isinstance(observation, dict):
            if observation.get("coordination_signal"):
                self.coordinated_actions += 1
                self.coordination_history.append({
                    "signal": observation["coordination_signal"],
                    "timestamp": datetime.now().isoformat()
                })
        
        # Use standard observe pattern
        return standard_observe_pattern(
            self,
            observation,
            importance=0.7,
            additional_data={
                "conventions_applied": self.conventions_applied,
                "coordinated_actions": self.coordinated_actions,
                "active_conventions": len([c for c in self.conventions if c.active])
            }
        )
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run task with convention-based actions.
        
        Args:
            task: Task description
            context: Optional context
            
        Returns:
            Final result
        """
        from ..common.agent_utils import standard_run_pattern
        
        # Prepare context
        if context is None:
            context = {}
        
        context["conventions_enabled"] = True
        context["conventions_count"] = len(self.conventions)
        
        # Use standard pattern
        result = standard_run_pattern(self, task, context)
        
        # Add convention information
        result["conventions_summary"] = {
            "conventions_applied": self.conventions_applied,
            "coordinated_actions": self.coordinated_actions,
            "protocol_followed": self.protocol_followed,
            "active_conventions": len([c for c in self.conventions if c.active]),
            "convention_types": {
                ctype.value: len([c for c in self.conventions if c.convention_type == ctype])
                for ctype in ConventionType
            }
        }
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "conventions_applied": self.conventions_applied,
            "coordinated_actions": self.coordinated_actions,
            "protocol_followed": self.protocol_followed,
            "active_conventions": len([c for c in self.conventions if c.active]),
            "total_conventions": len(self.conventions)
        })
    
    def add_convention(self, convention: Convention):
        """Add a new convention."""
        self.conventions.append(convention)
    
    def get_convention_report(self) -> Dict[str, Any]:
        """Get convention usage report."""
        return {
            "total_conventions": len(self.conventions),
            "active_conventions": len([c for c in self.conventions if c.active]),
            "conventions_applied": self.conventions_applied,
            "coordinated_actions": self.coordinated_actions,
            "protocol_usage": {
                protocol.value: len([
                    ac for ac in self.action_conventions
                    if ac.convention_applied.protocol == protocol
                ])
                for protocol in ActionProtocol
            },
            "recent_conventions": [
                {
                    "convention_id": ac.convention_applied.convention_id,
                    "protocol": ac.convention_applied.protocol.value,
                    "timestamp": ac.timestamp.isoformat()
                }
                for ac in self.action_conventions[-10:]
            ]
        }


