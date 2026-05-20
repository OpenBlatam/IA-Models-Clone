"""
Meaningful Human Control: Actionable
===================================

Paper: "Meaningful human control: actionable..."

Key concepts:
- Human oversight and control
- Actionable control mechanisms
- Human-in-the-loop systems
- Control interfaces
- Meaningful human intervention
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus
from ..common.memory import EpisodicMemory, SemanticMemory


class ControlLevel(Enum):
    """Levels of human control."""
    FULL_AUTONOMY = "full_autonomy"
    SUPERVISED = "supervised"
    ASSISTED = "assisted"
    MANUAL = "manual"
    OVERRIDE = "override"


class InterventionType(Enum):
    """Types of human interventions."""
    APPROVAL = "approval"
    REJECTION = "rejection"
    MODIFICATION = "modification"
    PAUSE = "pause"
    RESUME = "resume"
    TERMINATE = "terminate"


@dataclass
class ControlRequest:
    """Request for human control."""
    request_id: str
    action: Dict[str, Any]
    reason: str
    urgency: str = "normal"  # low, normal, high, critical
    requested_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, approved, rejected, modified


@dataclass
class HumanIntervention:
    """Human intervention record."""
    intervention_id: str
    intervention_type: InterventionType
    action_id: str
    human_feedback: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    outcome: Optional[str] = None


class HumanControlAgent(BaseAgent):
    """
    Agent with meaningful human control mechanisms.
    
    Implements human oversight, actionable control interfaces,
    and meaningful intervention capabilities.
    """
    
    def __init__(
        self,
        name: str,
        control_level: ControlLevel = ControlLevel.SUPERVISED,
        require_approval: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize human control agent.
        
        Args:
            name: Agent name
            control_level: Level of human control required
            require_approval: Whether to require approval for actions
            config: Additional configuration
        """
        super().__init__(name, config)
        self.control_level = control_level
        self.require_approval = require_approval
        
        # Control management
        self.control_requests: List[ControlRequest] = []
        self.human_interventions: List[HumanIntervention] = []
        self.pending_approvals: List[str] = []
        
        # Metrics
        self.actions_requiring_approval = 0
        self.approvals_granted = 0
        self.approvals_denied = 0
        self.interventions_count = 0
        
        # Memory systems
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        
        # Control state
        self.control_state: Dict[str, Any] = {
            "human_available": True,
            "last_intervention": None,
            "autonomy_reduced": False
        }
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about task with human control considerations.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Thinking result with control analysis
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Assess if human control needed
        control_needed = self._assess_control_needs(task, context)
        
        # Determine control requirements
        control_requirements = self._determine_control_requirements(task, control_needed)
        
        result = {
            "task": task,
            "control_needed": control_needed,
            "control_requirements": control_requirements,
            "control_level": self.control_level.value,
            "require_approval": self.require_approval
        }
        
        self.state.add_step("think", result)
        return result
    
    def _assess_control_needs(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Assess if human control is needed."""
        if self.control_level == ControlLevel.FULL_AUTONOMY:
            return False
        
        if self.control_level == ControlLevel.MANUAL:
            return True
        
        # Check for high-risk keywords
        high_risk_keywords = ["delete", "modify", "execute", "transfer", "critical"]
        if any(keyword in task.lower() for keyword in high_risk_keywords):
            return True
        
        # Check context
        if context and context.get("high_risk", False):
            return True
        
        return self.require_approval
    
    def _determine_control_requirements(
        self,
        task: str,
        control_needed: bool
    ) -> Dict[str, Any]:
        """Determine specific control requirements."""
        requirements = {
            "approval_required": control_needed,
            "intervention_type": None,
            "urgency": "normal"
        }
        
        if control_needed:
            # Determine urgency
            if any(word in task.lower() for word in ["critical", "urgent", "emergency"]):
                requirements["urgency"] = "critical"
            elif any(word in task.lower() for word in ["important", "significant"]):
                requirements["urgency"] = "high"
            
            # Determine intervention type
            if self.control_level == ControlLevel.SUPERVISED:
                requirements["intervention_type"] = InterventionType.APPROVAL.value
            elif self.control_level == ControlLevel.ASSISTED:
                requirements["intervention_type"] = InterventionType.MODIFICATION.value
        
        return requirements
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action with human control checks.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        # Check if approval needed
        if self._requires_approval(action):
            control_request = self._request_control(action)
            self.actions_requiring_approval += 1
            
            return {
                "status": "pending_approval",
                "request_id": control_request.request_id,
                "action": action,
                "reason": control_request.reason
            }
        
        # Execute action
        result = self._execute_with_control(action)
        
        self.state.add_step("act", {
            "action": action,
            "result": result,
            "control_applied": True
        })
        
        return result
    
    def _requires_approval(self, action: Dict[str, Any]) -> bool:
        """Check if action requires approval."""
        if not self.require_approval:
            return False
        
        if self.control_level == ControlLevel.FULL_AUTONOMY:
            return False
        
        # Check action type
        action_type = action.get("type", "").lower()
        if action_type in ["delete", "modify", "execute", "transfer"]:
            return True
        
        return False
    
    def _request_control(self, action: Dict[str, Any]) -> ControlRequest:
        """Request human control for action."""
        request = ControlRequest(
            request_id=f"request_{datetime.now().timestamp()}",
            action=action,
            reason=f"Action '{action.get('type')}' requires human approval",
            urgency="high" if action.get("type") in ["delete", "execute"] else "normal"
        )
        
        self.control_requests.append(request)
        self.pending_approvals.append(request.request_id)
        
        return request
    
    def _execute_with_control(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action with control mechanisms."""
        return {
            "status": "executed",
            "action": action,
            "human_controlled": self.control_level != ControlLevel.FULL_AUTONOMY,
            "timestamp": datetime.now().isoformat()
        }
    
    def approve_action(self, request_id: str, approved: bool, modifications: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Approve or reject a pending action.
        
        Args:
            request_id: Control request ID
            approved: Whether action is approved
            modifications: Optional modifications to action
            
        Returns:
            Approval result
        """
        # Find request
        request = next((r for r in self.control_requests if r.request_id == request_id), None)
        if not request:
            return {"status": "error", "reason": "Request not found"}
        
        # Update request status
        if approved:
            request.status = "approved"
            self.approvals_granted += 1
            
            # Record intervention
            intervention = HumanIntervention(
                intervention_id=f"intervention_{datetime.now().timestamp()}",
                intervention_type=InterventionType.APPROVAL,
                action_id=request_id,
                outcome="approved"
            )
            self.human_interventions.append(intervention)
            self.interventions_count += 1
            
            # Execute action (with modifications if provided)
            action = request.action.copy()
            if modifications:
                action.update(modifications)
                request.status = "modified"
            
            return self._execute_with_control(action)
        else:
            request.status = "rejected"
            self.approvals_denied += 1
            
            # Record intervention
            intervention = HumanIntervention(
                intervention_id=f"intervention_{datetime.now().timestamp()}",
                intervention_type=InterventionType.REJECTION,
                action_id=request_id,
                outcome="rejected"
            )
            self.human_interventions.append(intervention)
            self.interventions_count += 1
            
            return {
                "status": "rejected",
                "request_id": request_id,
                "reason": "Human approval denied"
            }
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation and update control state.
        
        Args:
            observation: Observation data
            
        Returns:
            Processed observation
        """
        from ..common.agent_utils import standard_observe_pattern
        
        self.state.status = AgentStatus.OBSERVING
        
        # Check for human feedback
        if isinstance(observation, dict):
            if observation.get("human_feedback"):
                self._process_human_feedback(observation["human_feedback"])
        
        # Use standard observe pattern
        return standard_observe_pattern(
            self,
            observation,
            importance=0.7,
            additional_data={
                "control_level": self.control_level.value,
                "pending_approvals": len(self.pending_approvals),
                "interventions_count": self.interventions_count
            }
        )
    
    def _process_human_feedback(self, feedback: Dict[str, Any]):
        """Process human feedback."""
        # Update control state based on feedback
        if feedback.get("reduce_autonomy"):
            if self.control_level == ControlLevel.FULL_AUTONOMY:
                self.control_level = ControlLevel.SUPERVISED
                self.control_state["autonomy_reduced"] = True
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run task with human control.
        
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
        
        context["control_level"] = self.control_level.value
        context["require_approval"] = self.require_approval
        
        # Use standard pattern
        result = standard_run_pattern(self, task, context)
        
        # Add human control information
        result["human_control_summary"] = {
            "control_level": self.control_level.value,
            "actions_requiring_approval": self.actions_requiring_approval,
            "approvals_granted": self.approvals_granted,
            "approvals_denied": self.approvals_denied,
            "interventions_count": self.interventions_count,
            "pending_approvals": len(self.pending_approvals)
        }
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "control_level": self.control_level.value,
            "actions_requiring_approval": self.actions_requiring_approval,
            "approvals_granted": self.approvals_granted,
            "approvals_denied": self.approvals_denied,
            "interventions_count": self.interventions_count,
            "pending_approvals": len(self.pending_approvals)
        })
    
    def get_control_report(self) -> Dict[str, Any]:
        """Get human control report."""
        return {
            "total_requests": len(self.control_requests),
            "pending_requests": len([r for r in self.control_requests if r.status == "pending"]),
            "approved_requests": len([r for r in self.control_requests if r.status == "approved"]),
            "rejected_requests": len([r for r in self.control_requests if r.status == "rejected"]),
            "approval_rate": self.approvals_granted / self.actions_requiring_approval
            if self.actions_requiring_approval > 0 else 0.0,
            "interventions": len(self.human_interventions),
            "recent_interventions": [
                {
                    "type": i.intervention_type.value,
                    "outcome": i.outcome,
                    "timestamp": i.timestamp.isoformat()
                }
                for i in self.human_interventions[-10:]
            ],
            "control_level": self.control_level.value
        }


