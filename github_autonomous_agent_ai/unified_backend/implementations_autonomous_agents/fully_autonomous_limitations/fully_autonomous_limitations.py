"""
Fully Autonomous AI Agents Should Not be Developed
==================================================

Paper: "Fully Autonomous AI Agents Should Not be Developed"

Key concepts:
- Safety concerns with fully autonomous agents
- Need for human oversight and control
- Risk assessment and mitigation
- Ethical boundaries for autonomy
- Guardrails and limitations
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus
from ..common.memory import EpisodicMemory, SemanticMemory


class RiskLevel(Enum):
    """Risk levels for agent actions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AutonomyConstraint(Enum):
    """Types of autonomy constraints."""
    HUMAN_APPROVAL_REQUIRED = "human_approval_required"
    SUPERVISED_EXECUTION = "supervised_execution"
    LIMITED_SCOPE = "limited_scope"
    TIME_BOUNDED = "time_bounded"
    RESOURCE_LIMITED = "resource_limited"


@dataclass
class SafetyCheck:
    """Safety check result."""
    check_id: str
    check_type: str
    passed: bool
    risk_level: RiskLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ActionLimitation:
    """Limitation on agent action."""
    limitation_id: str
    constraint_type: AutonomyConstraint
    description: str
    applies_to: List[str] = field(default_factory=list)
    active: bool = True


class FullyAutonomousLimitationsAgent(BaseAgent):
    """
    Agent with built-in limitations and safety constraints.
    
    Implements safety checks, human oversight, and ethical boundaries
    to prevent fully autonomous operation without safeguards.
    """
    
    def __init__(
        self,
        name: str,
        max_autonomy_level: float = 0.7,  # Max 70% autonomy
        require_human_approval: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize agent with limitations.
        
        Args:
            name: Agent name
            max_autonomy_level: Maximum allowed autonomy (0.0-1.0)
            require_human_approval: Whether to require human approval for actions
            config: Additional configuration
        """
        super().__init__(name, config)
        self.max_autonomy_level = max_autonomy_level
        self.require_human_approval = require_human_approval
        
        # Safety and limitations
        self.safety_checks: List[SafetyCheck] = []
        self.limitations: List[ActionLimitation] = []
        self.blocked_actions: List[Dict[str, Any]] = []
        self.human_approvals: List[Dict[str, Any]] = []
        
        # Memory systems
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        
        # Risk tracking
        self.risk_history: List[Dict[str, Any]] = []
        self.current_risk_level = RiskLevel.LOW
        
        # Initialize default limitations
        self._initialize_default_limitations()
    
    def _initialize_default_limitations(self):
        """Initialize default safety limitations."""
        self.limitations = [
            ActionLimitation(
                limitation_id="critical_actions",
                constraint_type=AutonomyConstraint.HUMAN_APPROVAL_REQUIRED,
                description="Critical actions require human approval",
                applies_to=["delete", "modify", "execute", "transfer"]
            ),
            ActionLimitation(
                limitation_id="resource_limits",
                constraint_type=AutonomyConstraint.RESOURCE_LIMITED,
                description="Resource usage is limited",
                applies_to=["all"]
            ),
            ActionLimitation(
                limitation_id="time_bounds",
                constraint_type=AutonomyConstraint.TIME_BOUNDED,
                description="Actions have time limits",
                applies_to=["all"]
            )
        ]
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about task with safety considerations.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Thinking result with safety assessment
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Assess risk
        risk_assessment = self._assess_risk(task, context)
        self.current_risk_level = risk_assessment["risk_level"]
        
        # Check limitations
        limitation_checks = self._check_limitations(task)
        
        # Determine if human approval needed
        needs_approval = self._needs_human_approval(task, risk_assessment)
        
        result = {
            "task": task,
            "risk_assessment": risk_assessment,
            "limitation_checks": limitation_checks,
            "needs_human_approval": needs_approval,
            "max_autonomy_allowed": self.max_autonomy_level,
            "can_proceed": not needs_approval and risk_assessment["risk_level"] != RiskLevel.CRITICAL
        }
        
        self.state.add_step("think", result)
        return result
    
    def _assess_risk(self, task: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess risk level of a task."""
        risk_level = RiskLevel.LOW
        risk_factors = []
        
        # Check for high-risk keywords
        high_risk_keywords = ["delete", "remove", "modify", "transfer", "execute", "install"]
        if any(keyword in task.lower() for keyword in high_risk_keywords):
            risk_level = RiskLevel.HIGH
            risk_factors.append("Contains high-risk action keywords")
        
        # Check for critical operations
        critical_keywords = ["system", "admin", "root", "database", "network"]
        if any(keyword in task.lower() for keyword in critical_keywords):
            if risk_level == RiskLevel.HIGH:
                risk_level = RiskLevel.CRITICAL
            else:
                risk_level = RiskLevel.MEDIUM
            risk_factors.append("Involves critical system operations")
        
        # Check context for additional risk
        if context:
            if context.get("sensitive_data", False):
                risk_level = RiskLevel.HIGH
                risk_factors.append("Involves sensitive data")
            if context.get("irreversible", False):
                risk_level = RiskLevel.CRITICAL
                risk_factors.append("Action is irreversible")
        
        assessment = {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "risk_score": self._calculate_risk_score(risk_level)
        }
        
        self.risk_history.append({
            "task": task,
            "assessment": assessment,
            "timestamp": datetime.now().isoformat()
        })
        
        return assessment
    
    def _calculate_risk_score(self, risk_level: RiskLevel) -> float:
        """Calculate numeric risk score."""
        scores = {
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.8,
            RiskLevel.CRITICAL: 1.0
        }
        return scores.get(risk_level, 0.0)
    
    def _check_limitations(self, task: str) -> List[Dict[str, Any]]:
        """Check if task violates any limitations."""
        checks = []
        
        for limitation in self.limitations:
            if not limitation.active:
                continue
            
            # Check if limitation applies
            applies = (
                "all" in limitation.applies_to or
                any(keyword in task.lower() for keyword in limitation.applies_to)
            )
            
            if applies:
                check = {
                    "limitation_id": limitation.limitation_id,
                    "constraint_type": limitation.constraint_type.value,
                    "description": limitation.description,
                    "violated": False,
                    "message": f"Limitation applies: {limitation.description}"
                }
                checks.append(check)
        
        return checks
    
    def _needs_human_approval(self, task: str, risk_assessment: Dict[str, Any]) -> bool:
        """Determine if task needs human approval."""
        if not self.require_human_approval:
            return False
        
        # Always require approval for critical risk
        if risk_assessment["risk_level"] == RiskLevel.CRITICAL:
            return True
        
        # Require approval for high risk if autonomy level is low
        if risk_assessment["risk_level"] == RiskLevel.HIGH:
            return self.max_autonomy_level < 0.8
        
        # Check limitations
        for limitation in self.limitations:
            if limitation.constraint_type == AutonomyConstraint.HUMAN_APPROVAL_REQUIRED:
                if any(keyword in task.lower() for keyword in limitation.applies_to):
                    return True
        
        return False
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action with safety checks.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        # Perform safety checks
        safety_result = self._perform_safety_checks(action)
        
        if not safety_result["safe"]:
            # Block unsafe action
            self.blocked_actions.append({
                "action": action,
                "reason": safety_result["reason"],
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "status": "blocked",
                "reason": safety_result["reason"],
                "risk_level": safety_result["risk_level"].value,
                "action": action
            }
        
        # Check if approval needed
        if safety_result.get("needs_approval", False):
            return {
                "status": "pending_approval",
                "action": action,
                "approval_request_id": self._request_approval(action)
            }
        
        # Execute action (with limitations)
        result = self._execute_with_limitations(action)
        
        # Record action
        self.state.add_step("act", {
            "action": action,
            "result": result,
            "safety_checks": safety_result
        })
        
        return result
    
    def _perform_safety_checks(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Perform safety checks on action."""
        action_type = action.get("type", "unknown")
        
        # Check 1: Risk level
        risk_level = RiskLevel.LOW
        if action_type in ["delete", "modify", "execute"]:
            risk_level = RiskLevel.HIGH
        
        # Check 2: Resource limits
        resource_check = self._check_resource_limits(action)
        
        # Check 3: Time bounds
        time_check = self._check_time_bounds(action)
        
        # Check 4: Scope limitations
        scope_check = self._check_scope_limitations(action)
        
        # Overall safety
        safe = (
            risk_level != RiskLevel.CRITICAL and
            resource_check["within_limits"] and
            time_check["within_bounds"] and
            scope_check["allowed"]
        )
        
        safety_check = SafetyCheck(
            check_id=f"check_{datetime.now().timestamp()}",
            check_type="comprehensive",
            passed=safe,
            risk_level=risk_level,
            message="Safety checks completed"
        )
        self.safety_checks.append(safety_check)
        
        return {
            "safe": safe,
            "risk_level": risk_level,
            "needs_approval": risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL],
            "reason": "Action passed safety checks" if safe else "Action failed safety checks",
            "resource_check": resource_check,
            "time_check": time_check,
            "scope_check": scope_check
        }
    
    def _check_resource_limits(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Check if action respects resource limits."""
        # Placeholder: In real implementation, check actual resource usage
        return {
            "within_limits": True,
            "message": "Resource limits respected"
        }
    
    def _check_time_bounds(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Check if action respects time bounds."""
        # Placeholder: In real implementation, check time constraints
        return {
            "within_bounds": True,
            "message": "Time bounds respected"
        }
    
    def _check_scope_limitations(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Check if action is within allowed scope."""
        action_type = action.get("type", "unknown")
        
        # Check limitations
        for limitation in self.limitations:
            if limitation.constraint_type == AutonomyConstraint.LIMITED_SCOPE:
                if action_type in limitation.applies_to or "all" in limitation.applies_to:
                    return {
                        "allowed": False,
                        "message": f"Action type '{action_type}' is outside allowed scope"
                    }
        
        return {
            "allowed": True,
            "message": "Action is within allowed scope"
        }
    
    def _execute_with_limitations(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action with limitations applied."""
        # Apply resource limits
        # Apply time bounds
        # Apply scope restrictions
        
        return {
            "status": "executed",
            "action": action,
            "limitations_applied": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def _request_approval(self, action: Dict[str, Any]) -> str:
        """Request human approval for action."""
        approval_id = f"approval_{datetime.now().timestamp()}"
        
        self.human_approvals.append({
            "approval_id": approval_id,
            "action": action,
            "status": "pending",
            "requested_at": datetime.now().isoformat()
        })
        
        return approval_id
    
    def approve_action(self, approval_id: str, approved: bool) -> Dict[str, Any]:
        """
        Approve or reject a pending action.
        
        Args:
            approval_id: Approval request ID
            approved: Whether action is approved
            
        Returns:
            Approval result
        """
        for approval in self.human_approvals:
            if approval["approval_id"] == approval_id:
                approval["status"] = "approved" if approved else "rejected"
                approval["decided_at"] = datetime.now().isoformat()
                
                if approved:
                    # Execute the action
                    action = approval["action"]
                    return self._execute_with_limitations(action)
                else:
                    return {
                        "status": "rejected",
                        "approval_id": approval_id,
                        "reason": "Human approval denied"
                    }
        
        return {"error": "Approval request not found"}
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation and update safety state.
        
        Args:
            observation: Observation data
            
        Returns:
            Processed observation
        """
        from ..common.agent_utils import standard_observe_pattern
        
        self.state.status = AgentStatus.OBSERVING
        
        # Update risk assessment if needed
        if isinstance(observation, dict) and observation.get("error"):
            self.current_risk_level = RiskLevel.MEDIUM
        
        # Use standard observe pattern
        return standard_observe_pattern(
            self,
            observation,
            importance=0.6,
            additional_data={
                "current_risk_level": self.current_risk_level.value
            }
        )
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run task with safety limitations.
        
        Args:
            task: Task description
            context: Optional context
            
        Returns:
            Final result
        """
        from ..common.agent_utils import standard_run_pattern
        
        # Prepare context with safety information
        if context is None:
            context = {}
        
        context["safety_mode"] = True
        context["max_autonomy"] = self.max_autonomy_level
        
        # Use standard pattern
        result = standard_run_pattern(self, task, context)
        
        # Add safety-specific information
        result["safety_summary"] = {
            "blocked_actions": len(self.blocked_actions),
            "pending_approvals": len([a for a in self.human_approvals if a["status"] == "pending"]),
            "current_risk_level": self.current_risk_level.value,
            "safety_checks_performed": len(self.safety_checks)
        }
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "max_autonomy_level": self.max_autonomy_level,
            "current_risk_level": self.current_risk_level.value,
            "blocked_actions_count": len(self.blocked_actions),
            "pending_approvals": len([a for a in self.human_approvals if a["status"] == "pending"]),
            "safety_checks_performed": len(self.safety_checks),
            "active_limitations": len([l for l in self.limitations if l.active])
        })
    
    def add_limitation(self, limitation: ActionLimitation):
        """Add a new limitation."""
        self.limitations.append(limitation)
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Get comprehensive safety report."""
        return {
            "total_safety_checks": len(self.safety_checks),
            "passed_checks": len([c for c in self.safety_checks if c.passed]),
            "failed_checks": len([c for c in self.safety_checks if not c.passed]),
            "blocked_actions": len(self.blocked_actions),
            "pending_approvals": len([a for a in self.human_approvals if a["status"] == "pending"]),
            "current_risk_level": self.current_risk_level.value,
            "active_limitations": len([l for l in self.limitations if l.active]),
            "risk_history": self.risk_history[-10:]  # Last 10 risk assessments
        }


