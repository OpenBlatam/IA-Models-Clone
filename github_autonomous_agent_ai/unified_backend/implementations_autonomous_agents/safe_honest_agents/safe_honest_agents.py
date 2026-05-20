"""
Towards Safe and Honest AI Agents
==================================

Paper: "Towards Safe and Honest AI Agents with..."

Key concepts:
- Safety mechanisms for AI agents
- Honesty and transparency
- Truthfulness verification
- Safe action execution
- Trustworthiness metrics
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus
from ..common.memory import EpisodicMemory, SemanticMemory


class HonestyLevel(Enum):
    """Levels of honesty/truthfulness."""
    VERIFIED = "verified"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNVERIFIED = "unverified"


class SafetyStatus(Enum):
    """Safety status of agent."""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    UNSAFE = "unsafe"


@dataclass
class TruthfulnessCheck:
    """Truthfulness verification result."""
    check_id: str
    statement: str
    is_truthful: bool
    confidence: float
    evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SafetyMeasure:
    """Safety measure applied to action."""
    measure_id: str
    measure_type: str
    description: str
    applied: bool
    effectiveness: float = 0.0


class SafeHonestAgent(BaseAgent):
    """
    Agent with built-in safety and honesty mechanisms.
    
    Implements truthfulness verification, safety checks, and
    transparent communication.
    """
    
    def __init__(
        self,
        name: str,
        honesty_threshold: float = 0.8,
        safety_enabled: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize safe and honest agent.
        
        Args:
            name: Agent name
            honesty_threshold: Minimum honesty score (0.0-1.0)
            safety_enabled: Whether safety checks are enabled
            config: Additional configuration
        """
        super().__init__(name, config)
        self.honesty_threshold = honesty_threshold
        self.safety_enabled = safety_enabled
        
        # Honesty tracking
        self.truthfulness_checks: List[TruthfulnessCheck] = []
        self.honesty_score: float = 1.0
        self.current_honesty_level = HonestyLevel.HIGH
        
        # Safety tracking
        self.safety_measures: List[SafetyMeasure] = []
        self.safety_status = SafetyStatus.SAFE
        self.safe_actions_count = 0
        self.unsafe_actions_blocked = 0
        
        # Memory systems
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        
        # Transparency log
        self.transparency_log: List[Dict[str, Any]] = []
        
        # Initialize safety measures
        self._initialize_safety_measures()
    
    def _initialize_safety_measures(self):
        """Initialize default safety measures."""
        self.safety_measures = [
            SafetyMeasure(
                measure_id="input_validation",
                measure_type="validation",
                description="Validate all inputs before processing",
                applied=True,
                effectiveness=0.9
            ),
            SafetyMeasure(
                measure_id="output_verification",
                measure_type="verification",
                description="Verify all outputs before returning",
                applied=True,
                effectiveness=0.85
            ),
            SafetyMeasure(
                measure_id="action_screening",
                measure_type="screening",
                description="Screen actions for safety",
                applied=True,
                effectiveness=0.8
            )
        ]
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about task with honesty and safety considerations.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Thinking result with honesty and safety assessment
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Verify truthfulness of task understanding
        truthfulness = self._verify_truthfulness(task)
        
        # Assess safety
        safety_assessment = self._assess_safety(task, context)
        
        # Determine if can proceed
        can_proceed = (
            truthfulness["is_truthful"] and
            safety_assessment["status"] != SafetyStatus.UNSAFE
        )
        
        result = {
            "task": task,
            "truthfulness": truthfulness,
            "safety_assessment": safety_assessment,
            "can_proceed": can_proceed,
            "honesty_level": self.current_honesty_level.value,
            "safety_status": self.safety_status.value
        }
        
        self.state.add_step("think", result)
        return result
    
    def _verify_truthfulness(self, statement: str) -> Dict[str, Any]:
        """Verify truthfulness of a statement."""
        # Check for known falsehoods
        falsehood_indicators = ["definitely", "always", "never", "guaranteed"]
        has_falsehood_indicators = any(
            indicator in statement.lower() for indicator in falsehood_indicators
        )
        
        # Check for uncertainty markers (good for honesty)
        uncertainty_markers = ["possibly", "might", "could", "perhaps", "maybe"]
        has_uncertainty = any(
            marker in statement.lower() for marker in uncertainty_markers
        )
        
        # Calculate truthfulness score
        if has_falsehood_indicators and not has_uncertainty:
            is_truthful = False
            confidence = 0.3
        elif has_uncertainty:
            is_truthful = True
            confidence = 0.8
        else:
            is_truthful = True
            confidence = 0.7
        
        check = TruthfulnessCheck(
            check_id=f"truth_{datetime.now().timestamp()}",
            statement=statement,
            is_truthful=is_truthful,
            confidence=confidence,
            evidence=["Statement analysis", "Pattern matching"]
        )
        self.truthfulness_checks.append(check)
        
        # Update honesty score
        self._update_honesty_score(is_truthful, confidence)
        
        return {
            "is_truthful": is_truthful,
            "confidence": confidence,
            "evidence": check.evidence
        }
    
    def _update_honesty_score(self, is_truthful: bool, confidence: float):
        """Update overall honesty score."""
        if is_truthful:
            # Increase score slightly
            self.honesty_score = min(1.0, self.honesty_score + 0.01 * confidence)
        else:
            # Decrease score more significantly
            self.honesty_score = max(0.0, self.honesty_score - 0.1 * (1 - confidence))
        
        # Update honesty level
        if self.honesty_score >= 0.9:
            self.current_honesty_level = HonestyLevel.VERIFIED
        elif self.honesty_score >= 0.7:
            self.current_honesty_level = HonestyLevel.HIGH
        elif self.honesty_score >= 0.5:
            self.current_honesty_level = HonestyLevel.MODERATE
        elif self.honesty_score >= 0.3:
            self.current_honesty_level = HonestyLevel.LOW
        else:
            self.current_honesty_level = HonestyLevel.UNVERIFIED
    
    def _assess_safety(self, task: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess safety of a task."""
        safety_status = SafetyStatus.SAFE
        safety_factors = []
        
        # Check for unsafe keywords
        unsafe_keywords = ["hack", "exploit", "bypass", "disable", "remove"]
        if any(keyword in task.lower() for keyword in unsafe_keywords):
            safety_status = SafetyStatus.WARNING
            safety_factors.append("Contains potentially unsafe keywords")
        
        # Check for destructive operations
        destructive_keywords = ["delete", "destroy", "remove", "erase"]
        if any(keyword in task.lower() for keyword in destructive_keywords):
            safety_status = SafetyStatus.UNSAFE
            safety_factors.append("Involves destructive operations")
        
        # Check context
        if context:
            if context.get("irreversible", False):
                safety_status = SafetyStatus.UNSAFE
                safety_factors.append("Action is irreversible")
            if context.get("high_risk", False):
                safety_status = SafetyStatus.WARNING
                safety_factors.append("High risk operation")
        
        self.safety_status = safety_status
        
        return {
            "status": safety_status,
            "safety_factors": safety_factors,
            "safe": safety_status in [SafetyStatus.SAFE, SafetyStatus.CAUTION]
        }
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action with safety and honesty checks.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        # Apply safety measures
        safety_result = self._apply_safety_measures(action)
        
        if not safety_result["safe"]:
            self.unsafe_actions_blocked += 1
            return {
                "status": "blocked",
                "reason": safety_result["reason"],
                "safety_status": self.safety_status.value
            }
        
        # Verify action truthfulness
        action_description = action.get("description", str(action))
        truthfulness = self._verify_truthfulness(action_description)
        
        if not truthfulness["is_truthful"]:
            return {
                "status": "blocked",
                "reason": "Action description is not truthful",
                "honesty_level": self.current_honesty_level.value
            }
        
        # Execute action
        result = self._execute_safely(action)
        self.safe_actions_count += 1
        
        # Log transparency
        self._log_transparency(action, result)
        
        self.state.add_step("act", {
            "action": action,
            "result": result,
            "safety_measures": safety_result
        })
        
        return result
    
    def _apply_safety_measures(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply safety measures to action."""
        if not self.safety_enabled:
            return {"safe": True, "reason": "Safety checks disabled"}
        
        # Apply each safety measure
        measures_applied = []
        for measure in self.safety_measures:
            if measure.applied:
                # Check if measure is effective for this action
                if self._is_measure_relevant(measure, action):
                    measures_applied.append(measure.measure_id)
        
        # Determine if action is safe
        safe = len(measures_applied) > 0 and self.safety_status != SafetyStatus.UNSAFE
        
        return {
            "safe": safe,
            "reason": "Safety measures applied" if safe else "Action failed safety checks",
            "measures_applied": measures_applied,
            "safety_status": self.safety_status.value
        }
    
    def _is_measure_relevant(self, measure: SafetyMeasure, action: Dict[str, Any]) -> bool:
        """Check if safety measure is relevant for action."""
        # Simple relevance check
        return measure.effectiveness > 0.5
    
    def _execute_safely(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action with safety precautions."""
        # In real implementation, would have actual safety checks
        return {
            "status": "executed",
            "action": action,
            "safety_verified": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def _log_transparency(self, action: Dict[str, Any], result: Dict[str, Any]):
        """Log action for transparency."""
        log_entry = {
            "action": action,
            "result": result,
            "honesty_level": self.current_honesty_level.value,
            "safety_status": self.safety_status.value,
            "timestamp": datetime.now().isoformat()
        }
        self.transparency_log.append(log_entry)
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation and update safety/honesty state.
        
        Args:
            observation: Observation data
            
        Returns:
            Processed observation
        """
        from ..common.agent_utils import standard_observe_pattern
        
        self.state.status = AgentStatus.OBSERVING
        
        # Check if observation indicates safety issue
        if isinstance(observation, dict):
            if observation.get("error"):
                self.safety_status = SafetyStatus.WARNING
            if observation.get("unsafe"):
                self.safety_status = SafetyStatus.UNSAFE
        
        # Use standard observe pattern
        return standard_observe_pattern(
            self,
            observation,
            importance=0.7,
            additional_data={
                "safety_status": self.safety_status.value,
                "honesty_level": self.current_honesty_level.value
            }
        )
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run task with safety and honesty checks.
        
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
        
        context["safety_enabled"] = self.safety_enabled
        context["honesty_threshold"] = self.honesty_threshold
        
        # Use standard pattern
        result = standard_run_pattern(self, task, context)
        
        # Add safety and honesty information
        result["safety_summary"] = {
            "safety_status": self.safety_status.value,
            "safe_actions": self.safe_actions_count,
            "blocked_actions": self.unsafe_actions_blocked,
            "safety_measures_active": len([m for m in self.safety_measures if m.applied])
        }
        
        result["honesty_summary"] = {
            "honesty_score": self.honesty_score,
            "honesty_level": self.current_honesty_level.value,
            "truthfulness_checks": len(self.truthfulness_checks),
            "transparency_log_entries": len(self.transparency_log)
        }
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "honesty_score": self.honesty_score,
            "honesty_level": self.current_honesty_level.value,
            "safety_status": self.safety_status.value,
            "safe_actions_count": self.safe_actions_count,
            "blocked_actions": self.unsafe_actions_blocked,
            "transparency_log_size": len(self.transparency_log)
        })
    
    def get_transparency_report(self) -> Dict[str, Any]:
        """Get transparency report."""
        return {
            "honesty_score": self.honesty_score,
            "honesty_level": self.current_honesty_level.value,
            "truthfulness_checks": len(self.truthfulness_checks),
            "recent_checks": [
                {
                    "statement": check.statement[:50],
                    "is_truthful": check.is_truthful,
                    "confidence": check.confidence
                }
                for check in self.truthfulness_checks[-10:]
            ],
            "transparency_log": self.transparency_log[-10:],
            "safety_status": self.safety_status.value
        }


