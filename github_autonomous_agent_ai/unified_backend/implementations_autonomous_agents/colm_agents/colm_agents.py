"""
COLM 2025 Workshop on AI Agents: Capabilities and Safety
========================================================

Paper: "Published as a paper at COLM 2025 Workshop on AI Agents: Capabilities and Safety"

Key concepts:
- AI agent capabilities assessment
- Safety evaluation for AI agents
- Capability-safety trade-offs
- Agent benchmarking
- Safety frameworks
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus
from ..common.memory import EpisodicMemory, SemanticMemory


class CapabilityType(Enum):
    """Types of agent capabilities."""
    REASONING = "reasoning"
    PLANNING = "planning"
    TOOL_USE = "tool_use"
    MULTI_AGENT = "multi_agent"
    ADAPTATION = "adaptation"
    LEARNING = "learning"


class SafetyLevel(Enum):
    """Safety levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CapabilityAssessment:
    """Assessment of agent capability."""
    capability_id: str
    capability_type: CapabilityType
    score: float  # 0.0 to 1.0
    description: str
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SafetyEvaluation:
    """Safety evaluation result."""
    evaluation_id: str
    safety_level: SafetyLevel
    risk_score: float  # 0.0 to 1.0
    vulnerabilities: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class COLM2025Agent(BaseAgent):
    """
    Agent for COLM 2025 Workshop on AI Agents: Capabilities and Safety.
    
    Evaluates agent capabilities and safety, manages trade-offs,
    and provides comprehensive assessment frameworks.
    """
    
    def __init__(
        self,
        name: str,
        safety_threshold: float = 0.7,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize COLM 2025 agent.
        
        Args:
            name: Agent name
            safety_threshold: Minimum safety score required (0.0-1.0)
            config: Additional configuration
        """
        super().__init__(name, config)
        self.safety_threshold = safety_threshold
        
        # Assessment tracking
        self.capability_assessments: List[CapabilityAssessment] = []
        self.safety_evaluations: List[SafetyEvaluation] = []
        self.trade_off_analyses: List[Dict[str, Any]] = []
        
        # Metrics
        self.capabilities_evaluated = 0
        self.safety_checks_performed = 0
        self.trade_offs_analyzed = 0
        
        # Memory systems
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        
        # Capability and safety tracking
        self.capability_scores: Dict[CapabilityType, float] = {
            ctype: 0.0 for ctype in CapabilityType
        }
        self.safety_history: List[Dict[str, Any]] = []
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about capability or safety task.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Thinking result with analysis
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Determine task type
        task_type = self._determine_task_type(task)
        
        # Analyze requirements
        if task_type == "capability":
            analysis = self._analyze_capability_requirements(task, context)
        elif task_type == "safety":
            analysis = self._analyze_safety_requirements(task, context)
        else:
            analysis = self._analyze_trade_off_requirements(task, context)
        
        result = {
            "task": task,
            "task_type": task_type,
            "analysis": analysis,
            "safety_threshold": self.safety_threshold
        }
        
        self.state.add_step("think", result)
        return result
    
    def _determine_task_type(self, task: str) -> str:
        """Determine type of task."""
        task_lower = task.lower()
        
        if "capability" in task_lower or "capabilities" in task_lower:
            return "capability"
        elif "safety" in task_lower or "safe" in task_lower:
            return "safety"
        else:
            return "trade_off"
    
    def _analyze_capability_requirements(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze capability requirements."""
        return {
            "capabilities_to_evaluate": [ct.value for ct in CapabilityType],
            "evaluation_depth": "comprehensive",
            "benchmark_required": True
        }
    
    def _analyze_safety_requirements(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze safety requirements."""
        return {
            "safety_levels_to_check": [sl.value for sl in SafetyLevel],
            "risk_assessment_required": True,
            "vulnerability_scan": True
        }
    
    def _analyze_trade_off_requirements(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze trade-off requirements."""
        return {
            "capability_safety_balance": True,
            "optimization_required": True
        }
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute capability or safety action.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        action_type = action.get("type", "evaluate")
        
        if action_type == "evaluate_capability":
            result = self._evaluate_capability(action)
        elif action_type == "evaluate_safety":
            result = self._evaluate_safety(action)
        elif action_type == "analyze_trade_off":
            result = self._analyze_trade_off(action)
        else:
            result = self._execute_generic_action(action)
        
        self.state.add_step("act", {
            "action": action,
            "result": result
        })
        
        return result
    
    def _evaluate_capability(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate agent capability."""
        capability_type = CapabilityType(action.get("capability_type", CapabilityType.REASONING.value))
        description = action.get("description", f"Evaluation of {capability_type.value}")
        
        # Perform evaluation (placeholder)
        score = 0.75  # Placeholder score
        
        assessment = CapabilityAssessment(
            capability_id=f"cap_{datetime.now().timestamp()}",
            capability_type=capability_type,
            score=score,
            description=description,
            test_results=action.get("test_results", [])
        )
        
        self.capability_assessments.append(assessment)
        self.capabilities_evaluated += 1
        self.capability_scores[capability_type] = score
        
        return {
            "status": "completed",
            "capability_id": assessment.capability_id,
            "capability_type": capability_type.value,
            "score": score,
            "meets_threshold": score >= 0.7
        }
    
    def _evaluate_safety(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate agent safety."""
        safety_level = SafetyLevel(action.get("safety_level", SafetyLevel.MEDIUM.value))
        
        # Perform safety evaluation (placeholder)
        risk_score = 0.3  # Lower is better
        vulnerabilities = action.get("vulnerabilities", [])
        mitigations = action.get("mitigations", [])
        
        evaluation = SafetyEvaluation(
            evaluation_id=f"safe_{datetime.now().timestamp()}",
            safety_level=safety_level,
            risk_score=risk_score,
            vulnerabilities=vulnerabilities,
            mitigations=mitigations
        )
        
        self.safety_evaluations.append(evaluation)
        self.safety_checks_performed += 1
        
        # Record in history
        self.safety_history.append({
            "evaluation_id": evaluation.evaluation_id,
            "risk_score": risk_score,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "status": "completed",
            "evaluation_id": evaluation.evaluation_id,
            "safety_level": safety_level.value,
            "risk_score": risk_score,
            "meets_threshold": risk_score <= (1.0 - self.safety_threshold),
            "vulnerabilities_count": len(vulnerabilities)
        }
    
    def _analyze_trade_off(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze capability-safety trade-off."""
        capability_scores = action.get("capability_scores", self.capability_scores)
        safety_score = action.get("safety_score", 0.7)
        
        # Calculate trade-off
        avg_capability = sum(capability_scores.values()) / len(capability_scores) if capability_scores else 0.0
        trade_off_score = (avg_capability + safety_score) / 2.0
        
        trade_off_analysis = {
            "analysis_id": f"tradeoff_{datetime.now().timestamp()}",
            "capability_score": avg_capability,
            "safety_score": safety_score,
            "trade_off_score": trade_off_score,
            "recommendation": self._get_trade_off_recommendation(avg_capability, safety_score),
            "timestamp": datetime.now().isoformat()
        }
        
        self.trade_off_analyses.append(trade_off_analysis)
        self.trade_offs_analyzed += 1
        
        return {
            "status": "completed",
            "trade_off_analysis": trade_off_analysis
        }
    
    def _get_trade_off_recommendation(self, capability: float, safety: float) -> str:
        """Get recommendation based on trade-off."""
        if capability >= 0.8 and safety >= 0.8:
            return "optimal_balance"
        elif capability < 0.6:
            return "improve_capabilities"
        elif safety < 0.6:
            return "improve_safety"
        else:
            return "balanced_improvement"
    
    def _execute_generic_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic action."""
        return {
            "status": "executed",
            "action": action,
            "timestamp": datetime.now().isoformat()
        }
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation and update assessment state.
        
        Args:
            observation: Observation data
            
        Returns:
            Processed observation
        """
        from ..common.agent_utils import standard_observe_pattern
        
        self.state.status = AgentStatus.OBSERVING
        
        # Update assessments if applicable
        if isinstance(observation, dict):
            if observation.get("capability_update"):
                self._update_capability_scores(observation["capability_update"])
            if observation.get("safety_update"):
                self._update_safety_assessment(observation["safety_update"])
        
        # Use standard observe pattern
        return standard_observe_pattern(
            self,
            observation,
            importance=0.7,
            additional_data={
                "capabilities_evaluated": self.capabilities_evaluated,
                "safety_checks_performed": self.safety_checks_performed,
                "trade_offs_analyzed": self.trade_offs_analyzed
            }
        )
    
    def _update_capability_scores(self, update: Dict[str, Any]):
        """Update capability scores."""
        for ctype_str, score in update.items():
            try:
                ctype = CapabilityType(ctype_str)
                self.capability_scores[ctype] = score
            except ValueError:
                pass
    
    def _update_safety_assessment(self, update: Dict[str, Any]):
        """Update safety assessment."""
        # Update safety history
        self.safety_history.append({
            "update": update,
            "timestamp": datetime.now().isoformat()
        })
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run capability or safety task.
        
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
        
        context["safety_threshold"] = self.safety_threshold
        
        # Use standard pattern
        result = standard_run_pattern(self, task, context)
        
        # Add COLM-specific information
        result["colm_summary"] = {
            "capabilities_evaluated": self.capabilities_evaluated,
            "safety_checks_performed": self.safety_checks_performed,
            "trade_offs_analyzed": self.trade_offs_analyzed,
            "average_capability_score": sum(self.capability_scores.values()) / len(self.capability_scores) if self.capability_scores else 0.0,
            "safety_threshold": self.safety_threshold,
            "capability_scores": {
                ctype.value: score for ctype, score in self.capability_scores.items()
            }
        }
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "capabilities_evaluated": self.capabilities_evaluated,
            "safety_checks_performed": self.safety_checks_performed,
            "trade_offs_analyzed": self.trade_offs_analyzed,
            "safety_threshold": self.safety_threshold
        })
    
    def get_comprehensive_assessment(self) -> Dict[str, Any]:
        """Get comprehensive capability and safety assessment."""
        avg_capability = sum(self.capability_scores.values()) / len(self.capability_scores) if self.capability_scores else 0.0
        
        latest_safety = self.safety_evaluations[-1] if self.safety_evaluations else None
        avg_risk = sum(e.risk_score for e in self.safety_evaluations) / len(self.safety_evaluations) if self.safety_evaluations else 0.0
        
        return {
            "capability_assessment": {
                "average_score": avg_capability,
                "scores_by_type": {
                    ctype.value: score for ctype, score in self.capability_scores.items()
                },
                "total_evaluations": len(self.capability_assessments)
            },
            "safety_assessment": {
                "average_risk_score": avg_risk,
                "latest_evaluation": {
                    "safety_level": latest_safety.safety_level.value,
                    "risk_score": latest_safety.risk_score
                } if latest_safety else None,
                "total_evaluations": len(self.safety_evaluations)
            },
            "trade_off_analysis": {
                "total_analyses": len(self.trade_off_analyses),
                "latest_recommendation": self.trade_off_analyses[-1].get("recommendation") if self.trade_off_analyses else None
            }
        }
