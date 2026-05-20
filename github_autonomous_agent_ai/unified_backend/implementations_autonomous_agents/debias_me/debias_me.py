"""
DeBiasMe: De-biasing Human-AI Interactions with Metacognitive AIED
===================================================================

Paper: "DeBiasMe: De-biasing Human-AI Interactions with Metacognitive AIED (AI in Education)"

Key concepts:
- Bias detection and mitigation in AI interactions
- Metacognitive awareness
- Debiasing strategies
- Fair interaction patterns
- Educational AI applications
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus
from ..common.memory import EpisodicMemory, SemanticMemory


class BiasType(Enum):
    """Types of biases."""
    COGNITIVE = "cognitive"
    CONFIRMATION = "confirmation"
    ANCHORING = "anchoring"
    AVAILABILITY = "availability"
    STEREOTYPING = "stereotyping"
    GENDER = "gender"
    RACIAL = "racial"
    CULTURAL = "cultural"
    SELECTION = "selection"


class DebiasingStrategy(Enum):
    """Debiasing strategies."""
    COUNTER_EXAMPLES = "counter_examples"
    PERSPECTIVE_TAKING = "perspective_taking"
    EXPLICIT_QUESTIONING = "explicit_questioning"
    DIVERSITY_PROMOTION = "diversity_promotion"
    CALIBRATION = "calibration"
    METACOGNITIVE_REFLECTION = "metacognitive_reflection"


@dataclass
class BiasDetection:
    """Bias detection result."""
    detection_id: str
    bias_type: BiasType
    severity: float
    context: str
    detected_at: datetime = field(default_factory=datetime.now)
    mitigated: bool = False


@dataclass
class DebiasingAction:
    """Debiasing action taken."""
    action_id: str
    strategy: DebiasingStrategy
    target_bias: BiasType
    description: str
    effectiveness: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class DeBiasMeAgent(BaseAgent):
    """
    Agent for detecting and mitigating biases in human-AI interactions.
    
    Implements metacognitive awareness and debiasing strategies
    for fair and unbiased interactions.
    """
    
    def __init__(
        self,
        name: str,
        debiasing_enabled: bool = True,
        metacognitive_mode: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize DeBiasMe agent.
        
        Args:
            name: Agent name
            debiasing_enabled: Whether debiasing is enabled
            metacognitive_mode: Whether to use metacognitive reflection
            config: Additional configuration
        """
        super().__init__(name, config)
        self.debiasing_enabled = debiasing_enabled
        self.metacognitive_mode = metacognitive_mode
        
        # Bias tracking
        self.bias_detections: List[BiasDetection] = []
        self.debiasing_actions: List[DebiasingAction] = []
        
        # Metrics
        self.biases_detected = 0
        self.biases_mitigated = 0
        self.interaction_count = 0
        self.fair_interactions = 0
        
        # Memory systems
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        
        # Metacognitive state
        self.metacognitive_awareness: Dict[str, Any] = {
            "bias_awareness_level": 0.5,
            "reflection_frequency": 0.3,
            "self_correction_count": 0
        }
        
        # Debiasing strategies configuration
        self.strategy_config: Dict[str, Any] = {
            "counter_examples_enabled": True,
            "perspective_taking_enabled": True,
            "explicit_questioning_enabled": True,
            "diversity_promotion_enabled": True
        }
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about task with bias awareness.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Thinking result with bias analysis
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Detect biases
        bias_analysis = self._detect_biases(task, context)
        
        # Metacognitive reflection
        if self.metacognitive_mode:
            metacognitive_analysis = self._metacognitive_reflection(task, bias_analysis)
        else:
            metacognitive_analysis = {"reflection_performed": False}
        
        # Determine debiasing needs
        debiasing_needed = bias_analysis["biases_found"] > 0
        
        result = {
            "task": task,
            "bias_analysis": bias_analysis,
            "metacognitive_analysis": metacognitive_analysis,
            "debiasing_needed": debiasing_needed,
            "debiasing_enabled": self.debiasing_enabled
        }
        
        self.state.add_step("think", result)
        return result
    
    def _detect_biases(self, task: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect biases in task or context."""
        detected_biases = []
        
        # Check for stereotyping
        stereotyping_keywords = ["always", "never", "all", "none", "typical"]
        if any(keyword in task.lower() for keyword in stereotyping_keywords):
            bias = BiasDetection(
                detection_id=f"bias_{datetime.now().timestamp()}",
                bias_type=BiasType.STEREOTYPING,
                severity=0.6,
                context=task
            )
            detected_biases.append(bias)
            self.bias_detections.append(bias)
            self.biases_detected += 1
        
        # Check for gender bias
        gender_keywords = ["he should", "she should", "men are", "women are"]
        if any(keyword in task.lower() for keyword in gender_keywords):
            bias = BiasDetection(
                detection_id=f"bias_{datetime.now().timestamp()}",
                bias_type=BiasType.GENDER,
                severity=0.7,
                context=task
            )
            detected_biases.append(bias)
            self.bias_detections.append(bias)
            self.biases_detected += 1
        
        # Check for confirmation bias
        if context and context.get("confirmation_seeking", False):
            bias = BiasDetection(
                detection_id=f"bias_{datetime.now().timestamp()}",
                bias_type=BiasType.CONFIRMATION,
                severity=0.5,
                context=str(context)
            )
            detected_biases.append(bias)
            self.bias_detections.append(bias)
            self.biases_detected += 1
        
        return {
            "biases_found": len(detected_biases),
            "biases": [
                {
                    "type": b.bias_type.value,
                    "severity": b.severity,
                    "context": b.context[:100]
                }
                for b in detected_biases
            ]
        }
    
    def _metacognitive_reflection(self, task: str, bias_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform metacognitive reflection on task and biases."""
        reflection = {
            "reflection_performed": True,
            "bias_awareness": self.metacognitive_awareness["bias_awareness_level"],
            "insights": []
        }
        
        if bias_analysis["biases_found"] > 0:
            reflection["insights"].append("Biases detected - need to apply debiasing strategies")
            self.metacognitive_awareness["bias_awareness_level"] = min(
                1.0,
                self.metacognitive_awareness["bias_awareness_level"] + 0.1
            )
        
        # Self-correction if needed
        if bias_analysis["biases_found"] > 2:
            reflection["insights"].append("Multiple biases detected - high priority for correction")
            self.metacognitive_awareness["self_correction_count"] += 1
        
        return reflection
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action with debiasing.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        # Apply debiasing if needed
        if self.debiasing_enabled:
            debiased_action = self._apply_debiasing(action)
        else:
            debiased_action = action
        
        # Execute action
        result = self._execute_action(debiased_action)
        
        # Track interaction
        self.interaction_count += 1
        if not self._has_bias(result):
            self.fair_interactions += 1
        
        self.state.add_step("act", {
            "original_action": action,
            "debiased_action": debiased_action,
            "result": result
        })
        
        return result
    
    def _apply_debiasing(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply debiasing strategies to action."""
        action_str = str(action).lower()
        debiased = action.copy()
        
        # Check for biases in action
        for bias in self.bias_detections:
            if not bias.mitigated and bias.bias_type.value in action_str:
                # Apply appropriate strategy
                strategy = self._select_debiasing_strategy(bias.bias_type)
                
                if strategy:
                    debiasing_action = DebiasingAction(
                        action_id=f"debiasing_{datetime.now().timestamp()}",
                        strategy=strategy,
                        target_bias=bias.bias_type,
                        description=f"Applied {strategy.value} to mitigate {bias.bias_type.value}",
                        effectiveness=0.7
                    )
                    self.debiasing_actions.append(debiasing_action)
                    
                    # Mark bias as mitigated
                    bias.mitigated = True
                    self.biases_mitigated += 1
                    
                    # Modify action based on strategy
                    if strategy == DebiasingStrategy.COUNTER_EXAMPLES:
                        debiased["counter_examples_provided"] = True
                    elif strategy == DebiasingStrategy.PERSPECTIVE_TAKING:
                        debiased["multiple_perspectives"] = True
                    elif strategy == DebiasingStrategy.EXPLICIT_QUESTIONING:
                        debiased["explicit_questions"] = True
        
        return debiased
    
    def _select_debiasing_strategy(self, bias_type: BiasType) -> Optional[DebiasingStrategy]:
        """Select appropriate debiasing strategy for bias type."""
        strategy_map = {
            BiasType.STEREOTYPING: DebiasingStrategy.COUNTER_EXAMPLES,
            BiasType.CONFIRMATION: DebiasingStrategy.PERSPECTIVE_TAKING,
            BiasType.GENDER: DebiasingStrategy.DIVERSITY_PROMOTION,
            BiasType.RACIAL: DebiasingStrategy.DIVERSITY_PROMOTION,
            BiasType.CULTURAL: DebiasingStrategy.PERSPECTIVE_TAKING,
            BiasType.ANCHORING: DebiasingStrategy.EXPLICIT_QUESTIONING,
            BiasType.AVAILABILITY: DebiasingStrategy.CALIBRATION
        }
        
        strategy = strategy_map.get(bias_type)
        
        # Check if strategy is enabled
        if strategy:
            strategy_key = f"{strategy.value}_enabled"
            if self.strategy_config.get(strategy_key, True):
                return strategy
        
        return DebiasingStrategy.METACOGNITIVE_REFLECTION
    
    def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action."""
        return {
            "status": "executed",
            "action": action,
            "debiased": action.get("counter_examples_provided") or action.get("multiple_perspectives", False),
            "timestamp": datetime.now().isoformat()
        }
    
    def _has_bias(self, result: Dict[str, Any]) -> bool:
        """Check if result contains bias."""
        # Simple check - in real implementation would be more sophisticated
        return False
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation and update bias awareness.
        
        Args:
            observation: Observation data
            
        Returns:
            Processed observation
        """
        from ..common.agent_utils import standard_observe_pattern
        
        self.state.status = AgentStatus.OBSERVING
        
        # Check for bias feedback in observation
        if isinstance(observation, dict):
            if observation.get("bias_feedback"):
                self._update_bias_awareness(observation["bias_feedback"])
        
        # Use standard observe pattern
        return standard_observe_pattern(
            self,
            observation,
            importance=0.7,
            additional_data={
                "bias_awareness_level": self.metacognitive_awareness["bias_awareness_level"],
                "fair_interaction_rate": self.fair_interactions / self.interaction_count if self.interaction_count > 0 else 0.0
            }
        )
    
    def _update_bias_awareness(self, feedback: Dict[str, Any]):
        """Update bias awareness based on feedback."""
        if feedback.get("bias_detected"):
            self.metacognitive_awareness["bias_awareness_level"] = min(
                1.0,
                self.metacognitive_awareness["bias_awareness_level"] + 0.05
            )
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run task with debiasing.
        
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
        
        context["debiasing_enabled"] = self.debiasing_enabled
        context["metacognitive_mode"] = self.metacognitive_mode
        
        # Use standard pattern
        result = standard_run_pattern(self, task, context)
        
        # Add debiasing information
        result["debiasing_summary"] = {
            "biases_detected": self.biases_detected,
            "biases_mitigated": self.biases_mitigated,
            "debiasing_actions": len(self.debiasing_actions),
            "fair_interaction_rate": self.fair_interactions / self.interaction_count if self.interaction_count > 0 else 0.0,
            "bias_awareness_level": self.metacognitive_awareness["bias_awareness_level"]
        }
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "biases_detected": self.biases_detected,
            "biases_mitigated": self.biases_mitigated,
            "interaction_count": self.interaction_count,
            "fair_interactions": self.fair_interactions,
            "bias_awareness_level": self.metacognitive_awareness["bias_awareness_level"],
            "debiasing_enabled": self.debiasing_enabled
        })
    
    def get_debiasing_report(self) -> Dict[str, Any]:
        """Get comprehensive debiasing report."""
        return {
            "total_biases_detected": self.biases_detected,
            "total_biases_mitigated": self.biases_mitigated,
            "mitigation_rate": self.biases_mitigated / self.biases_detected if self.biases_detected > 0 else 0.0,
            "debiasing_actions": len(self.debiasing_actions),
            "bias_distribution": {
                bias_type.value: len([b for b in self.bias_detections if b.bias_type == bias_type])
                for bias_type in BiasType
            },
            "strategy_usage": {
                strategy.value: len([a for a in self.debiasing_actions if a.strategy == strategy])
                for strategy in DebiasingStrategy
            },
            "metacognitive_awareness": self.metacognitive_awareness,
            "fair_interaction_rate": self.fair_interactions / self.interaction_count if self.interaction_count > 0 else 0.0
        }


