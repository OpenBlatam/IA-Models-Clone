"""
Human-Computer Interaction for Autonomous Agents
================================================

Paper: "arXiv:2503.09794v1 [cs.HC] 12 Mar 2025"

Key concepts:
- Human-computer interaction patterns
- User interface design for agents
- Interaction modalities
- Feedback mechanisms
- Usability evaluation
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus
from ..common.memory import EpisodicMemory, SemanticMemory


class InteractionModality(Enum):
    """Interaction modalities."""
    TEXT = "text"
    VOICE = "voice"
    GESTURE = "gesture"
    VISUAL = "visual"
    MULTIMODAL = "multimodal"


class FeedbackType(Enum):
    """Types of feedback."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    CORRECTIVE = "corrective"
    CONFIRMATION = "confirmation"


@dataclass
class Interaction:
    """A human-computer interaction."""
    interaction_id: str
    modality: InteractionModality
    user_input: str
    agent_response: str
    feedback: Optional[FeedbackType] = None
    timestamp: datetime = field(default_factory=datetime.now)
    duration: Optional[float] = None  # seconds


@dataclass
class UsabilityMetric:
    """Usability metric."""
    metric_id: str
    metric_name: str
    value: float
    description: str
    timestamp: datetime = field(default_factory=datetime.now)


class HCIAgent(BaseAgent):
    """
    Agent for Human-Computer Interaction.
    
    Manages interactions with users, evaluates usability,
    and adapts interface based on feedback.
    """
    
    def __init__(
        self,
        name: str,
        preferred_modality: InteractionModality = InteractionModality.TEXT,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize HCI agent.
        
        Args:
            name: Agent name
            preferred_modality: Preferred interaction modality
            config: Additional configuration
        """
        super().__init__(name, config)
        self.preferred_modality = preferred_modality
        
        # Interaction management
        self.interactions: List[Interaction] = []
        self.usability_metrics: List[UsabilityMetric] = []
        self.feedback_history: List[Dict[str, Any]] = []
        
        # Metrics
        self.interactions_count = 0
        self.positive_feedback_count = 0
        self.negative_feedback_count = 0
        self.usability_evaluations = 0
        
        # Memory systems
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        
        # Interaction patterns
        self.interaction_patterns: Dict[str, Any] = {}
        self.user_preferences: Dict[str, Any] = {}
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about interaction task.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Thinking result with interaction analysis
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Analyze interaction needs
        interaction_analysis = self._analyze_interaction_needs(task, context)
        
        # Determine best modality
        recommended_modality = self._recommend_modality(task, context)
        
        result = {
            "task": task,
            "interaction_analysis": interaction_analysis,
            "recommended_modality": recommended_modality.value,
            "preferred_modality": self.preferred_modality.value
        }
        
        self.state.add_step("think", result)
        return result
    
    def _analyze_interaction_needs(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze interaction needs for task."""
        needs = {
            "requires_feedback": True,
            "interaction_complexity": "medium",
            "estimated_duration": 30  # seconds
        }
        
        if "complex" in task.lower() or "detailed" in task.lower():
            needs["interaction_complexity"] = "high"
            needs["estimated_duration"] = 60
        
        return needs
    
    def _recommend_modality(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> InteractionModality:
        """Recommend interaction modality."""
        # Check user preferences
        if context and context.get("user_preferred_modality"):
            try:
                return InteractionModality(context["user_preferred_modality"])
            except ValueError:
                pass
        
        # Default to preferred modality
        return self.preferred_modality
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute interaction action.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        action_type = action.get("type", "interact")
        
        if action_type == "interact":
            result = self._process_interaction(action)
        elif action_type == "evaluate_usability":
            result = self._evaluate_usability(action)
        elif action_type == "adapt_interface":
            result = self._adapt_interface(action)
        else:
            result = self._execute_generic_action(action)
        
        self.state.add_step("act", {
            "action": action,
            "result": result
        })
        
        return result
    
    def _process_interaction(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Process human-computer interaction."""
        user_input = action.get("user_input", "")
        modality = InteractionModality(action.get("modality", self.preferred_modality.value))
        
        # Generate agent response (placeholder)
        agent_response = f"Agent response to: {user_input[:50]}"
        
        # Create interaction
        interaction = Interaction(
            interaction_id=f"interaction_{datetime.now().timestamp()}",
            modality=modality,
            user_input=user_input,
            agent_response=agent_response
        )
        
        self.interactions.append(interaction)
        self.interactions_count += 1
        
        return {
            "status": "completed",
            "interaction_id": interaction.interaction_id,
            "modality": modality.value,
            "agent_response": agent_response
        }
    
    def _evaluate_usability(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate usability."""
        metric_name = action.get("metric_name", "overall_usability")
        
        # Calculate usability score (placeholder)
        usability_score = 0.8  # Based on interaction history
        
        metric = UsabilityMetric(
            metric_id=f"metric_{datetime.now().timestamp()}",
            metric_name=metric_name,
            value=usability_score,
            description=f"Usability evaluation for {metric_name}"
        )
        
        self.usability_metrics.append(metric)
        self.usability_evaluations += 1
        
        return {
            "status": "completed",
            "metric_id": metric.metric_id,
            "metric_name": metric_name,
            "usability_score": usability_score
        }
    
    def _adapt_interface(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt interface based on feedback."""
        adaptation_type = action.get("adaptation_type", "modality")
        
        # Adapt based on feedback history
        if self.negative_feedback_count > self.positive_feedback_count:
            # Switch modality if negative feedback is high
            if self.preferred_modality == InteractionModality.TEXT:
                self.preferred_modality = InteractionModality.MULTIMODAL
        
        return {
            "status": "completed",
            "adaptation_type": adaptation_type,
            "new_preferred_modality": self.preferred_modality.value
        }
    
    def _execute_generic_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic action."""
        return {
            "status": "executed",
            "action": action,
            "timestamp": datetime.now().isoformat()
        }
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation and update interaction state.
        
        Args:
            observation: Observation data
            
        Returns:
            Processed observation
        """
        from ..common.agent_utils import standard_observe_pattern
        
        self.state.status = AgentStatus.OBSERVING
        
        # Process feedback if provided
        if isinstance(observation, dict):
            if observation.get("feedback"):
                self._process_feedback(observation["feedback"])
            if observation.get("user_preference"):
                self.user_preferences.update(observation["user_preference"])
        
        # Use standard observe pattern
        return standard_observe_pattern(
            self,
            observation,
            importance=0.7,
            additional_data={
                "interactions_count": self.interactions_count,
                "positive_feedback": self.positive_feedback_count,
                "negative_feedback": self.negative_feedback_count
            }
        )
    
    def _process_feedback(self, feedback: Dict[str, Any]):
        """Process user feedback."""
        feedback_type = FeedbackType(feedback.get("type", FeedbackType.NEUTRAL.value))
        
        # Update feedback counts
        if feedback_type == FeedbackType.POSITIVE:
            self.positive_feedback_count += 1
        elif feedback_type == FeedbackType.NEGATIVE:
            self.negative_feedback_count += 1
        
        # Record in history
        self.feedback_history.append({
            "type": feedback_type.value,
            "content": feedback.get("content", ""),
            "timestamp": datetime.now().isoformat()
        })
        
        # Update last interaction if available
        if self.interactions:
            self.interactions[-1].feedback = feedback_type
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run interaction task.
        
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
        
        context["preferred_modality"] = self.preferred_modality.value
        
        # Use standard pattern
        result = standard_run_pattern(self, task, context)
        
        # Add HCI-specific information
        result["hci_summary"] = {
            "interactions_count": self.interactions_count,
            "positive_feedback": self.positive_feedback_count,
            "negative_feedback": self.negative_feedback_count,
            "usability_evaluations": self.usability_evaluations,
            "preferred_modality": self.preferred_modality.value,
            "feedback_ratio": self.positive_feedback_count / max(self.negative_feedback_count, 1)
        }
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "interactions_count": self.interactions_count,
            "positive_feedback": self.positive_feedback_count,
            "negative_feedback": self.negative_feedback_count,
            "usability_evaluations": self.usability_evaluations,
            "preferred_modality": self.preferred_modality.value
        })
    
    def get_usability_report(self) -> Dict[str, Any]:
        """Get comprehensive usability report."""
        avg_usability = sum(m.value for m in self.usability_metrics) / len(self.usability_metrics) if self.usability_metrics else 0.0
        
        return {
            "total_interactions": len(self.interactions),
            "average_usability": avg_usability,
            "feedback_summary": {
                "positive": self.positive_feedback_count,
                "negative": self.negative_feedback_count,
                "ratio": self.positive_feedback_count / max(self.negative_feedback_count, 1)
            },
            "modality_usage": {
                modality.value: len([i for i in self.interactions if i.modality == modality])
                for modality in InteractionModality
            },
            "user_preferences": self.user_preferences,
            "recent_interactions": [
                {
                    "modality": i.modality.value,
                    "feedback": i.feedback.value if i.feedback else None,
                    "timestamp": i.timestamp.isoformat()
                }
                for i in self.interactions[-10:]
            ]
        }
