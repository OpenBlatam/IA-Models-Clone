"""
MORPHEUS: Modeling Role from Personalized Dialogue History
===========================================================

Paper: "MORPHEUS*: Modeling Role from Personalized Dialogue History by Exploring and Utilizing Latent Space"

Key concepts:
- Role modeling from dialogue history
- Personalized agent behavior
- Latent space exploration
- Context-aware role adaptation
- Dialogue-based personality inference
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus
from ..common.memory import EpisodicMemory, SemanticMemory


class RoleType(Enum):
    """Types of roles an agent can take."""
    ASSISTANT = "assistant"
    EXPERT = "expert"
    COLLABORATOR = "collaborator"
    MENTOR = "mentor"
    PEER = "peer"
    ADVISOR = "advisor"
    FACILITATOR = "facilitator"


class DialogueContext(Enum):
    """Dialogue context types."""
    FORMAL = "formal"
    INFORMAL = "informal"
    TECHNICAL = "technical"
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    EDUCATIONAL = "educational"


@dataclass
class DialogueTurn:
    """A single dialogue turn."""
    turn_id: str
    speaker: str
    utterance: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Optional[DialogueContext] = None


@dataclass
class RoleProfile:
    """Role profile inferred from dialogue."""
    role_id: str
    role_type: RoleType
    confidence: float
    characteristics: Dict[str, Any] = field(default_factory=dict)
    inferred_from: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class MorpheusAgent(BaseAgent):
    """
    Agent that models its role from personalized dialogue history.
    
    Infers role and adapts behavior based on dialogue patterns
    and latent space exploration.
    """
    
    def __init__(
        self,
        name: str,
        initial_role: Optional[RoleType] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Morpheus agent.
        
        Args:
            name: Agent name
            initial_role: Initial role (if None, will be inferred)
            config: Additional configuration
        """
        super().__init__(name, config)
        
        # Role management
        self.current_role: Optional[RoleType] = initial_role
        self.role_profiles: List[RoleProfile] = []
        self.dialogue_history: List[DialogueTurn] = []
        
        # Latent space representation
        self.latent_representation: Dict[str, Any] = {}
        
        # Role adaptation
        self.role_adaptations: List[Dict[str, Any]] = []
        self.adaptation_count = 0
        
        # Memory systems
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        
        # Dialogue analysis
        self.dialogue_patterns: Dict[str, Any] = {}
        self.context_awareness: Dict[str, Any] = {}
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about task considering current role.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Thinking result with role consideration
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Infer role if not set
        if self.current_role is None:
            inferred_role = self._infer_role_from_dialogue()
            if inferred_role:
                self.current_role = inferred_role
        
        # Analyze dialogue context
        dialogue_context = self._analyze_dialogue_context(task, context)
        
        # Adapt role if needed
        role_adaptation = self._adapt_role_if_needed(task, dialogue_context)
        
        result = {
            "task": task,
            "current_role": self.current_role.value if self.current_role else None,
            "dialogue_context": dialogue_context,
            "role_adaptation": role_adaptation,
            "latent_features": self._extract_latent_features(task)
        }
        
        self.state.add_step("think", result)
        return result
    
    def _infer_role_from_dialogue(self) -> Optional[RoleType]:
        """Infer role from dialogue history."""
        if not self.dialogue_history:
            return RoleType.ASSISTANT  # Default
        
        # Analyze dialogue patterns
        expert_keywords = ["explain", "analyze", "evaluate", "recommend"]
        assistant_keywords = ["help", "assist", "support", "guide"]
        collaborator_keywords = ["work together", "collaborate", "team", "joint"]
        
        dialogue_text = " ".join([turn.utterance for turn in self.dialogue_history[-10:]])
        dialogue_lower = dialogue_text.lower()
        
        if any(keyword in dialogue_lower for keyword in expert_keywords):
            return RoleType.EXPERT
        elif any(keyword in dialogue_lower for keyword in collaborator_keywords):
            return RoleType.COLLABORATOR
        elif any(keyword in dialogue_lower for keyword in assistant_keywords):
            return RoleType.ASSISTANT
        
        return RoleType.ASSISTANT
    
    def _analyze_dialogue_context(self, task: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze dialogue context."""
        context_type = DialogueContext.FORMAL
        
        # Determine context from task
        if any(word in task.lower() for word in ["please", "would you", "could you"]):
            context_type = DialogueContext.FORMAL
        elif any(word in task.lower() for word in ["hey", "can you", "help me"]):
            context_type = DialogueContext.INFORMAL
        elif any(word in task.lower() for word in ["explain", "analyze", "evaluate"]):
            context_type = DialogueContext.TECHNICAL
        
        return {
            "context_type": context_type.value,
            "formality_level": 0.7 if context_type == DialogueContext.FORMAL else 0.3,
            "dialogue_turns": len(self.dialogue_history)
        }
    
    def _adapt_role_if_needed(self, task: str, dialogue_context: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt role based on task and context."""
        adaptation = {
            "adapted": False,
            "previous_role": self.current_role.value if self.current_role else None,
            "new_role": None
        }
        
        # Adapt based on context
        if dialogue_context["context_type"] == DialogueContext.TECHNICAL:
            if self.current_role != RoleType.EXPERT:
                self.current_role = RoleType.EXPERT
                adaptation["adapted"] = True
                adaptation["new_role"] = RoleType.EXPERT.value
        elif dialogue_context["context_type"] == DialogueContext.EDUCATIONAL:
            if self.current_role != RoleType.MENTOR:
                self.current_role = RoleType.MENTOR
                adaptation["adapted"] = True
                adaptation["new_role"] = RoleType.MENTOR.value
        
        if adaptation["adapted"]:
            self.adaptation_count += 1
            self.role_adaptations.append({
                "task": task,
                "adaptation": adaptation,
                "timestamp": datetime.now().isoformat()
            })
        
        return adaptation
    
    def _extract_latent_features(self, task: str) -> Dict[str, Any]:
        """Extract latent features from task."""
        # Placeholder for latent space features
        return {
            "complexity": 0.5,
            "domain": "general",
            "interaction_style": "collaborative"
        }
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action according to current role.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        # Adapt action based on role
        role_adapted_action = self._adapt_action_to_role(action)
        
        # Execute action
        result = self._execute_role_action(role_adapted_action)
        
        self.state.add_step("act", {
            "original_action": action,
            "role_adapted_action": role_adapted_action,
            "result": result,
            "role": self.current_role.value if self.current_role else None
        })
        
        return result
    
    def _adapt_action_to_role(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt action based on current role."""
        adapted = action.copy()
        
        if self.current_role == RoleType.EXPERT:
            adapted["tone"] = "authoritative"
            adapted["detail_level"] = "high"
        elif self.current_role == RoleType.ASSISTANT:
            adapted["tone"] = "helpful"
            adapted["detail_level"] = "medium"
        elif self.current_role == RoleType.MENTOR:
            adapted["tone"] = "educational"
            adapted["detail_level"] = "high"
            adapted["encouraging"] = True
        
        return adapted
    
    def _execute_role_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action with role-specific behavior."""
        return {
            "status": "executed",
            "action": action,
            "role": self.current_role.value if self.current_role else None,
            "timestamp": datetime.now().isoformat()
        }
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation and update dialogue history.
        
        Args:
            observation: Observation data
            
        Returns:
            Processed observation
        """
        from ..common.agent_utils import standard_observe_pattern
        
        self.state.status = AgentStatus.OBSERVING
        
        # Extract dialogue from observation
        if isinstance(observation, dict) and observation.get("dialogue"):
            self._add_dialogue_turn(observation["dialogue"])
        
        # Use standard observe pattern
        return standard_observe_pattern(
            self,
            observation,
            importance=0.7,
            additional_data={
                "current_role": self.current_role.value if self.current_role else None,
                "dialogue_turns": len(self.dialogue_history)
            }
        )
    
    def _add_dialogue_turn(self, dialogue_data: Dict[str, Any]):
        """Add dialogue turn to history."""
        turn = DialogueTurn(
            turn_id=f"turn_{datetime.now().timestamp()}",
            speaker=dialogue_data.get("speaker", "user"),
            utterance=dialogue_data.get("utterance", ""),
            context=DialogueContext(dialogue_data.get("context", "formal"))
        )
        self.dialogue_history.append(turn)
        
        # Keep only recent history
        if len(self.dialogue_history) > 100:
            self.dialogue_history = self.dialogue_history[-100:]
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run task with role-aware behavior.
        
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
        
        context["current_role"] = self.current_role.value if self.current_role else None
        
        # Use standard pattern
        result = standard_run_pattern(self, task, context)
        
        # Add role-specific information
        result["role_summary"] = {
            "current_role": self.current_role.value if self.current_role else None,
            "role_adaptations": self.adaptation_count,
            "dialogue_turns": len(self.dialogue_history),
            "role_profiles": len(self.role_profiles)
        }
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "current_role": self.current_role.value if self.current_role else None,
            "role_adaptations": self.adaptation_count,
            "dialogue_turns": len(self.dialogue_history),
            "role_profiles": len(self.role_profiles)
        })
    
    def create_role_profile(self, role_type: RoleType, confidence: float = 0.8) -> RoleProfile:
        """Create a new role profile."""
        profile = RoleProfile(
            role_id=f"role_{datetime.now().timestamp()}",
            role_type=role_type,
            confidence=confidence,
            inferred_from=[turn.utterance[:50] for turn in self.dialogue_history[-5:]]
        )
        self.role_profiles.append(profile)
        return profile
    
    def get_role_analysis(self) -> Dict[str, Any]:
        """Get role analysis from dialogue."""
        return {
            "current_role": self.current_role.value if self.current_role else None,
            "role_profiles": len(self.role_profiles),
            "dialogue_history_length": len(self.dialogue_history),
            "role_adaptations": self.adaptation_count,
            "recent_adaptations": self.role_adaptations[-5:],
            "dialogue_patterns": self.dialogue_patterns
        }


