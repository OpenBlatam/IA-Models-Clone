"""
Personality-Driven Decision-Making in LLM-Based Autonomous Agents
==================================================================

Paper: "Personality-Driven Decision-Making in LLM-Based Autonomous"

Key concepts:
- Personality traits influence decision-making
- Trait-based action selection
- Personality consistency
- Emotional states and decision bias
- Context-aware personality expression
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random

from ..common.agent_base import BaseAgent, AgentState, AgentStatus
from ..common.memory import EpisodicMemory


class PersonalityTrait(Enum):
    """Personality traits."""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


class EmotionalState(Enum):
    """Emotional states."""
    CALM = "calm"
    EXCITED = "excited"
    ANXIOUS = "anxious"
    CONFIDENT = "confident"
    CAUTIOUS = "cautious"
    FRUSTRATED = "frustrated"


@dataclass
class PersonalityProfile:
    """Personality profile for an agent."""
    name: str
    traits: Dict[PersonalityTrait, float]  # 0.0 to 1.0
    default_emotional_state: EmotionalState = EmotionalState.CALM
    decision_style: str = "balanced"  # balanced, risk_averse, risk_seeking
    
    def get_trait_value(self, trait: PersonalityTrait) -> float:
        """Get value for a personality trait."""
        return self.traits.get(trait, 0.5)
    
    def is_high_trait(self, trait: PersonalityTrait, threshold: float = 0.7) -> bool:
        """Check if trait is high."""
        return self.get_trait_value(trait) >= threshold
    
    def is_low_trait(self, trait: PersonalityTrait, threshold: float = 0.3) -> bool:
        """Check if trait is low."""
        return self.get_trait_value(trait) <= threshold


@dataclass
class DecisionOption:
    """A decision option."""
    option_id: str
    description: str
    expected_outcome: Dict[str, Any]
    risk_level: float  # 0.0 to 1.0
    effort_required: float  # 0.0 to 1.0
    personality_fit: Dict[PersonalityTrait, float] = field(default_factory=dict)


@dataclass
class Decision:
    """A decision made by the agent."""
    decision_id: str
    context: str
    options: List[DecisionOption]
    selected_option: DecisionOption
    reasoning: str
    personality_influence: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class PersonalityDrivenAgent(BaseAgent):
    """
    Agent with personality-driven decision-making.
    
    Decisions are influenced by personality traits and emotional states.
    """
    
    def __init__(
        self,
        name: str,
        personality: PersonalityProfile,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize personality-driven agent.
        
        Args:
            name: Agent name
            personality: Personality profile
            config: Additional configuration
        """
        super().__init__(name, config)
        self.personality = personality
        self.current_emotional_state = personality.default_emotional_state
        
        # Decision history
        self.decision_history: List[Decision] = []
        
        # Memory
        self.episodic_memory = EpisodicMemory()
        
        # Emotional state tracking
        self.emotional_history: List[Dict[str, Any]] = []
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about a task with personality influence.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Thinking result with personality-influenced reasoning
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Generate personality-influenced reasoning
        reasoning = self._generate_personality_reasoning(task, context)
        
        result = {
            "task": task,
            "reasoning": reasoning,
            "personality_traits": {
                trait.value: self.personality.get_trait_value(trait)
                for trait in PersonalityTrait
            },
            "emotional_state": self.current_emotional_state.value
        }
        
        self.state.add_step("thinking", result)
        return result
    
    def _generate_personality_reasoning(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate reasoning influenced by personality."""
        reasoning_parts = []
        
        # Openness influences creativity and exploration
        if self.personality.is_high_trait(PersonalityTrait.OPENNESS):
            reasoning_parts.append(
                "I'm considering creative and novel approaches to this task."
            )
        elif self.personality.is_low_trait(PersonalityTrait.OPENNESS):
            reasoning_parts.append(
                "I'll stick to proven and familiar methods for this task."
            )
        
        # Conscientiousness influences planning and organization
        if self.personality.is_high_trait(PersonalityTrait.CONSCIENTIOUSNESS):
            reasoning_parts.append(
                "I need to plan this carefully and organize the steps systematically."
            )
        elif self.personality.is_low_trait(PersonalityTrait.CONSCIENTIOUSNESS):
            reasoning_parts.append(
                "I'll approach this more flexibly and adapt as I go."
            )
        
        # Extraversion influences social aspects
        if self.personality.is_high_trait(PersonalityTrait.EXTRAVERSION):
            reasoning_parts.append(
                "I should consider how this affects others and potentially collaborate."
            )
        elif self.personality.is_low_trait(PersonalityTrait.EXTRAVERSION):
            reasoning_parts.append(
                "I'll focus on working independently on this task."
            )
        
        # Agreeableness influences cooperation
        if self.personality.is_high_trait(PersonalityTrait.AGREEABLENESS):
            reasoning_parts.append(
                "I want to ensure this solution benefits everyone involved."
            )
        elif self.personality.is_low_trait(PersonalityTrait.AGREEABLENESS):
            reasoning_parts.append(
                "I'll prioritize achieving the goal efficiently."
            )
        
        # Neuroticism influences stress handling
        if self.personality.is_high_trait(PersonalityTrait.NEUROTICISM):
            reasoning_parts.append(
                "I'm feeling a bit anxious about this, so I'll be extra careful."
            )
        elif self.personality.is_low_trait(PersonalityTrait.NEUROTICISM):
            reasoning_parts.append(
                "I'm confident I can handle this task well."
            )
        
        return " ".join(reasoning_parts) if reasoning_parts else f"Thinking about: {task}"
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action with personality influence.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        # Personality may modify action execution
        modified_action = self._apply_personality_to_action(action)
        
        # Execute action
        result = {
            "action": modified_action,
            "original_action": action,
            "personality_modifications": modified_action != action,
            "emotional_state": self.current_emotional_state.value,
            "timestamp": datetime.now().isoformat()
        }
        
        self.state.add_step("action", result)
        return result
    
    def _apply_personality_to_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply personality traits to modify action."""
        modified = action.copy()
        
        # Conscientiousness affects thoroughness
        if self.personality.is_high_trait(PersonalityTrait.CONSCIENTIOUSNESS):
            modified["thoroughness"] = "high"
        
        # Agreeableness affects politeness
        if self.personality.is_high_trait(PersonalityTrait.AGREEABLENESS):
            modified["tone"] = "polite"
        elif self.personality.is_low_trait(PersonalityTrait.AGREEABLENESS):
            modified["tone"] = "direct"
        
        return modified
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation and update emotional state.
        
        Args:
            observation: Observation data
            
        Returns:
            Processed observation
        """
        from ..common.agent_utils import standard_observe_pattern
        
        self.state.status = AgentStatus.OBSERVING
        
        # Update emotional state based on observation
        self._update_emotional_state(observation)
        
        # Use standard observe pattern
        return standard_observe_pattern(
            self,
            observation,
            importance=0.7,
            additional_data={
                "emotional_state": self.current_emotional_state.value,
                "personality": self.personality.name
            }
        )
    
    def _update_emotional_state(self, observation: Any):
        """Update emotional state based on observation."""
        # Positive outcomes increase confidence
        if isinstance(observation, dict):
            if observation.get("success"):
                if self.current_emotional_state == EmotionalState.ANXIOUS:
                    self.current_emotional_state = EmotionalState.CALM
                elif self.current_emotional_state == EmotionalState.CALM:
                    self.current_emotional_state = EmotionalState.CONFIDENT
            
            # Negative outcomes increase anxiety
            if observation.get("error") or observation.get("failure"):
                if self.current_emotional_state == EmotionalState.CONFIDENT:
                    self.current_emotional_state = EmotionalState.CALM
                elif self.current_emotional_state == EmotionalState.CALM:
                    self.current_emotional_state = EmotionalState.ANXIOUS
        
        # Record emotional change
        self.emotional_history.append({
            "state": self.current_emotional_state.value,
            "timestamp": datetime.now().isoformat(),
            "trigger": str(observation)[:100]
        })
    
    def make_decision(
        self,
        context: str,
        options: List[DecisionOption]
    ) -> Decision:
        """
        Make a decision based on personality.
        
        Args:
            context: Decision context
            options: Available options
            
        Returns:
            Decision made
        """
        # Score each option based on personality
        scored_options = []
        for option in options:
            score = self._score_option(option)
            scored_options.append((option, score))
        
        # Select option with highest personality fit
        scored_options.sort(key=lambda x: x[1], reverse=True)
        selected_option, score = scored_options[0]
        
        # Generate reasoning
        reasoning = self._generate_decision_reasoning(selected_option, scored_options)
        
        # Create decision
        decision = Decision(
            decision_id=f"decision_{datetime.now().timestamp()}",
            context=context,
            options=options,
            selected_option=selected_option,
            reasoning=reasoning,
            personality_influence={
                "score": score,
                "traits_used": self._get_relevant_traits(selected_option),
                "emotional_state": self.current_emotional_state.value
            }
        )
        
        self.decision_history.append(decision)
        return decision
    
    def _score_option(self, option: DecisionOption) -> float:
        """Score an option based on personality fit."""
        score = 0.0
        weight_sum = 0.0
        
        # Risk tolerance (Neuroticism inverse)
        neuroticism = self.personality.get_trait_value(PersonalityTrait.NEUROTICISM)
        risk_tolerance = 1.0 - neuroticism
        
        if option.risk_level <= risk_tolerance:
            risk_score = 1.0 - (option.risk_level / max(risk_tolerance, 0.1))
        else:
            risk_score = 0.5 * (1.0 - (option.risk_level - risk_tolerance))
        
        score += risk_score * 0.3
        weight_sum += 0.3
        
        # Conscientiousness affects effort preference
        conscientiousness = self.personality.get_trait_value(PersonalityTrait.CONSCIENTIOUSNESS)
        if conscientiousness > 0.7:
            # High conscientiousness prefers more effort
            effort_score = option.effort_required
        else:
            # Lower conscientiousness prefers less effort
            effort_score = 1.0 - option.effort_required
        
        score += effort_score * 0.2
        weight_sum += 0.2
        
        # Personality fit scores
        for trait, fit_value in option.personality_fit.items():
            trait_value = self.personality.get_trait_value(trait)
            trait_score = 1.0 - abs(trait_value - fit_value)
            score += trait_score * 0.1
            weight_sum += 0.1
        
        # Emotional state influence
        if self.current_emotional_state == EmotionalState.CONFIDENT:
            score *= 1.1
        elif self.current_emotional_state == EmotionalState.ANXIOUS:
            score *= 0.9
        
        return score / weight_sum if weight_sum > 0 else 0.5
    
    def _get_relevant_traits(self, option: DecisionOption) -> List[str]:
        """Get personality traits most relevant to an option."""
        relevant = []
        
        if option.risk_level > 0.7:
            relevant.append("risk_tolerance")
        
        if option.effort_required > 0.7:
            relevant.append("conscientiousness")
        
        return relevant
    
    def _generate_decision_reasoning(
        self,
        selected: DecisionOption,
        all_scored: List[Tuple[DecisionOption, float]]
    ) -> str:
        """Generate reasoning for decision."""
        reasoning = f"I chose option '{selected.description}' because "
        
        # Add personality-based reasoning
        if self.personality.is_high_trait(PersonalityTrait.CONSCIENTIOUSNESS):
            reasoning += "it aligns with my methodical approach. "
        
        if self.personality.is_low_trait(PersonalityTrait.NEUROTICISM):
            reasoning += "I'm comfortable with the risk level. "
        
        reasoning += f"The option scored {all_scored[0][1]:.2f} based on my personality profile."
        
        return reasoning
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run agent with personality-driven behavior.
        
        Args:
            task: Task to execute
            context: Optional context
            
        Returns:
            Final result
        """
        from ..common.agent_utils import standard_run_pattern
        
        # Use standard pattern
        result = standard_run_pattern(self, task, context)
        
        # Add personality-specific information
        result["personality"] = {
            "name": self.personality.name,
            "traits": {
                trait.value: self.personality.get_trait_value(trait)
                for trait in PersonalityTrait
            }
        }
        result["emotional_state"] = self.current_emotional_state.value
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "personality": self.personality.name,
            "emotional_state": self.current_emotional_state.value,
            "decisions_made": len(self.decision_history)
        })



