"""
Theory of Mind for Autonomous Agents
======================================

Paper: "Autonomous Agents Modelling Other Agents: A Comprehensive Survey"

Theory of Mind (ToM) enables agents to:
- Model other agents' mental states
- Predict other agents' actions
- Understand beliefs, desires, and intentions
- Adapt behavior based on other agents' models
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from ..common.agent_base import BaseAgent, AgentStatus
from ..common.memory import EpisodicMemory


@dataclass
class BeliefState:
    """Represents an agent's belief about another agent's mental state."""
    agent_id: str
    beliefs: Dict[str, Any] = field(default_factory=dict)
    desires: List[str] = field(default_factory=list)
    intentions: List[str] = field(default_factory=list)
    knowledge: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update(self, new_beliefs: Dict[str, Any], confidence: float = 0.5):
        """Update belief state."""
        self.beliefs.update(new_beliefs)
        self.confidence = confidence
        self.last_updated = datetime.now()


@dataclass
class AgentModel:
    """
    Model of another agent.
    
    Contains:
    - Belief state about the agent
    - Behavioral patterns
    - Action history
    - Predicted next actions
    """
    agent_id: str
    belief_state: BeliefState
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    behavioral_patterns: Dict[str, Any] = field(default_factory=dict)
    predicted_actions: List[str] = field(default_factory=list)
    
    def add_action(self, action: Dict[str, Any]):
        """Add observed action to history."""
        self.action_history.append(action)
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
    
    def predict_next_action(self) -> Optional[str]:
        """Predict next action based on patterns."""
        if self.predicted_actions:
            return self.predicted_actions[0]
        return None


class IntentionTracker:
    """Tracks and infers intentions of other agents."""
    
    def __init__(self):
        """Initialize intention tracker."""
        self.intentions: Dict[str, List[str]] = {}
        self.intention_confidence: Dict[str, float] = {}
    
    def infer_intention(self, agent_id: str, actions: List[Dict[str, Any]]) -> List[str]:
        """
        Infer intentions from action sequence.
        
        Args:
            agent_id: ID of agent being modeled
            actions: Sequence of observed actions
            
        Returns:
            List of inferred intentions
        """
        # Simple pattern-based intention inference
        intentions = []
        
        # Analyze action patterns
        if len(actions) >= 3:
            # Check for goal-directed behavior
            if self._is_goal_directed(actions):
                intentions.append("pursuing_goal")
            
            # Check for exploration
            if self._is_exploring(actions):
                intentions.append("exploring")
            
            # Check for cooperation
            if self._is_cooperating(actions):
                intentions.append("cooperating")
        
        self.intentions[agent_id] = intentions
        return intentions
    
    def _is_goal_directed(self, actions: List[Dict[str, Any]]) -> bool:
        """Check if actions show goal-directed behavior."""
        # Simplified: check for consistent direction
        return len(actions) > 2
    
    def _is_exploring(self, actions: List[Dict[str, Any]]) -> bool:
        """Check if agent is exploring."""
        # Simplified: check for diverse actions
        unique_actions = len(set(str(a) for a in actions))
        return unique_actions > len(actions) * 0.5
    
    def _is_cooperating(self, actions: List[Dict[str, Any]]) -> bool:
        """Check if agent is cooperating."""
        # Simplified heuristic
        return False


class TheoryOfMindAgent(BaseAgent):
    """
    Agent with Theory of Mind capabilities.
    
    Can model, predict, and adapt to other agents' mental states.
    """
    
    def __init__(
        self,
        name: str = "ToMAgent",
        llm: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Theory of Mind agent.
        
        Args:
            name: Agent name
            llm: Language model for reasoning
            config: Additional configuration
        """
        super().__init__(name=name, llm=llm, config=config)
        self.agent_models: Dict[str, AgentModel] = {}
        self.intention_tracker = IntentionTracker()
        self.memory = EpisodicMemory()
    
    def observe_agent(
        self,
        other_agent_id: str,
        action: Dict[str, Any],
        context: Optional[Dict] = None
    ):
        """
        Observe another agent's action and update model.
        
        Args:
            other_agent_id: ID of observed agent
            action: Action taken by the agent
            context: Additional context
        """
        # Get or create agent model
        if other_agent_id not in self.agent_models:
            belief_state = BeliefState(agent_id=other_agent_id)
            self.agent_models[other_agent_id] = AgentModel(
                agent_id=other_agent_id,
                belief_state=belief_state
            )
        
        agent_model = self.agent_models[other_agent_id]
        
        # Add action to history
        agent_model.add_action(action)
        
        # Update belief state
        self._update_belief_state(agent_model, action, context)
        
        # Infer intentions
        intentions = self.intention_tracker.infer_intention(
            other_agent_id,
            agent_model.action_history[-10:]  # Last 10 actions
        )
        agent_model.belief_state.intentions = intentions
        
        # Predict next actions
        predicted = self._predict_actions(agent_model)
        agent_model.predicted_actions = predicted
    
    def predict_agent_action(self, other_agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Predict next action of another agent.
        
        Args:
            other_agent_id: ID of agent to predict
            
        Returns:
            Predicted action or None
        """
        if other_agent_id not in self.agent_models:
            return None
        
        agent_model = self.agent_models[other_agent_id]
        predicted_action = agent_model.predict_next_action()
        
        if predicted_action:
            return {
                "action": predicted_action,
                "confidence": agent_model.belief_state.confidence,
                "based_on": "behavioral_patterns"
            }
        
        return None
    
    def adapt_to_agent(self, other_agent_id: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Adapt own behavior based on model of another agent.
        
        Args:
            other_agent_id: ID of agent to adapt to
            context: Current context
            
        Returns:
            Adapted action
        """
        if other_agent_id not in self.agent_models:
            return {"action": "default", "reason": "no_model"}
        
        agent_model = self.agent_models[other_agent_id]
        
        # Analyze agent's intentions
        if "cooperating" in agent_model.belief_state.intentions:
            return {"action": "cooperate", "reason": "agent_is_cooperating"}
        elif "pursuing_goal" in agent_model.belief_state.intentions:
            return {"action": "support", "reason": "agent_has_goal"}
        else:
            return {"action": "observe", "reason": "uncertain_intentions"}
    
    def think(self, observation: str, context: Optional[Dict] = None) -> str:
        """
        Think about current situation, including other agents.
        
        Args:
            observation: Current observation
            context: Additional context
            
        Returns:
            Thought/reasoning
        """
        # Consider other agents in reasoning
        agent_considerations = []
        for agent_id, model in self.agent_models.items():
            predicted = self.predict_agent_action(agent_id)
            if predicted:
                agent_considerations.append(
                    f"Agent {agent_id} likely to: {predicted['action']}"
                )
        
        thought = f"Observing: {observation}"
        if agent_considerations:
            thought += f"\nOther agents: {'; '.join(agent_considerations)}"
        
        return thought
    
    def act(self, thought: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Act based on thought, considering other agents.
        
        Args:
            thought: Reasoning thought
            context: Additional context
            
        Returns:
            Action result
        """
        # Check if we need to adapt to other agents
        if context and "other_agents" in context:
            other_agent_id = context.get("other_agents", [None])[0]
            if other_agent_id:
                adapted_action = self.adapt_to_agent(other_agent_id, context)
                return {
                    "action": adapted_action["action"],
                    "reason": adapted_action["reason"],
                    "complete": False
                }
        
        # Default action
        return {
            "action": "default_action",
            "reason": thought,
            "complete": False
        }
    
    def observe(self, action_result: Dict[str, Any]) -> str:
        """Process action result."""
        return f"Action result: {action_result.get('action')}"
    
    def _update_belief_state(
        self,
        agent_model: AgentModel,
        action: Dict[str, Any],
        context: Optional[Dict]
    ):
        """Update belief state about an agent."""
        # Infer beliefs from action
        new_beliefs = {}
        
        if "goal" in str(action).lower():
            new_beliefs["has_goal"] = True
        if "cooperate" in str(action).lower():
            new_beliefs["is_cooperative"] = True
        
        if new_beliefs:
            agent_model.belief_state.update(new_beliefs, confidence=0.7)
    
    def _predict_actions(self, agent_model: AgentModel) -> List[str]:
        """Predict future actions based on patterns."""
        if len(agent_model.action_history) < 3:
            return []
        
        # Simple pattern-based prediction
        recent_actions = [a.get("action", str(a)) for a in agent_model.action_history[-3:]]
        
        # Predict continuation of pattern
        if len(recent_actions) >= 2:
            # Simple: predict similar action
            return [recent_actions[-1]]
        
        return []
    
    def get_agent_model(self, agent_id: str) -> Optional[AgentModel]:
        """Get model of a specific agent."""
        return self.agent_models.get(agent_id)
    
    def get_all_models(self) -> Dict[str, AgentModel]:
        """Get all agent models."""
        return self.agent_models.copy()



