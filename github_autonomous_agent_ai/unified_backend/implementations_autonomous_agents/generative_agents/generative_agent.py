"""
Generative Agent Implementation
=================================

Paper: "Generative Agents: Interactive Simulacra of Human Behavior"
Stanford University

Key components:
1. Memory: Episodic memory for experiences, semantic memory for knowledge
2. Reflection: Periodically reflect on memories to form insights
3. Planning: Generate plans based on current situation and memories
4. Action: Execute actions based on plans
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from ..common.agent_base import BaseAgent, AgentStatus
from ..common.memory import EpisodicMemory, SemanticMemory


@dataclass
class AgentProfile:
    """Profile/characteristics of a generative agent."""
    name: str
    age: int = 25
    traits: List[str] = field(default_factory=list)
    occupation: str = ""
    relationships: Dict[str, str] = field(default_factory=dict)
    current_location: str = ""
    current_activity: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "name": self.name,
            "age": self.age,
            "traits": self.traits,
            "occupation": self.occupation,
            "relationships": self.relationships,
            "current_location": self.current_location,
            "current_activity": self.current_activity
        }


class GenerativeAgent(BaseAgent):
    """
    Generative Agent that simulates human-like behavior.
    
    Features:
    - Maintains episodic and semantic memory
    - Reflects on experiences to form insights
    - Plans activities based on memories and current situation
    - Executes actions and updates memory
    """
    
    def __init__(
        self,
        profile: AgentProfile,
        llm: Optional[Any] = None,
        reflection_threshold: int = 3,
        planning_horizon: int = 3,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize generative agent.
        
        Args:
            profile: Agent profile/characteristics
            llm: Language model
            reflection_threshold: Number of memories before reflection
            planning_horizon: Number of planned activities ahead
            config: Additional configuration
        """
        super().__init__(name=profile.name, llm=llm, config=config)
        self.profile = profile
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self.reflection_threshold = reflection_threshold
        self.planning_horizon = planning_horizon
        self.current_plan: List[Dict[str, Any]] = []
        self.insights: List[str] = []
        self.last_reflection: Optional[datetime] = None
    
    def think(self, observation: str, context: Optional[Dict] = None) -> str:
        """
        Think about current situation.
        
        Combines:
        - Current observation
        - Relevant memories
        - Current plan
        - Agent profile/traits
        """
        # Retrieve relevant memories
        relevant_memories = self.episodic_memory.retrieve(observation, top_k=5)
        
        # Build context for thinking
        context_str = self._build_thinking_context(observation, relevant_memories, context)
        
        if self.llm:
            thought = self._call_llm_for_thinking(context_str)
        else:
            thought = self._simple_thinking(observation, relevant_memories)
        
        return thought
    
    def act(self, thought: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute an action based on thought and current plan.
        
        Actions are determined by:
        - Current plan
        - Agent traits/preferences
        - Current situation
        """
        # Check if we need to plan
        if not self.current_plan:
            self._generate_plan(thought, context)
        
        # Execute next action in plan
        if self.current_plan:
            action = self.current_plan[0]
            self.current_plan = self.current_plan[1:]  # Remove executed action
            
            # Record action in memory
            self.episodic_memory.add(
                f"Executed action: {action.get('action')}",
                importance=0.7,
                metadata=action
            )
            
            return {
                "action": action.get("action"),
                "description": action.get("description"),
                "location": action.get("location"),
                "duration": action.get("duration", 60),
                "complete": False
            }
        else:
            # No plan, generate idle action
            return {
                "action": "idle",
                "description": "No planned activities",
                "complete": True
            }
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation and update memory.
        
        Also triggers reflection if threshold reached.
        
        Args:
            observation: Observation data (can be Dict or other)
            
        Returns:
            Processed observation as dictionary
        """
        from ..common.agent_utils import standard_observe_pattern
        
        self.state.status = AgentStatus.OBSERVING
        
        # Convert to dict if needed
        if isinstance(observation, dict):
            action_result = observation
        else:
            action_result = {"description": str(observation)}
        
        # Store observation in episodic memory
        observation_text = f"Action: {action_result.get('action')}, Result: {action_result.get('description')}"
        importance = 0.8 if action_result.get("action") != "idle" else 0.3
        
        self.episodic_memory.add(
            observation_text,
            importance=importance,
            metadata=action_result
        )
        
        # Update profile
        if action_result.get("location"):
            self.profile.current_location = action_result.get("location")
        if action_result.get("action"):
            self.profile.current_activity = action_result.get("action")
        
        # Check if reflection needed
        recent_memories = self.episodic_memory.get_recent(self.reflection_threshold)
        if len(recent_memories) >= self.reflection_threshold:
            if not self.last_reflection or (datetime.now() - self.last_reflection).seconds > 3600:
                self._reflect()
        
        # Use standard observe pattern
        return standard_observe_pattern(
            self,
            observation,
            importance=importance,
            additional_data={
                "observation_text": observation_text,
                "reflection_triggered": len(recent_memories) >= self.reflection_threshold,
                "current_location": self.profile.current_location,
                "current_activity": self.profile.current_activity
            }
        )
    
    def _generate_plan(self, thought: str, context: Optional[Dict] = None):
        """Generate a plan of activities."""
        # Retrieve relevant memories for planning
        relevant_memories = self.episodic_memory.retrieve(thought, top_k=10)
        
        if self.llm:
            plan = self._call_llm_for_planning(thought, relevant_memories)
        else:
            plan = self._simple_planning(thought, relevant_memories)
        
        self.current_plan = plan[:self.planning_horizon]
    
    def _reflect(self):
        """
        Reflect on recent memories to form insights.
        
        Reflection helps agents:
        - Understand patterns in their experiences
        - Form higher-level knowledge
        - Update semantic memory
        """
        recent_memories = self.episodic_memory.get_recent(self.reflection_threshold * 2)
        
        if self.llm:
            insights = self._call_llm_for_reflection(recent_memories)
        else:
            insights = self._simple_reflection(recent_memories)
        
        self.insights.extend(insights)
        self.last_reflection = datetime.now()
        
        # Store insights in semantic memory
        for i, insight in enumerate(insights):
            self.semantic_memory.add_fact(
                f"insight_{len(self.insights) - len(insights) + i}",
                insight
            )
    
    def _build_thinking_context(
        self,
        observation: str,
        memories: List,
        context: Optional[Dict]
    ) -> str:
        """Build context string for thinking."""
        context_parts = [
            f"Agent: {self.profile.name}",
            f"Traits: {', '.join(self.profile.traits)}",
            f"Location: {self.profile.current_location}",
            f"Current activity: {self.profile.current_activity}",
            "",
            "Relevant memories:",
            "\n".join([f"- {m.content}" for m in memories]),
            "",
            f"Current observation: {observation}",
            "",
            "Recent insights:",
            "\n".join([f"- {insight}" for insight in self.insights[-3:]])
        ]
        
        return "\n".join(context_parts)
    
    def _call_llm_for_thinking(self, context: str) -> str:
        """Call LLM for thinking (placeholder)."""
        return f"Based on my memories and current situation, I should consider: {context[:100]}"
    
    def _call_llm_for_planning(self, thought: str, memories: List) -> List[Dict[str, Any]]:
        """Call LLM for planning (placeholder)."""
        return [
            {"action": "work", "description": "Continue current work", "location": "office", "duration": 120},
            {"action": "lunch", "description": "Have lunch", "location": "cafeteria", "duration": 60},
            {"action": "meeting", "description": "Team meeting", "location": "conference room", "duration": 90}
        ]
    
    def _call_llm_for_reflection(self, memories: List) -> List[str]:
        """Call LLM for reflection (placeholder)."""
        return [
            "I've been working on similar tasks recently",
            "I should focus more on collaboration",
            "My productivity is higher in the morning"
        ]
    
    def _simple_thinking(self, observation: str, memories: List) -> str:
        """Simple template-based thinking."""
        return f"Considering {observation} in context of my {len(memories)} relevant memories"
    
    def _simple_planning(self, thought: str, memories: List) -> List[Dict[str, Any]]:
        """Simple template-based planning."""
        return [
            {"action": "plan_step_1", "description": "First planned activity", "location": "current", "duration": 60},
            {"action": "plan_step_2", "description": "Second planned activity", "location": "current", "duration": 60},
            {"action": "plan_step_3", "description": "Third planned activity", "location": "current", "duration": 60}
        ]
    
    def _simple_reflection(self, memories: List) -> List[str]:
        """Simple template-based reflection."""
        return [
            f"Reflecting on {len(memories)} recent experiences",
            "Noting patterns in my activities"
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "profile": self.profile.to_dict(),
            "current_plan": self.current_plan,
            "insights": self.insights,
            "memory_count": len(self.episodic_memory.memories),
            "semantic_facts": len(self.semantic_memory.facts)
        })



