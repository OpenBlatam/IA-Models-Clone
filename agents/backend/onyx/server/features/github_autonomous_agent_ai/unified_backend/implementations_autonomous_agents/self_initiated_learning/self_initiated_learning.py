"""
AI Autonomy: Self-Initiated Open-World Continual Learning
==========================================================

Paper: "AI Autonomy: Self-Initiated Open-World Continual Learning and Adaptation"

Key concepts:
- Self-initiated learning
- Open-world adaptation
- Continual learning
- Autonomous task discovery
- Self-directed exploration
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus
from ..common.memory import EpisodicMemory, SemanticMemory


class LearningTrigger(Enum):
    """Triggers for self-initiated learning."""
    CURIOSITY = "curiosity"
    PERFORMANCE_DROP = "performance_drop"
    NOVEL_ENCOUNTER = "novel_encounter"
    TASK_COMPLETION = "task_completion"
    ERROR_DETECTION = "error_detection"


@dataclass
class LearningTask:
    """A self-initiated learning task."""
    task_id: str
    trigger: LearningTrigger
    description: str
    priority: float
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class LearningExperience:
    """A learning experience."""
    experience_id: str
    task_id: str
    knowledge_gained: Dict[str, Any]
    performance_improvement: float
    timestamp: datetime = field(default_factory=datetime.now)


class SelfInitiatedLearningAgent(BaseAgent):
    """
    Agent with self-initiated continual learning capabilities.
    
    Can autonomously identify learning opportunities and adapt.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize self-initiated learning agent.
        
        Args:
            name: Agent name
            config: Configuration parameters
        """
        super().__init__(name, config)
        
        # Memory systems
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        
        # Learning tasks
        self.learning_tasks: List[LearningTask] = []
        self.active_learning_task: Optional[LearningTask] = None
        
        # Learning experiences
        self.learning_experiences: List[LearningExperience] = []
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.baseline_performance: float = 0.5
        
        # Learning parameters
        self.curiosity_threshold = config.get("curiosity_threshold", 0.7)
        self.performance_drop_threshold = config.get("performance_drop_threshold", 0.2)
        self.learning_rate = config.get("learning_rate", 0.1)
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about task and identify learning opportunities.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Thinking result with potential learning opportunities
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Check for learning triggers
        learning_opportunities = self._identify_learning_opportunities(task, context)
        
        # Create learning tasks if opportunities found
        for opportunity in learning_opportunities:
            self._create_learning_task(opportunity)
        
        result = {
            "task": task,
            "learning_opportunities": len(learning_opportunities),
            "active_learning_tasks": len([t for t in self.learning_tasks if t.status == "active"]),
            "reasoning": f"Analyzing task: {task}. Found {len(learning_opportunities)} learning opportunities."
        }
        
        self.state.add_step("thinking", result)
        return result
    
    def _identify_learning_opportunities(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify learning opportunities."""
        opportunities = []
        
        # Check for novel encounters
        if self._is_novel_task(task):
            opportunities.append({
                "trigger": LearningTrigger.NOVEL_ENCOUNTER,
                "description": f"Novel task encountered: {task}",
                "priority": 0.8
            })
        
        # Check performance
        current_performance = self._evaluate_performance()
        if current_performance < self.baseline_performance - self.performance_drop_threshold:
            opportunities.append({
                "trigger": LearningTrigger.PERFORMANCE_DROP,
                "description": f"Performance dropped to {current_performance:.2f}",
                "priority": 0.9
            })
        
        # Check curiosity (unexplored areas)
        if self._has_curiosity_trigger(task):
            opportunities.append({
                "trigger": LearningTrigger.CURIOSITY,
                "description": f"Curious about: {task}",
                "priority": 0.6
            })
        
        return opportunities
    
    def _is_novel_task(self, task: str) -> bool:
        """Check if task is novel."""
        # Check semantic memory for similar tasks
        similar_tasks = self.semantic_memory.retrieve(
            query=task,
            top_k=1
        )
        
        return len(similar_tasks) == 0
    
    def _evaluate_performance(self) -> float:
        """Evaluate current performance."""
        if not self.performance_history:
            return self.baseline_performance
        
        # Average of recent performance
        recent_performance = self.performance_history[-10:]
        if recent_performance:
            return sum(p.get("score", 0.5) for p in recent_performance) / len(recent_performance)
        
        return self.baseline_performance
    
    def _has_curiosity_trigger(self, task: str) -> bool:
        """Check if curiosity should trigger learning."""
        # Simplified: check if task contains unknown concepts
        known_concepts = self.semantic_memory.get_all_facts()
        
        # If task mentions concepts not in memory, trigger curiosity
        task_words = set(task.lower().split())
        known_words = set()
        for fact in known_concepts:
            known_words.update(fact.get("content", "").lower().split())
        
        unknown_words = task_words - known_words
        return len(unknown_words) > 3  # Threshold for curiosity
    
    def _create_learning_task(self, opportunity: Dict[str, Any]):
        """Create a learning task from opportunity."""
        task = LearningTask(
            task_id=f"learn_{datetime.now().timestamp()}",
            trigger=opportunity["trigger"],
            description=opportunity["description"],
            priority=opportunity["priority"]
        )
        
        self.learning_tasks.append(task)
        
        # Activate if high priority
        if task.priority > 0.8 and not self.active_learning_task:
            self.active_learning_task = task
            task.status = "active"
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action, potentially incorporating learning.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        # Execute action
        result = {
            "action": action,
            "status": "executed",
            "timestamp": datetime.now().isoformat()
        }
        
        # If learning task is active, incorporate learning
        if self.active_learning_task:
            learning_result = self._incorporate_learning(action, result)
            result["learning"] = learning_result
        
        self.state.add_step("action", result)
        return result
    
    def _incorporate_learning(
        self,
        action: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Incorporate learning from action."""
        if not self.active_learning_task:
            return {}
        
        # Learn from action and result
        knowledge = {
            "action_type": action.get("type"),
            "outcome": result.get("status"),
            "context": action.get("context", {})
        }
        
        # Store in semantic memory
        self.semantic_memory.add_fact(
            content=str(knowledge),
            relationships={"task": self.active_learning_task.description}
        )
        
        # Update performance
        performance_improvement = self.learning_rate * 0.1  # Small improvement
        self.baseline_performance += performance_improvement
        self.baseline_performance = min(1.0, self.baseline_performance)
        
        # Create learning experience
        experience = LearningExperience(
            experience_id=f"exp_{datetime.now().timestamp()}",
            task_id=self.active_learning_task.task_id,
            knowledge_gained=knowledge,
            performance_improvement=performance_improvement
        )
        self.learning_experiences.append(experience)
        
        # Complete learning task if sufficient learning occurred
        if len(self.learning_experiences) >= 3:
            self.active_learning_task.status = "completed"
            self.active_learning_task.completed_at = datetime.now()
            self.active_learning_task = None
        
        return {
            "knowledge_gained": knowledge,
            "performance_improvement": performance_improvement
        }
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation and update learning.
        
        Args:
            observation: Observation data
            
        Returns:
            Processed observation
        """
        from ..common.agent_utils import standard_observe_pattern
        
        self.state.status = AgentStatus.OBSERVING
        
        # Check for errors that trigger learning
        if isinstance(observation, dict) and observation.get("error"):
            self._create_learning_task({
                "trigger": LearningTrigger.ERROR_DETECTION,
                "description": f"Error detected: {observation['error']}",
                "priority": 0.9
            })
        
        # Update performance history
        if isinstance(observation, dict) and "performance" in observation:
            self.performance_history.append({
                "score": observation["performance"],
                "timestamp": datetime.now().isoformat()
            })
        
        # Use standard observe pattern
        return standard_observe_pattern(
            self,
            observation,
            importance=0.7,
            additional_data={
                "learning_active": self.active_learning_task is not None,
                "learning_status": self.get_learning_status()
            }
        )
    
    def initiate_learning(self, trigger: LearningTrigger, description: str):
        """
        Manually initiate a learning task.
        
        Args:
            trigger: Learning trigger
            description: Task description
        """
        opportunity = {
            "trigger": trigger,
            "description": description,
            "priority": 0.8
        }
        self._create_learning_task(opportunity)
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status."""
        return {
            "active_learning_task": self.active_learning_task.description if self.active_learning_task else None,
            "pending_tasks": len([t for t in self.learning_tasks if t.status == "pending"]),
            "completed_tasks": len([t for t in self.learning_tasks if t.status == "completed"]),
            "learning_experiences": len(self.learning_experiences),
            "baseline_performance": self.baseline_performance,
            "current_performance": self._evaluate_performance()
        }
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run agent with self-initiated learning.
        
        Args:
            task: Task to execute
            context: Optional context
            
        Returns:
            Final result
        """
        from ..common.agent_utils import standard_run_pattern
        
        # Use standard pattern
        result = standard_run_pattern(self, task, context)
        
        # Add learning-specific information
        result["learning_status"] = self.get_learning_status()
        
        return result



