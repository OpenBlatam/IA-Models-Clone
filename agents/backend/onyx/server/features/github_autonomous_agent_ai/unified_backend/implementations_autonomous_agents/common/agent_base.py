"""
Base Agent Class
================

Base class for all autonomous agent implementations.
Provides common functionality for state management, action execution, and observation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class AgentState:
    """Represents the current state of an agent."""
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    step_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_step(self, step_type: str, content: Any, metadata: Optional[Dict] = None):
        """Add a step to the history."""
        step = {
            "type": step_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "step": self.step_count,
            "metadata": metadata or {}
        }
        self.history.append(step)
        self.step_count += 1
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "status": self.status.value,
            "current_task": self.current_task,
            "history": self.history,
            "context": self.context,
            "step_count": self.step_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class BaseAgent(ABC):
    """
    Base class for autonomous agents.
    
    All agent implementations should inherit from this class and implement
    the abstract methods for thinking, acting, and observing.
    """
    
    def __init__(
        self,
        name: str = "BaseAgent",
        llm: Optional[Any] = None,
        tools: Optional[List[Any]] = None,
        memory: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Agent name/identifier
            llm: Language model instance (optional)
            tools: List of available tools (optional)
            memory: Memory system (optional)
            config: Additional configuration
        """
        self.name = name
        self.llm = llm
        self.tools = tools or []
        self.memory = memory
        self.config = config or {}
        self.state = AgentState()
        
    @abstractmethod
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about a task or observation.
        
        Args:
            task: Task description or observation
            context: Additional context
            
        Returns:
            Thinking result as dictionary
        """
        pass
    
    @abstractmethod
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action.
        
        Args:
            action: Action dictionary with 'type' and other parameters
            
        Returns:
            Action result dictionary
        """
        pass
    
    @abstractmethod
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation.
        
        Args:
            observation: Observation data (can be dict, string, or other)
            
        Returns:
            Processed observation as dictionary
        """
        pass
    
    def run(self, task: str, max_steps: int = 10, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the agent on a task.
        
        This is a default implementation that can be overridden by subclasses.
        Many agents override this with their own implementation.
        
        Args:
            task: Task description
            max_steps: Maximum number of reasoning-action-observation cycles
            context: Optional context
            
        Returns:
            Final result dictionary
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        self.state.add_step("task_start", {"task": task})
        
        # Use standard pattern if available
        try:
            from .agent_utils import standard_run_pattern
            return standard_run_pattern(self, task, context)
        except ImportError:
            # Fallback to basic implementation
            pass
        
        # Basic implementation
        thinking = self.think(task, context)
        action = {"type": "execute"}
        action_result = self.act(action)
        observation = {}
        observation_result = self.observe(observation)
        
        self.state.status = AgentStatus.COMPLETED
        
        return {
            "task": task,
            "thinking": thinking,
            "action": action_result,
            "observation": observation_result,
            "final_state": self.state.to_dict(),
            "completed": self.state.status == AgentStatus.COMPLETED
        }
    
    def _is_complete(self, observation: Any, action_result: Dict[str, Any]) -> bool:
        """
        Determine if the task is complete.
        Can be overridden by subclasses for custom completion logic.
        """
        # Default: check if action result indicates completion
        if isinstance(observation, dict):
            return observation.get("complete", False) or action_result.get("complete", False)
        elif isinstance(observation, str):
            return "complete" in observation.lower() or action_result.get("complete", False)
        return action_result.get("complete", False)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state."""
        return self.state.to_dict()
    
    def reset(self):
        """Reset agent state."""
        self.state = AgentState()



