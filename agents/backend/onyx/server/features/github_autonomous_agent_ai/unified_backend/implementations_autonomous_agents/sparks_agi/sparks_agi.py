"""
Sparks of Artificial General Intelligence
=========================================

Paper: "Sparks of Artificial General Intelligence:"

Key concepts:
- AGI capabilities and emergence
- Multi-modal understanding
- Reasoning and planning
- Tool use and manipulation
- Code generation and execution
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus


class AGICapability(Enum):
    """AGI capabilities."""
    MULTI_MODAL_UNDERSTANDING = "multi_modal_understanding"
    REASONING = "reasoning"
    PLANNING = "planning"
    TOOL_USE = "tool_use"
    CODE_GENERATION = "code_generation"
    CREATIVE_TASKS = "creative_tasks"
    PROBLEM_SOLVING = "problem_solving"


class EmergenceLevel(Enum):
    """Levels of emergent behavior."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    AGI_LIKE = "agi_like"


@dataclass
class AGITask:
    """An AGI task."""
    task_id: str
    task_type: str
    description: str
    required_capabilities: List[AGICapability]
    complexity: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AGIPerformance:
    """Performance metrics for AGI capabilities."""
    capability: AGICapability
    performance_score: float
    emergence_level: EmergenceLevel
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


class SparksAGIAgent(BaseAgent):
    """
    Agent demonstrating sparks of AGI capabilities.
    
    Implements various AGI-like behaviors and capabilities.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize AGI agent.
        
        Args:
            name: Agent name
            config: Configuration parameters
        """
        super().__init__(name, config)
        
        # AGI components
        self.capabilities: Dict[AGICapability, float] = {}
        self.performance_history: List[AGIPerformance] = []
        self.task_history: List[AGITask] = []
        
        # Initialize capabilities
        self._initialize_capabilities()
        
        # AGI parameters
        self.reasoning_depth = config.get("reasoning_depth", 5)
        self.planning_horizon = config.get("planning_horizon", 10)
    
    def _initialize_capabilities(self):
        """Initialize AGI capabilities."""
        for capability in AGICapability:
            # Initialize with base level
            self.capabilities[capability] = 0.5
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about task using AGI capabilities.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Thinking result with AGI reasoning
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Analyze task requirements
        required_capabilities = self._analyze_task_requirements(task)
        
        # Multi-step reasoning
        reasoning_steps = self._multi_step_reasoning(task, context)
        
        # Generate plan
        plan = self._generate_plan(task, reasoning_steps)
        
        result = {
            "task": task,
            "required_capabilities": [c.value for c in required_capabilities],
            "reasoning_steps": reasoning_steps,
            "plan": plan,
            "agi_confidence": self._calculate_agi_confidence(required_capabilities),
            "reasoning": f"Applied AGI capabilities: {[c.value for c in required_capabilities]}"
        }
        
        self.state.add_step("thinking", result)
        return result
    
    def _analyze_task_requirements(self, task: str) -> List[AGICapability]:
        """Analyze what capabilities are needed for task."""
        required = []
        
        task_lower = task.lower()
        
        if any(word in task_lower for word in ["image", "visual", "see", "picture"]):
            required.append(AGICapability.MULTI_MODAL_UNDERSTANDING)
        
        if any(word in task_lower for word in ["why", "reason", "explain", "because"]):
            required.append(AGICapability.REASONING)
        
        if any(word in task_lower for word in ["plan", "steps", "sequence", "order"]):
            required.append(AGICapability.PLANNING)
        
        if any(word in task_lower for word in ["use", "tool", "api", "call"]):
            required.append(AGICapability.TOOL_USE)
        
        if any(word in task_lower for word in ["code", "program", "function", "script"]):
            required.append(AGICapability.CODE_GENERATION)
        
        if any(word in task_lower for word in ["create", "generate", "design", "write"]):
            required.append(AGICapability.CREATIVE_TASKS)
        
        # Default: problem solving
        if not required:
            required.append(AGICapability.PROBLEM_SOLVING)
        
        return required
    
    def _multi_step_reasoning(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Perform multi-step reasoning."""
        steps = []
        
        # Step 1: Understand task
        steps.append({
            "step": 1,
            "type": "understanding",
            "content": f"Understanding task: {task}",
            "capability": AGICapability.MULTI_MODAL_UNDERSTANDING.value
        })
        
        # Step 2: Break down
        steps.append({
            "step": 2,
            "type": "decomposition",
            "content": "Breaking down task into sub-tasks",
            "capability": AGICapability.REASONING.value
        })
        
        # Step 3: Plan
        steps.append({
            "step": 3,
            "type": "planning",
            "content": "Creating execution plan",
            "capability": AGICapability.PLANNING.value
        })
        
        return steps
    
    def _generate_plan(
        self,
        task: str,
        reasoning_steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate execution plan."""
        return {
            "plan_id": f"plan_{datetime.now().timestamp()}",
            "steps": [
                {"action": "analyze", "description": "Analyze task requirements"},
                {"action": "reason", "description": "Apply reasoning"},
                {"action": "execute", "description": "Execute solution"},
                {"action": "verify", "description": "Verify results"}
            ],
            "estimated_complexity": len(reasoning_steps)
        }
    
    def _calculate_agi_confidence(self, capabilities: List[AGICapability]) -> float:
        """Calculate confidence in AGI capabilities."""
        if not capabilities:
            return 0.0
        
        # Average capability scores
        avg_score = sum(self.capabilities.get(cap, 0.5) for cap in capabilities) / len(capabilities)
        return avg_score
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action using AGI capabilities.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        action_type = action.get("type", "execute")
        
        # Execute with AGI capabilities
        result = {
            "action": action,
            "action_type": action_type,
            "status": "executed",
            "agi_capabilities_used": action.get("capabilities", []),
            "timestamp": datetime.now().isoformat()
        }
        
        self.state.add_step("action", result)
        return result
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation and update capabilities.
        
        Args:
            observation: Observation data
            
        Returns:
            Processed observation
        """
        self.state.status = AgentStatus.OBSERVING
        
        # Update capabilities based on performance
        if isinstance(observation, dict) and "performance" in observation:
            self._update_capabilities(observation["performance"])
        
        processed = {
            "observation": observation,
            "current_capabilities": {k.value: v for k, v in self.capabilities.items()},
            "timestamp": datetime.now().isoformat()
        }
        
        self.state.add_step("observation", processed)
        return processed
    
    def _update_capabilities(self, performance_data: Dict[str, Any]):
        """Update capabilities based on performance."""
        for capability_str, score in performance_data.items():
            try:
                capability = AGICapability(capability_str)
                # Update with learning rate
                learning_rate = 0.1
                self.capabilities[capability] = (
                    self.capabilities.get(capability, 0.5) * (1 - learning_rate) +
                    score * learning_rate
                )
            except ValueError:
                continue
    
    def evaluate_capability(self, capability: AGICapability) -> AGIPerformance:
        """
        Evaluate a specific AGI capability.
        
        Args:
            capability: Capability to evaluate
            
        Returns:
            Performance metrics
        """
        score = self.capabilities.get(capability, 0.5)
        
        # Determine emergence level
        if score >= 0.9:
            level = EmergenceLevel.AGI_LIKE
        elif score >= 0.7:
            level = EmergenceLevel.ADVANCED
        elif score >= 0.5:
            level = EmergenceLevel.INTERMEDIATE
        else:
            level = EmergenceLevel.BASIC
        
        performance = AGIPerformance(
            capability=capability,
            performance_score=score,
            emergence_level=level,
            metrics={
                "score": score,
                "level": level.value
            }
        )
        
        self.performance_history.append(performance)
        return performance
    
    def get_agi_status(self) -> Dict[str, Any]:
        """Get overall AGI status."""
        avg_capability = sum(self.capabilities.values()) / len(self.capabilities) if self.capabilities else 0.0
        
        return {
            "name": self.name,
            "average_capability": avg_capability,
            "capabilities": {k.value: v for k, v in self.capabilities.items()},
            "tasks_completed": len(self.task_history),
            "performance_evaluations": len(self.performance_history)
        }
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run AGI task.
        
        Args:
            task: Task description
            context: Optional context
            
        Returns:
            Final result
        """
        from ..common.agent_utils import standard_run_pattern, create_status_dict
        
        # Use standard pattern
        result = standard_run_pattern(self, task, context)
        
        # Add AGI-specific information
        result["agi_status"] = self.get_agi_status()
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {"agi_status": self.get_agi_status()})



