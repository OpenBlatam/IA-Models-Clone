"""
From LLM Reasoning to Autonomous AI Agents
===========================================

Paper: "From LLM Reasoning to Autonomous AI Agents"

Key concepts:
- Transition from reasoning to autonomous action
- Progressive autonomy levels
- Reasoning-to-action pipeline
- Goal decomposition and planning
- Self-monitoring and adaptation
"""

from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from ..common.agent_base import BaseAgent, AgentState, AgentStatus
from ..common.memory import EpisodicMemory, SemanticMemory
from ..common.tools import ToolRegistry


class AutonomyLevel(Enum):
    """Levels of agent autonomy."""
    REASONING_ONLY = "reasoning_only"  # Only generates reasoning, no actions
    ASSISTED = "assisted"  # Requires human approval for actions
    SEMI_AUTONOMOUS = "semi_autonomous"  # Can act with constraints
    FULLY_AUTONOMOUS = "fully_autonomous"  # Complete autonomy


@dataclass
class ReasoningStep:
    """A single reasoning step."""
    step_id: str
    reasoning: str
    confidence: float
    dependencies: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ActionPlan:
    """Plan of actions derived from reasoning."""
    plan_id: str
    goal: str
    steps: List[Dict[str, Any]]
    reasoning_chain: List[ReasoningStep]
    confidence: float
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Goal:
    """Agent goal representation."""
    goal_id: str
    description: str
    priority: int
    status: str = "pending"
    sub_goals: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class LLMToAutonomousAgent(BaseAgent):
    """
    Agent that transitions from LLM reasoning to autonomous action.
    
    Implements progressive autonomy with reasoning-to-action pipeline.
    """
    
    def __init__(
        self,
        name: str,
        autonomy_level: AutonomyLevel = AutonomyLevel.SEMI_AUTONOMOUS,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize LLM-to-Autonomous agent.
        
        Args:
            name: Agent name
            autonomy_level: Level of autonomy
            tool_registry: Registry of available tools
            config: Additional configuration
        """
        super().__init__(name, config)
        self.autonomy_level = autonomy_level
        self.tool_registry = tool_registry or ToolRegistry()
        
        # Reasoning and planning components
        self.reasoning_chain: List[ReasoningStep] = []
        self.current_plan: Optional[ActionPlan] = None
        self.goals: List[Goal] = []
        
        # Memory systems
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        
        # Monitoring
        self.action_history: List[Dict[str, Any]] = []
        self.success_rate: float = 0.0
        self.adaptation_history: List[Dict[str, Any]] = []
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate reasoning for a task.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Reasoning result with steps and confidence
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Generate reasoning steps
        reasoning_steps = self._generate_reasoning_steps(task, context)
        self.reasoning_chain.extend(reasoning_steps)
        
        # Evaluate reasoning quality
        confidence = self._evaluate_reasoning_quality(reasoning_steps)
        
        result = {
            "reasoning_steps": [
                {
                    "step_id": step.step_id,
                    "reasoning": step.reasoning,
                    "confidence": step.confidence
                }
                for step in reasoning_steps
            ],
            "overall_confidence": confidence,
            "task": task
        }
        
        self.state.add_step("reasoning", result)
        return result
    
    def _generate_reasoning_steps(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> List[ReasoningStep]:
        """Generate reasoning steps for a task."""
        steps = []
        
        # Step 1: Understand task
        step1 = ReasoningStep(
            step_id="understand",
            reasoning=f"Understanding task: {task}. Analyzing requirements and constraints.",
            confidence=0.8
        )
        steps.append(step1)
        
        # Step 2: Decompose goal
        step2 = ReasoningStep(
            step_id="decompose",
            reasoning=f"Decomposing task into sub-goals and action steps.",
            confidence=0.75,
            dependencies=[step1.step_id]
        )
        steps.append(step2)
        
        # Step 3: Plan actions
        step3 = ReasoningStep(
            step_id="plan",
            reasoning=f"Creating action plan based on decomposed goals.",
            confidence=0.7,
            dependencies=[step2.step_id]
        )
        steps.append(step3)
        
        # Step 4: Evaluate feasibility
        step4 = ReasoningStep(
            step_id="evaluate",
            reasoning=f"Evaluating plan feasibility and resource requirements.",
            confidence=0.75,
            dependencies=[step3.step_id]
        )
        steps.append(step4)
        
        return steps
    
    def _evaluate_reasoning_quality(self, steps: List[ReasoningStep]) -> float:
        """Evaluate overall quality of reasoning."""
        if not steps:
            return 0.0
        
        avg_confidence = sum(step.confidence for step in steps) / len(steps)
        
        # Check for logical dependencies
        dependency_score = 1.0 if all(
            step.dependencies for step in steps[1:]
        ) else 0.8
        
        return (avg_confidence + dependency_score) / 2
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action based on reasoning.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        # Check autonomy level
        if self.autonomy_level == AutonomyLevel.REASONING_ONLY:
            return {
                "status": "blocked",
                "reason": "Agent is in reasoning-only mode",
                "action": action
            }
        
        # Check if action requires approval
        if self.autonomy_level == AutonomyLevel.ASSISTED:
            # In real implementation, would request human approval
            pass
        
        # Execute action
        action_type = action.get("type", "unknown")
        action_params = action.get("params", {})
        
        if action_type == "tool_call":
            tool_name = action_params.get("tool")
            tool_args = action_params.get("args", {})
            
            if self.tool_registry.has_tool(tool_name):
                tool = self.tool_registry.get_tool(tool_name)
                result = tool(**tool_args)
            else:
                result = {"error": f"Tool {tool_name} not found"}
        else:
            result = {"status": "executed", "action": action}
        
        # Record action
        action_record = {
            "action": action,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "autonomy_level": self.autonomy_level.value
        }
        self.action_history.append(action_record)
        self.state.add_step("action", action_record)
        
        return result
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation and update state.
        
        Args:
            observation: Observation data
            
        Returns:
            Processed observation
        """
        from ..common.agent_utils import standard_observe_pattern
        
        self.state.status = AgentStatus.OBSERVING
        
        # Update success rate if applicable
        if isinstance(observation, dict) and "success" in observation:
            self._update_success_rate(observation["success"])
        
        # Adapt if needed
        if self._should_adapt(observation):
            self._adapt(observation)
        
        # Use standard observe pattern
        return standard_observe_pattern(
            self,
            observation,
            importance=0.7,
            additional_data={
                "stored_in_memory": True,
                "success_rate": self.success_rate,
                "autonomy_level": self.autonomy_level.value
            }
        )
    
    def _update_success_rate(self, success: bool):
        """Update success rate based on action outcomes."""
        if not self.action_history:
            return
        
        recent_actions = self.action_history[-10:]  # Last 10 actions
        success_count = sum(
            1 for action in recent_actions
            if action.get("result", {}).get("success", False)
        )
        self.success_rate = success_count / len(recent_actions) if recent_actions else 0.0
    
    def _should_adapt(self, observation: Any) -> bool:
        """Determine if agent should adapt based on observation."""
        # Adapt if success rate is low
        if self.success_rate < 0.5 and len(self.action_history) > 5:
            return True
        
        # Adapt if observation indicates failure
        if isinstance(observation, dict) and observation.get("error"):
            return True
        
        return False
    
    def _adapt(self, observation: Any):
        """Adapt agent behavior based on observations."""
        adaptation = {
            "timestamp": datetime.now().isoformat(),
            "trigger": str(observation),
            "changes": []
        }
        
        # Lower confidence in similar future actions
        if self.current_plan:
            self.current_plan.confidence *= 0.9
        
        # Adjust autonomy level if needed
        if self.success_rate < 0.3:
            if self.autonomy_level == AutonomyLevel.FULLY_AUTONOMOUS:
                self.autonomy_level = AutonomyLevel.SEMI_AUTONOMOUS
                adaptation["changes"].append("Reduced autonomy level")
        
        self.adaptation_history.append(adaptation)
    
    def create_plan(self, goal: str) -> ActionPlan:
        """
        Create an action plan from reasoning.
        
        Args:
            goal: Goal description
            
        Returns:
            Action plan
        """
        # Generate reasoning
        reasoning_result = self.think(goal)
        reasoning_steps = [
            ReasoningStep(
                step_id=step["step_id"],
                reasoning=step["reasoning"],
                confidence=step["confidence"]
            )
            for step in reasoning_result["reasoning_steps"]
        ]
        
        # Generate action steps
        action_steps = self._generate_action_steps(goal, reasoning_steps)
        
        # Create plan
        plan = ActionPlan(
            plan_id=f"plan_{datetime.now().timestamp()}",
            goal=goal,
            steps=action_steps,
            reasoning_chain=reasoning_steps,
            confidence=reasoning_result["overall_confidence"]
        )
        
        self.current_plan = plan
        return plan
    
    def _generate_action_steps(
        self,
        goal: str,
        reasoning_steps: List[ReasoningStep]
    ) -> List[Dict[str, Any]]:
        """Generate action steps from reasoning."""
        steps = []
        
        # Extract action steps from reasoning
        for i, reasoning_step in enumerate(reasoning_steps):
            if "plan" in reasoning_step.step_id.lower():
                # Generate concrete actions
                steps.append({
                    "step_id": f"action_{i+1}",
                    "type": "tool_call",
                    "params": {
                        "tool": "search",
                        "args": {"query": goal}
                    },
                    "description": f"Execute action based on {reasoning_step.step_id}"
                })
        
        return steps
    
    def execute_plan(self, plan: Optional[ActionPlan] = None) -> Dict[str, Any]:
        """
        Execute an action plan.
        
        Args:
            plan: Plan to execute (uses current plan if None)
            
        Returns:
            Execution results
        """
        if plan is None:
            plan = self.current_plan
        
        if plan is None:
            return {"error": "No plan available"}
        
        results = []
        for step in plan.steps:
            result = self.act(step)
            results.append(result)
            
            # Observe result
            self.observe(result)
        
        return {
            "plan_id": plan.plan_id,
            "goal": plan.goal,
            "steps_executed": len(results),
            "results": results,
            "success": all(
                r.get("status") != "error" for r in results
            )
        }
    
    def add_goal(self, description: str, priority: int = 5) -> Goal:
        """
        Add a new goal.
        
        Args:
            description: Goal description
            priority: Goal priority (1-10)
            
        Returns:
            Created goal
        """
        goal = Goal(
            goal_id=f"goal_{datetime.now().timestamp()}",
            description=description,
            priority=priority
        )
        self.goals.append(goal)
        self.goals.sort(key=lambda g: g.priority, reverse=True)
        return goal
    
    def run(self, task: str) -> Dict[str, Any]:
        """
        Run complete reasoning-to-action pipeline.
        
        Args:
            task: Task to execute
            
        Returns:
            Final result
        """
        # Step 1: Think (reasoning)
        reasoning = self.think(task)
        
        # Step 2: Create plan
        plan = self.create_plan(task)
        
        # Step 3: Execute plan
        execution_result = self.execute_plan(plan)
        
        # Step 4: Final observation
        final_observation = self.observe(execution_result)
        
        self.state.status = AgentStatus.COMPLETED
        
        return {
            "task": task,
            "reasoning": reasoning,
            "plan": {
                "plan_id": plan.plan_id,
                "goal": plan.goal,
                "confidence": plan.confidence
            },
            "execution": execution_result,
            "final_observation": final_observation,
            "autonomy_level": self.autonomy_level.value
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "autonomy_level": self.autonomy_level.value,
            "goals_count": len(self.goals),
            "success_rate": self.success_rate,
            "reasoning_steps": len(self.reasoning_chain),
            "actions_executed": len(self.action_history),
            "adaptations": len(self.adaptation_history)
        })



