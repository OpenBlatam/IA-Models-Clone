"""
ReAct Agent Implementation
===========================

ReAct: Synergizing Reasoning and Acting in Language Models
Paper: https://arxiv.org/abs/2210.03629

The ReAct framework interleaves reasoning and acting:
1. Thought: Agent reasons about what to do
2. Action: Agent takes an action using a tool
3. Observation: Agent observes the result
4. Repeat until task is complete

Improvements:
- Better LLM integration with multiple providers
- Enhanced action parsing with multiple formats
- Robust error handling and retry logic
- Memory and context management
- Performance monitoring
- Streaming support for thoughts
- Better prompt engineering
"""

import logging
import re
import json
import time
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field

from ..common.agent_base import BaseAgent, AgentStatus
from ..common.tools import ToolRegistry, Tool

# Import helper classes
from .react_constants import (
    Defaults,
    ReActPatterns,
    ErrorMessages,
    ObservationTemplates,
    FinishKeywords,
    SearchKeywords,
    CalculationKeywords,
    ReadKeywords,
    WriteKeywords
)
from .react_result_builder import ResultBuilder
from .react_llm_adapter import LLMAdapter
from .react_action_parser import ActionParser, ActionFormat
from .react_observation_formatter import ObservationFormatter
from .react_retry_executor import RetryExecutor

logger = logging.getLogger(__name__)


@dataclass
class ReActStep:
    """Represents a single ReAct step (Thought -> Action -> Observation)."""
    step_number: int
    thought: str
    action: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    success: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_number": self.step_number,
            "thought": self.thought,
            "action": self.action,
            "observation": self.observation,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "success": self.success
        }


class ReActAgent(BaseAgent):
    """
    ReAct Agent: Interleaves reasoning and acting.
    
    The agent follows a pattern:
    Thought: [reasoning about what to do]
    Action: [tool_name]([parameters])
    Observation: [result from action]
    ... (repeat)
    
    Improvements:
    - Multiple LLM provider support
    - Enhanced action parsing
    - Error recovery and retry logic
    - Performance tracking
    - Memory integration
    - Streaming support
    """
    
    def __init__(
        self,
        name: str = "ReActAgent",
        llm: Optional[Any] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_iterations: int = 10,
        config: Optional[Dict[str, Any]] = None,
        enable_streaming: bool = False,
        retry_on_error: bool = True,
        max_retries: int = 3,
        timeout: Optional[float] = None
    ):
        """
        Initialize ReAct agent.
        
        Args:
            name: Agent name
            llm: Language model for reasoning (supports OpenAI, Anthropic, etc.)
            tool_registry: Registry of available tools
            max_iterations: Maximum reasoning-action cycles
            config: Additional configuration
            enable_streaming: Enable streaming of thoughts
            retry_on_error: Retry failed actions
            max_retries: Maximum retry attempts
            timeout: Timeout per step in seconds
        """
        super().__init__(name=name, llm=llm, config=config)
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_iterations = max_iterations
        self.enable_streaming = enable_streaming
        self.retry_on_error = retry_on_error
        self.max_retries = max_retries
        self.timeout = timeout
        
        # History tracking
        self.thought_history: List[str] = []
        self.action_history: List[Dict[str, Any]] = []
        self.steps: List[ReActStep] = []
        
        # Performance tracking
        self.step_times: List[float] = []
        self.total_tokens_used: int = 0
        
        # Configuration
        self.verbose = config.get("verbose", False) if config else False
        
        # Initialize helper classes
        self.llm_adapter = LLMAdapter(llm, config) if llm else None
        self.action_parser = ActionParser()
        self.observation_formatter = ObservationFormatter()
        self.retry_executor = RetryExecutor(
            max_retries=max_retries,
            retry_on_error=retry_on_error,
            timeout=timeout
        )
    
    def think(
        self, 
        observation: str, 
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate a thought/reasoning step.
        
        The thought should reason about:
        - What information is needed
        - What action to take next
        - How to interpret previous observations
        
        Args:
            observation: Current observation
            context: Additional context
            
        Returns:
            Thought/reasoning as string
        """
        start_time = time.time()
        
        # Build enhanced prompt for reasoning
        prompt = self._build_reasoning_prompt(observation, context)
        
        try:
            if self.llm_adapter:
                # Use LLM adapter to generate thought
                thought = self.llm_adapter.generate(prompt, stream=self.enable_streaming)
            else:
                # Fallback: enhanced template-based reasoning
                thought = self._enhanced_reasoning(observation, context)
            
            # Validate thought format
            thought = self._validate_thought(thought)
            
            self.thought_history.append(thought)
            
            if self.verbose:
                logger.info(f"[{self.name}] Thought: {thought[:100]}...")
            
            return thought
            
        except Exception as e:
            logger.error(f"Error in think step: {e}", exc_info=True)
            # Fallback to simple reasoning
            thought = self._simple_reasoning(observation, context)
            self.thought_history.append(thought)
            return thought
        finally:
            duration = time.time() - start_time
            self.step_times.append(duration)
    
    def act(
        self, 
        thought: str, 
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extract action from thought and execute it.
        
        Supports multiple action formats:
        - Function call: tool_name(param1=value1, param2=value2)
        - JSON: {"tool": "name", "parameters": {...}}
        - Natural language: "I will use tool_name with parameters..."
        
        Args:
            thought: The reasoning/thought
            context: Additional context
            
        Returns:
            Action result dictionary
        """
        start_time = time.time()
        
        try:
            # Parse action using helper
            action = self.action_parser.parse(thought)
            
            if not action:
                # No action found - task might be complete
                return ResultBuilder.finish_result(
                    reason=ErrorMessages.NO_ACTION_FOUND
                )
            
            tool_name = action.get("tool")
            parameters = action.get("parameters", {})
            
            # Validate tool exists
            if not self.tool_registry.get(tool_name):
                return ResultBuilder.tool_not_found_result(
                    tool_name=tool_name,
                    parameters=parameters,
                    available_tools=self.tool_registry.list_tools()
                )
            
            # Execute tool with retry logic using helper
            def execute_tool():
                return self.tool_registry.execute(tool_name, **parameters)
            
            result = self.retry_executor.execute_with_retry(execute_tool)
            
            # Build result using helper
            duration = time.time() - start_time
            action_result = ResultBuilder.success_result(
                tool=tool_name,
                parameters=parameters,
                result=result,
                duration=duration
            )
            
            self.action_history.append(action_result)
            
            if self.verbose:
                logger.info(
                    f"[{self.name}] Action: {tool_name}({parameters}) -> "
                    f"{'Success' if result.get('success') else 'Failed'}"
                )
            
            return action_result
            
        except Exception as e:
            logger.error(f"Error in act step: {e}", exc_info=True)
            return ResultBuilder.error_result(
                tool=None,
                parameters={},
                error=str(e),
                duration=time.time() - start_time
            )
    
    def observe(
        self, 
        action_result: Dict[str, Any]
    ) -> str:
        """
        Generate observation from action result.
        
        Converts the action result into a natural language observation
        that can be used in the next reasoning step. Includes error
        handling and context enrichment.
        
        Args:
            action_result: Result from action execution
            
        Returns:
            Natural language observation
        """
        if action_result.get("complete"):
            return "Task completed successfully."
        
        tool = action_result.get("tool")
        result = action_result.get("result", {})
        
        # Enhanced observation generation
        if result.get("success"):
            observation = self._format_success_observation(tool, result)
        else:
            observation = self._format_error_observation(tool, result)
        
        # Add context if available
        if "context" in result:
            observation += f" Context: {result['context']}"
        
        return observation
    
    def _build_reasoning_prompt(
        self, 
        observation: str, 
        context: Optional[Dict] = None
    ) -> str:
        """
        Build enhanced prompt for LLM reasoning.
        
        Includes:
        - Available tools with descriptions
        - Previous step history
        - Current observation
        - Task context
        - Format instructions
        """
        # Get available tools with descriptions
        available_tools = []
        if self.tool_registry:
            tools = self.tool_registry.list_tools()
            for tool_name in tools:
                tool = self.tool_registry.get(tool_name)
                if tool:
                    available_tools.append(
                        f"- {tool_name}: {tool.description}"
                    )
        
        # Build history context
        history_context = self._format_history()
        
        # Build prompt
        prompt = f"""You are a ReAct agent. You reason about tasks and take actions using tools.

Available tools:
{chr(10).join(available_tools) if available_tools else 'No tools available'}

Previous steps:
{history_context}

Current observation: {observation}

Task context: {context.get('task', 'N/A') if context else 'N/A'}

Think step by step about what to do next. Your response must follow this format:

Thought: [your reasoning about the current situation and what action to take]

Action: [choose one]
  Option 1: tool_name(param1="value1", param2="value2")
  Option 2: {{"tool": "tool_name", "parameters": {{"param1": "value1", "param2": "value2"}}}}
  Option 3: finish (if task is complete)

Your response:"""
        
        return prompt
    
    def _format_history(self) -> str:
        """Format thought and action history with enhanced details."""
        if not self.steps:
            return "No previous steps."
        
        history_lines = []
        for step in self.steps[-5:]:  # Last 5 steps for context
            history_lines.append(f"Step {step.step_number}:")
            history_lines.append(f"  Thought: {step.thought[:200]}...")
            if step.action:
                tool = step.action.get("tool", "N/A")
                params = step.action.get("parameters", {})
                history_lines.append(f"  Action: {tool}({params})")
            if step.observation:
                history_lines.append(f"  Observation: {step.observation[:200]}...")
            history_lines.append("")
        
        return "\n".join(history_lines)
    
    # Note: Action parsing, parameter parsing, and tool execution retry logic
    # have been moved to helper classes (ActionParser, RetryExecutor)
    # Observation formatting has been moved to ObservationFormatter
    
    def _validate_thought(self, thought: str) -> str:
        """Validate and clean thought string."""
        from .react_constants import ErrorMessages
        
        if not thought or not thought.strip():
            return ErrorMessages.INVALID_THOUGHT
        
        # Remove excessive whitespace
        thought = re.sub(r'\s+', ' ', thought).strip()
        
        # Ensure it starts with "Thought:" if it doesn't
        if not thought.lower().startswith("thought:"):
            thought = f"Thought: {thought}"
        
        return thought
    
    # Note: LLM calling logic has been moved to LLMAdapter helper class
    
    def _enhanced_reasoning(
        self, 
        observation: str, 
        context: Optional[Dict] = None
    ) -> str:
        """
        Enhanced template-based reasoning (fallback when no LLM).
        
        Uses pattern matching and heuristics to generate reasonable thoughts.
        """
        observation_lower = observation.lower()
        
        # Pattern-based reasoning using constants
        if any(word in observation_lower for word in SearchKeywords.KEYWORDS):
            query = self._extract_query(observation)
            return f"Thought: I need to search for information. Action: search(query=\"{query}\")"
        
        elif any(word in observation_lower for word in CalculationKeywords.KEYWORDS):
            expression = self._extract_expression(observation)
            return f"Thought: I need to perform a calculation. Action: calculator(expression=\"{expression}\")"
        
        elif any(word in observation_lower for word in ReadKeywords.KEYWORDS):
            resource = self._extract_resource(observation)
            return f"Thought: I need to retrieve information. Action: read(resource=\"{resource}\")"
        
        elif any(word in observation_lower for word in WriteKeywords.KEYWORDS):
            return f"Thought: I need to save information. Action: write(data=\"...\")"
        
        else:
            return "Thought: I need to understand the task better. Let me gather more information. Action: finish"
    
    def _simple_reasoning(
        self, 
        observation: str, 
        context: Optional[Dict] = None
    ) -> str:
        """Simple template-based reasoning (minimal fallback)."""
        return self._enhanced_reasoning(observation, context)
    
    def _extract_query(self, text: str) -> str:
        """Extract search query from text."""
        # Use patterns from constants
        for pattern in ReActPatterns.QUERY_EXTRACTION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Fallback: use first few words
        words = text.split()[:5]
        return " ".join(words)
    
    def _extract_expression(self, text: str) -> str:
        """Extract mathematical expression from text."""
        # Use pattern from constants
        match = re.search(ReActPatterns.MATH_EXPRESSION_PATTERN, text)
        if match:
            return match.group(1)
        return "2+2"
    
    def _extract_resource(self, text: str) -> str:
        """Extract resource identifier from text."""
        # Use patterns from constants
        for pattern in ReActPatterns.RESOURCE_EXTRACTION_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return "resource"
    
    def run(
        self, 
        task: str, 
        max_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run ReAct agent on a task with enhanced tracking.
        
        Args:
            task: Task description
            max_steps: Maximum steps (overrides max_iterations if provided)
            
        Returns:
            Final result with detailed execution information
        """
        start_time = time.time()
        max_steps = max_steps or self.max_iterations
        
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        self.state.add_step("task_start", {"task": task, "max_steps": max_steps})
        
        observation = f"Task: {task}"
        
        logger.info(f"[{self.name}] Starting task: {task}")
        
        for step_num in range(max_steps):
            step_start = time.time()
            step = ReActStep(step_number=step_num + 1)
            
            try:
                # Thought
                self.state.status = AgentStatus.THINKING
                thought = self.think(observation, {"step": step_num, "task": task})
                step.thought = thought
                self.state.add_step("think", thought)
                
                # Action
                self.state.status = AgentStatus.ACTING
                action_result = self.act(thought, {"step": step_num, "task": task})
                step.action = action_result
                self.state.add_step("act", action_result)
                
                # Observation
                self.state.status = AgentStatus.OBSERVING
                observation = self.observe(action_result)
                step.observation = observation
                step.success = action_result.get("result", {}).get("success", False)
                self.state.add_step("observe", observation)
                
                # Calculate step duration
                step.duration = time.time() - step_start
                
                # Store step
                self.steps.append(step)
                
                # Check completion
                if action_result.get("complete") or "complete" in observation.lower():
                    self.state.status = AgentStatus.COMPLETED
                    self.state.add_step(
                        "task_complete", 
                        {
                            "final_observation": observation,
                            "steps_taken": step_num + 1
                        }
                    )
                    logger.info(f"[{self.name}] Task completed in {step_num + 1} steps")
                    break
                
                # Check for timeout
                if self.timeout and (time.time() - start_time) > self.timeout:
                    logger.warning(f"[{self.name}] Task timeout after {self.timeout}s")
                    self.state.status = AgentStatus.ERROR
                    self.state.add_step("timeout", {"timeout": self.timeout})
                    break
                    
            except Exception as e:
                logger.error(f"[{self.name}] Error in step {step_num + 1}: {e}", exc_info=True)
                self.state.status = AgentStatus.ERROR
                step.success = False
                step.observation = f"Error: {str(e)}"
                self.state.add_step("error", {"error": str(e), "step": step_num + 1})
                self.steps.append(step)
                break
        
        total_duration = time.time() - start_time
        
        result = {
            "task": task,
            "final_state": self.state.to_dict(),
            "completed": self.state.status == AgentStatus.COMPLETED,
            "thoughts": self.thought_history,
            "actions": self.action_history,
            "steps": [step.to_dict() for step in self.steps],
            "performance": {
                "total_duration": total_duration,
                "steps_taken": len(self.steps),
                "avg_step_time": sum(self.step_times) / len(self.step_times) if self.step_times else 0,
                "total_tokens": self.total_tokens_used,
                "success_rate": sum(1 for s in self.steps if s.success) / len(self.steps) if self.steps else 0
            }
        }
        
        logger.info(
            f"[{self.name}] Task finished: "
            f"completed={result['completed']}, "
            f"steps={len(self.steps)}, "
            f"duration={total_duration:.2f}s"
        )
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_steps": len(self.steps),
            "avg_step_time": sum(self.step_times) / len(self.step_times) if self.step_times else 0,
            "total_tokens": self.total_tokens_used,
            "success_rate": sum(1 for s in self.steps if s.success) / len(self.steps) if self.steps else 0,
            "thoughts_count": len(self.thought_history),
            "actions_count": len(self.action_history)
        }
    
    def reset(self):
        """Reset agent state and history."""
        super().reset()
        self.thought_history.clear()
        self.action_history.clear()
        self.steps.clear()
        self.step_times.clear()
        self.total_tokens_used = 0
        logger.debug(f"[{self.name}] Agent reset")



