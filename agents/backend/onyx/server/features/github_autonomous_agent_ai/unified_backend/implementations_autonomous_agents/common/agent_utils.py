"""
Agent Utilities
===============

Common utility functions for agent implementations.
"""

from typing import Dict, Any, Optional
from .agent_base import BaseAgent, AgentStatus


def standard_run_pattern(
    agent: BaseAgent,
    task: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Standard run pattern for agents.
    
    Implements the common think -> act -> observe pattern.
    
    Args:
        agent: Agent instance
        task: Task description
        context: Optional context
        
    Returns:
        Result dictionary
    """
    context = context or {}
    
    # Think
    thinking = agent.think(task, context)
    
    # Prepare action from thinking or context
    action = context.get("action", {})
    if not action or "type" not in action:
        # Extract action from thinking if available
        if isinstance(thinking, dict):
            action_type = thinking.get("action_type") or thinking.get("selected_action") or "execute"
            action = {
                "type": action_type,
                **thinking.get("action_params", {})
            }
        else:
            action = {"type": "execute"}
    
    # Act
    action_result = agent.act(action)
    
    # Prepare observation
    observation = context.get("observation", {})
    if not observation:
        # Create default observation
        observation = {
            "result": "completed",
            "action_result": action_result
        }
    
    # Observe
    observation_result = agent.observe(observation)
    
    agent.state.status = AgentStatus.COMPLETED
    
    return {
        "task": task,
        "thinking": thinking,
        "action": action_result,
        "observation": observation_result,
        "final_state": agent.state.to_dict(),
        "completed": agent.state.status == AgentStatus.COMPLETED
    }


def create_status_dict(
    agent: BaseAgent,
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create standard status dictionary for agent.
    
    Args:
        agent: Agent instance
        additional_info: Optional additional information to include
        
    Returns:
        Status dictionary
    """
    status = {
        "name": agent.name,
        "status": agent.state.status.value,
        "current_task": agent.state.current_task,
        "steps_count": len(agent.state.history)
    }
    
    if additional_info:
        status.update(additional_info)
    
    return status


def validate_agent_config(
    config: Optional[Dict[str, Any]],
    defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate and complete agent configuration.
    
    Args:
        config: Configuration dictionary
        defaults: Default values
        
    Returns:
        Validated and completed configuration
    """
    config = config or {}
    defaults = defaults or {}
    
    # Merge defaults with provided config
    validated = defaults.copy()
    validated.update(config)
    
    return validated


def standard_observe_pattern(
    agent: BaseAgent,
    observation: Any,
    importance: float = 0.7,
    additional_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Standard observe pattern for agents.
    
    Handles common observation processing:
    - Stores in episodic memory
    - Updates agent state
    - Returns processed observation
    
    Args:
        agent: Agent instance
        observation: Observation data
        importance: Importance score for memory (0.0-1.0)
        additional_data: Optional additional data to include in result
        
    Returns:
        Processed observation dictionary
    """
    from datetime import datetime
    
    # Store in episodic memory if available
    if hasattr(agent, 'episodic_memory'):
        agent.episodic_memory.add_experience(
            content=str(observation),
            importance=importance
        )
    
    # Prepare processed observation
    processed = {
        "observation": observation,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add additional data if provided
    if additional_data:
        processed.update(additional_data)
    
    # Add step to agent state
    agent.state.add_step("observe", processed)
    
    return processed


