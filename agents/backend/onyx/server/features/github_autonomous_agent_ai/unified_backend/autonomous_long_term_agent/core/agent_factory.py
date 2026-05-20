"""
Agent Factory - Factory for creating agent instances
Supports both standard and enhanced versions
"""

import logging
from typing import Optional, Dict, Any, Union

from ..config import settings
from .agent import AutonomousLongTermAgent
from .agent_enhanced import EnhancedAutonomousAgent

logger = logging.getLogger(__name__)


def create_agent(
    agent_id: Optional[str] = None,
    instruction: Optional[str] = None,
    enhanced: bool = True,
    **kwargs
) -> Union[AutonomousLongTermAgent, EnhancedAutonomousAgent]:
    """
    Create an agent instance (standard or enhanced).
    
    Args:
        agent_id: Optional agent ID
        instruction: Optional agent instruction
        enhanced: Use enhanced version with papers (default: True)
        **kwargs: Additional parameters
    
    Returns:
        Agent instance (AutonomousLongTermAgent or EnhancedAutonomousAgent)
        
    Note:
        If enhanced=True and EnhancedAutonomousAgent creation fails,
        automatically falls back to standard AutonomousLongTermAgent.
    """
    if not enhanced:
        return AutonomousLongTermAgent(
            agent_id=agent_id,
            instruction=instruction,
            **kwargs
        )
    
    # Try to create enhanced agent, fallback to standard on error
    try:
        return EnhancedAutonomousAgent(
            agent_id=agent_id,
            instruction=instruction,
            enable_papers=True,
            **kwargs
        )
    except Exception as e:
        logger.warning(
            f"Failed to create enhanced agent, falling back to standard: {e}",
            exc_info=True
        )
        return AutonomousLongTermAgent(
            agent_id=agent_id,
            instruction=instruction,
            **kwargs
        )


def create_standard_agent(
    agent_id: Optional[str] = None,
    instruction: Optional[str] = None,
    **kwargs
) -> AutonomousLongTermAgent:
    """
    Create a standard agent instance.
    
    Note: This is a convenience wrapper around create_agent(enhanced=False).
    For new code, prefer using create_agent(enhanced=False) directly.
    """
    return create_agent(
        agent_id=agent_id,
        instruction=instruction,
        enhanced=False,
        **kwargs
    )


def create_enhanced_agent(
    agent_id: Optional[str] = None,
    instruction: Optional[str] = None,
    **kwargs
) -> AutonomousLongTermAgent:
    """
    Create an enhanced agent instance with paper optimizations.
    
    Note: This is a convenience wrapper around create_agent(enhanced=True).
    For new code, prefer using create_agent(enhanced=True) directly.
    """
    return create_agent(
        agent_id=agent_id,
        instruction=instruction,
        enhanced=True,
        **kwargs
    )




