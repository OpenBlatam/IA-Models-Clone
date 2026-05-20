"""
Autonomous Operation Handler
Handles autonomous operations when no tasks are available
"""

import logging
from typing import Optional

from .learning_engine import LearningEngine
from .world_model import ContinualWorldModel
from .async_helpers import safe_async_call
from ..config import get_config

logger = logging.getLogger(__name__)


class AutonomousOperationHandler:
    """
    Handles autonomous operations when no tasks are available
    
    Separated from AutonomousLongTermAgent for better organization.
    Responsible for:
    - Self-initiated learning operations
    - World model-based planning
    - Autonomous decision making
    
    This allows the agent to continue operating and learning even when
    there are no explicit tasks in the queue.
    """
    
    def __init__(
        self,
        learning_engine: LearningEngine,
        world_model: Optional[ContinualWorldModel],
        agent_id: str,
        instruction: str
    ):
        """
        Initialize AutonomousOperationHandler
        
        Args:
            learning_engine: Engine for self-initiated learning
            world_model: Optional world model for planning
            agent_id: Agent identifier for logging
            instruction: Agent's main instruction/goal
        """
        self.learning_engine = learning_engine
        self.world_model = world_model
        self.agent_id = agent_id
        self.instruction = instruction
        self.config = get_config()
    
    async def execute(self) -> None:
        """
        Execute autonomous operation when no tasks are available.
        
        Coordinates two types of autonomous operations:
        1. Self-initiated learning: Records autonomous operation events
        2. World-based planning: Uses world model to plan next actions
        
        This allows the agent to continue operating and learning even when
        there are no explicit tasks in the queue.
        """
        # Self-initiated learning and adaptation
        if self.config.autonomous.learning_enabled:
            await self._perform_self_initiated_learning()
        
        # Self-planning based on world model (EvoAgent paper)
        if self.world_model:
            await self._perform_world_based_planning()
    
    async def _perform_self_initiated_learning(self) -> None:
        """
        Perform self-initiated learning operations.
        
        Records autonomous operation events in the learning engine to track
        when the agent operates without explicit tasks. This helps the agent
        learn from its autonomous behavior patterns.
        """
        await safe_async_call(
            self.learning_engine.record_event,
            "autonomous_operation",
            {"agent_id": self.agent_id},
            "success",
            error_message=f"Error in self-initiated learning for agent {self.agent_id}"
        )
    
    async def _perform_world_based_planning(self) -> None:
        """
        Perform self-planning based on world model (EvoAgent paper).
        
        Uses the continual world model to plan next actions based on:
        - Current world state
        - Agent's instruction/goal
        - Historical patterns
        
        If a plan is generated, recommended actions are logged for visibility.
        """
        plan = await safe_async_call(
            self.world_model.plan_based_on_world,
            goal=self.instruction,
            error_message=f"Error in autonomous planning for agent {self.agent_id}"
        )
        
        if plan:
            logger.debug(f"Autonomous planning: {plan.get('planning_strategy')}")
            
            # Execute recommended actions if any
            recommended_actions = plan.get("recommended_actions", [])
            if recommended_actions:
                logger.info(f"Recommended actions: {recommended_actions}")
