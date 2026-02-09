"""
Validators
Input validation utilities for agent operations
"""

import logging
from typing import Dict, Any, Optional, List

from .exceptions import AgentServiceError
from ..config import settings

logger = logging.getLogger(__name__)


class AgentValidator:
    """
    Validator for agent-related operations
    Centralizes validation logic
    """
    
    @staticmethod
    def validate_agent_id(agent_id: str) -> None:
        """
        Validate agent ID format
        
        Args:
            agent_id: Agent ID to validate
        
        Raises:
            AgentServiceError: If agent ID is invalid
        """
        if not agent_id or not isinstance(agent_id, str):
            raise AgentServiceError("Agent ID must be a non-empty string")
        
        if len(agent_id) > 100:
            raise AgentServiceError("Agent ID must be 100 characters or less")
    
    @staticmethod
    def validate_instruction(instruction: Optional[str]) -> None:
        """
        Validate agent instruction
        
        Args:
            instruction: Instruction to validate
        
        Raises:
            AgentServiceError: If instruction is invalid
        """
        if instruction is not None:
            if not isinstance(instruction, str):
                raise AgentServiceError("Instruction must be a string")
            
            if len(instruction) > 10000:
                raise AgentServiceError("Instruction must be 10000 characters or less")
    
    @staticmethod
    def validate_parallel_agent_count(count: int) -> None:
        """
        Validate number of parallel agents
        
        Args:
            count: Number of agents to create
        
        Raises:
            AgentServiceError: If count is invalid
        """
        if not isinstance(count, int):
            raise AgentServiceError("Count must be an integer")
        
        if count < 1:
            raise AgentServiceError("Count must be at least 1")
        
        max_instances = settings.agent_max_parallel_instances
        if count > max_instances:
            raise AgentServiceError(
                f"Count ({count}) exceeds maximum parallel instances ({max_instances})"
            )
    
    @staticmethod
    def validate_task_instruction(instruction: str) -> None:
        """
        Validate task instruction
        
        Args:
            instruction: Task instruction to validate
        
        Raises:
            AgentServiceError: If instruction is invalid
        """
        if not instruction or not isinstance(instruction, str):
            raise AgentServiceError("Task instruction must be a non-empty string")
        
        if len(instruction) > 5000:
            raise AgentServiceError("Task instruction must be 5000 characters or less")
    
    @staticmethod
    def validate_metadata(metadata: Optional[Dict[str, Any]]) -> None:
        """
        Validate task metadata
        
        Args:
            metadata: Metadata to validate
        
        Raises:
            AgentServiceError: If metadata is invalid
        """
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise AgentServiceError("Metadata must be a dictionary")
            
            # Check for circular references or very deep nesting
            if len(str(metadata)) > 10000:
                raise AgentServiceError("Metadata is too large (max 10000 characters when serialized)")


class ServiceRequestValidator:
    """
    Validator for service request parameters
    """
    
    @staticmethod
    def validate_create_agent_request(
        agent_id: Optional[str],
        instruction: Optional[str],
        enhanced: bool
    ) -> None:
        """
        Validate create agent request
        
        Args:
            agent_id: Optional agent ID
            instruction: Optional instruction
            enhanced: Whether to use enhanced version
        
        Raises:
            AgentServiceError: If validation fails
        """
        if agent_id is not None:
            AgentValidator.validate_agent_id(agent_id)
        
        AgentValidator.validate_instruction(instruction)
        
        if not isinstance(enhanced, bool):
            raise AgentServiceError("Enhanced must be a boolean")
    
    @staticmethod
    def validate_parallel_agents_request(
        count: int,
        instruction: Optional[str],
        enhanced: bool
    ) -> None:
        """
        Validate parallel agents request
        
        Args:
            count: Number of agents
            instruction: Optional instruction
            enhanced: Whether to use enhanced version
        
        Raises:
            AgentServiceError: If validation fails
        """
        AgentValidator.validate_parallel_agent_count(count)
        AgentValidator.validate_instruction(instruction)
        
        if not isinstance(enhanced, bool):
            raise AgentServiceError("Enhanced must be a boolean")
    
    @staticmethod
    def validate_add_task_request(
        agent_id: str,
        instruction: str,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """
        Validate add task request
        
        Args:
            agent_id: Agent ID
            instruction: Task instruction
            metadata: Optional metadata
        
        Raises:
            AgentServiceError: If validation fails
        """
        AgentValidator.validate_agent_id(agent_id)
        AgentValidator.validate_task_instruction(instruction)
        AgentValidator.validate_metadata(metadata)

