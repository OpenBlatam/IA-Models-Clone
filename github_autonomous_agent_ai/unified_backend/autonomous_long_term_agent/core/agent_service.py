"""
Agent Service
Business logic layer for agent operations
Separates business logic from API controllers and provides centralized error handling
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, TypeVar, Callable, Awaitable

from .agent import AutonomousLongTermAgent
from .agent_factory import create_agent
from .agent_registry import get_registry
from .task_converter import TaskConverter
from .exceptions import (
    AgentNotFoundError,
    TaskNotFoundError,
    AgentServiceError
)
from .validators import ServiceRequestValidator
from .async_helpers import safe_async_call
from ..api.v1.schemas.responses import TaskResponse
from ..config import settings

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AgentService:
    """
    Service layer for agent operations
    Separates business logic from API controllers and provides centralized error handling
    """
    
    def __init__(self):
        self.registry = get_registry()
        self._task_converter = TaskConverter()
    
    async def _get_agent_or_raise(self, agent_id: str) -> AutonomousLongTermAgent:
        """
        Get agent by ID or raise AgentNotFoundError
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent instance
            
        Raises:
            AgentNotFoundError: If agent not found
        """
        agent = await self.registry.get(agent_id)
        if not agent:
            raise AgentNotFoundError(agent_id)
        return agent
    
    async def _execute_with_service_error_handling(
        self,
        operation: Callable[[], Awaitable[T]],
        operation_name: str,
        agent_id: Optional[str] = None,
        **context
    ) -> T:
        """
        Execute an operation with consistent service error handling
        
        Args:
            operation: Async callable to execute
            operation_name: Name of operation for error messages
            agent_id: Optional agent ID for context
            **context: Additional context for error messages
            
        Returns:
            Result of the operation
            
        Raises:
            AgentNotFoundError: If agent not found (re-raised)
            TaskNotFoundError: If task not found (re-raised)
            AgentServiceError: If operation fails
        """
        try:
            return await operation()
        except (AgentNotFoundError, TaskNotFoundError):
            # Re-raise domain exceptions as-is
            raise
        except Exception as e:
            logger.error(
                f"Error in {operation_name}",
                exc_info=True,
                extra={"agent_id": agent_id, **context}
            )
            raise AgentServiceError(
                f"Failed to {operation_name}: {str(e)}",
                operation=operation_name,
                agent_id=agent_id,
                **context
            )
    
    async def create_and_start_agent(
        self,
        agent_id: Optional[str] = None,
        instruction: Optional[str] = None,
        enhanced: bool = True
    ) -> AutonomousLongTermAgent:
        """
        Create and start a new agent
        
        Args:
            agent_id: Optional custom agent ID
            instruction: Agent instruction
            enhanced: Use enhanced version
        
        Returns:
            Created agent instance
        
        Raises:
            AgentServiceError: If agent creation fails
        """
        # Validate request
        ServiceRequestValidator.validate_create_agent_request(
            agent_id, instruction, enhanced
        )
        
        async def _create():
            agent = create_agent(
                agent_id=agent_id,
                instruction=instruction,
                enhanced=enhanced
            )
            await self.registry.register(agent)
            await agent.start()
            logger.info(f"Created and started agent {agent.agent_id}")
            return agent
        
        return await self._execute_with_service_error_handling(
            _create,
            "create and start agent",
            agent_id=agent_id,
            enhanced=enhanced
        )
    
    async def get_agent(self, agent_id: str) -> AutonomousLongTermAgent:
        """
        Get agent by ID
        
        Args:
            agent_id: Agent ID
        
        Returns:
            Agent instance
        
        Raises:
            AgentNotFoundError: If agent not found
        """
        return await self._get_agent_or_raise(agent_id)
    
    async def stop_agent(self, agent_id: str) -> None:
        """
        Stop an agent
        
        Args:
            agent_id: Agent ID
        
        Raises:
            AgentNotFoundError: If agent not found
            AgentServiceError: If stop fails
        """
        async def _stop():
            agent = await self._get_agent_or_raise(agent_id)
            await agent.stop()
            await self.registry.remove(agent_id)
            logger.info(f"Stopped agent {agent_id}")
        
        await self._execute_with_service_error_handling(
            _stop,
            "stop agent",
            agent_id=agent_id
        )
    
    async def _execute_agent_operation(
        self,
        agent_id: str,
        operation_name: str,
        operation: Callable[[AutonomousLongTermAgent], Awaitable[T]],
        **context
    ) -> T:
        """
        Execute an operation on an agent with consistent error handling.
        Centralizes the pattern of getting agent and executing operation.
        
        Args:
            agent_id: Agent ID
            operation_name: Name of operation for logging/errors
            operation: Async function that takes agent and returns result
            **context: Additional context for error messages
            
        Returns:
            Result of the operation
            
        Raises:
            AgentNotFoundError: If agent not found
            AgentServiceError: If operation fails
        """
        async def _execute():
            agent = await self._get_agent_or_raise(agent_id)
            result = await operation(agent)
            logger.info(f"{operation_name} agent {agent_id}")
            return result
        
        return await self._execute_with_service_error_handling(
            _execute,
            operation_name,
            agent_id=agent_id,
            **context
        )
    
    async def pause_agent(self, agent_id: str) -> None:
        """
        Pause an agent
        
        Args:
            agent_id: Agent ID
        
        Raises:
            AgentNotFoundError: If agent not found
            AgentServiceError: If pause fails
        """
        await self._execute_agent_operation(
            agent_id,
            "pause agent",
            lambda agent: agent.pause()
        )
    
    async def resume_agent(self, agent_id: str) -> None:
        """
        Resume a paused agent
        
        Args:
            agent_id: Agent ID
        
        Raises:
            AgentNotFoundError: If agent not found
            AgentServiceError: If resume fails
        """
        await self._execute_agent_operation(
            agent_id,
            "resume agent",
            lambda agent: agent.resume()
        )
    
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get agent status
        
        Args:
            agent_id: Agent ID
        
        Returns:
            Agent status dictionary
        
        Raises:
            AgentNotFoundError: If agent not found
            AgentServiceError: If status retrieval fails
        """
        return await self._execute_agent_operation(
            agent_id,
            "get agent status",
            lambda agent: agent.get_status()
        )
    
    async def get_agent_health(self, agent_id: str) -> Dict[str, Any]:
        """
        Get agent health
        
        Args:
            agent_id: Agent ID
        
        Returns:
            Agent health dictionary
        
        Raises:
            AgentNotFoundError: If agent not found
            AgentServiceError: If health check fails
        """
        return await self._execute_agent_operation(
            agent_id,
            "get agent health",
            lambda agent: agent.get_health()
        )
    
    async def add_task(
        self,
        agent_id: str,
        instruction: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add task to agent
        
        Args:
            agent_id: Agent ID
            instruction: Task instruction
            metadata: Optional task metadata
        
        Returns:
            Task ID
        
        Raises:
            AgentServiceError: If validation fails
        """
        # Validate request
        ServiceRequestValidator.validate_add_task_request(
            agent_id, instruction, metadata
        )
        
        Raises:
            AgentNotFoundError: If agent not found
            AgentServiceError: If task addition fails
        """
        async def _add_task(agent: AutonomousLongTermAgent) -> str:
            task_id = await agent.add_task(instruction, metadata)
            logger.info(f"Added task {task_id} to agent {agent_id}")
            return task_id
        
        return await self._execute_agent_operation(
            agent_id,
            "add task",
            _add_task,
            has_metadata=metadata is not None
        )
    
    async def get_task(self, agent_id: str, task_id: str) -> TaskResponse:
        """
        Get task by ID
        
        Args:
            agent_id: Agent ID
            task_id: Task ID
        
        Returns:
            TaskResponse
        
        Raises:
            AgentNotFoundError: If agent not found
            TaskNotFoundError: If task not found
            AgentServiceError: If task retrieval fails
        """
        async def _get_task():
            agent = await self._get_agent_or_raise(agent_id)
            task = await agent.task_queue.get_task(task_id)
            if not task:
                raise TaskNotFoundError(task_id, agent_id)
            return self._task_converter.to_response(task)
        
        return await self._execute_with_service_error_handling(
            _get_task,
            "get task",
            agent_id=agent_id,
            task_id=task_id
        )
    
    async def list_tasks(
        self,
        agent_id: str,
        status: Optional[str] = None
    ) -> List[TaskResponse]:
        """
        List tasks for an agent
        
        Args:
            agent_id: Agent ID
            status: Optional status filter
        
        Returns:
            List of TaskResponse
        
        Raises:
            AgentNotFoundError: If agent not found
            AgentServiceError: If task listing fails or status is invalid
        """
        async def _list_tasks():
            agent = await self._get_agent_or_raise(agent_id)
            task_status = self._task_converter.parse_status(status)
            tasks = await agent.task_queue.list_tasks(status=task_status)
            return self._task_converter.to_response_list(tasks)
        
        try:
            return await self._execute_with_service_error_handling(
                _list_tasks,
                "list tasks",
                agent_id=agent_id,
                status_filter=status
            )
        except ValueError as e:
            # Convert ValueError to AgentServiceError for invalid status
            raise AgentServiceError(
                f"Invalid status: {str(e)}",
                operation="list_tasks",
                agent_id=agent_id,
                invalid_status=status
            )
    
    async def list_all_agents(self) -> List[Dict[str, Any]]:
        """
        List all agents with their status
        
        Returns:
            List of agent status dictionaries
        """
        agents = await self.registry.list_all()
        agents_list = []
        
        for agent in agents:
            status = await safe_async_call(
                agent.get_status,
                error_message=f"Error getting status for agent {agent.agent_id}"
            )
            if status:
                agents_list.append(status)
        
        return agents_list
    
    async def stop_all_agents(self) -> int:
        """
        Stop all agents
        
        Returns:
            Number of agents stopped
        """
        agents = await self.registry.list_all()
        stopped_count = 0
        
        for agent in agents:
            result = await safe_async_call(
                agent.stop,
                error_message=f"Error stopping agent {agent.agent_id}"
            )
            if result is not None:
                stopped_count += 1
        
        await self.registry.clear()
        logger.info(f"Stopped {stopped_count} agents")
        return stopped_count
    
    async def create_parallel_agents(
        self,
        count: int,
        instruction: Optional[str] = None,
        enhanced: bool = True
    ) -> List[str]:
        """
        Create multiple agents in parallel (actually parallel, not sequential)
        
        Args:
            count: Number of agents to create
            instruction: Instruction for all agents
            enhanced: Use enhanced version
        
        Returns:
            List of agent IDs
        
        Raises:
            AgentServiceError: If count exceeds limit or creation fails
        """
        # Validate request
        ServiceRequestValidator.validate_parallel_agents_request(
            count, instruction, enhanced
        )
        
        if count > settings.agent_max_parallel_instances:
            raise AgentServiceError(
                f"Maximum {settings.agent_max_parallel_instances} parallel agents allowed",
                operation="create_parallel_agents",
                requested_count=count,
                max_allowed=settings.agent_max_parallel_instances
            )
        
        async def _create_agent_with_index(index: int) -> str:
            """Create a single agent and return its ID"""
            agent = await self.create_and_start_agent(
                instruction=instruction,
                enhanced=enhanced
            )
            logger.info(f"Created parallel agent {agent.agent_id} ({index+1}/{count})")
            return agent.agent_id
        
        # Create all agents in parallel using asyncio.gather
        try:
            agent_ids = await asyncio.gather(
                *[_create_agent_with_index(i) for i in range(count)],
                return_exceptions=False
            )
            logger.info(f"Successfully created {len(agent_ids)} parallel agents")
            return agent_ids
        except Exception as e:
            logger.error(f"Error creating parallel agents: {e}", exc_info=True)
            raise AgentServiceError(
                f"Failed to create parallel agents: {str(e)}",
                operation="create_parallel_agents",
                count=count,
                enhanced=enhanced
            )


# Global service instance
_global_service = AgentService()


def get_agent_service() -> AgentService:
    """Get global agent service"""
    return _global_service

