"""
Agent Registry
Manages agent instances and provides thread-safe access
"""

import asyncio
import logging
from typing import Dict, Optional, List
from .agent import AutonomousLongTermAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Thread-safe registry for managing agent instances
    """
    
    def __init__(self):
        self._agents: Dict[str, AutonomousLongTermAgent] = {}
        self._lock = asyncio.Lock()
    
    async def register(self, agent: AutonomousLongTermAgent) -> None:
        """
        Register an agent
        
        Args:
            agent: Agent instance to register
        """
        async with self._lock:
            self._agents[agent.agent_id] = agent
            logger.info(f"Registered agent {agent.agent_id}")
    
    async def get(self, agent_id: str) -> Optional[AutonomousLongTermAgent]:
        """
        Get agent by ID
        
        Args:
            agent_id: Agent ID
        
        Returns:
            Agent instance or None if not found
        """
        async with self._lock:
            return self._agents.get(agent_id)
    
    async def remove(self, agent_id: str) -> bool:
        """
        Remove agent from registry
        
        Args:
            agent_id: Agent ID
        
        Returns:
            True if removed, False if not found
        """
        async with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
                logger.info(f"Removed agent {agent_id} from registry")
                return True
            return False
    
    async def list_all(self) -> List[AutonomousLongTermAgent]:
        """
        List all registered agents
        
        Returns:
            List of agent instances
        """
        async with self._lock:
            return list(self._agents.values())
    
    async def get_all_ids(self) -> List[str]:
        """
        Get all agent IDs
        
        Returns:
            List of agent IDs
        """
        async with self._lock:
            return list(self._agents.keys())
    
    async def count(self) -> int:
        """
        Get number of registered agents
        
        Returns:
            Number of agents
        """
        async with self._lock:
            return len(self._agents)
    
    async def clear(self) -> int:
        """
        Clear all agents from registry
        
        Returns:
            Number of agents removed
        """
        async with self._lock:
            count = len(self._agents)
            self._agents.clear()
            logger.info(f"Cleared {count} agents from registry")
            return count
    
    async def exists(self, agent_id: str) -> bool:
        """
        Check if agent exists
        
        Args:
            agent_id: Agent ID
        
        Returns:
            True if agent exists
        """
        async with self._lock:
            return agent_id in self._agents


# Global registry instance
_global_registry = AgentRegistry()


def get_registry() -> AgentRegistry:
    """Get global agent registry"""
    return _global_registry




