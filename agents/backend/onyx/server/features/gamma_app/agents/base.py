"""
Agents Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import uuid4
from dataclasses import dataclass


class AgentState(str, Enum):
    """Agent state"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"


class AgentCapability(str, Enum):
    """Agent capabilities"""
    TEXT_GENERATION = "text_generation"
    CODE_EXECUTION = "code_execution"
    WEB_SEARCH = "web_search"
    DATA_ANALYSIS = "data_analysis"
    FILE_PROCESSING = "file_processing"
    TOOL_USAGE = "tool_usage"


@dataclass
class AgentMessage:
    """Message between agents"""
    id: str
    from_agent: str
    to_agent: str
    content: Dict[str, Any]
    timestamp: datetime
    message_type: str = "standard"


class AgentBase(ABC):
    """Base interface for agents"""
    
    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task"""
        pass
    
    @abstractmethod
    async def get_state(self) -> AgentState:
        """Get current agent state"""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the agent"""
        pass


class Agent:
    """Agent definition"""
    
    def __init__(
        self,
        name: str,
        agent_type: str,
        capabilities: List[AgentCapability],
        config: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid4())
        self.name = name
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.config = config or {}
        self.state = AgentState.IDLE
        self.created_at = datetime.utcnow()

