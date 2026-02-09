from typing import Dict, Any, Optional, TYPE_CHECKING
import structlog
from ..schemas import AgentContext

if TYPE_CHECKING:
    from ..core.event_bus import EventBus
    from ..core.memory import SharedMemory

logger = structlog.get_logger()

class BaseAgent:
    """
    Base class for all agents in the system.
    Provides standardized logging, event bus access, and shared memory.
    """

    def __init__(self, name: str, role: str, event_bus: Optional["EventBus"] = None, memory: Optional["SharedMemory"] = None):
        """
        Initialize the agent.

        Args:
            name (str): The name of the agent instance.
            role (str): The role description of the agent.
            event_bus (EventBus): Optional reference to the system event bus.
            memory (SharedMemory): Optional reference to the system shared memory.
        """
        self.name = name
        self.role = role
        self.event_bus = event_bus
        self.memory = memory
        self.logger = logger.bind(agent_name=name, agent_role=role)
        
    def log(self, message: str, level: str = "info", **kwargs):
        """
        Logs a message using the agent's logger.

        Args:
            message (str): The message to log.
            level (str): The log level ('debug', 'info', 'warning', 'error'). Defaults to 'info'.
            **kwargs: Additional context to log.
        """
        lvl = level.lower()
        if lvl == "debug":
            self.logger.debug(message, **kwargs)
        elif lvl == "warning":
            self.logger.warning(message, **kwargs)
        elif lvl == "error":
            self.logger.error(message, **kwargs)
        else:
            self.logger.info(message, **kwargs)

    def publish_event(self, topic: str, data: Dict[str, Any]):
        """
        Publishes an event to the system event bus.
        """
        if self.event_bus:
            self.event_bus.publish(topic, {"source": self.name, **data})

    def remember(self, content: str, tags: list = None):
        """
        Stores a memory in the shared memory stream.
        """
        if self.memory:
            self.memory.add_memory(content, self.name, tags)

    def recall(self, query: str) -> list:
        """
        Retrieves relevant memories from the shared memory stream.
        """
        if self.memory:
            return self.memory.retrieve_relevant(query)
        return []

    def run(self, context: AgentContext) -> Dict[str, Any]:
        """
        Main execution method for the agent.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Agents must implement the run method.")
