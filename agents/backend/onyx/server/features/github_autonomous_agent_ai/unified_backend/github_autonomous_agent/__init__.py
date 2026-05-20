"""
GitHub Autonomous Agent AI
===========================

Agente autónomo que se conecta a cualquier repositorio de GitHub y ejecuta
instrucciones de forma continua hasta que el usuario le indique detenerse.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "Autonomous AI agent for GitHub repositories"

# Try to import main components with error handling
try:
    from .core.agent import GitHubAutonomousAgent
except ImportError:
    GitHubAutonomousAgent = None

try:
    from .core.task_executor import TaskExecutor
except ImportError:
    TaskExecutor = None

try:
    from .api.agent_api import AgentAPI
except ImportError:
    AgentAPI = None

__all__ = [
    "GitHubAutonomousAgent",
    "TaskExecutor",
    "AgentAPI",
]

