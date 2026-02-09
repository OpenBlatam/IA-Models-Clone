"""
GitHub Autonomous Agent AI
===========================

Agente autónomo que se conecta a cualquier repositorio de GitHub y ejecuta
instrucciones de forma continua hasta que el usuario le indique parar.

Características:
- Conexión a cualquier repositorio de GitHub
- Ejecución continua de tareas
- Funciona incluso con la computadora apagada (ejecución en servidor/cloud)
- Control desde frontend
- Sistema de cola de tareas persistente
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "Autonomous AI agent for GitHub repositories with continuous task execution and persistent queue system"

# Try to import components with error handling
try:
    from .core.agent import GitHubAutonomousAgent
except ImportError:
    GitHubAutonomousAgent = None

try:
    from .api import create_app
except ImportError:
    create_app = None

__all__ = [
    "GitHubAutonomousAgent",
    "create_app",
]
